"""Compile a normalized instruction document into a real parametric pattern.

This is the part that makes the dataset trustworthy. Each drafting action is
turned into **the same tool calls the agent emits** and executed through the
same :func:`backend.agent.dispatch_tool` dispatcher against a live
:class:`~seamly_engine.operations.PatternSession`. So a step only lands in the
dataset if the engine actually built the geometry it describes: the tool name,
the argument shape, and the resulting coordinates are all verified rather than
asserted.

That gives three artefacts from one pass:

1. a parametric ``.sm2d`` for the garment (measurements become ``#`` variables,
   so it re-drafts for any wearer);
2. per-step ``(instruction, state_before, tool_calls)`` records — few-shot
   examples and an eval set in the agent's own vocabulary;
3. an exact list of the steps the document under-determines.

Conventions worth knowing: screen angles are degrees counter-clockwise with
**y pointing down**, so `270` is downward and `0` is to the right. Panels become
separate draft blocks laid out side by side, and point names are suffixed per
panel (``C`` → ``C_L``) because Seamly point names are pattern-global.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "engine", _ROOT / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from seamly_engine import geometry as geo  # noqa: E402
from seamly_engine.model import DraftBlock, Pattern  # noqa: E402
from seamly_engine.operations import PatternSession  # noqa: E402
from seamly_engine.state import compact_state  # noqa: E402

from .normalize import Document  # noqa: E402
from .schema import MEASUREMENT_DEFAULTS, DraftAction, Issue  # noqa: E402

#: Panels are laid out left-to-right this far apart (source units) so blocks
#: don't overlap in the render.
PANEL_PITCH = 45.0

_SUFFIX = {"back": "", "front_left": "_L", "front_right": "_R", "bottom": "_Q"}


@dataclass
class Panel:
    name: str
    block_index: int
    suffix: str
    flip_y: bool = False
    ids: dict[str, int] = field(default_factory=dict)
    midpoint_of: dict[str, tuple[str, str]] = field(default_factory=dict)
    steps: list[DraftAction] = field(default_factory=list)  # replayed when traced


@dataclass
class CompiledStep:
    action: DraftAction
    instruction: str
    tool_calls: list[dict]
    ok: bool
    message: str
    state_before: dict | None = None
    skipped: str = ""
    #: Deep copy of the session as it stood *after* this step. Only populated
    #: when ``keep_sessions=True`` — the evaluator needs it to hand an agent the
    #: exact pattern the reference had at each point in the episode.
    session_after: PatternSession | None = None

    def as_dict(self) -> dict:
        return {
            "garment": self.action.garment, "panel": self.action.panel,
            "step_index": self.action.step_index,
            "instruction": self.instruction,
            "source_image": self.action.source_image,
            "tool_calls": self.tool_calls,
            "verified": self.ok,
            "message": self.message,
            "skipped": self.skipped,
            "issues": [i.as_dict() for i in self.action.issues],
        }


@dataclass
class CompileResult:
    session: PatternSession
    steps: list[CompiledStep]
    variables: dict[str, float]
    #: The session before any step ran — blocks and declared variables, no
    #: geometry. Only populated with ``keep_sessions=True``; the evaluator needs
    #: it as the starting state of an episode.
    initial_session: PatternSession | None = None

    @property
    def verified(self) -> list[CompiledStep]:
        return [s for s in self.steps if s.ok]

    @property
    def unverified(self) -> list[CompiledStep]:
        return [s for s in self.steps if not s.ok]


def compile_document(doc: Document, *, name: str | None = None,
                     unit: str = "inch", capture_state: bool = True,
                     keep_sessions: bool = False) -> CompileResult:
    return _Compiler(doc, name=name, unit=unit, capture_state=capture_state,
                     keep_sessions=keep_sessions).run()


class _Compiler:
    def __init__(self, doc: Document, *, name: str | None, unit: str,
                 capture_state: bool, keep_sessions: bool = False):
        self.doc = doc
        self.capture_state = capture_state
        self.keep_sessions = keep_sessions
        pattern = Pattern(unit=unit, pattern_name=name or doc.garment.replace("_", " ").title())
        self.session = PatternSession(pattern)
        self.panels: dict[str, Panel] = {}
        self.steps: list[CompiledStep] = []
        self.variables: dict[str, float] = {}
        self._replaying = False
        self._prebuilt: str | None = None  # set when a handler already executed
        self._notes: list[Issue] = []      # non-blocking findings for this step

    # --- driver ----------------------------------------------------------
    def run(self) -> CompileResult:
        self._declare_variables()
        for panel_name in self.doc.panels:
            self._make_panel(panel_name)
        initial = copy.deepcopy(self.session) if self.keep_sessions else None
        for action in self.doc.actions:
            self._apply(action)
        return CompileResult(self.session, self.steps, self.variables, initial)

    def _declare_variables(self) -> None:
        """Every measurement the document names becomes a pattern variable.

        Values are the placeholders from :mod:`dataset.schema` — the compiled
        pattern is parametric, so changing one variable re-drafts the garment.
        """
        for measure in sorted(self.doc.measurements):
            value = MEASUREMENT_DEFAULTS.get(measure)
            if value is None:
                value = 1.0
            res = self.session.set_variable(
                f"#{measure}", f"{value:g}",
                description=f"{measure.replace('_', ' ')} (placeholder — replace "
                            f"with the wearer's measurement)")
            if res.ok:
                self.variables[f"#{measure}"] = value

    def _make_panel(self, name: str) -> Panel:
        if name in self.panels:
            return self.panels[name]
        index = len(self.session.pattern.draft_blocks)
        self.session.pattern.draft_blocks.append(DraftBlock(name=name))
        suffix = _SUFFIX.get(name, f"_{name[:2].upper()}")
        flip = any(a.panel == name and a.action == "TRANSFORM"
                   and "flip" in (a.value_raw or "") for a in self.doc.actions)
        panel = Panel(name=name, block_index=index, suffix=suffix, flip_y=flip)
        self.panels[name] = panel
        return panel

    # --- per-action ------------------------------------------------------
    def _apply(self, action: DraftAction) -> None:
        panel = self._make_panel(action.panel)
        if action.duplicate_of is not None:
            self._record(action, [], False,
                         skipped=f"restates step {action.duplicate_of:g}")
            return
        if action.skipped:
            self._record(action, [], False,
                         skipped=str(action.overrides.get("why", "retired by override")))
            return

        handler = getattr(self, f"_do_{action.action.lower()}", None)
        if handler is None:
            self._record(action, [], False, skipped=f"no compiler for {action.action}")
            return

        state_before = self._state() if self.capture_state else None
        self._prebuilt = None
        self._notes = []
        try:
            calls = handler(panel, action)
        except _Undetermined as exc:
            action.issues.extend(self._notes)
            action.issues.append(exc.issue)
            self._record(action, [], False, skipped=str(exc), state_before=state_before)
            return

        action.issues.extend(self._notes)
        if self._prebuilt is not None:
            # A handler that had to execute as it went (tracing needs each
            # point's id before it can build the next) reports its own result.
            ok, message, self._prebuilt = True, self._prebuilt, None
        else:
            ok, message = self._execute(calls, panel, action)
        self._record(action, calls, ok, message=message, state_before=state_before)
        if ok and not self._replaying:
            panel.steps.append(action)

    def _do_set_origin(self, panel: Panel, a: DraftAction) -> list[dict]:
        x = PANEL_PITCH * panel.block_index
        return [self._point("single", panel, a.outputs[0], {"x": f"{x:g}", "y": "0"})]

    def _do_define_point_distance(self, panel: Panel, a: DraftAction) -> list[dict]:
        base = self._ref(panel, a, 0)
        if a.direction == "cross":
            # x from one point, y from another — how a drafter finds where a
            # horizontal guide meets a vertical one.
            return [self._point("intersectXY", panel, a.outputs[0], {
                "firstPoint": str(base), "secondPoint": str(self._ref(panel, a, 1))})]
        length = self._measure(a)
        out = a.outputs[0]
        if a.direction == "along_line":
            along = a.overrides.get("along")
            if not along or len(along) < 2:
                raise _Undetermined(Issue(
                    "missing_along_line",
                    f"{a.raw_text!r} measures along a line the row doesn't name",
                    needs="along"))
            return [self._point("alongLine", panel, out, {
                "firstPoint": str(self._id(panel, along[0])),
                "secondPoint": str(self._id(panel, along[1])),
                "length": length}),
                self._line(base, out)]
        return [self._point("endLine", panel, out, {
            "basePoint": str(base), "angle": self._angle(panel, a), "length": length}),
            self._line(base, out)]

    _do_apply_seam_allowance = _do_define_point_distance
    _do_define_point_offset = _do_define_point_distance

    def _do_define_point_midpoint(self, panel: Panel, a: DraftAction) -> list[dict]:
        first, second = a.inputs[0], a.inputs[1]
        panel.midpoint_of[a.outputs[0]] = (first, second)
        n1, n2 = self._name(panel, first), self._name(panel, second)
        return [self._point("alongLine", panel, a.outputs[0], {
            "firstPoint": str(self._id(panel, first)),
            "secondPoint": str(self._id(panel, second)),
            "length": f"Line_{n1}_{n2}/2"})]

    def _do_define_point_ratio_split(self, panel: Panel, a: DraftAction) -> list[dict]:
        """Two points a measured span apart, centred on a reference point."""
        centre = a.inputs[0]
        ends = a.overrides.get("toward") or panel.midpoint_of.get(centre)
        if not ends:
            raise _Undetermined(Issue(
                "missing_split_axis",
                f"{a.raw_text!r} splits around {centre!r} but nothing says along "
                f"which line",
                needs="toward"))
        half = f"({self._measure(a)})/2"
        calls = []
        for out, end in zip(a.outputs, ends):
            calls.append(self._point("alongLine", panel, out, {
                "firstPoint": str(self._id(panel, centre)),
                "secondPoint": str(self._id(panel, end)),
                "length": half}))
        return calls

    def _do_draw_curve(self, panel: Panel, a: DraftAction) -> list[dict]:
        return self._spline_through(panel, a, a.inputs, a.outputs[0])

    def _do_annotate_requirement(self, panel: Panel, a: DraftAction) -> list[dict]:
        """A spec-driven value: always a variable, plus a point when the row
        names one (*"C - F → neck depth as per requirement"*)."""
        var = _variable_name(a)
        formula = a.formula or (f"{a.value:g}" if a.value is not None
                                else f"{MEASUREMENT_DEFAULTS.get(var.lstrip('#'), 1.0):g}")
        calls = [{"name": "add_variable",
                  "input": {"name": var, "formula": formula,
                            "description": a.instruction()}}]
        if a.inputs and a.outputs and _is_point_ref(a.outputs[0]):
            base = self._ref(panel, a, 0)
            calls.append(self._point("endLine", panel, a.outputs[0], {
                "basePoint": str(base), "angle": self._angle(panel, a), "length": var}))
            calls.append(self._line(base, a.outputs[0]))
        return calls

    def _do_trace_reference(self, panel: Panel, a: DraftAction) -> list[dict]:
        """*"Place the back pattern and trace it"* — rebuild the source panel's
        construction in this block so the copy stays parametric, rather than
        freezing coordinates the way tracing on cloth does."""
        source = self._source_panel(a)
        if source is None:
            raise _Undetermined(Issue(
                "unknown_trace_source", f"no compiled panel matches {a.inputs!r}",
                needs="inputs"))
        calls: list[dict] = []
        self._replaying = True
        try:
            for step in list(source.steps):
                copy = _rebind(step, panel.name)
                handler = getattr(self, f"_do_{copy.action.lower()}", None)
                if handler is None:
                    continue
                sub = handler(panel, copy)
                ok, _msg = self._execute(sub, panel, copy)
                if ok:
                    calls.extend(sub)
                    panel.steps.append(copy)
        finally:
            self._replaying = False
        if not calls:
            raise _Undetermined(Issue("empty_trace",
                                      f"panel {source.name!r} produced nothing to trace"))
        self._prebuilt = (f"traced {len(calls)} objects from panel {source.name!r}"
                          + (" (flipped)" if panel.flip_y else ""))
        return calls

    def _do_transform(self, panel: Panel, a: DraftAction) -> list[dict]:
        """Reorienting a traced block is handled when the panel is created (the
        flip changes which way later steps measure), so there is nothing to
        emit — but it is recorded so the dataset shows the step was understood."""
        panel.flip_y = "flip" in (a.value_raw or "")
        return []

    # --- geometry helpers -------------------------------------------------
    def _spline_through(self, panel: Panel, a: DraftAction, refs: list[str],
                        label: str) -> list[dict]:
        """Fit a smooth cubic path through existing points.

        The document says *"connect E, E₂, E₁ & H as an arm curve"* and leaves
        the shape to the drafter's hand. We use Catmull-Rom tangents, which give
        the roundest curve through the points, and materialise each Bézier
        handle as a point placed **relative to its on-curve neighbour** — so the
        curve follows when the measurements change.
        """
        ids = [self._id(panel, r) for r in refs]
        pts = [self.session.evaluated.points.get(i) for i in ids]
        if any(p is None for p in pts):
            raise _Undetermined(Issue("curve_points_unresolved",
                                      "some curve points have no geometry yet"))
        n = len(pts)
        tangents = []
        for i in range(n):
            prev, nxt = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
            span = 1 if (i == 0 or i == n - 1) else 2
            tangents.append(geo.Point((nxt.x - prev.x) / span, (nxt.y - prev.y) / span))

        calls: list[dict] = []
        path: list[str] = [str(ids[0])]
        for i in range(n - 1):
            h1 = geo.Point(pts[i].x + tangents[i].x / 3, pts[i].y + tangents[i].y / 3)
            h2 = geo.Point(pts[i + 1].x - tangents[i + 1].x / 3,
                           pts[i + 1].y - tangents[i + 1].y / 3)
            for anchor_idx, handle, tag in ((i, h1, "a"), (i + 1, h2, "b")):
                nm = f"{_slug_name(label)}{i + 1}{tag}"
                calls.append(self._point("endLine", panel, nm, {
                    "basePoint": str(ids[anchor_idx]),
                    "angle": f"{geo.line_angle(pts[anchor_idx], handle):.4f}",
                    "length": f"{_dist(pts[anchor_idx], handle):.4f}",
                    "showPointName": "false"}, control=True))
                path.append("$" + nm)
            path.append(str(ids[i + 1]))
        calls.append({"name": "add_curve",
                      "input": {"kind": "spline", "spline_type": "cubicBezierPath",
                                "path_points": path, "__label__": label}})
        return calls

    # --- execution --------------------------------------------------------
    def _execute(self, calls: list[dict], panel: Panel, a: DraftAction) -> tuple[bool, str]:
        """Run the tool calls through the agent's own dispatcher."""
        from agent import dispatch_tool

        if not calls:
            return True, "no geometry required"
        messages = []
        for call in calls:
            args = _resolve_placeholders(dict(call["input"]), panel)
            label = args.pop("__label__", None)
            local = args.pop("__ref__", None)
            res = dispatch_tool(self.session, call["name"], args)
            messages.append(res.message)
            if not res.ok:
                return False, "; ".join(messages)
            new_id = max(self.session.added_ids) if self.session.added_ids else None
            if local and new_id is not None:
                panel.ids[local] = new_id
            if label and new_id is not None:
                panel.ids[label] = new_id
            # Keep the resolved arguments: the dataset must record the ids the
            # engine was actually given, not the compiler's forward references.
            call["input"]["__resolved__"] = args
        return True, "; ".join(messages)

    def _point(self, tool_type: str, panel: Panel, ref: str, attrs: dict,
               *, control: bool = False) -> dict:
        name = ref if control else self._unique_name(panel, ref)
        payload = {"tool_type": tool_type,
                   "attrs": {"name": name, **attrs},
                   "__ref__": ref}
        return {"name": "add_point", "input": payload}

    def _line(self, base_id: int, to_ref: str) -> dict:
        """The notation *"A - B"* means a line as well as a point, so draw it —
        that is what makes the compiled pattern legible next to the figure."""
        return {"name": "add_line",
                "input": {"firstPoint": str(base_id), "secondPoint": "$" + to_ref}}

    def _measure(self, a: DraftAction) -> str:
        m = a.measure
        if m is None:
            raise _Undetermined(Issue("missing_measure", "no distance given",
                                      needs="measure"))
        return m

    def _angle(self, panel: Panel, a: DraftAction) -> str:
        """Bearing in engine convention (0 = right, 270 = down)."""
        if "angle" in a.overrides:            # a human correction always wins
            return str(a.overrides["angle"])
        if a.angle is not None:               # then a bearing read off the figure
            return f"{a.angle:g}"
        if a.direction == "vertical":
            return "90" if panel.flip_y else "270"
        if a.direction == "horizontal":
            return "0"
        raise _Undetermined(Issue(
            "missing_direction",
            f"{a.raw_text!r} needs a bearing; only figure {a.source_image} has it",
            needs="angle"))

    def _ref(self, panel: Panel, a: DraftAction, index: int) -> int:
        if index >= len(a.inputs):
            raise _Undetermined(Issue("missing_inputs", "no reference point",
                                      needs="inputs"))
        return self._id(panel, a.inputs[index])

    def _id(self, panel: Panel, ref: str) -> int:
        if ref in panel.ids:
            return panel.ids[ref]
        raise _Undetermined(Issue(
            "undefined_input", f"point {ref!r} is not defined in panel {panel.name!r}",
            needs="inputs"))

    def _name(self, panel: Panel, ref: str) -> str:
        return f"{ref}{panel.suffix}"

    def _unique_name(self, panel: Panel, ref: str) -> str:
        """Figures restart their labels: the front panel's "F" is not the "F"
        traced from the back. When a label is reused we keep both points and
        give the new one a numbered name, noting it on the step — the geometry
        stays correct and the reuse shows up in the dataset."""
        name = self._name(panel, ref)
        if ref not in panel.ids:
            return name
        taken = {o.raw.get("name") for o in self.session.pattern.all_objects()}
        n = 2
        while f"{ref}{n}{panel.suffix}" in taken:
            n += 1
        renamed = f"{ref}{n}{panel.suffix}"
        self._notes.append(Issue(
            "label_reused",
            f"panel {panel.name!r} redefines {ref!r}, which it already traced; "
            f"the new point is named {renamed!r}", blocking=False))
        return renamed

    def _source_panel(self, a: DraftAction) -> Panel | None:
        for ref in a.inputs:
            stem = ref.replace("_pattern", "").replace("_block", "")
            if stem in self.panels and self.panels[stem].steps:
                return self.panels[stem]
        return None

    def _state(self) -> dict:
        return compact_state(self.session.pattern, self.session.evaluated)

    def _record(self, a: DraftAction, calls: list[dict], ok: bool, *,
                message: str = "", skipped: str = "", state_before: dict | None = None) -> None:
        if self._replaying:
            return
        self.steps.append(CompiledStep(
            action=a, instruction=a.instruction(),
            tool_calls=[_public(c) for c in calls], ok=ok,
            message=message or skipped, state_before=state_before, skipped=skipped,
            session_after=copy.deepcopy(self.session) if self.keep_sessions else None))


class _Undetermined(Exception):
    """The document doesn't pin this step down. Carries the issue to report."""

    def __init__(self, issue: Issue):
        super().__init__(issue.detail)
        self.issue = issue


# --- small helpers -----------------------------------------------------------
def _resolve_placeholders(args: dict, panel: Panel) -> dict:
    """``$Name`` placeholders in path_points become real ids once created."""
    out = dict(args)
    pts = out.get("path_points")
    if pts:
        out["path_points"] = [panel.ids[p[1:]] if isinstance(p, str) and p.startswith("$")
                              else int(p) for p in pts]
    for key in ("firstPoint", "secondPoint"):
        v = out.get(key)
        if isinstance(v, str) and v.startswith("$"):
            out[key] = str(panel.ids[v[1:]])
    return out


def _public(call: dict) -> dict:
    """Strip compiler bookkeeping from a tool call before it goes in the dataset."""
    source = call["input"].get("__resolved__", call["input"])
    args = {k: v for k, v in source.items() if not k.startswith("__")}
    if "attrs" in args:
        args["attrs"] = {k: v for k, v in args["attrs"].items() if not k.startswith("__")}
    return {"name": call["name"], "input": args}


def _rebind(step: DraftAction, panel: str) -> DraftAction:
    import copy as _copy
    clone = _copy.deepcopy(step)
    clone.panel = panel
    return clone


def _is_point_ref(ref: str) -> bool:
    """Point labels are short (``A``, ``C1``, ``x2``); measurement names aren't."""
    return len(ref) <= 3 and ref[:1].isalpha() and "_" not in ref


def _variable_name(a: DraftAction) -> str:
    if a.outputs and not _is_point_ref(a.outputs[0]):
        # "Required measurement: Waist" extracts as `waist_measurement`; it names
        # the same quantity every formula calls `waist`.
        name = a.outputs[0]
        for suffix in ("_measurement", "_value"):
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)]
        return "#" + name
    text = (a.raw_text or a.notes or "value").split("->")[-1]
    for stop in ("as per requirement", "(same as given in left part)", "as per"):
        text = text.replace(stop, "")
    return "#" + _slug_name(text) or "#value"


def _slug_name(text: str) -> str:
    keep = [c if (c.isalnum() or c == "_") else " " for c in text.strip()]
    return "_".join("".join(keep).split()).lower()[:32] or "value"


def _dist(a: geo.Point, b: geo.Point) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
