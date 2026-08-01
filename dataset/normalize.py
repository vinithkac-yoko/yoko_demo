"""Normalize an extracted drafting-action JSONL into the strict IR.

Extraction from a PDF is lossy and inconsistent; this pass makes it uniform and,
more importantly, **says what is wrong**. On the Angrakha Maxi source it finds,
without being told:

* ``output_ref`` sometimes a string and sometimes a list (steps 3 and 5 of the
  bottom panel) — split into one action per output;
* a step that measures from a point (``G``) no other step ever defines;
* the same point (``H₁``) defined twice — the second is a restatement carried
  over from the next figure, not a new step;
* distances given with no direction, where only the figure knows which way;
* free-text values (*"2, 3, or 4 as required"*) that are really a parameter.

Everything it cannot decide becomes an :class:`~dataset.schema.Issue` with a
``needs`` key. A per-document overrides file answers those keys once, with
provenance, and the document then compiles deterministically forever.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .schema import ACTIONS, DIRECTIONS, DraftAction, Issue, required_fields

# 12" / 12in / 12 inch — the documents mix all three.
_UNIT_SUFFIX = re.compile(r'(?<=[\d.])\s*(?:in\b|inch(?:es)?\b|")')
_PAIR_LENGTH = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)_to_([A-Za-z][A-Za-z0-9]*)\b")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# "neck width along A-C", "measured along A - C"
_ALONG = re.compile(r"along\s+([A-Za-z][A-Za-z0-9]*)\s*[-–]\s*([A-Za-z][A-Za-z0-9]*)",
                    re.IGNORECASE)

_FUNCTIONS = {"sin", "cos", "tan", "sinD", "cosD", "tanD", "sqrt", "abs",
              "min", "max", "log", "log2", "exp", "fmod", "atan2"}


@dataclass
class Document:
    """A normalized instruction document."""

    garment: str
    actions: list[DraftAction] = field(default_factory=list)
    measurements: set[str] = field(default_factory=set)
    panels: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> list[DraftAction]:
        return [a for a in self.actions if a.blocked]

    def issues(self) -> list[tuple[str, Issue]]:
        return [(a.key, i) for a in self.actions for i in a.issues]


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_overrides(path: str | Path | None) -> tuple[dict, list[dict]]:
    """Return (per-step resolutions, rows to insert).

    ``insert`` rows are steps the prose omits but the figure shows — a point the
    text measures from without ever defining, or the block a "changes only"
    section silently assumes. They carry fractional ``step_index`` values so
    they sort into place.
    """
    if not path or not Path(path).exists():
        return {}, []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "steps" not in data and "insert" not in data:
        return data, []
    inserts = [i["row"] for i in data.get("insert", []) if i.get("row")]
    return data.get("steps", {}), inserts


# --- expression normalization ------------------------------------------------
def normalize_formula(expr: str | None, known_points: set[str]) -> tuple[str | None, set[str]]:
    """Rewrite an extracted formula into engine (qmuparser) grammar.

    * ``1in`` / ``1"`` → ``1`` (the compiled pattern's unit *is* inches).
    * ``A_to_B`` → ``Line_A_B``, the engine pseudo-variable for the live
      distance between two points — so a flare that is "twice A-B" stays
      parametric instead of freezing to a number.
    * every other bare identifier is a measurement → ``#name`` increment.
    """
    if not expr:
        return None, set()
    out = _UNIT_SUFFIX.sub("", expr)
    out = _PAIR_LENGTH.sub(lambda m: f"Line_{m.group(1)}_{m.group(2)}", out)

    used: set[str] = set()

    def sub_ident(m: re.Match) -> str:
        name = m.group(0)
        if name.startswith("Line_") or name in _FUNCTIONS or name in known_points:
            return name
        used.add(name)
        return "#" + name

    out = _IDENT.sub(sub_ident, out)
    return re.sub(r"\s+", "", out), used


def normalize_value(raw) -> tuple[float | None, str | None, Issue | None]:
    """Parse an extracted ``value``. Returns (number, raw, issue)."""
    if raw is None:
        return None, None, None
    text = str(raw).strip()
    stripped = _UNIT_SUFFIX.sub("", text)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
        return float(stripped), text, None
    if _NUMBER.search(text):
        return None, text, Issue(
            "value_not_a_number",
            f"value {text!r} names a choice, not a measurement",
            needs="measure", blocking=True)
    return None, text, None  # e.g. "flip_vertical" — a keyword, not a length


# --- the pass ----------------------------------------------------------------
def normalize(rows: list[dict], overrides: dict | None = None,
              inserts: list[dict] | None = None) -> Document:
    overrides = overrides or {}
    rows = _merge_inserts(rows, inserts or [])
    garment = rows[0].get("garment", "pattern") if rows else "pattern"
    doc = Document(garment=garment)

    # Symbols defined per panel. A traced panel inherits its source's symbols,
    # which is the only reason `front_left`'s "C - F" resolves at all.
    defined: dict[str, set[str]] = {}
    traced_from: dict[str, str] = {}
    seen: dict[tuple, int] = {}

    for row in rows:
        panel = row.get("panel") or "main"
        if panel not in defined:
            defined[panel] = set()
            doc.panels.append(panel)
        for act in _split_row(row, garment, panel):
            _normalize_action(act, defined, traced_from, seen, overrides, doc)
            doc.actions.append(act)
    return doc


def _as_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _merge_inserts(rows: list[dict], inserts: list[dict]) -> list[dict]:
    """Fold figure-derived rows into the document, keeping each panel's order."""
    if not inserts:
        return rows
    order: dict[str, int] = {}
    for r in rows:
        order.setdefault(r.get("panel") or "main", len(order))
    for r in inserts:
        order.setdefault(r.get("panel") or "main", len(order))
    merged = list(rows) + list(inserts)
    return sorted(merged, key=lambda r: (order[r.get("panel") or "main"],
                                         float(r.get("step_index", 0))))


def _split_row(row: dict, garment: str, panel: str) -> list[DraftAction]:
    """One row → one action, except a row whose ``output_ref`` is a list of
    independent points (``A - C = B - D``), which is really two steps."""
    out = row.get("output_ref")
    outputs = [o for o in (out if isinstance(out, list) else [out]) if o]
    inputs = list(row.get("input_refs") or [])
    action = row.get("action", "")

    def make(o: list[str], i: list[str]) -> DraftAction:
        return DraftAction(
            garment=garment, panel=panel, step_index=float(row.get("step_index", 0)),
            action=action, outputs=o, inputs=i,
            formula_raw=row.get("formula"), value_raw=row.get("value"),
            direction=row.get("direction"),
            angle=_as_float(row.get("angle")),
            along=list(row.get("along") or []),
            confidence=row.get("confidence") or "",
            page=row.get("page"),
            notes=row.get("notes") or "",
            raw_text=row.get("raw_text") or "", source_image=row.get("source_image") or "",
        )

    # A multi-output row only splits when the outputs pair up 1:1 with inputs.
    # RATIO_SPLIT's two outputs share one input and stay together.
    if len(outputs) > 1 and len(outputs) == len(inputs) and action != "DEFINE_POINT_RATIO_SPLIT":
        return [make([o], [i]) for o, i in zip(outputs, inputs)]
    return [make(outputs, inputs)]


def _normalize_action(a: DraftAction, defined: dict[str, set[str]],
                      traced_from: dict[str, str], seen: dict[tuple, int],
                      overrides: dict, doc: Document) -> None:
    if a.action not in ACTIONS:
        a.issues.append(Issue("unknown_action", f"verb {a.action!r} is not in the vocabulary"))
        return

    known = defined[a.panel]
    a.formula, used = normalize_formula(a.formula_raw, known)
    doc.measurements |= used
    a.value, a.value_raw, val_issue = normalize_value(a.value_raw)

    if a.direction not in DIRECTIONS:
        a.issues.append(Issue("unknown_direction", f"direction {a.direction!r} unrecognized",
                              needs="direction"))

    # Overrides answer the open questions; apply before validating so a resolved
    # step stops being reported.
    a.overrides = dict(overrides.get(a.key, {}))
    if "measure" in a.overrides:
        a.formula, extra = normalize_formula(str(a.overrides["measure"]), known)
        doc.measurements |= extra
        val_issue = None
    if "direction" in a.overrides:
        a.direction = a.overrides["direction"]
    if "inputs" in a.overrides:
        a.inputs = list(a.overrides["inputs"])
    if a.overrides.get("skip"):
        a.issues.append(Issue("skipped_by_override",
                              str(a.overrides.get("why", "skipped")), blocking=False))
        a.skipped = True
        return
    if val_issue:
        a.issues.append(val_issue)

    # An "along_line" step needs the line it runs along; the note usually says.
    if a.direction == "along_line" and "along" not in a.overrides:
        if a.along:
            a.overrides["along"] = list(a.along)
        else:
            m = _ALONG.search(f"{a.notes} {a.raw_text}")
            if m:
                a.overrides["along"] = [m.group(1), m.group(2)]

    _check_refs(a, defined, traced_from)
    _check_required(a)
    _check_duplicate(a, seen)

    if not a.blocked or a.action == "TRACE_REFERENCE":
        known.update(a.outputs)
    if a.action == "TRACE_REFERENCE" and a.inputs:
        src = a.inputs[0].replace("_pattern", "").replace("_block", "")
        traced_from[a.panel] = src
        if src in defined:
            known.update(defined[src])


def _check_refs(a: DraftAction, defined: dict[str, set[str]],
                traced_from: dict[str, str]) -> None:
    """Inputs must already exist in this panel (or in the panel it traced)."""
    if a.action in ("TRACE_REFERENCE", "ANNOTATE_REQUIREMENT", "TRANSFORM", "SET_ORIGIN"):
        return
    known = defined[a.panel]
    for ref in a.inputs:
        if ref not in known:
            src = traced_from.get(a.panel)
            hint = f" (panel traces {src!r})" if src else ""
            a.issues.append(Issue(
                "undefined_input", f"step measures from {ref!r}, which no earlier "
                                   f"step in panel {a.panel!r} defines{hint}",
                needs="inputs"))


def _check_required(a: DraftAction) -> None:
    for need in required_fields(a.action):
        if need == "inputs" and not a.inputs:
            a.issues.append(Issue("missing_inputs", "no reference point", needs="inputs"))
        elif need == "inputs2" and len(a.inputs) < 2:
            a.issues.append(Issue("missing_inputs", "needs two reference points",
                                  needs="inputs"))
        elif need == "inputs3" and len(a.inputs) < 3:
            a.issues.append(Issue("missing_inputs", "a curve needs three or more points",
                                  needs="inputs"))
        elif need == "measure" and a.direction == "cross":
            if len(a.inputs) < 2:
                a.issues.append(Issue("missing_inputs",
                                      "a crossing needs two reference points",
                                      needs="inputs"))
        elif need == "measure" and a.measure is None:
            a.issues.append(Issue("missing_measure", "no formula and no numeric value",
                                  needs="measure"))
        elif need == "direction" and a.direction == "cross":
            pass
        elif need == "direction" and a.angle is not None:
            pass
        elif need == "direction" and a.direction in (None, "offset"):
            a.issues.append(Issue(
                "missing_direction",
                f"{a.raw_text!r} gives a distance but not a bearing — only the "
                f"figure ({a.source_image}) knows which way",
                needs="angle"))
        elif need == "value_raw" and not a.value_raw:
            a.issues.append(Issue("missing_value", "no transform named", needs="measure"))

    # An offset/inward step is a distance with no bearing unless overridden.
    if a.action in ("DEFINE_POINT_OFFSET", "APPLY_SEAM_ALLOWANCE") \
            and "angle" not in a.overrides and a.angle is None:
        a.issues.append(Issue(
            "missing_direction",
            f"{a.raw_text!r} offsets by a distance but the direction is only in "
            f"the figure ({a.source_image})",
            needs="angle"))


def _check_duplicate(a: DraftAction, seen: dict[tuple, int]) -> None:
    if not a.outputs or a.action in ("ANNOTATE_REQUIREMENT", "TRANSFORM"):
        return
    sig = (a.panel, a.action, tuple(a.outputs), a.formula, a.value)
    if sig in seen:
        a.duplicate_of = seen[sig]
        a.issues.append(Issue("duplicate_step",
                              f"restates step {seen[sig]} of the same panel",
                              blocking=False))
    else:
        seen[sig] = a.step_index
