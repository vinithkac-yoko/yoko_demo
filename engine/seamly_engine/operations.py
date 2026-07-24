"""Mutation layer — the action API the agent's tools call.

A :class:`PatternSession` wraps a parsed pattern + measurements and exposes the
edit operations a VLA performs, re-evaluating after each so callers always see
fresh geometry and an updated state export. Two invariants matter here:

* **Delete is block-and-report.** Deleting an object that others depend on is
  refused; the full dependent chain is returned so the model can decide whether
  to delete those first. We never silently cascade.
* **Edits re-evaluate and can be rolled back.** ``edit_formula`` recomputes the
  whole DAG; if the edit leaves previously-resolved objects unresolved, the
  caller can roll back to the snapshot.

This is intentionally small — the full parity mutation set (every add_* tool) is
on the roadmap — but it nails the semantics the rest of the system relies on.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .evaluator import evaluate_pattern
from .measurements import MeasurementTable
from .model import Evaluated, Pattern, PatternObject
from .parser import _collect_refs
from .state import export_state


@dataclass
class OpResult:
    ok: bool
    message: str
    blocked_by: list[int] | None = None  # dependents that blocked a delete


class PatternSession:
    """A persistent, multi-turn editing session over one pattern."""

    def __init__(self, pattern: Pattern, measurements: MeasurementTable | None = None,
                 *, size: float | None = None, height: float | None = None):
        self.pattern = pattern
        self.measurements = measurements
        self.size = size
        self.height = height
        self.added_ids: set[int] = set()  # objects created this session (for rendering)
        self.evaluated: Evaluated = self._evaluate()

    def _evaluate(self) -> Evaluated:
        return evaluate_pattern(self.pattern, self.measurements,
                                size=self.size, height=self.height)

    # --- inspection ----------------------------------------------------------
    def state(self) -> dict:
        return export_state(self.pattern, self.evaluated, self.measurements)

    def dependents_of(self, object_id: int) -> list[int]:
        """Direct dependents of an object (who references it)."""
        return [o.id for o in self.pattern.all_objects() if object_id in o.refs]

    def transitive_dependents(self, object_id: int) -> list[int]:
        """All objects that would be invalidated by removing ``object_id``."""
        by_dep: dict[int, list[int]] = {}
        for o in self.pattern.all_objects():
            for r in o.refs:
                by_dep.setdefault(r, []).append(o.id)
        seen: set[int] = set()
        stack = list(by_dep.get(object_id, []))
        while stack:
            oid = stack.pop()
            if oid in seen:
                continue
            seen.add(oid)
            stack.extend(by_dep.get(oid, []))
        return sorted(seen)

    # --- mutation ------------------------------------------------------------
    def _next_id(self) -> int:
        ids = [0]
        for o in self.pattern.all_objects():
            ids.append(o.id)
            for a in ("point1", "point2"):
                v = o.raw.get(a, "")
                if v.isdigit():
                    ids.append(int(v))
            for ch in o.children:
                d = ch.get("dst", "")
                if d.isdigit():
                    ids.append(int(d))
        for db in self.pattern.draft_blocks:
            for m in db.modeling:
                ids += [m.id, m.id_object]
            for ip in db.internal_paths:
                ids.append(ip.id)
        for pc in self.pattern.pieces:
            ids.append(pc.id)
        return max(ids) + 1

    def add_object(self, tag: str, tool_type: str, attrs: dict,
                   children: list[dict] | None = None, block_index: int = 0) -> OpResult:
        """Add a new construction object (any Seamly tool type). Assigns a fresh
        id, re-evaluates, and rolls back if the object can't be computed or breaks
        the pattern. ``trueDarts`` automatically reserves two output-point ids."""
        if not self.pattern.draft_blocks:
            return OpResult(False, "no draft block to add to")
        new_id = self._next_id()
        raw = {k: str(v) for k, v in (attrs or {}).items()}
        raw["id"] = str(new_id)
        if tool_type:
            raw["type"] = tool_type
        kids = [dict(c) for c in (children or [])]
        if tool_type == "trueDarts":
            raw.setdefault("point1", str(new_id + 1))
            raw.setdefault("point2", str(new_id + 2))
        obj = PatternObject(id=new_id, tag=tag, tool_type=tool_type, raw=raw, children=kids)
        obj.refs = _collect_refs(raw, kids)

        before = set(self.evaluated.unresolved)
        snapshot = copy.deepcopy(self.pattern)
        self.pattern.draft_blocks[block_index].objects.append(obj)
        new_ev = self._evaluate()
        newly = set(new_ev.unresolved) - before

        makes_geometry = tag in ("point", "arc", "elArc", "spline")
        resolved_self = (
            (not makes_geometry)
            or new_id in new_ev.points
            or new_id in new_ev.curves
            or (tool_type == "trueDarts" and int(raw["point1"]) in new_ev.points)
        )
        if newly or not resolved_self:
            self.pattern = snapshot
            reason = new_ev.unresolved.get(new_id, "it broke dependent objects")
            return OpResult(False, f"could not add {tool_type or tag} #{new_id}: {reason}",
                            blocked_by=sorted(newly))
        self.evaluated = new_ev
        self.added_ids.add(new_id)
        if tool_type == "trueDarts":
            self.added_ids.update({int(raw["point1"]), int(raw["point2"])})
        nm = raw.get("name", "")
        return OpResult(True, f"added {tool_type or tag} {nm} (#{new_id})".replace("  ", " "))

    def edit_object(self, object_id: int, attrs: dict) -> OpResult:
        """Set one or more attributes of an object (generalizes edit_formula).
        Re-evaluates and rolls back if it breaks previously-valid objects."""
        obj = self.pattern.object_by_id().get(object_id)
        if obj is None:
            return OpResult(False, f"no object with id {object_id}")
        before = set(self.evaluated.unresolved)
        snapshot = copy.deepcopy(self.pattern)
        for k, v in attrs.items():
            obj.raw[k] = str(v)
        obj.refs = _collect_refs(obj.raw, obj.children)
        new_ev = self._evaluate()
        newly = set(new_ev.unresolved) - before
        if newly:
            self.pattern = snapshot
            return OpResult(
                False,
                f"edit rolled back: it left {len(newly)} object(s) unresolvable "
                f"({self._names(sorted(newly))}).",
                blocked_by=sorted(newly),
            )
        self.evaluated = new_ev
        return OpResult(True, f"updated {self._name(object_id)}: {attrs}")

    def delete_object(self, object_id: int) -> OpResult:
        """Delete an object. Refused (block + report) if anything depends on it."""
        obj = self.pattern.object_by_id().get(object_id)
        if obj is None:
            return OpResult(False, f"no object with id {object_id}")
        deps = self.dependents_of(object_id)
        if deps:
            names = self._names(deps)
            return OpResult(
                False,
                f"cannot delete {self._name(object_id)}: "
                f"{len(deps)} object(s) depend on it ({names}). "
                f"Delete those first, or choose a different edit.",
                blocked_by=deps,
            )
        for db in self.pattern.draft_blocks:
            db.objects = [o for o in db.objects if o.id != object_id]
        self.evaluated = self._evaluate()
        return OpResult(True, f"deleted {self._name(object_id)}")

    def edit_formula(self, object_id: int, attr: str, new_formula: str) -> OpResult:
        """Change a formula attribute (e.g. a length), re-evaluate, and roll
        back if it breaks previously-resolved objects."""
        obj = self.pattern.object_by_id().get(object_id)
        if obj is None:
            return OpResult(False, f"no object with id {object_id}")
        if attr not in obj.raw:
            return OpResult(False, f"{self._name(object_id)} has no attribute {attr!r}")

        before_unresolved = set(self.evaluated.unresolved)
        old = obj.raw[attr]
        snapshot = copy.deepcopy(self.pattern)
        obj.raw[attr] = new_formula
        new_ev = self._evaluate()
        newly_broken = set(new_ev.unresolved) - before_unresolved
        if newly_broken:
            self.pattern = snapshot
            return OpResult(
                False,
                f"edit rolled back: setting {attr}={new_formula!r} on "
                f"{self._name(object_id)} left {len(newly_broken)} object(s) "
                f"unresolvable ({self._names(sorted(newly_broken))}).",
                blocked_by=sorted(newly_broken),
            )
        self.evaluated = new_ev
        return OpResult(True, f"set {attr}={new_formula!r} on {self._name(object_id)} "
                              f"(was {old!r})")

    # --- helpers -------------------------------------------------------------
    def _name(self, oid: int) -> str:
        o = self.pattern.object_by_id().get(oid)
        nm = o.raw.get("name") if o else None
        return f"{nm} (#{oid})" if nm else f"#{oid}"

    def _names(self, ids: list[int], limit: int = 6) -> str:
        shown = [self._name(i) for i in ids[:limit]]
        more = "" if len(ids) <= limit else f", +{len(ids) - limit} more"
        return ", ".join(shown) + more
