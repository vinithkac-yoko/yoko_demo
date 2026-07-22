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
from .model import Evaluated, Pattern
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
