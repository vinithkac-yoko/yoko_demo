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

    def __init__(
        self,
        pattern: Pattern,
        measurements: MeasurementTable | None = None,
        *,
        size: float | None = None,
        height: float | None = None,
    ):
        self.pattern = pattern
        self.measurements = measurements
        self.size = size
        self.height = height
        self.added_ids: set[int] = set()  # objects created this session (for rendering)
        self.evaluated: Evaluated = self._evaluate()

    def _evaluate(self) -> Evaluated:
        return evaluate_pattern(self.pattern, self.measurements, size=self.size, height=self.height)

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

    def add_object(
        self,
        tag: str,
        tool_type: str,
        attrs: dict,
        children: list[dict] | None = None,
        block_index: int = 0,
    ) -> OpResult:
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
        if tag == "operation":
            # Each source produces a transformed copy; reserve a destination id
            # for any child that doesn't already name one.
            nxt = new_id + 1
            for ch in kids:
                if not str(ch.get("dst", "")).isdigit():
                    ch["dst"] = str(nxt)
                    nxt += 1
        obj = PatternObject(id=new_id, tag=tag, tool_type=tool_type, raw=raw, children=kids)
        obj.refs = _collect_refs(raw, kids)

        before = set(self.evaluated.unresolved)
        snapshot = copy.deepcopy(self.pattern)
        self.pattern.draft_blocks[block_index].objects.append(obj)
        new_ev = self._evaluate()
        newly = set(new_ev.unresolved) - before

        makes_geometry = tag in ("point", "arc", "elArc", "spline")
        dest_ids = [int(c["dst"]) for c in kids if str(c.get("dst", "")).isdigit()]
        resolved_self = (
            (not makes_geometry and tag != "operation")
            or new_id in new_ev.points
            or new_id in new_ev.curves
            or (tool_type == "trueDarts" and int(raw["point1"]) in new_ev.points)
            or (
                tag == "operation"
                and dest_ids
                and all(d in new_ev.points or d in new_ev.curves for d in dest_ids)
            )
        )
        if newly or not resolved_self:
            self.pattern = snapshot
            reason = new_ev.unresolved.get(new_id, "it broke dependent objects")
            return OpResult(
                False,
                f"could not add {tool_type or tag} #{new_id}: {reason}",
                blocked_by=sorted(newly),
            )
        self.evaluated = new_ev
        self.added_ids.add(new_id)
        if tool_type == "trueDarts":
            self.added_ids.update({int(raw["point1"]), int(raw["point2"])})
        self.added_ids.update(dest_ids)
        nm = raw.get("name", "")
        return OpResult(True, f"added {tool_type or tag} {nm} (#{new_id})".replace("  ", " "))

    def create_piece(
        self,
        name: str,
        node_ids: list[int],
        *,
        seam_allowance: bool = True,
        width: str = "1",
        internal_paths: list[dict] | None = None,
        grainline_anchor: int | None = None,
    ) -> OpResult:
        """Define a new cut piece from existing construction objects.

        The nodes must be given in outline order (walking the seam), mixing
        points and curves. Rolls back if the piece can't be built or doesn't
        produce a usable outline."""
        from .pieces import build_piece, piece_outline_points

        if len(node_ids) < 3:
            return OpResult(False, "a piece needs at least 3 outline nodes")
        snapshot = copy.deepcopy(self.pattern)
        try:
            piece = build_piece(
                self.pattern,
                self._next_id,
                name,
                node_ids,
                seam_allowance=seam_allowance,
                width=width,
                internal_paths=internal_paths,
                grainline_anchor=grainline_anchor,
            )
        except Exception as e:
            self.pattern = snapshot
            return OpResult(False, f"could not create piece: {e}")

        outline = piece_outline_points(self.pattern, self.evaluated, piece)
        if len(outline) < 3:
            self.pattern = snapshot
            return OpResult(False, "piece outline could not be resolved from those nodes")
        self.added_ids.update({piece.id} | {n.object_id for n in piece.nodes})
        return OpResult(
            True, f"created piece “{name}” (#{piece.id}) with {len(piece.nodes)} outline nodes"
        )

    def split_piece(
        self,
        piece_id: int,
        point_a_id: int,
        point_b_id: int,
        *,
        name_a: str,
        name_b: str,
        seam_allowance: bool | None = None,
        width: str | None = None,
    ) -> OpResult:
        """Cut a piece into two along the straight line between two points.
        Each point may be an existing outline vertex, or a point you just
        constructed that sits exactly on one of the piece's straight edges.
        See :func:`seamly_engine.pieces.split_piece` for the geometric rules.
        Rolls back if either resulting outline doesn't resolve to a real
        shape; never partially mutates."""
        from .pieces import piece_outline_points, split_piece

        snapshot = copy.deepcopy(self.pattern)
        try:
            piece_a, piece_b, dropped = split_piece(
                self.pattern,
                self.evaluated,
                self._next_id,
                piece_id,
                point_a_id,
                point_b_id,
                name_a=name_a,
                name_b=name_b,
                seam_allowance=seam_allowance,
                width=width,
            )
        except ValueError as e:
            self.pattern = snapshot
            return OpResult(False, f"could not split: {e}")

        out_a = piece_outline_points(self.pattern, self.evaluated, piece_a)
        out_b = piece_outline_points(self.pattern, self.evaluated, piece_b)
        if len(out_a) < 3 or len(out_b) < 3:
            self.pattern = snapshot
            return OpResult(False, "the split didn't leave two real outlines")

        self.added_ids.update({piece_a.id, piece_b.id})
        self.added_ids.update(n.object_id for n in piece_a.nodes + piece_b.nodes)
        msg = f"split into “{name_a}” (#{piece_a.id}) and “{name_b}” (#{piece_b.id})"
        if dropped:
            msg += (
                f"; {len(dropped)} internal path(s) straddled the cut and were "
                f"dropped ({', '.join(dropped)}) — re-add them if needed"
            )
        return OpResult(True, msg)

    def merge_piece(
        self,
        piece_a_id: int,
        piece_b_id: int,
        edge_point_1: int,
        edge_point_2: int,
        *,
        name: str,
        seam_allowance: bool | None = None,
        width: str | None = None,
    ) -> OpResult:
        """Combine two pieces into one along a seam they share, named by its
        two endpoints. See :func:`seamly_engine.pieces.merge_piece`. Rolls
        back if the pieces don't actually share that edge, or the merged
        outline doesn't resolve."""
        from .pieces import merge_piece, piece_outline_points

        snapshot = copy.deepcopy(self.pattern)
        try:
            merged = merge_piece(
                self.pattern,
                self._next_id,
                piece_a_id,
                piece_b_id,
                edge_point_1,
                edge_point_2,
                name=name,
                seam_allowance=seam_allowance,
                width=width,
            )
        except ValueError as e:
            self.pattern = snapshot
            return OpResult(False, f"could not merge: {e}")

        outline = piece_outline_points(self.pattern, self.evaluated, merged)
        if len(outline) < 3:
            self.pattern = snapshot
            return OpResult(False, "the merge didn't resolve to a real outline")

        self.added_ids.update({merged.id} | {n.object_id for n in merged.nodes})
        return OpResult(True, f"merged into “{name}” (#{merged.id})")

    def delete_piece(self, piece_id: int) -> OpResult:
        """Remove a piece. Its construction geometry is untouched — a piece is
        a view onto construction objects, not their owner."""
        from .pieces import delete_piece, piece_by_id

        piece = piece_by_id(self.pattern, piece_id)
        if piece is None:
            return OpResult(False, f"no piece with id {piece_id}")
        name = piece.name
        delete_piece(self.pattern, piece_id)
        return OpResult(True, f"deleted piece “{name}” (#{piece_id})")

    def edit_piece(
        self,
        piece_id: int,
        *,
        name: str | None = None,
        seam_allowance: bool | None = None,
        width: str | None = None,
    ) -> OpResult:
        """Rename a piece, or change its seam allowance / width."""
        from .pieces import edit_piece

        try:
            piece = edit_piece(
                self.pattern, piece_id, name=name, seam_allowance=seam_allowance, width=width
            )
        except ValueError as e:
            return OpResult(False, str(e))
        return OpResult(True, f"updated piece “{piece.name}” (#{piece.id})")

    def set_variable(self, name: str, formula: str, description: str = "") -> OpResult:
        """Add or update a pattern variable (increment). Re-evaluates and rolls
        back if the new value breaks any formula that uses it."""
        from .model import Increment

        before = set(self.evaluated.unresolved)
        snapshot = copy.deepcopy(self.pattern)
        existing = next((i for i in self.pattern.increments if i.name == name), None)
        if existing is not None:
            existing.formula = formula
            if description:
                existing.description = description
        else:
            self.pattern.increments.append(
                Increment(name=name, formula=formula, description=description)
            )
        new_ev = self._evaluate()
        if name not in new_ev.increment_values:
            self.pattern = snapshot
            return OpResult(False, f"could not evaluate {name} = {formula!r}")
        newly = set(new_ev.unresolved) - before
        if newly:
            self.pattern = snapshot
            return OpResult(
                False, f"{name}={formula!r} broke {len(newly)} object(s)", blocked_by=sorted(newly)
            )
        self.evaluated = new_ev
        val = round(new_ev.increment_values[name], 4)
        verb = "updated" if existing is not None else "added"
        return OpResult(True, f"{verb} variable {name} = {formula} ({val})")

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
        return OpResult(
            True, f"set {attr}={new_formula!r} on {self._name(object_id)} (was {old!r})"
        )

    # --- helpers -------------------------------------------------------------
    def _name(self, oid: int) -> str:
        o = self.pattern.object_by_id().get(oid)
        nm = o.raw.get("name") if o else None
        return f"{nm} (#{oid})" if nm else f"#{oid}"

    def _names(self, ids: list[int], limit: int = 6) -> str:
        shown = [self._name(i) for i in ids[:limit]]
        more = "" if len(ids) <= limit else f", +{len(ids) - limit} more"
        return ", ".join(shown) + more
