"""VLA state representation exporter.

This is the layer that lets a vision-language-action model *understand* a
pattern beyond a screenshot. It flattens the evaluated pattern into a
JSON-serializable structure where every object is semantically tagged:

* ``role`` — anchor / seamline / dart / drill_hole / grainline / notch /
  curve_control / guide / construction — derived from how the object is used
  (piece membership, internal paths, whether anything references it).
* ``is_construction`` vs ``is_final_outline`` — whether the object is
  scaffolding or part of a cut/sewn piece. Derived from the modeling→piece
  chain, not guessed.
* ``built_from`` / ``dependents`` — the DAG edges, so the model can reason about
  what an edit or a delete will affect (delete is *block + report dependents*).
* ``formula`` — raw expression **and** its resolved numeric value.

Nothing here re-derives geometry; it reads the :class:`Evaluated` result.
"""

from __future__ import annotations

from .formula import identifiers
from .geometry import Arc, BezierPath, CubicBezier, Point
from .measurements import MeasurementTable
from .model import Evaluated, Pattern

_VISIBLE_LINE = "lineType"


def export_state(pattern: Pattern, ev: Evaluated,
                 measurements: MeasurementTable | None = None) -> dict:
    objs = pattern.all_objects()
    by_id = {o.id: o for o in objs}
    name_of = {o.id: o.raw.get("name", "") for o in objs if o.raw.get("name")}

    # trueDarts output points have their own ids/names declared inline.
    for o in objs:
        if o.tool_type == "trueDarts":
            for pid_a, nm_a in (("point1", "name1"), ("point2", "name2")):
                pid, nm = o.raw.get(pid_a), o.raw.get(nm_a)
                if pid and pid.isdigit() and nm:
                    name_of[int(pid)] = nm

    dependents = _build_dependents(pattern)
    piece_map, roles = _semantic_roles(pattern)

    objects_out = []
    for o in objs:
        objects_out.append(_object_record(o, ev, dependents, roles, piece_map, name_of))

    total = len(objs)
    resolved = total - len(ev.unresolved)
    return {
        "pattern": {
            "name": pattern.pattern_name,
            "unit": pattern.unit,
            "version": pattern.version,
            "base_size": measurements.base_size if measurements else None,
            "base_height": measurements.base_height if measurements else None,
        },
        "measurements": (measurements.resolve() if measurements else {}),
        "variables": {
            inc.name: {
                "formula": inc.formula,
                "value": ev.increment_values.get(inc.name),
                "description": inc.description,
            }
            for inc in pattern.increments
        },
        "pieces": [
            {
                "id": p.id,
                "name": p.name,
                "seam_allowance": p.seam_allowance,
                "outline_object_ids": sorted(piece_map.get(p.id, set())),
            }
            for p in pattern.pieces
        ],
        "objects": objects_out,
        "coverage": {
            "total": total,
            "resolved": resolved,
            "fraction": round(resolved / total, 4) if total else 1.0,
            "unresolved": ev.unresolved,
        },
    }


def _build_dependents(pattern: Pattern) -> dict[int, list[int]]:
    dep: dict[int, set[int]] = {}
    for o in pattern.all_objects():
        for r in o.refs:
            dep.setdefault(r, set()).add(o.id)
    return {k: sorted(v) for k, v in dep.items()}


def _semantic_roles(pattern: Pattern) -> tuple[dict[int, set[int]], dict[int, str]]:
    """Return (piece_id -> set of calc object ids in its outline,
    calc object id -> role string)."""
    # modeling id -> calculation object id
    modeling_to_calc: dict[int, int] = {}
    internal_paths: dict[int, object] = {}
    for db in pattern.draft_blocks:
        for m in db.modeling:
            modeling_to_calc[m.id] = m.id_object
        for ip in db.internal_paths:
            internal_paths[ip.id] = ip

    roles: dict[int, str] = {}
    piece_map: dict[int, set[int]] = {}

    for piece in pattern.pieces:
        calc_ids: set[int] = set()
        # main seam/cut outline
        for node in piece.nodes:
            calc = modeling_to_calc.get(node.object_id)
            if calc is not None:
                calc_ids.add(calc)
                roles.setdefault(calc, "seamline")
        # internal paths (darts, guide lines, drill holes)
        for ipid in piece.internal_path_ids:
            ip = internal_paths.get(ipid)
            if ip is None:
                continue
            role = _classify_path(ip.name, ip.line_type)
            for mid in ip.node_ids:
                calc = modeling_to_calc.get(mid)
                if calc is not None:
                    calc_ids.add(calc)
                    roles.setdefault(calc, role)
        # grainline + anchors
        if piece.grainline_anchor is not None:
            calc = modeling_to_calc.get(piece.grainline_anchor)
            if calc is not None:
                roles[calc] = "grainline"
        for aid in piece.anchor_ids:
            calc = modeling_to_calc.get(aid)
            if calc is not None:
                roles.setdefault(calc, "anchor")
        piece_map[piece.id] = calc_ids

    return piece_map, roles


def _classify_path(name: str, line_type: str) -> str:
    low = name.lower()
    if "dart" in low:
        return "dart"
    if "drill" in low or "hole" in low:
        return "drill_hole"
    if "notch" in low:
        return "notch"
    if "grain" in low:
        return "grainline"
    return "guide"


def _object_record(o, ev: Evaluated, dependents, roles, piece_map, name_of) -> dict:
    in_pieces = [pid for pid, ids in piece_map.items() if o.id in ids]
    is_final = bool(in_pieces)
    line_type = o.raw.get(_VISIBLE_LINE, "")
    is_construction = (not is_final) or line_type == "none"

    role = roles.get(o.id)
    if role is None:
        role = _fallback_role(o, dependents, line_type)

    rec = {
        "id": o.id,
        "name": o.raw.get("name", name_of.get(o.id, "")),
        "kind": o.tag,
        "tool_type": o.tool_type,
        "role": role,
        "is_construction": is_construction,
        "is_final_outline": is_final,
        "built_from": o.refs,
        "dependents": dependents.get(o.id, []),
        "visual": {
            "line_type": line_type or None,
            "color": o.raw.get("lineColor") or o.raw.get("color"),
            "weight": o.raw.get("lineWeight"),
        },
        "piece_membership": in_pieces,
        "geometry": _geometry_of(o, ev),
        "formula": _formula_of(o, ev),
    }
    return rec


def _fallback_role(o, dependents, line_type: str) -> str:
    if o.tool_type == "single":
        return "anchor"
    if o.tag in ("spline", "arc"):
        return "curve"
    if o.tag == "line":
        return "construction_line"
    if o.tag == "operation":
        return "operation"
    # A point used only as a spline control handle (invisible, feeds a curve).
    if line_type == "none" and dependents.get(o.id):
        return "curve_control"
    return "construction"


def _geometry_of(o, ev: Evaluated) -> dict | None:
    p = ev.points.get(o.id)
    if p is not None:
        return {"type": "point", "x": round(p.x, 5), "y": round(p.y, 5)}
    curve = ev.curves.get(o.id)
    if curve is not None:
        return _curve_geometry(curve)
    # trueDarts / operations: expose their produced points
    if o.tool_type == "trueDarts":
        out = {}
        for pid_a, nm_a in (("point1", "name1"), ("point2", "name2")):
            pid = o.raw.get(pid_a)
            if pid and pid.isdigit() and int(pid) in ev.points:
                q = ev.points[int(pid)]
                out[o.raw.get(nm_a, pid)] = {"x": round(q.x, 5), "y": round(q.y, 5)}
        return {"type": "double_point", "points": out} if out else None
    return None


def _curve_geometry(curve) -> dict:
    poly = [[round(p.x, 4), round(p.y, 4)] for p in curve.polyline()]
    if isinstance(curve, Arc):
        return {"type": "arc", "center": [curve.center.x, curve.center.y],
                "radius": curve.radius, "angle1": curve.angle1, "angle2": curve.angle2,
                "polyline": poly}
    if isinstance(curve, CubicBezier):
        return {"type": "cubic_bezier",
                "control_points": [[c.x, c.y] for c in (curve.p0, curve.p1, curve.p2, curve.p3)],
                "polyline": poly}
    if isinstance(curve, BezierPath):
        return {"type": "bezier_path",
                "control_points": [[c.x, c.y] for c in curve.on_and_controls],
                "polyline": poly}
    return {"type": "curve", "polyline": poly}


def _formula_of(o, ev: Evaluated) -> dict:
    out: dict[str, dict] = {}
    vals = ev.values.get(o.id, {})
    for attr in ("length", "angle", "radius", "angle1", "angle2"):
        raw = o.raw.get(attr)
        if raw is None:
            continue
        entry: dict[str, object] = {"raw": raw}
        if attr in vals:
            entry["resolved"] = round(vals[attr], 5)
        try:
            refs = sorted(identifiers(raw))
            if refs:
                entry["references"] = refs
        except Exception:
            pass
        out[attr] = entry
    return out
