"""Piece (block) helpers — resolve a final piece to its real geometry.

A piece's outline is defined by an ordered list of ``<nodes>`` that reference
*modeling* objects, which in turn map back to *calculation* objects. Walking that
node list — points as vertices, spline/arc nodes as polylines (respecting the
``reverse`` flag) — reconstructs the actual seam/cut outline. This is the
authoritative "real pattern" shape; anything not reachable here is construction.

Both the renderer (draw one block cleanly) and the state exporter (scope the
agent to one block) build on these helpers.
"""

from __future__ import annotations

from . import geometry as geo
from .model import Evaluated, InternalPath, Pattern, Piece


def modeling_map(pattern: Pattern) -> dict[int, int]:
    """modeling object id -> calculation object id."""
    out: dict[int, int] = {}
    for db in pattern.draft_blocks:
        for m in db.modeling:
            out[m.id] = m.id_object
    return out


def internal_paths_by_id(pattern: Pattern) -> dict[int, InternalPath]:
    return {ip.id: ip for db in pattern.draft_blocks for ip in db.internal_paths}


def piece_by_id(pattern: Pattern, piece_id: int) -> Piece | None:
    return next((p for p in pattern.pieces if p.id == piece_id), None)


def classify_path(name: str, line_type: str = "") -> str:
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


def piece_calc_ids(pattern: Pattern, piece: Piece) -> set[int]:
    """The calculation object ids that make up the piece (outline + internal
    paths + anchors + grainline). These are the *real* objects of the piece."""
    mm = modeling_map(pattern)
    ips = internal_paths_by_id(pattern)
    ids: set[int] = set()
    for node in piece.nodes:
        calc = mm.get(node.object_id)
        if calc is not None:
            ids.add(calc)
    for ipid in piece.internal_path_ids:
        ip = ips.get(ipid)
        if ip:
            for mid in ip.node_ids:
                calc = mm.get(mid)
                if calc is not None:
                    ids.add(calc)
    for aid in piece.anchor_ids + ([piece.grainline_anchor] if piece.grainline_anchor else []):
        calc = mm.get(aid) if aid is not None else None
        if calc is not None:
            ids.add(calc)
    return ids


def _append(pts: list[geo.Point], p: geo.Point, eps: float = 1e-6) -> None:
    if pts and pts[-1].dist(p) < eps:
        return
    pts.append(p)


def piece_outline_points(pattern: Pattern, ev: Evaluated, piece: Piece) -> list[geo.Point]:
    """Ordered boundary points of the piece's main seam/cut outline (closed)."""
    mm = modeling_map(pattern)
    pts: list[geo.Point] = []
    for node in piece.nodes:
        calc = mm.get(node.object_id)
        if calc is None:
            continue
        if node.node_type == "NodePoint":
            p = ev.points.get(calc)
            if p is not None:
                _append(pts, p)
        else:  # NodeSpline / NodeSplinePath / NodeArc
            curve = ev.curves.get(calc)
            if curve is None:
                continue
            poly = curve.polyline()
            if node.reverse:
                poly = list(reversed(poly))
            for p in poly:
                _append(pts, p)
    if pts:
        _append(pts, pts[0])  # close the loop
    return pts


def piece_outline_vertices(pattern: Pattern, ev: Evaluated, piece: Piece) -> list[tuple[str, geo.Point]]:
    """(name, point) for the NodePoint vertices of the outline — for labels."""
    mm = modeling_map(pattern)
    by_id = pattern.object_by_id()
    out: list[tuple[str, geo.Point]] = []
    for node in piece.nodes:
        if node.node_type != "NodePoint":
            continue
        calc = mm.get(node.object_id)
        p = ev.points.get(calc) if calc is not None else None
        if p is None:
            continue
        name = by_id[calc].raw.get("name", "") if calc in by_id else ""
        out.append((name, p))
    return out


def piece_internal_paths(pattern: Pattern, ev: Evaluated, piece: Piece
                         ) -> list[tuple[str, str, list[geo.Point]]]:
    """(role, name, points) for each internal path (dart, drill hole, guide…)."""
    mm = modeling_map(pattern)
    ips = internal_paths_by_id(pattern)
    out: list[tuple[str, str, list[geo.Point]]] = []
    for ipid in piece.internal_path_ids:
        ip = ips.get(ipid)
        if ip is None:
            continue
        pts: list[geo.Point] = []
        for mid in ip.node_ids:
            calc = mm.get(mid)
            if calc is None:
                continue
            if calc in ev.points:
                pts.append(ev.points[calc])
            elif calc in ev.curves:
                pts.extend(ev.curves[calc].polyline())
        if len(pts) >= 2:
            out.append((classify_path(ip.name, ip.line_type), ip.name, pts))
    return out


def piece_grainline(pattern: Pattern, ev: Evaluated, piece: Piece
                    ) -> tuple[geo.Point, geo.Point] | None:
    """Endpoints of the grainline, if the piece has one."""
    if piece.grainline_anchor is None:
        return None
    calc = modeling_map(pattern).get(piece.grainline_anchor)
    center = ev.points.get(calc) if calc is not None else None
    if center is None or piece.grainline_length <= 0:
        return None
    half = piece.grainline_length / 2.0
    a = geo.from_polar(center, piece.grainline_rotation, half)
    b = geo.from_polar(center, piece.grainline_rotation + 180.0, half)
    return (a, b)
