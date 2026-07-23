"""SVG renderer for an evaluated pattern.

The render is deliberately *legible to a VLA and to a human on a phone*:
construction geometry is dimmed and thin, final-piece outlines are bold, darts
and drill holes use their own strokes, and points carry their names. It is the
same image the model reasons over (alongside the structured state) and the view
the phone shows. Pure Python string building — no external deps.
"""

from __future__ import annotations

from .geometry import Arc, BezierPath, CubicBezier, Point
from .model import Evaluated, Pattern
from .state import export_state

# Role -> stroke styling. Final-outline strokes read boldest.
_STYLE = {
    "seamline": ("#111", 2.2, None),
    "dart": ("#c026d3", 1.6, "4,3"),
    "drill_hole": ("#0891b2", 1.2, None),
    "grainline": ("#16a34a", 1.4, None),
    "guide": ("#9ca3af", 0.8, "2,2"),
    "curve": ("#111", 1.6, None),
    "construction_line": ("#c9c9c9", 0.6, "3,3"),
    "construction": ("#d1d5db", 0.6, None),
    "curve_control": ("#e5e7eb", 0.5, None),
    "anchor": ("#111", 0.6, None),
}


def render_svg(pattern: Pattern, ev: Evaluated, *, width: int = 900,
               labels: bool = True, padding: float = 4.0) -> str:
    state = export_state(pattern, ev)
    role_of = {o["id"]: o["role"] for o in state["objects"]}
    final_of = {o["id"]: o["is_final_outline"] for o in state["objects"]}

    xs = [p.x for p in ev.points.values()]
    ys = [p.y for p in ev.points.values()]
    for c in ev.curves.values():
        for p in c.polyline(8):
            xs.append(p.x); ys.append(p.y)
    if not xs:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'
    minx, maxx = min(xs) - padding, max(xs) + padding
    miny, maxy = min(ys) - padding, max(ys) + padding
    span_x = maxx - minx or 1.0
    span_y = maxy - miny or 1.0
    scale = width / span_x
    height = int(span_y * scale)

    def sx(x: float) -> float:
        return (x - minx) * scale

    def sy(y: float) -> float:
        return (y - miny) * scale

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#fbfbfb"/>',
    ]

    by_id = pattern.object_by_id()

    # curves first (under points)
    for oid, curve in ev.curves.items():
        role = role_of.get(oid, "curve")
        color, w, dash = _STYLE.get(role, _STYLE["curve"])
        if final_of.get(oid):
            w = max(w, 2.0)
        poly = curve.polyline()
        d = "M " + " L ".join(f"{sx(p.x):.1f},{sy(p.y):.1f}" for p in poly)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{da}/>')

    # lines
    for o in pattern.all_objects():
        if o.tag != "line":
            continue
        p1 = ev.points.get(_ref(o, "firstPoint"))
        p2 = ev.points.get(_ref(o, "secondPoint"))
        if not (p1 and p2):
            continue
        role = role_of.get(o.id, "construction_line")
        color, w, dash = _STYLE.get(role, _STYLE["construction_line"])
        if final_of.get(o.id):
            color, w = "#111", max(w, 2.0)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{sx(p1.x):.1f}" y1="{sy(p1.y):.1f}" x2="{sx(p2.x):.1f}" '
            f'y2="{sy(p2.y):.1f}" stroke="{color}" stroke-width="{w}"{da}/>')

    # points + labels
    for oid, p in ev.points.items():
        role = role_of.get(oid, "construction")
        is_final = final_of.get(oid)
        r = 2.4 if is_final else 1.4
        fill = "#111" if is_final else "#b0b0b0"
        parts.append(f'<circle cx="{sx(p.x):.1f}" cy="{sy(p.y):.1f}" r="{r}" fill="{fill}"/>')
        if labels and is_final:
            nm = by_id.get(oid, None)
            name = nm.raw.get("name", "") if nm else ""
            if name:
                parts.append(
                    f'<text x="{sx(p.x)+3:.1f}" y="{sy(p.y)-3:.1f}" font-size="7" '
                    f'fill="#444">{name}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _ref(o, attr: str) -> int:
    v = o.raw.get(attr, "")
    return int(v) if v.lstrip("-").isdigit() else -1


# --- single-piece (block) render --------------------------------------------
_PATH_STROKE = {
    "dart": ("#c026d3", 1.6, "5,3"),
    "drill_hole": ("#0891b2", 1.3, None),
    "notch": ("#ea580c", 1.4, None),
    "grainline": ("#16a34a", 1.4, None),
    "guide": ("#9ca3af", 1.0, "3,3"),
}


def render_piece_svg(pattern: Pattern, ev: Evaluated, piece, *, width: int = 900,
                     labels: bool = True, padding: float = 3.0) -> str:
    """Render ONE piece (block): its real seam/cut outline bold, internal paths
    (darts/drill-holes/guides) styled, grainline drawn, outline vertices labeled.
    Construction scaffolding is intentionally omitted for a clean, legible view.
    """
    from . import pieces as P

    outline = P.piece_outline_points(pattern, ev, piece)
    ipaths = P.piece_internal_paths(pattern, ev, piece)
    grain = P.piece_grainline(pattern, ev, piece)
    verts = P.piece_outline_vertices(pattern, ev, piece)

    all_pts = list(outline)
    for _, _, pts in ipaths:
        all_pts += pts
    if grain:
        all_pts += list(grain)
    if not all_pts:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'

    xs = [p.x for p in all_pts]
    ys = [p.y for p in all_pts]
    minx, maxx = min(xs) - padding, max(xs) + padding
    miny, maxy = min(ys) - padding, max(ys) + padding
    span_x = maxx - minx or 1.0
    span_y = maxy - miny or 1.0
    scale = width / span_x
    height = int(span_y * scale)

    def sx(x: float) -> float:
        return (x - minx) * scale

    def sy(y: float) -> float:
        return (y - miny) * scale

    def path_d(pts):
        return "M " + " L ".join(f"{sx(p.x):.1f},{sy(p.y):.1f}" for p in pts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#fbfbfb"/>',
    ]
    if outline:
        parts.append(f'<path d="{path_d(outline)} Z" fill="#eef1f6" stroke="#111" '
                     f'stroke-width="2.4" stroke-linejoin="round"/>')
    for role, _name, pts in ipaths:
        color, w, dash = _PATH_STROKE.get(role, _PATH_STROKE["guide"])
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<path d="{path_d(pts)}" fill="none" stroke="{color}" '
                     f'stroke-width="{w}"{da}/>')
    if grain:
        a, b = grain
        parts.append(f'<line x1="{sx(a.x):.1f}" y1="{sy(a.y):.1f}" x2="{sx(b.x):.1f}" '
                     f'y2="{sy(b.y):.1f}" stroke="#16a34a" stroke-width="1.4"/>')
    for name, p in verts:
        parts.append(f'<circle cx="{sx(p.x):.1f}" cy="{sy(p.y):.1f}" r="2.4" fill="#111"/>')
        if labels and name:
            parts.append(f'<text x="{sx(p.x)+3:.1f}" y="{sy(p.y)-3:.1f}" font-size="8" '
                         f'fill="#444">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)
