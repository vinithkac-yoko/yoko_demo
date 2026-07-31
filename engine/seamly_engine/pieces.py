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
from .model import Evaluated, InternalPath, ModelingObject, Pattern, Piece, PieceNode


def _node_kinds(obj) -> tuple[str, str, str]:
    """(modeling_type, node_type, xml_tag) for a calculation object used in a piece."""
    if obj.tag == "spline":
        if obj.tool_type == "cubicBezierPath":
            return "modelingPath", "NodeSplinePath", "spline"
        return "modelingSpline", "NodeSpline", "spline"
    if obj.tag in ("arc", "elArc"):
        return "modeling", "NodeArc", "arc"
    return "modeling", "NodePoint", "point"


def build_piece(pattern: Pattern, next_id, name: str, node_ids: list[int], *,
                seam_allowance: bool = True, width: str = "1",
                internal_paths: list[dict] | None = None,
                grainline_anchor: int | None = None,
                block_index: int = 0) -> Piece:
    """Create a cut piece from construction objects.

    Seamly represents a piece in two layers: a ``<modeling>`` copy of each
    construction object it uses, and a ``<piece>`` whose ``<nodes>`` reference
    those copies in outline order. This builds both, so the result is a real
    piece — it renders, exports, and reopens in Seamly2D.

    ``next_id`` is a callable returning a fresh object id.
    """
    db = pattern.draft_blocks[block_index]
    by_id = pattern.object_by_id()

    def _model(calc_id: int) -> tuple[int, str]:
        """Add (or reuse) a modeling copy of a calculation object."""
        obj = by_id.get(calc_id)
        if obj is None:
            raise ValueError(f"no object with id {calc_id}")
        mtype, ntype, tag = _node_kinds(obj)
        for m in db.modeling:                      # reuse an existing copy
            if m.id_object == calc_id and m.modeling_type == mtype:
                return m.id, ntype
        mid = next_id()
        db.modeling.append(ModelingObject(id=mid, id_object=calc_id,
                                          modeling_type=mtype, tag=tag))
        return mid, ntype

    piece = Piece(id=next_id(), name=name, seam_allowance=seam_allowance, width=width)
    piece.raw = {"name": name, "seamAllowance": str(seam_allowance).lower(),
                 "width": width, "united": "false", "inLayout": "true",
                 "forbidFlipping": "false", "hideMainPath": "false", "version": "2"}

    for calc_id in node_ids:
        mid, ntype = _model(calc_id)
        piece.nodes.append(PieceNode(object_id=mid, node_type=ntype))

    for spec in (internal_paths or []):
        ids = [_model(i)[0] for i in spec.get("node_ids", [])]
        if len(ids) < 2:
            continue
        ip = InternalPath(id=next_id(), name=spec.get("name", "path"),
                          line_type=spec.get("line_type", "dashDotLine"), node_ids=ids)
        db.internal_paths.append(ip)
        piece.internal_path_ids.append(ip.id)

    if grainline_anchor is not None:
        aid = next_id()
        db.modeling.append(ModelingObject(id=aid, id_object=grainline_anchor,
                                          modeling_type="anchor", tag="point"))
        piece.grainline_anchor = aid
        piece.anchor_ids.append(aid)
        piece.grainline_length = 15.0

    pattern.pieces.append(piece)
    return piece


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


def group_pieces(pattern: Pattern) -> list[dict]:
    """Group pieces into **blocks** (garments). Piece names are like
    ``"A - Skirt Back"`` / ``"A - Skirt Front"`` — the ``A``/``B``/``C`` prefix is
    the block, and Front/Back are its pieces. Returns one entry per block with a
    friendly label and the member piece ids."""
    groups: dict[str, list[Piece]] = {}
    order: list[str] = []
    for p in pattern.pieces:
        key = p.name.split(" - ", 1)[0].strip() if " - " in p.name else p.name
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(p)

    out: list[dict] = []
    for key in order:
        ps = groups[key]
        label = _block_label(ps)
        out.append({
            "key": key,
            "label": label,
            "piece_ids": [p.id for p in ps],
            "pieces": [p.name for p in ps],
        })
    return out


def _block_label(ps: list[Piece]) -> str:
    """Derive a block name from its pieces by stripping the prefix and Front/Back."""
    bases: list[str] = []
    for p in ps:
        n = p.name.split(" - ", 1)[1] if " - " in p.name else p.name
        for w in (" Front", " Back", " front", " back"):
            n = n.replace(w, "")
        bases.append(n.strip())
    return bases[0] if bases else "Block"


def pieces_for_key(pattern: Pattern, key: str) -> list[Piece]:
    return [p for p in pattern.pieces
            if (p.name.split(" - ", 1)[0].strip() if " - " in p.name else p.name) == key]


def scoped_ids(pattern: Pattern, target_pieces: list[Piece]) -> set[int]:
    """Calc object ids for a set of pieces (a block): their real objects plus the
    construction geometry that drives them (transitive dependencies)."""
    by_id = pattern.object_by_id()
    # inline output points (trueDarts / operation destinations) -> producer element
    producer: dict[int, int] = {}
    for o in pattern.all_objects():
        if o.tool_type == "trueDarts":
            for a in ("point1", "point2"):
                v = o.raw.get(a, "")
                if v.isdigit():
                    producer[int(v)] = o.id
        elif o.tag == "operation":
            for ch in o.children:
                d = ch.get("dst", "")
                if d.isdigit():
                    producer[int(d)] = o.id

    relevant: set[int] = set()
    for pc in target_pieces:
        relevant |= piece_calc_ids(pattern, pc)
    stack = list(relevant)
    while stack:
        oid = stack.pop()
        o = by_id.get(oid)
        if o is None:
            prod = producer.get(oid)
            if prod is not None and prod not in relevant:
                relevant.add(prod); stack.append(prod)
            continue
        for r in o.refs:
            if r not in relevant:
                relevant.add(r); stack.append(r)
    return relevant


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
