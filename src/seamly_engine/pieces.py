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


def build_piece(
    pattern: Pattern,
    next_id,
    name: str,
    node_ids: list[int | tuple[int, bool]],
    *,
    seam_allowance: bool = True,
    width: str = "1",
    internal_paths: list[dict] | None = None,
    grainline_anchor: int | None = None,
    block_index: int = 0,
) -> Piece:
    """Create a cut piece from construction objects.

    Seamly represents a piece in two layers: a ``<modeling>`` copy of each
    construction object it uses, and a ``<piece>`` whose ``<nodes>`` reference
    those copies in outline order. This builds both, so the result is a real
    piece — it renders, exports, and reopens in Seamly2D.

    ``next_id`` is a callable returning a fresh object id. Each entry of
    ``node_ids`` is either a bare calculation id (curve nodes default to
    ``reverse=False``) or an ``(id, reverse)`` pair — :func:`split_piece` and
    :func:`merge_piece` need the pair form to carry a curve's traversal
    direction through unchanged when they copy part of an existing outline.
    """
    db = pattern.draft_blocks[block_index]
    by_id = pattern.object_by_id()

    def _model(calc_id: int) -> tuple[int, str]:
        """Add (or reuse) a modeling copy of a calculation object."""
        obj = by_id.get(calc_id)
        if obj is None:
            raise ValueError(f"no object with id {calc_id}")
        mtype, ntype, tag = _node_kinds(obj)
        for m in db.modeling:  # reuse an existing copy
            if m.id_object == calc_id and m.modeling_type == mtype:
                return m.id, ntype
        mid = next_id()
        db.modeling.append(ModelingObject(id=mid, id_object=calc_id, modeling_type=mtype, tag=tag))
        return mid, ntype

    piece = Piece(id=next_id(), name=name, seam_allowance=seam_allowance, width=width)
    piece.raw = {
        "name": name,
        "seamAllowance": str(seam_allowance).lower(),
        "width": width,
        "united": "false",
        "inLayout": "true",
        "forbidFlipping": "false",
        "hideMainPath": "false",
        "version": "2",
    }

    for entry in node_ids:
        calc_id, reverse = entry if isinstance(entry, tuple) else (entry, False)
        mid, ntype = _model(calc_id)
        piece.nodes.append(PieceNode(object_id=mid, node_type=ntype, reverse=reverse))

    for spec in internal_paths or []:
        ids = [_model(i)[0] for i in spec.get("node_ids", [])]
        if not ids:
            continue  # a drill hole is legitimately a single arc/circle reference
        ip = InternalPath(
            id=next_id(),
            name=spec.get("name", "path"),
            line_type=spec.get("line_type", "dashDotLine"),
            node_ids=ids,
        )
        db.internal_paths.append(ip)
        piece.internal_path_ids.append(ip.id)

    if grainline_anchor is not None:
        aid = next_id()
        db.modeling.append(
            ModelingObject(id=aid, id_object=grainline_anchor, modeling_type="anchor", tag="point")
        )
        piece.grainline_anchor = aid
        piece.anchor_ids.append(aid)
        piece.grainline_length = 15.0

    pattern.pieces.append(piece)
    return piece


# --- splitting and merging -----------------------------------------------------
# A piece's outline is a cycle of NodePoint/curve entries (see the module
# docstring). Splitting and merging both come down to the same move: find two
# named vertices on the cycle, and work with the two arcs between them —
# ``_arc_between`` walks one, forward, with wraparound.


#: An outline entry, independent of any real ``PieceNode`` — (calc_id,
#: node_type, reverse). Working over plain tuples (rather than ``piece.nodes``
#: directly) is what lets :func:`_splice_point` insert a point that was never
#: literally in the piece's node list.
_Entry = tuple[int, str, bool]


def _outline_entries(piece: Piece, mm: dict[int, int]) -> list[_Entry]:
    out: list[_Entry] = []
    for node in piece.nodes:
        calc = mm.get(node.object_id)
        reverse = node.reverse if node.node_type != "NodePoint" else False
        out.append((calc, node.node_type, reverse))
    return out


def _entry_index(entries: list[_Entry], calc_id: int) -> int:
    for i, (c, t, _r) in enumerate(entries):
        if t == "NodePoint" and c == calc_id:
            return i
    return -1


def _splice_point(entries: list[_Entry], ev: Evaluated, calc_id: int) -> list[_Entry] | None:
    """A split endpoint doesn't have to already be an outline vertex: a point
    freshly constructed with the ordinary point tools works too, as long as it
    sits exactly on one of the piece's *straight* edges (there is no object to
    cut when the edge is implicit — inserting the vertex is enough). A point
    on a curved edge isn't supported: splitting the underlying curve itself is
    a bigger operation this doesn't attempt. Returns ``None`` if the point is
    usable neither as-is nor by splicing.
    """
    if _entry_index(entries, calc_id) != -1:
        return entries
    p = ev.points.get(calc_id)
    if p is None:
        return None
    n = len(entries)
    for i in range(n):
        c1, t1, _ = entries[i]
        c2, t2, _ = entries[(i + 1) % n]
        if t1 != "NodePoint" or t2 != "NodePoint":
            continue  # a curve occupies this edge; splicing isn't supported
        p1, p2 = ev.points.get(c1), ev.points.get(c2)
        if p1 is None or p2 is None:
            continue
        if geo.point_on_segment(p, p1, p2):
            spliced = list(entries)
            spliced.insert(i + 1, (calc_id, "NodePoint", False))
            return spliced
    return None


def _arc_between(entries: list[_Entry], i_start: int, i_end: int) -> list[tuple[int, bool]]:
    """Walk ``entries`` forward from index ``i_start`` to ``i_end`` (inclusive,
    wrapping around the cycle if needed), as (calc_id, reverse) pairs ready to
    feed back into :func:`build_piece`."""
    n = len(entries)
    idxs = [i_start]
    i = i_start
    while i != i_end:
        i = (i + 1) % n
        idxs.append(i)
    return [(entries[k][0], entries[k][2]) for k in idxs]


def _arc_polygon(ev: Evaluated, arc: list[tuple[int, bool]]) -> list[geo.Point]:
    """Resolved boundary points of an in-progress arc (not yet a real Piece),
    closed — for containment tests before the new pieces exist."""
    pts: list[geo.Point] = []
    for calc_id, reverse in arc:
        if calc_id in ev.points:
            _append(pts, ev.points[calc_id])
        elif calc_id in ev.curves:
            poly = ev.curves[calc_id].polyline()
            if reverse:
                poly = list(reversed(poly))
            for p in poly:
                _append(pts, p)
    if pts:
        _append(pts, pts[0])
    return pts


def _assign_by_containment(
    pattern: Pattern,
    ev: Evaluated,
    piece: Piece,
    poly_a: list[geo.Point],
    poly_b: list[geo.Point],
) -> tuple[list[str], list[str], list[str]]:
    """Sort a piece's internal paths and grainline into "belongs to A", "belongs
    to B", or "dropped" (straddles the cut, or can't be resolved), by testing
    every point of each against both new outlines. Returns (ids_a, ids_b, dropped_names)
    for internal paths; the grainline is handled by the caller separately."""
    ips = internal_paths_by_id(pattern)
    mm = modeling_map(pattern)
    ids_a: list[str] = []
    ids_b: list[str] = []
    dropped: list[str] = []
    for ipid in piece.internal_path_ids:
        ip = ips.get(ipid)
        if ip is None:
            continue
        pts: list[geo.Point] = []
        ok = True
        for mid in ip.node_ids:
            calc = mm.get(mid)
            if calc in ev.points:
                pts.append(ev.points[calc])
            elif calc in ev.curves:
                pts.extend(ev.curves[calc].polyline())
            else:
                ok = False
                break
        if ok and pts and all(geo.point_in_polygon(p, poly_a) for p in pts):
            ids_a.append(str(ipid))
        elif ok and pts and all(geo.point_in_polygon(p, poly_b) for p in pts):
            ids_b.append(str(ipid))
        else:
            dropped.append(ip.name)
    return ids_a, ids_b, dropped


def split_piece(
    pattern: Pattern,
    ev: Evaluated,
    next_id,
    piece_id: int,
    point_a_id: int,
    point_b_id: int,
    *,
    name_a: str,
    name_b: str,
    seam_allowance: bool | None = None,
    width: str | None = None,
    block_index: int = 0,
) -> tuple[Piece, Piece, list[str]]:
    """Cut a piece into two along the straight line between two of its own
    outline vertices.

    ``point_a_id`` / ``point_b_id`` are the *construction* ids of the cut's two
    ends. Each is either an existing outline vertex, or a point you've already
    built (with the ordinary point tools) that sits exactly on one of the
    piece's straight edges — it gets spliced into the outline as part of the
    split. A point on a *curved* edge isn't supported: that would mean cutting
    the curve object itself, which this doesn't attempt. The new edge is a
    straight line between the two points: the same implicit-straight-edge
    convention Seamly already uses between any two consecutive outline
    vertices, so splitting creates no new construction geometry, only two new
    pieces.

    Internal paths (darts, drill holes) are assigned to whichever new piece
    geometrically contains them; a grainline follows its anchor the same way.
    Anything that straddles the cut is dropped — its name comes back in the
    third element of the result so the caller can report it — rather than
    silently kept on the wrong side or split itself.

    Raises ``ValueError`` (never partially mutates) if either point can't be
    placed on the outline, or the cut is degenerate.
    """
    piece = piece_by_id(pattern, piece_id)
    if piece is None:
        raise ValueError(f"no piece with id {piece_id}")
    mm = modeling_map(pattern)
    entries = _outline_entries(piece, mm)
    entries = _splice_point(entries, ev, point_a_id)
    if entries is None:
        raise ValueError(
            f"point {point_a_id} isn't on {piece.name!r}'s outline (or sits on a "
            "curved edge, which isn't supported)"
        )
    entries = _splice_point(entries, ev, point_b_id)
    if entries is None:
        raise ValueError(
            f"point {point_b_id} isn't on {piece.name!r}'s outline (or sits on a "
            "curved edge, which isn't supported)"
        )
    i = _entry_index(entries, point_a_id)
    j = _entry_index(entries, point_b_id)
    if i == j:
        raise ValueError("a split needs two different points")

    arc_a = _arc_between(entries, i, j)  # point_a -> ... -> point_b
    arc_b = _arc_between(entries, j, i)  # point_b -> ... -> point_a
    # A piece needs at least 3 outline nodes (matches create_piece's own
    # minimum) — an arc of exactly the 2 cut endpoints, with nothing between
    # them, would close back on itself as a doubled line, not a real shape.
    if len(arc_a) < 3 or len(arc_b) < 3:
        raise ValueError("that cut doesn't leave two real outlines")

    sa = piece.seam_allowance if seam_allowance is None else seam_allowance
    w = piece.width if width is None else width

    poly_a = _arc_polygon(ev, arc_a)
    poly_b = _arc_polygon(ev, arc_b)
    ids_a, ids_b, dropped = _assign_by_containment(pattern, ev, piece, poly_a, poly_b)
    ip_by_id = internal_paths_by_id(pattern)

    def _copy_paths(ids: list[str]) -> list[dict]:
        specs = []
        for sid in ids:
            ip = ip_by_id[int(sid)]
            calc_ids = [mm[n] for n in ip.node_ids if mm.get(n) is not None]
            specs.append({"name": ip.name, "line_type": ip.line_type, "node_ids": calc_ids})
        return specs

    anchor_a = anchor_b = None
    if piece.grainline_anchor is not None:
        gp = ev.points.get(mm.get(piece.grainline_anchor))
        if gp is not None:
            if geo.point_in_polygon(gp, poly_a):
                anchor_a = mm.get(piece.grainline_anchor)
            elif geo.point_in_polygon(gp, poly_b):
                anchor_b = mm.get(piece.grainline_anchor)
            else:
                dropped.append("grainline")

    piece_a = build_piece(
        pattern,
        next_id,
        name_a,
        arc_a,
        seam_allowance=sa,
        width=w,
        internal_paths=_copy_paths(ids_a),
        grainline_anchor=anchor_a,
        block_index=block_index,
    )
    piece_b = build_piece(
        pattern,
        next_id,
        name_b,
        arc_b,
        seam_allowance=sa,
        width=w,
        internal_paths=_copy_paths(ids_b),
        grainline_anchor=anchor_b,
        block_index=block_index,
    )
    delete_piece(pattern, piece_id)
    return piece_a, piece_b, dropped


def merge_piece(
    pattern: Pattern,
    next_id,
    piece_a_id: int,
    piece_b_id: int,
    edge_point_1: int,
    edge_point_2: int,
    *,
    name: str,
    seam_allowance: bool | None = None,
    width: str | None = None,
    block_index: int = 0,
) -> Piece:
    """Combine two pieces into one along a seam they share.

    ``edge_point_1`` / ``edge_point_2`` name the two ends of the shared edge —
    both pieces must already be built from the *same* construction objects
    along that edge (this is the exact-inverse case of :func:`split_piece`; it
    is also what you get from two pieces deliberately drafted to share a seam).
    The shared arc is dissolved and the two remaining arcs are stitched into
    one closed outline.

    Internal paths and the grainline carry over from whichever source piece
    had them (both pieces' are kept — a merge cannot straddle-drop anything,
    since nothing is being cut).

    Raises ``ValueError`` if the two pieces don't share an identical sequence
    of construction points between the named endpoints (in either winding
    direction) — the everyday reason is that "the same seam" was drafted with
    two different sets of points that only happen to land on the same
    coordinates; :func:`split_piece` never produces that problem, an
    independently-drafted pair of pieces sometimes will.
    """
    pa = piece_by_id(pattern, piece_a_id)
    pb = piece_by_id(pattern, piece_b_id)
    if pa is None:
        raise ValueError(f"no piece with id {piece_a_id}")
    if pb is None:
        raise ValueError(f"no piece with id {piece_b_id}")
    mm = modeling_map(pattern)
    entries_a = _outline_entries(pa, mm)
    entries_b = _outline_entries(pb, mm)

    ia1 = _entry_index(entries_a, edge_point_1)
    ia2 = _entry_index(entries_a, edge_point_2)
    ib1 = _entry_index(entries_b, edge_point_1)
    ib2 = _entry_index(entries_b, edge_point_2)
    if -1 in (ia1, ia2):
        raise ValueError(f"the shared edge isn't on {pa.name!r}'s outline")
    if -1 in (ib1, ib2):
        raise ValueError(f"the shared edge isn't on {pb.name!r}'s outline")

    a_12 = _arc_between(entries_a, ia1, ia2)
    a_21 = _arc_between(entries_a, ia2, ia1)
    b_12 = _arc_between(entries_b, ib1, ib2)
    b_21 = _arc_between(entries_b, ib2, ib1)

    def ids_only(arc: list[tuple[int, bool]]) -> list[int]:
        return [c for c, _r in arc]

    # The shared seam runs opposite ways around each piece's boundary: find
    # which of piece_a's two arcs is the exact reverse (by construction id)
    # of one of piece_b's, and dissolve that pair.
    kept_a = kept_b = None
    if ids_only(a_12) == list(reversed(ids_only(b_12))):
        kept_a, kept_b = a_21, b_21
    elif ids_only(a_12) == list(reversed(ids_only(b_21))):
        kept_a, kept_b = a_21, b_12
    elif ids_only(a_21) == list(reversed(ids_only(b_12))):
        kept_a, kept_b = a_12, b_21
    elif ids_only(a_21) == list(reversed(ids_only(b_21))):
        kept_a, kept_b = a_12, b_12

    if kept_a is None:
        raise ValueError(
            f"{pa.name!r} and {pb.name!r} don't reference the same construction "
            "points between those two edges — not the same seam"
        )

    # kept_a ends where kept_b starts (both at one of the shared endpoints);
    # drop kept_b's first entry so that junction point isn't duplicated.
    merged_nodes = kept_a + kept_b[1:]
    sa = pa.seam_allowance if seam_allowance is None else seam_allowance
    w = pa.width if width is None else width

    def _copy_paths(src: Piece) -> list[dict]:
        ip_by_id = internal_paths_by_id(pattern)
        specs = []
        for pid in src.internal_path_ids:
            ip = ip_by_id.get(int(pid))
            if ip is None:
                continue
            calc_ids = [mm[n] for n in ip.node_ids if mm.get(n) is not None]
            specs.append({"name": ip.name, "line_type": ip.line_type, "node_ids": calc_ids})
        return specs

    grainline_anchor = None
    if pa.grainline_anchor is not None:
        grainline_anchor = mm.get(pa.grainline_anchor)
    elif pb.grainline_anchor is not None:
        grainline_anchor = mm.get(pb.grainline_anchor)

    merged = build_piece(
        pattern,
        next_id,
        name,
        merged_nodes,
        seam_allowance=sa,
        width=w,
        internal_paths=_copy_paths(pa) + _copy_paths(pb),
        grainline_anchor=grainline_anchor,
        block_index=block_index,
    )
    delete_piece(pattern, piece_a_id)
    delete_piece(pattern, piece_b_id)
    return merged


def delete_piece(pattern: Pattern, piece_id: int) -> bool:
    """Remove a piece. Its construction geometry is untouched — a piece is a
    view onto construction objects, not their owner — so this never cascades.
    Returns False if no such piece existed."""
    before = len(pattern.pieces)
    pattern.pieces = [p for p in pattern.pieces if p.id != piece_id]
    return len(pattern.pieces) < before


def edit_piece(
    pattern: Pattern,
    piece_id: int,
    *,
    name: str | None = None,
    seam_allowance: bool | None = None,
    width: str | None = None,
) -> Piece:
    """Rename a piece or change its seam allowance / width in place."""
    piece = piece_by_id(pattern, piece_id)
    if piece is None:
        raise ValueError(f"no piece with id {piece_id}")
    if name is not None:
        piece.name = name
    if seam_allowance is not None:
        piece.seam_allowance = seam_allowance
    if width is not None:
        piece.width = width
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
        out.append(
            {
                "key": key,
                "label": label,
                "piece_ids": [p.id for p in ps],
                "pieces": [p.name for p in ps],
            }
        )
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
    return [
        p
        for p in pattern.pieces
        if (p.name.split(" - ", 1)[0].strip() if " - " in p.name else p.name) == key
    ]


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
                relevant.add(prod)
                stack.append(prod)
            continue
        for r in o.refs:
            if r not in relevant:
                relevant.add(r)
                stack.append(r)
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


def piece_outline_vertices(
    pattern: Pattern, ev: Evaluated, piece: Piece
) -> list[tuple[str, geo.Point]]:
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


def piece_internal_paths(
    pattern: Pattern, ev: Evaluated, piece: Piece
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


def piece_grainline(
    pattern: Pattern, ev: Evaluated, piece: Piece
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
