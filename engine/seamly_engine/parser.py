"""Parse a Seamly2D ``.sm2d`` pattern (newer dialect: ``<draftBlock>`` /
``<calculation>`` / ``lineType`` / ``spline type="cubicBezier"``) into the
object model. The parser is deliberately permissive and lossless: it records
every attribute in ``raw`` so the writer can reproduce the file, and it derives
``refs`` (the ids each object is built from) for dependency ordering.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .model import (
    DraftBlock,
    Increment,
    InternalPath,
    ModelingObject,
    Pattern,
    PatternObject,
    Piece,
    PieceNode,
)

# Attributes whose values are object-id references, by element/tool.
_REF_ATTRS = (
    "basePoint", "firstPoint", "secondPoint", "thirdPoint",
    "center", "curve", "p1Line", "p2Line",
    "baseLineP1", "baseLineP2", "dartP1", "dartP2", "dartP3",
    "point1", "point2", "point3", "point4",
)


def _collect_refs(attrs: dict[str, str], children: list[dict[str, str]]) -> list[int]:
    refs: list[int] = []
    for a in _REF_ATTRS:
        v = attrs.get(a)
        if v and v.lstrip("-").isdigit():
            refs.append(int(v))
    for ch in children:
        for key in ("pSpline",):
            v = ch.get(key)
            if v and v.isdigit():
                refs.append(int(v))
    return refs


def parse_pattern(path_or_text: str, *, is_text: bool = False) -> Pattern:
    root = ET.fromstring(path_or_text) if is_text else ET.parse(path_or_text).getroot()

    pat = Pattern()
    pat.version = root.findtext("version", default="").strip()
    pat.unit = (root.findtext("unit") or "cm").strip()
    pat.pattern_name = (root.findtext("patternName") or "").strip()
    pat.measurements_file = (root.findtext("measurements") or "").strip()

    inc_root = root.find("increments")
    if inc_root is not None:
        for inc in inc_root.findall("increment"):
            pat.increments.append(Increment(
                name=inc.get("name", ""),
                formula=inc.get("formula", "0"),
                description=inc.get("description", ""),
            ))

    for block in root.findall("draftBlock"):
        db = DraftBlock(name=block.get("name", ""))
        calc = block.find("calculation")
        if calc is not None:
            for el in calc:
                db.objects.append(_parse_object(el))

        # <modeling>: piece-local copies of calculation objects + internal paths.
        modeling = block.find("modeling")
        if modeling is not None:
            for el in modeling:
                if el.tag == "path":
                    db.internal_paths.append(_parse_internal_path(el))
                elif el.get("idObject"):
                    db.modeling.append(ModelingObject(
                        id=int(el.get("id", "0")),
                        id_object=int(el.get("idObject", "0")),
                        modeling_type=el.get("type", ""),
                    ))

        # <pieces>: the final pattern pieces (seam/cut outline + metadata).
        pieces = block.find("pieces")
        if pieces is not None:
            for piece in pieces.findall("piece"):
                pat.pieces.append(_parse_piece(piece))

        pat.draft_blocks.append(db)

    return pat


def _parse_internal_path(el: ET.Element) -> InternalPath:
    node_ids: list[int] = []
    nodes = el.find("nodes")
    if nodes is not None:
        for node in nodes.findall("node"):
            oid = node.get("idObject", "0")
            if oid.isdigit():
                node_ids.append(int(oid))
    return InternalPath(
        id=int(el.get("id", "0")),
        name=el.get("name", ""),
        line_type=el.get("lineType", ""),
        node_ids=node_ids,
    )


def _parse_object(el: ET.Element) -> PatternObject:
    attrs = dict(el.attrib)
    if el.tag == "operation":
        return _parse_operation(el, attrs)
    children = [dict(c.attrib) | {"__tag__": c.tag} for c in el]
    obj = PatternObject(
        id=int(attrs.get("id", "0")),
        tag=el.tag,
        tool_type=attrs.get("type", ""),
        raw=attrs,
        children=children,
    )
    obj.refs = _collect_refs(attrs, children)
    return obj


def _parse_operation(el: ET.Element, attrs: dict[str, str]) -> PatternObject:
    """Operations (rotation/moving/flippingByLine/flippingByAxis) transform a
    list of source objects into a parallel list of destination objects. We keep
    source/destination items paired by position as children so the evaluator can
    map each source id to the new destination id it produces."""
    children: list[dict[str, str]] = []
    sources = [dict(i.attrib) for i in el.findall("./source/item")]
    dests = [dict(i.attrib) for i in el.findall("./destination/item")]
    for src, dst in zip(sources, dests):
        children.append({
            "src": src.get("idObject", ""),
            "dst": dst.get("idObject", ""),
        })
    obj = PatternObject(
        id=int(attrs.get("id", "0")),
        tag="operation",
        tool_type=attrs.get("type", ""),
        raw=attrs,
        children=children,
    )
    obj.refs = _collect_refs(attrs, [])
    obj.refs += [int(c["src"]) for c in children if c["src"].isdigit()]
    return obj


def _parse_piece(piece_el: ET.Element) -> Piece:
    piece = Piece(
        id=int(piece_el.get("id", "0")),
        name=piece_el.get("name", ""),
        seam_allowance=piece_el.get("seamAllowance", "false") == "true",
        width=piece_el.get("width", ""),
        raw=dict(piece_el.attrib),
    )
    nodes = piece_el.find("nodes")
    if nodes is not None:
        for node in nodes.findall("node"):
            oid = node.get("idObject") or node.get("id") or "0"
            piece.nodes.append(PieceNode(
                object_id=int(oid) if oid.isdigit() else 0,
                node_type=node.get("type", ""),
                reverse=node.get("reverse", "0") == "1",
            ))
    ipaths = piece_el.find("iPaths")
    if ipaths is not None:
        for rec in ipaths.findall("record"):
            p = rec.get("path", "")
            if p.isdigit():
                piece.internal_path_ids.append(int(p))
    grain = piece_el.find("grainline")
    if grain is not None and grain.get("centerAnchor", "").isdigit():
        piece.grainline_anchor = int(grain.get("centerAnchor"))
        try:
            piece.grainline_rotation = float(grain.get("rotation", "90"))
            piece.grainline_length = float(grain.get("length", "0"))
        except ValueError:
            pass
    anchors = piece_el.find("anchors")
    if anchors is not None:
        for rec in anchors.findall("record"):
            txt = (rec.text or "").strip()
            if txt.isdigit():
                piece.anchor_ids.append(int(txt))
    return piece
