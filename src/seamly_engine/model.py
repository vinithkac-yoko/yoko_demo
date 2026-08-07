"""In-memory object model for a Seamly2D pattern.

The model is a faithful, lossless-as-possible representation of the ``.sm2d``
XML: every construction object keeps its raw attributes (so the writer can round
-trip) plus a normalized shape the evaluator and state exporter consume. The
pattern is a *directed acyclic graph* — every object references earlier objects
by integer id — evaluated in document order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Point


@dataclass
class Increment:
    """A user-defined variable (``<increment>``) with a formula."""

    name: str
    formula: str
    description: str = ""


@dataclass
class PatternObject:
    """Base for anything inside a ``<calculation>`` block.

    ``raw`` holds the original XML attributes verbatim for round-tripping;
    ``tag`` is the XML element name (``point``/``line``/``arc``/``spline``).
    """

    id: int
    tag: str
    tool_type: str  # value of the ``type`` attribute (e.g. "endLine", "simple")
    raw: dict[str, str] = field(default_factory=dict)
    # Ids of objects this one is constructed from (filled by the parser).
    refs: list[int] = field(default_factory=list)
    # Extra children (e.g. pathPoint ids for cubicBezierPath / trueDarts outputs)
    children: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DraftBlock:
    name: str
    objects: list[PatternObject] = field(default_factory=list)
    modeling: list[ModelingObject] = field(default_factory=list)
    internal_paths: list[InternalPath] = field(default_factory=list)
    # Raw XML of sections we parse but don't mutate (modeling / pieces / groups),
    # kept verbatim so writing an imported pattern round-trips faithfully.
    raw_sections: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelingObject:
    """A piece-local copy of a calculation object (``<modeling>`` section).

    ``id`` is the modeling object's own id; ``id_object`` points back to the
    calculation object it copies. Piece outlines reference modeling ids, so this
    map is what connects a final piece to the construction objects it's made of.
    """

    id: int
    id_object: int
    modeling_type: str  # modeling | modelingSpline | modelingPath | anchor
    tag: str = "point"  # point | spline | arc — the XML element to write


@dataclass
class InternalPath:
    """An internal path inside a piece (dart, hip line, drill hole, ...)."""

    id: int
    name: str
    line_type: str
    node_ids: list[int] = field(default_factory=list)  # modeling ids


@dataclass
class PieceNode:
    """One node in a piece main outline. ``object_id`` is a *modeling* id."""

    object_id: int
    node_type: str  # NodePoint | NodeArc | NodeSpline | NodeSplinePath
    reverse: bool = False  # traverse the curve backwards when walking the outline


@dataclass
class Piece:
    """A final pattern piece (``<piece>``): the cut/sew outline plus metadata.

    The calculation objects reachable through ``nodes`` (via the modeling map)
    are exactly what marks an object as part of the *final* pattern rather than
    construction scaffolding. ``internal_paths`` carry darts/notches/drill
    holes; ``grainline_anchor`` / ``anchor_ids`` mark grainline and anchors.
    """

    id: int
    name: str
    seam_allowance: bool = False
    width: str = ""
    nodes: list[PieceNode] = field(default_factory=list)
    internal_path_ids: list[int] = field(default_factory=list)
    grainline_anchor: int | None = None
    grainline_rotation: float = 90.0
    grainline_length: float = 0.0
    anchor_ids: list[int] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class Pattern:
    version: str = ""
    unit: str = "cm"
    pattern_name: str = ""
    measurements_file: str = ""
    increments: list[Increment] = field(default_factory=list)
    draft_blocks: list[DraftBlock] = field(default_factory=list)
    pieces: list[Piece] = field(default_factory=list)
    # Root-level attributes we do not model explicitly, kept for round-tripping.
    raw_header: dict[str, str] = field(default_factory=dict)
    # Verbatim XML of root-level elements we don't model (gradation, patternLabel…)
    raw_root_sections: list[str] = field(default_factory=list)
    description: str = ""
    notes: str = ""
    pattern_number: str = ""

    def all_objects(self) -> list[PatternObject]:
        return [o for b in self.draft_blocks for o in b.objects]

    def object_by_id(self) -> dict[int, PatternObject]:
        return {o.id: o for o in self.all_objects()}


@dataclass
class Evaluated:
    """Result of evaluating the pattern DAG.

    ``points`` maps object id -> resolved coordinate. ``curves`` maps object id
    -> a geometry object (Arc/CubicBezier/BezierPath). ``values`` records the
    resolved numeric value of every formula attribute for inspection/export.
    ``unresolved`` lists ids the engine could not compute yet (e.g. tool types
    not implemented) with the reason, so nothing crashes the whole evaluation.
    """

    points: dict[int, Point] = field(default_factory=dict)
    curves: dict[int, object] = field(default_factory=dict)
    # curve id -> ordered on-curve point ids (for SplPath_/Spl_ partial lengths)
    curve_nodes: dict[int, list[int]] = field(default_factory=dict)
    values: dict[int, dict[str, float]] = field(default_factory=dict)
    increment_values: dict[str, float] = field(default_factory=dict)
    unresolved: dict[int, str] = field(default_factory=dict)
