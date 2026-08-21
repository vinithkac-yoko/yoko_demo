"""Tests for splitting and merging pieces.

Two fixtures: a small hand-built rectangle (for controlled, exact-coordinate
assertions — splice behavior, rollback, merge mismatches) and the real Aldrich
Skirt Back piece (for a split/merge round-trip against production geometry,
including its five internal paths — two darts, a reference line, and two
drill holes represented as bare arcs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import seamly_engine as se
from seamly_engine.authoring import new_pattern
from seamly_engine.operations import PatternSession
from seamly_engine.pieces import internal_paths_by_id, piece_outline_points

FIX = Path(__file__).resolve().parent / "fixtures"


def _ids(session: PatternSession) -> dict[str, int]:
    return {o.raw.get("name"): o.id for o in session.pattern.all_objects() if o.raw.get("name")}


def _rounded(pts) -> list[tuple[float, float]]:
    return sorted((round(p.x, 4), round(p.y, 4)) for p in pts)


@pytest.fixture
def rectangle():
    """A 20x10 rectangle: A(0,0) B(20,0) C(20,10) D(0,10), plus a dart-like
    internal path near the AB edge and a grainline anchored at the centre."""
    session = PatternSession(new_pattern("rect", with_defaults=False))
    origin = _ids(session)["A"]

    def pt(name, base, angle, length):
        res = session.add_object(
            "point",
            "endLine",
            {"name": name, "basePoint": str(base), "angle": angle, "length": length},
        )
        assert res.ok, res.message

    pt("B", origin, "0", "20")
    pt("D", origin, "270", "10")
    ids = _ids(session)
    res = session.add_object(
        "point",
        "intersectXY",
        {"name": "C", "firstPoint": str(ids["B"]), "secondPoint": str(ids["D"])},
    )
    assert res.ok, res.message
    ids = _ids(session)

    # a small internal "dart" near the AB edge, apex pointing into the body
    pt("tip1", origin, "0", "8")
    ids = _ids(session)
    res = session.add_object(
        "point",
        "endLine",
        {"name": "apex2", "basePoint": str(ids["tip1"]), "angle": "270", "length": "3"},
    )
    assert res.ok, res.message
    pt("tip2", origin, "0", "12")
    ids = _ids(session)

    res = session.create_piece(
        "Rect",
        [origin, ids["B"], ids["C"], ids["D"]],
        internal_paths=[
            {"name": "TestDart", "node_ids": [ids["tip1"], ids["apex2"], ids["tip2"]]},
        ],
        grainline_anchor=ids["apex2"],
    )
    assert res.ok, res.message
    return session, ids


@pytest.fixture
def aldrich_skirt_back():
    session = PatternSession(
        se.load_pattern(str(FIX / "aldrich_basic.sm2d")),
        se.load_measurements(str(FIX / "aldrich_measurements.vst")),
    )
    piece = next(p for p in session.pattern.pieces if p.name == "A - Skirt Back")
    return session, piece


# --- splitting -----------------------------------------------------------------
def test_split_produces_two_real_pieces_and_retires_the_original(rectangle):
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")

    res = session.split_piece(original.id, ids["A"], ids["C"], name_a="Left", name_b="Right")
    assert res.ok, res.message

    names = {p.name for p in session.pattern.pieces}
    assert "Rect" not in names
    assert {"Left", "Right"} <= names
    assert session.evaluated.unresolved == {}


def test_split_between_named_endpoints_matches_hand_computed_geometry(rectangle):
    """A-B-C is one side of a diagonal split, D is on the other."""
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    session.split_piece(original.id, ids["A"], ids["C"], name_a="ABC", name_b="ADC")

    abc = next(p for p in session.pattern.pieces if p.name == "ABC")
    adc = next(p for p in session.pattern.pieces if p.name == "ADC")
    out_abc = piece_outline_points(session.pattern, session.evaluated, abc)
    out_adc = piece_outline_points(session.pattern, session.evaluated, adc)

    # ABC: triangle through A(0,0) B(20,0) C(20,10); ADC: A(0,0) D(0,10) C(20,10)
    assert (0.0, 0.0) in _rounded(out_abc) and (20.0, 0.0) in _rounded(out_abc)
    assert (0.0, 10.0) in _rounded(out_adc) and (20.0, 10.0) in _rounded(out_adc)


def test_split_endpoint_must_be_on_the_outline(rectangle):
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    res = session.split_piece(original.id, ids["A"], ids["apex2"], name_a="X", name_b="Y")
    assert res.ok is False
    assert "Rect" in {p.name for p in session.pattern.pieces}  # untouched


def test_split_rolls_back_cleanly_on_failure(rectangle):
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    before_pieces = list(session.pattern.pieces)
    session.split_piece(original.id, ids["A"], ids["apex2"], name_a="X", name_b="Y")
    assert [p.id for p in session.pattern.pieces] == [p.id for p in before_pieces]


def test_a_freshly_built_point_on_a_straight_edge_is_a_valid_split_endpoint(rectangle):
    """The point doesn't have to already be an outline vertex — a point that
    lands exactly on one of the piece's straight edges gets spliced in."""
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")

    # midpoint of A-B (y=0), a genuinely new construction point
    res = session.add_object(
        "point",
        "alongLine",
        {"name": "M", "firstPoint": str(ids["A"]), "secondPoint": str(ids["B"]), "length": "10"},
    )
    assert res.ok, res.message
    mid = _ids(session)["M"]

    res = session.split_piece(original.id, mid, ids["D"], name_a="Left", name_b="Right")
    assert res.ok, res.message
    left = next(p for p in session.pattern.pieces if p.name == "Left")
    out = piece_outline_points(session.pattern, session.evaluated, left)
    assert (10.0, 0.0) in _rounded(out)


def test_a_point_off_every_edge_cannot_be_spliced(rectangle):
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    # apex2 is interior, not on any edge of the rectangle
    res = session.split_piece(original.id, ids["A"], ids["apex2"], name_a="X", name_b="Y")
    assert res.ok is False
    assert "isn't on" in res.message


def test_internal_path_fully_on_one_side_follows_it(rectangle):
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    # The dart (tip1=(8,0), apex2=(8,3), tip2=(12,0)) sits near the A-B edge.
    # Cut along the other diagonal, B-D: the arc through C excludes A and the
    # dart entirely; the arc through A keeps both.
    res = session.split_piece(original.id, ids["B"], ids["D"], name_a="WithC", name_b="WithA")
    assert res.ok, res.message
    with_c = next(p for p in session.pattern.pieces if p.name == "WithC")
    with_a = next(p for p in session.pattern.pieces if p.name == "WithA")
    assert with_a.internal_path_ids and with_a.grainline_anchor is not None
    assert not with_c.internal_path_ids and with_c.grainline_anchor is None


def test_internal_path_straddling_the_cut_is_dropped_and_reported(rectangle):
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    # vertical split at x=10 cuts straight through the dart (tip1 x=8, tip2 x=12)
    res = session.add_object(
        "point",
        "endLine",
        {"name": "top_mid", "basePoint": str(ids["D"]), "angle": "0", "length": "10"},
    )
    assert res.ok, res.message
    res = session.add_object(
        "point",
        "endLine",
        {"name": "bot_mid", "basePoint": str(ids["A"]), "angle": "0", "length": "10"},
    )
    assert res.ok, res.message
    ids = _ids(session)
    res = session.split_piece(
        original.id, ids["bot_mid"], ids["top_mid"], name_a="Left", name_b="Right"
    )
    assert res.ok, res.message
    assert "straddled" in res.message and "TestDart" in res.message
    left = next(p for p in session.pattern.pieces if p.name == "Left")
    right = next(p for p in session.pattern.pieces if p.name == "Right")
    assert not left.internal_path_ids and not right.internal_path_ids


def test_split_and_merge_round_trip_a_real_production_piece(aldrich_skirt_back):
    """Split the Aldrich Skirt Back, then merge the two halves back along the
    same edge, and recover geometry identical to the original — including all
    five internal paths (two darts, a reference line, two drill-hole arcs)."""
    session, piece = aldrich_skirt_back
    names = _ids(session)
    a_id, b_id = names["A1"], names["A10"]

    orig_out = _rounded(piece_outline_points(session.pattern, session.evaluated, piece))
    orig_paths = {internal_paths_by_id(session.pattern)[i].name for i in piece.internal_path_ids}

    res = session.split_piece(piece.id, a_id, b_id, name_a="Waistband", name_b="Skirt Body")
    assert res.ok, res.message
    assert "dropped" not in res.message and "straddled" not in res.message

    wb = next(p for p in session.pattern.pieces if p.name == "Waistband")
    sk = next(p for p in session.pattern.pieces if p.name == "Skirt Body")
    assert session.evaluated.unresolved == {}

    res = session.merge_piece(wb.id, sk.id, a_id, b_id, name="Rejoined")
    assert res.ok, res.message
    merged = next(p for p in session.pattern.pieces if p.name == "Rejoined")

    merged_out = _rounded(piece_outline_points(session.pattern, session.evaluated, merged))
    merged_paths = {internal_paths_by_id(session.pattern)[i].name for i in merged.internal_path_ids}
    assert merged_out == orig_out
    assert merged_paths == orig_paths
    assert {p.name for p in session.pattern.pieces} & {"Waistband", "Skirt Body"} == set()
    assert session.evaluated.unresolved == {}


# --- merging ---------------------------------------------------------------
def test_merge_requires_the_same_edge_on_both_sides(rectangle):
    session, ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    session.split_piece(original.id, ids["A"], ids["C"], name_a="ABC", name_b="ADC")
    abc = next(p for p in session.pattern.pieces if p.name == "ABC")
    adc = next(p for p in session.pattern.pieces if p.name == "ADC")

    res = session.merge_piece(abc.id, adc.id, ids["A"], ids["B"], name="Bad")
    assert res.ok is False
    assert {p.name for p in session.pattern.pieces} == {"ABC", "ADC"}  # untouched


def test_merge_of_unrelated_pieces_is_refused(rectangle):
    session, ids = rectangle
    rect = next(p for p in session.pattern.pieces if p.name == "Rect")
    # a second, disjoint piece
    res = session.add_object(
        "point", "endLine", {"name": "Z", "basePoint": str(ids["A"]), "angle": "180", "length": "5"}
    )
    assert res.ok
    res = session.add_object(
        "point",
        "endLine",
        {"name": "Z2", "basePoint": str(_ids(session)["Z"]), "angle": "270", "length": "5"},
    )
    assert res.ok
    res = session.add_object(
        "point",
        "intersectXY",
        {"name": "Z3", "firstPoint": str(ids["A"]), "secondPoint": str(_ids(session)["Z2"])},
    )
    assert res.ok
    zids = _ids(session)
    res = session.create_piece("Other", [ids["A"], zids["Z"], zids["Z2"], zids["Z3"]])
    assert res.ok, res.message
    other = next(p for p in session.pattern.pieces if p.name == "Other")

    res = session.merge_piece(rect.id, other.id, ids["A"], ids["B"], name="Bad")
    assert res.ok is False


# --- delete / edit ---------------------------------------------------------
def test_delete_piece_leaves_construction_geometry_untouched(rectangle):
    session, _ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    before_points = dict(session.evaluated.points)

    res = session.delete_piece(original.id)
    assert res.ok, res.message
    assert session.pattern.pieces == []
    assert session.evaluated.points == before_points  # construction untouched


def test_delete_piece_reports_missing_id(rectangle):
    session, _ids = rectangle
    res = session.delete_piece(999999)
    assert res.ok is False


def test_edit_piece_changes_name_and_seam_allowance(rectangle):
    session, _ids = rectangle
    original = next(p for p in session.pattern.pieces if p.name == "Rect")
    res = session.edit_piece(original.id, name="Renamed", seam_allowance=False, width="2")
    assert res.ok, res.message
    piece = next(p for p in session.pattern.pieces if p.id == original.id)
    assert piece.name == "Renamed"
    assert piece.seam_allowance is False
    assert piece.width == "2"


# --- geometry primitives ----------------------------------------------------
def test_point_in_polygon_is_boundary_inclusive():
    from seamly_engine import geometry as geo

    square = [
        geo.Point(0, 0),
        geo.Point(10, 0),
        geo.Point(10, 10),
        geo.Point(0, 10),
        geo.Point(0, 0),
    ]
    assert geo.point_in_polygon(geo.Point(5, 5), square) is True
    assert geo.point_in_polygon(geo.Point(15, 5), square) is False
    assert geo.point_in_polygon(geo.Point(10, 5), square) is True  # on an edge
    assert geo.point_in_polygon(geo.Point(0, 0), square) is True  # on a vertex


def test_point_on_segment_excludes_endpoints():
    from seamly_engine import geometry as geo

    a, b = geo.Point(0, 0), geo.Point(10, 0)
    assert geo.point_on_segment(geo.Point(5, 0), a, b) is True
    assert geo.point_on_segment(geo.Point(0, 0), a, b) is False  # endpoint
    assert geo.point_on_segment(geo.Point(5, 1), a, b) is False  # off the line
    assert geo.point_on_segment(geo.Point(15, 0), a, b) is False  # beyond b
