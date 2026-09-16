"""Tests for the mutation API — :class:`seamly_engine.operations.PatternSession`.

Editing a pattern is where a parametric CAD model is easiest to corrupt: one
bad formula and objects downstream stop resolving, one careless delete and the
construction chain has a hole in it. These tests pin the two guarantees callers
build on — **every mutation re-evaluates and rolls back on breakage**, and
**delete is block-and-report, never cascade** — as well as the ordinary case of
actually building geometry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import seamly_engine as se
from seamly_engine.authoring import new_pattern
from seamly_engine.operations import PatternSession

FIX = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def blank():
    """A blank canvas: one draft block and an origin point ``A``.

    Built without the default increments — ``#CM`` is defined as
    ``height/#BaseHeight``, so it needs a measurement table these tests have no
    reason to carry.
    """
    return PatternSession(new_pattern("test", with_defaults=False))


@pytest.fixture
def aldrich():
    return PatternSession(
        se.load_pattern(str(FIX / "aldrich_basic.sm2d")),
        se.load_measurements(str(FIX / "aldrich_measurements.vst")),
    )


def _ids(session: PatternSession) -> dict[str, int]:
    return {o.raw.get("name"): o.id for o in session.pattern.all_objects() if o.raw.get("name")}


def _point(session, tool_type, **attrs):
    res = session.add_object("point", tool_type, attrs)
    assert res.ok, res.message
    return res


# --- building geometry --------------------------------------------------------
def test_points_lines_and_curves_build_a_block(blank):
    origin = _ids(blank)["A"]
    _point(blank, "endLine", name="B", basePoint=str(origin), angle="270", length="60")
    _point(blank, "endLine", name="C", basePoint=str(origin), angle="0", length="17")
    ids = _ids(blank)

    assert blank.add_object(
        "line", "", {"firstPoint": str(origin), "secondPoint": str(ids["B"])}
    ).ok
    assert blank.add_object(
        "arc",
        "simple",
        {"name": "hem", "center": str(origin), "radius": "10", "angle1": "0", "angle2": "90"},
    ).ok

    pts = blank.evaluated.points
    # Screen coordinates are y-down while angles are counter-clockwise, so 270
    # goes *down* the page and 0 goes right.
    assert pts[ids["B"]].y == pytest.approx(60.0)
    assert pts[ids["C"]].x == pytest.approx(17.0)
    assert blank.evaluated.unresolved == {}


def test_a_variable_drives_the_geometry_that_uses_it(blank):
    origin = _ids(blank)["A"]
    assert blank.set_variable("#Hem", "20").ok
    _point(blank, "endLine", name="H", basePoint=str(origin), angle="0", length="#Hem")

    hid = _ids(blank)["H"]
    assert blank.evaluated.points[hid].x == pytest.approx(20.0)

    assert blank.set_variable("#Hem", "25").ok
    assert blank.evaluated.points[hid].x == pytest.approx(25.0)


def test_operations_copy_rather_than_move(blank):
    origin = _ids(blank)["A"]
    _point(blank, "endLine", name="P", basePoint=str(origin), angle="0", length="10")
    pid = _ids(blank)["P"]

    res = blank.add_object(
        "operation",
        "flippingByAxis",
        {"center": str(origin), "axisType": "vertical", "suffix": "_m"},
        children=[{"src": str(pid)}],
    )
    assert res.ok, res.message
    assert blank.evaluated.points[pid].x == pytest.approx(10.0)  # source untouched
    mirrored = [
        i
        for i in blank.added_ids
        if i in blank.evaluated.points and blank.evaluated.points[i].x == pytest.approx(-10.0)
    ]
    assert mirrored


def test_a_dart_declares_two_output_points_inline(aldrich):
    """``trueDarts`` is the one tool that produces two points from one element:
    the leg ids are declared as ``point1``/``point2`` attributes and never
    appear as elements of their own, so an id allocator has to know about them.
    """
    ids = _ids(aldrich)
    res = aldrich.add_object(
        "point",
        "trueDarts",
        {
            "baseLineP1": str(ids["A1"]),
            "baseLineP2": str(ids["A9"]),
            "dartP1": str(ids["A11"]),
            "dartP2": str(ids["A13"]),
            "dartP3": str(ids["A12"]),
            "name1": "X1",
            "name2": "X2",
        },
    )
    assert res.ok, res.message
    assert len(aldrich.added_ids) == 3  # the tool plus its two legs

    dart = next(
        o
        for o in aldrich.pattern.all_objects()
        if o.tool_type == "trueDarts" and o.id in aldrich.added_ids
    )
    legs = [int(dart.raw["point1"]), int(dart.raw["point2"])]
    assert all(i in aldrich.evaluated.points for i in legs)
    assert [dart.raw["name1"], dart.raw["name2"]] == ["X1", "X2"]


def test_a_piece_can_be_cut_from_the_geometry(blank):
    origin = _ids(blank)["A"]
    _point(blank, "endLine", name="B", basePoint=str(origin), angle="270", length="60")
    _point(blank, "endLine", name="C", basePoint=str(origin), angle="0", length="17")
    ids = _ids(blank)
    _point(blank, "intersectXY", name="D", firstPoint=str(ids["C"]), secondPoint=str(ids["B"]))
    ids = _ids(blank)

    res = blank.create_piece(
        "Panel", [origin, ids["C"], ids["D"], ids["B"]], grainline_anchor=origin
    )
    assert res.ok, res.message
    assert [p.name for p in blank.pattern.pieces] == ["Panel"]


# --- the guarantees callers rely on -------------------------------------------
def test_an_edit_that_breaks_the_pattern_is_rolled_back(aldrich):
    ids = _ids(aldrich)
    before = dict(aldrich.evaluated.points)

    res = aldrich.edit_object(ids["A1"], {"length": "nonsense_var"})
    assert res.ok is False
    assert aldrich.evaluated.points == before


def test_an_object_that_cannot_be_computed_is_not_added(blank):
    before = len(blank.pattern.all_objects())
    res = blank.add_object(
        "point", "endLine", {"name": "Z", "basePoint": "999", "angle": "0", "length": "5"}
    )
    assert res.ok is False
    assert len(blank.pattern.all_objects()) == before


def test_delete_blocks_and_reports_instead_of_cascading(aldrich):
    ids = _ids(aldrich)
    res = aldrich.delete_object(ids["A1"])

    assert res.ok is False
    assert res.blocked_by, "the dependent chain must come back with the refusal"
    assert ids["A1"] in {o.id for o in aldrich.pattern.all_objects()}


def test_a_leaf_object_deletes_cleanly(blank):
    origin = _ids(blank)["A"]
    _point(blank, "endLine", name="Z", basePoint=str(origin), angle="0", length="5")
    zid = _ids(blank)["Z"]

    assert blank.delete_object(zid).ok
    assert "Z" not in _ids(blank)


def test_a_variable_that_breaks_dependents_is_rolled_back(aldrich):
    before = dict(aldrich.evaluated.points)
    res = aldrich.set_variable("#CM", "no_such_measurement")
    assert res.ok is False
    assert aldrich.evaluated.points == before


def test_dependents_are_reported_transitively(aldrich):
    ids = _ids(aldrich)
    direct = aldrich.dependents_of(ids["A1"])
    transitive = aldrich.transitive_dependents(ids["A1"])
    assert set(direct) <= set(transitive)
    assert len(transitive) > len(direct)


def test_edits_reach_a_real_production_pattern(aldrich):
    """The mutation API has to work on the file the engine was modelled from,
    not only on geometry built inside a test."""
    ids = _ids(aldrich)
    res = aldrich.edit_object(ids["A9"], {"length": "waist_circ/4+5*#CM"})
    assert res.ok, res.message
    assert aldrich.evaluated.unresolved == {}
