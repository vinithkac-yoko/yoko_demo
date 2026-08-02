"""Tests for the action space — the environment's contract with any policy.

Every tool is exercised through :func:`~seamly_engine.actions.dispatch_tool`,
the same entry point a model, a replayed document, or a human driver uses. The
two properties that matter most are the ones a policy depends on to recover:
dispatch never raises, and a mutation that would break the pattern is rolled
back rather than half-applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import seamly_engine as se  # noqa: E402
from seamly_engine.actions import POINT_TYPES, TOOLS, dispatch_tool  # noqa: E402
from seamly_engine.authoring import new_pattern  # noqa: E402
from seamly_engine.operations import PatternSession  # noqa: E402
from seamly_engine.pieces import pieces_for_key  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


@pytest.fixture
def blank():
    """A blank canvas: one draft block and an origin point `A`.

    Built without the default increments — `#CM` is defined as
    `height/#BaseHeight`, so it needs a measurement table this session has no
    reason to carry."""
    return PatternSession(new_pattern("test", with_defaults=False))


@pytest.fixture
def aldrich():
    return PatternSession(se.load_pattern(str(FIX / "aldrich_basic.sm2d")),
                          se.load_measurements(str(FIX / "aldrich_measurements.vst")))


def _ids(session):
    return {o.raw.get("name"): o.id for o in session.pattern.all_objects()
            if o.raw.get("name")}


def _add(session, tool, **args):
    res = dispatch_tool(session, tool, args)
    assert res.ok, res.message
    return res


# --- the schema itself --------------------------------------------------------
def test_every_tool_has_a_dispatchable_name():
    """A tool a model can see but not call is worse than no tool at all."""
    for tool in TOOLS:
        session = PatternSession(new_pattern("t", with_defaults=False))
        res = dispatch_tool(session, tool["name"], {})
        assert "unknown tool" not in res.message


def test_schemas_are_well_formed():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["description"] and schema["type"] == "object"
        for required in schema.get("required", []):
            assert required in schema["properties"], f"{tool['name']}: {required}"


def test_all_of_seamlys_point_tools_are_offered():
    assert len(POINT_TYPES) == 21
    enum = next(t for t in TOOLS if t["name"] == "add_point")
    assert enum["input_schema"]["properties"]["tool_type"]["enum"] == POINT_TYPES


# --- dispatch never raises ----------------------------------------------------
def test_unknown_tool_is_reported_not_raised(blank):
    res = dispatch_tool(blank, "make_it_nice", {})
    assert res.ok is False and "unknown tool" in res.message


def test_missing_argument_comes_back_correctable(blank):
    res = dispatch_tool(blank, "add_point", {"attrs": {}})
    assert res.ok is False and "missing required argument" in res.message


def test_unknown_curve_kind_is_reported(blank):
    res = dispatch_tool(blank, "add_curve", {"kind": "banana"})
    assert res.ok is False and "unknown kind" in res.message


def test_wrong_types_do_not_escape_as_exceptions(blank):
    for name, args in [
        ("edit_object", {"object_id": "not-a-number", "attrs": {}}),
        ("create_piece", {"name": "p", "node_ids": ["x", "y", "z"]}),
        ("add_operation", {"kind": "rotation", "source_ids": []}),
        ("add_point", {"tool_type": "endLine", "attrs": {"basePoint": "999"}}),
    ]:
        res = dispatch_tool(blank, name, args)
        assert res.ok is False and res.message


# --- the tools actually build geometry ----------------------------------------
def test_points_lines_and_curves_build_a_block(blank):
    origin = _ids(blank)["A"]
    _add(blank, "add_point", tool_type="endLine",
         attrs={"name": "B", "basePoint": str(origin), "angle": "270", "length": "60"})
    _add(blank, "add_point", tool_type="endLine",
         attrs={"name": "C", "basePoint": str(origin), "angle": "0", "length": "17"})
    ids = _ids(blank)
    _add(blank, "add_line", firstPoint=str(origin), secondPoint=str(ids["B"]))
    _add(blank, "add_curve", kind="arc",
         attrs={"name": "hem", "center": str(origin), "radius": "10",
                "angle1": "0", "angle2": "90"})

    pts = blank.evaluated.points
    assert pts[ids["B"]].y == pytest.approx(60.0)
    assert pts[ids["C"]].x == pytest.approx(17.0)
    assert blank.evaluated.unresolved == {}


def test_a_variable_drives_the_geometry_that_uses_it(blank):
    origin = _ids(blank)["A"]
    _add(blank, "add_variable", name="#Hem", formula="20")
    _add(blank, "add_point", tool_type="endLine",
         attrs={"name": "H", "basePoint": str(origin), "angle": "0", "length": "#Hem"})
    hid = _ids(blank)["H"]
    assert blank.evaluated.points[hid].x == pytest.approx(20.0)

    _add(blank, "add_variable", name="#Hem", formula="25")
    assert blank.evaluated.points[hid].x == pytest.approx(25.0)


def test_operations_copy_points(blank):
    origin = _ids(blank)["A"]
    _add(blank, "add_point", tool_type="endLine",
         attrs={"name": "P", "basePoint": str(origin), "angle": "0", "length": "10"})
    ids = _ids(blank)
    before = len(blank.pattern.all_objects())
    _add(blank, "add_operation", kind="flippingByAxis", source_ids=[ids["P"]],
         center=str(origin), axisType="vertical", suffix="_m")
    assert len(blank.pattern.all_objects()) > before
    mirrored = [i for i in blank.added_ids if i in blank.evaluated.points
                and blank.evaluated.points[i].x == pytest.approx(-10.0)]
    assert mirrored


def test_a_piece_can_be_cut_from_the_geometry(blank):
    origin = _ids(blank)["A"]
    for name, angle, length in (("B", "270", "60"), ("C", "0", "17")):
        _add(blank, "add_point", tool_type="endLine",
             attrs={"name": name, "basePoint": str(origin), "angle": angle,
                    "length": length})
    ids = _ids(blank)
    _add(blank, "add_point", tool_type="intersectXY",
         attrs={"name": "D", "firstPoint": str(ids["C"]), "secondPoint": str(ids["B"])})
    ids = _ids(blank)
    res = _add(blank, "create_piece", name="Panel",
               node_ids=[origin, ids["C"], ids["D"], ids["B"]],
               grainline_anchor=origin)
    assert "Panel" in res.message
    assert [p.name for p in blank.pattern.pieces] == ["Panel"]


def test_a_dart_produces_its_two_leg_points(aldrich):
    ids = _ids(aldrich)
    _add(aldrich, "add_dart", baseLineP1=str(ids["A1"]), baseLineP2=str(ids["A9"]),
         dartP1=str(ids["A11"]), dartP2=str(ids["A13"]), dartP3=str(ids["A12"]),
         name1="X1", name2="X2")
    assert len(aldrich.added_ids) == 3   # the tool plus its two legs
    # A dart declares its outputs inline (point1/name1, point2/name2) rather
    # than as separate objects, so look for the reserved ids' geometry.
    dart = next(o for o in aldrich.pattern.all_objects()
                if o.tool_type == "trueDarts" and o.id in aldrich.added_ids)
    legs = [int(dart.raw["point1"]), int(dart.raw["point2"])]
    assert all(i in aldrich.evaluated.points for i in legs)
    assert [dart.raw["name1"], dart.raw["name2"]] == ["X1", "X2"]


# --- the engine, not the tool, decides what is legal --------------------------
def test_an_edit_that_breaks_the_pattern_is_rolled_back(aldrich):
    ids = _ids(aldrich)
    before = dict(aldrich.evaluated.points)
    res = dispatch_tool(aldrich, "edit_object",
                        {"object_id": ids["A1"], "attrs": {"length": "nonsense_var"}})
    assert res.ok is False
    assert aldrich.evaluated.points == before


def test_delete_blocks_and_reports_instead_of_cascading(aldrich):
    ids = _ids(aldrich)
    res = dispatch_tool(aldrich, "delete_object", {"object_id": ids["A1"]})
    assert res.ok is False
    assert res.blocked_by, "the dependent chain must come back with the refusal"
    assert ids["A1"] in {o.id for o in aldrich.pattern.all_objects()}


def test_a_leaf_object_deletes_cleanly(blank):
    origin = _ids(blank)["A"]
    _add(blank, "add_point", tool_type="endLine",
         attrs={"name": "Z", "basePoint": str(origin), "angle": "0", "length": "5"})
    zid = _ids(blank)["Z"]
    assert dispatch_tool(blank, "delete_object", {"object_id": zid}).ok
    assert "Z" not in _ids(blank)


def test_edits_reach_a_real_production_pattern(aldrich):
    """The action space has to work on the file it was modelled from, not just
    on geometry we built ourselves."""
    pieces = pieces_for_key(aldrich.pattern, "A")
    assert pieces
    ids = _ids(aldrich)
    res = dispatch_tool(aldrich, "edit_object",
                        {"object_id": ids["A9"], "attrs": {"length": "waist_circ/4+5*#CM"}})
    assert res.ok, res.message
    assert aldrich.evaluated.unresolved == {}
