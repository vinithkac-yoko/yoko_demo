"""End-to-end tests against the real Aldrich basic pattern set.

These assert the engine parses, evaluates, and semantically tags a full
production Seamly2D file — not a toy — so regressions in the formula engine,
geometry kernel, DAG evaluator, or state exporter surface immediately.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import seamly_engine as se
from seamly_engine import formula
from seamly_engine.formula import Scope
from seamly_engine import geometry as geo

FIX = Path(__file__).parent / "fixtures"
PATTERN = FIX / "aldrich_basic.sm2d"
MEAS = FIX / "aldrich_measurements.vst"


@pytest.fixture(scope="module")
def evaluated():
    pat = se.load_pattern(str(PATTERN))
    meas = se.load_measurements(str(MEAS))
    ev = se.evaluate_pattern(pat, meas)
    return pat, meas, ev


# --- formula engine ----------------------------------------------------------
def _scope(mapping):
    return Scope(resolve=lambda n: mapping[n])


def test_formula_arithmetic_and_precedence():
    assert formula.evaluate("2+3*4", _scope({})) == 14
    assert formula.evaluate("(2+3)*4", _scope({})) == 20
    assert formula.evaluate("2^3^2", _scope({})) == 512  # right assoc


def test_formula_ternary_and_comparison():
    s = _scope({"size": 34})
    assert formula.evaluate("size>22?4.75:size>16?4.5:4", s) == 4.75
    assert formula.evaluate("size>40?1:0", s) == 0


def test_formula_degree_trig():
    assert math.isclose(formula.evaluate("cosD(0)", _scope({})), 1.0)
    assert math.isclose(formula.evaluate("sinD(90)", _scope({})), 1.0, abs_tol=1e-9)


def test_formula_increment_and_measurement_names():
    s = _scope({"#CM": 1.0, "hip_circ": 84})
    assert formula.evaluate("(hip_circ/2)+1.5*#CM", s) == 43.5


def test_formula_identifiers_extraction():
    ids = formula.identifiers("(hip_circ/2)+1.5*#CM")
    assert ids == {"hip_circ", "#CM"}


# --- geometry kernel ---------------------------------------------------------
def test_from_polar_screen_convention():
    base = geo.Point(0, 0)
    down = geo.from_polar(base, 270, 5)  # 270 == down in screen coords
    assert math.isclose(down.x, 0, abs_tol=1e-9)
    assert math.isclose(down.y, 5, abs_tol=1e-9)


def test_reflect_point_across_vertical_line():
    p = geo.Point(2, 0)
    r = geo.reflect_point(p, geo.Point(0, -1), geo.Point(0, 1))
    assert math.isclose(r.x, -2) and math.isclose(r.y, 0)


# --- parsing -----------------------------------------------------------------
def test_parse_structure(evaluated):
    pat, _, _ = evaluated
    assert pat.pattern_name == "Adrich 6th Ed Basic Pattern"
    assert len(pat.increments) == 17
    assert len(pat.all_objects()) == 425
    assert [p.name for p in pat.pieces] == [
        "A - Skirt Back", "A - Skirt Front", "B - Trousers Front",
        "B - Trousers Back", "C - Bodice Back", "C - Bodice Front",
        "D - 1 Piece Sleeve",
    ]


def test_increments_resolved_correctly(evaluated):
    _, _, ev = evaluated
    # #CM = height / #BaseHeight = 166 / 166 = 1.0 at base
    assert math.isclose(ev.increment_values["#CM"], 1.0)
    assert math.isclose(ev.increment_values["#BaseHeight"], 166.0)


# --- full DAG evaluation -----------------------------------------------------
def test_full_pattern_resolves(evaluated):
    pat, _, ev = evaluated
    # Every object in the real pattern must evaluate to finite geometry.
    assert ev.unresolved == {}
    for oid, p in ev.points.items():
        assert math.isfinite(p.x) and math.isfinite(p.y), oid


def test_single_point_matches_file(evaluated):
    pat, _, ev = evaluated
    ids = {o.raw.get("name"): o.id for o in pat.all_objects() if o.raw.get("name")}
    a = ev.points[ids["A"]]
    assert math.isclose(a.x, 0.79375, abs_tol=1e-4)
    assert math.isclose(a.y, 1.05833, abs_tol=1e-4)


def test_measurement_driven_distance(evaluated):
    pat, _, ev = evaluated
    ids = {o.raw.get("name"): o.id for o in pat.all_objects() if o.raw.get("name")}
    # A9 is waist_circ/4 + 4*#CM from A1 == 60/4 + 4 == 19 cm
    d = ev.points[ids["A1"]].dist(ev.points[ids["A9"]])
    assert math.isclose(d, 19.0, abs_tol=1e-6)


def test_true_darts_outputs_registered(evaluated):
    pat, _, ev = evaluated
    # trueDarts element 190 emits points 191 (Bba) and 192 (Bca).
    assert 191 in ev.points and 192 in ev.points


def test_flipping_operation_creates_destination(evaluated):
    pat, _, ev = evaluated
    # operation 570 mirrors source 569 to destination 571.
    assert 571 in ev.points


# --- state export ------------------------------------------------------------
def test_state_export_tags_construction_vs_final(evaluated):
    pat, meas, ev = evaluated
    state = se.export_state(pat, ev, meas)
    assert state["coverage"]["fraction"] == 1.0
    # There must be both final-outline and construction objects, and darts.
    roles = [o["role"] for o in state["objects"]]
    assert roles.count("seamline") > 50
    assert "dart" in roles
    assert "grainline" in roles
    final = [o for o in state["objects"] if o["is_final_outline"]]
    construction = [o for o in state["objects"] if o["is_construction"]]
    assert len(final) > 100
    assert len(construction) > len(final)


def test_state_export_has_dependents_for_delete(evaluated):
    pat, meas, ev = evaluated
    state = se.export_state(pat, ev, meas)
    by_id = {o["id"]: o for o in state["objects"]}
    # Point A (id 1) is the origin; many things ultimately build from it, so it
    # must report dependents (delete would be blocked and report them).
    assert by_id[1]["dependents"], "origin point should have dependents"


def test_state_export_is_json_serializable(evaluated):
    import json
    pat, meas, ev = evaluated
    state = se.export_state(pat, ev, meas)
    assert json.loads(json.dumps(state))["pattern"]["unit"] == "cm"


# --- mutation layer ----------------------------------------------------------
def _session():
    from seamly_engine.operations import PatternSession
    return PatternSession(se.load_pattern(str(PATTERN)), se.load_measurements(str(MEAS)))


def test_delete_blocks_and_reports_dependents():
    sess = _session()
    res = sess.delete_object(1)  # origin point A
    assert res.ok is False
    assert res.blocked_by and len(res.blocked_by) == 4  # A1, B, C, D build from A


def test_delete_leaf_succeeds():
    sess = _session()
    leaf = next(o.id for o in reversed(sess.pattern.all_objects())
                if o.tag == "line" and not sess.dependents_of(o.id))
    res = sess.delete_object(leaf)
    assert res.ok is True
    assert leaf not in sess.pattern.object_by_id()


def test_edit_formula_reevaluates():
    sess = _session()
    ids = {o.raw.get("name"): o.id for o in sess.pattern.all_objects() if o.raw.get("name")}
    res = sess.edit_formula(ids["A1"], "length", "6*#CM")
    assert res.ok is True
    assert sess.evaluated.unresolved == {}


def test_edit_formula_rolls_back_on_break():
    sess = _session()
    ids = {o.raw.get("name"): o.id for o in sess.pattern.all_objects() if o.raw.get("name")}
    # Reference an undefined variable -> should break and roll back.
    res = sess.edit_formula(ids["A1"], "length", "does_not_exist")
    assert res.ok is False
    # original formula restored
    a1 = sess.pattern.object_by_id()[ids["A1"]]
    assert a1.raw["length"] == "5*#CM"


# --- blocks / pieces ---------------------------------------------------------
def test_list_pieces(evaluated):
    from seamly_engine.state import list_pieces
    pat, _, _ = evaluated
    names = [p["name"] for p in list_pieces(pat)]
    assert names == [
        "A - Skirt Back", "A - Skirt Front", "B - Trousers Front",
        "B - Trousers Back", "C - Bodice Back", "C - Bodice Front",
        "D - 1 Piece Sleeve",
    ]
    assert all(p["object_count"] > 0 for p in list_pieces(pat))


def test_piece_state_is_scoped(evaluated):
    from seamly_engine.state import compact_state, piece_state
    pat, meas, ev = evaluated
    piece = next(p for p in pat.pieces if p.name == "C - Bodice Front")
    ps = piece_state(pat, ev, piece, meas)
    assert ps["block"] == "C - Bodice Front"
    # scoped to far fewer objects than the whole pattern
    assert 0 < ps["object_count"] < len(compact_state(pat, ev, meas)["objects"])
    # and it includes final-outline (seamline) objects of the block
    assert any(o["final_outline"] for o in ps["objects"])


def test_render_piece_svg_draws_outline(evaluated):
    from seamly_engine.render import render_piece_svg
    pat, _, ev = evaluated
    piece = next(p for p in pat.pieces if p.name == "C - Bodice Front")
    svg = render_piece_svg(pat, ev, piece, width=600)
    assert svg.startswith("<svg") and "<path" in svg  # a real outline path
    assert "Z" in svg  # closed outline
