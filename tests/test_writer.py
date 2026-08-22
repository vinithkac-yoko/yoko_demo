"""Writer + blank-canvas authoring tests.

The writer is what makes saving, versioning, and export possible, so the key
guarantee is a **lossless round-trip**: parse -> write -> parse must reproduce the
same objects and identical geometry.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import seamly_engine as se
from seamly_engine.authoring import new_pattern
from seamly_engine.operations import PatternSession
from seamly_engine.parser import parse_pattern
from seamly_engine.writer import pattern_to_xml

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def meas():
    return se.load_measurements(str(FIX / "aldrich_measurements.vst"))


def test_roundtrip_preserves_geometry(meas):
    pat = se.load_pattern(str(FIX / "aldrich_basic.sm2d"))
    ev1 = se.evaluate_pattern(pat, meas)

    pat2 = parse_pattern(pattern_to_xml(pat), is_text=True)
    ev2 = se.evaluate_pattern(pat2, meas)

    assert len(pat2.all_objects()) == len(pat.all_objects())
    assert len(pat2.pieces) == len(pat.pieces)
    assert ev2.unresolved == ev1.unresolved == {}
    assert set(ev2.points) == set(ev1.points)
    for oid, p in ev1.points.items():
        q = ev2.points[oid]
        assert math.isclose(p.x, q.x, abs_tol=1e-9) and math.isclose(p.y, q.y, abs_tol=1e-9)


def test_roundtrip_is_deterministic():
    pat = se.load_pattern(str(FIX / "aldrich_basic.sm2d"))
    assert pattern_to_xml(pat) == pattern_to_xml(parse_pattern(pattern_to_xml(pat), is_text=True))


def test_blank_pattern_has_origin(meas):
    pat = new_pattern("Fresh")
    sess = PatternSession(pat, meas)
    assert len(pat.all_objects()) == 1
    assert sess.evaluated.points[1].as_tuple() == (0.0, 0.0)
    assert sess.evaluated.unresolved == {}


def test_build_from_blank_and_roundtrip(meas):
    """Draft from nothing, then confirm the result saves and reloads."""
    sess = PatternSession(new_pattern("Fresh", measurements_file="m.vst"), meas)
    r1 = sess.add_object(
        "point", "endLine", {"basePoint": "1", "angle": "270", "length": "10", "name": "B"}
    )
    r2 = sess.add_object(
        "point",
        "endLine",
        {"basePoint": "1", "angle": "0", "length": "(waist_circ/4)+2*#CM", "name": "C"},
    )
    assert r1.ok and r2.ok
    assert math.isclose(sess.evaluated.points[2].y, 10.0, abs_tol=1e-6)  # 10cm down
    assert math.isclose(sess.evaluated.points[3].x, 17.0, abs_tol=1e-6)  # 60/4 + 2

    pat2 = parse_pattern(pattern_to_xml(sess.pattern), is_text=True)
    ev2 = se.evaluate_pattern(pat2, meas)
    assert len(pat2.all_objects()) == 3 and ev2.unresolved == {}


# --- piece creation ---------------------------------------------------------
def test_create_piece_from_blank_canvas(meas):
    """Draft a block from nothing, turn it into a cut piece, and reload it."""
    from seamly_engine.pieces import piece_outline_points

    sess = PatternSession(new_pattern("Skirt block"), meas)
    for attrs in (
        {"basePoint": "1", "angle": "0", "length": "17", "name": "B"},
        {"basePoint": "2", "angle": "270", "length": "60", "name": "C"},
        {"basePoint": "1", "angle": "270", "length": "60", "name": "D"},
    ):
        assert sess.add_object("point", "endLine", attrs).ok

    res = sess.create_piece(
        "Skirt Front",
        [1, 2, 3, 4],
        grainline_anchor=1,
        internal_paths=[{"name": "Dart 1", "node_ids": [1, 3]}],
    )
    assert res.ok, res.message
    piece = sess.pattern.pieces[0]
    assert len(piece.nodes) == 4 and piece.internal_path_ids and piece.grainline_anchor

    outline = piece_outline_points(sess.pattern, sess.evaluated, piece)
    assert len(outline) == 5  # 4 corners + closing point
    assert math.isclose(outline[1].x, 17.0, abs_tol=1e-6)
    assert math.isclose(outline[2].y, 60.0, abs_tol=1e-6)

    # survives save + reload
    pat2 = parse_pattern(pattern_to_xml(sess.pattern), is_text=True)
    ev2 = se.evaluate_pattern(pat2, meas)
    assert len(pat2.pieces) == 1 and pat2.pieces[0].name == "Skirt Front"
    assert len(pat2.pieces[0].nodes) == 4 and ev2.unresolved == {}
    reloaded = piece_outline_points(pat2, ev2, pat2.pieces[0])
    assert [(round(p.x, 4), round(p.y, 4)) for p in reloaded] == [
        (round(p.x, 4), round(p.y, 4)) for p in outline
    ]


def test_create_piece_rejects_too_few_nodes(meas):
    sess = PatternSession(new_pattern("X"), meas)
    assert sess.create_piece("bad", [1]).ok is False


def test_adding_piece_preserves_imported_pieces(meas):
    """A new piece must not disturb the pieces that came from the file."""
    pat = se.load_pattern(str(FIX / "aldrich_basic.sm2d"))
    sess = PatternSession(pat, meas)
    ids = {o.raw.get("name"): o.id for o in pat.all_objects() if o.raw.get("name")}
    before = [p.name for p in pat.pieces]

    assert sess.create_piece("New Panel", [ids["A1"], ids["A9"], ids["A10"], ids["A8"]]).ok

    pat2 = parse_pattern(pattern_to_xml(sess.pattern), is_text=True)
    assert [p.name for p in pat2.pieces] == [*before, "New Panel"]
    assert se.evaluate_pattern(pat2, meas).unresolved == {}
