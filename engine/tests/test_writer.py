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
    r1 = sess.add_object("point", "endLine",
                         {"basePoint": "1", "angle": "270", "length": "10", "name": "B"})
    r2 = sess.add_object("point", "endLine",
                         {"basePoint": "1", "angle": "0",
                          "length": "(waist_circ/4)+2*#CM", "name": "C"})
    assert r1.ok and r2.ok
    assert math.isclose(sess.evaluated.points[2].y, 10.0, abs_tol=1e-6)   # 10cm down
    assert math.isclose(sess.evaluated.points[3].x, 17.0, abs_tol=1e-6)   # 60/4 + 2

    pat2 = parse_pattern(pattern_to_xml(sess.pattern), is_text=True)
    ev2 = se.evaluate_pattern(pat2, meas)
    assert len(pat2.all_objects()) == 3 and ev2.unresolved == {}
