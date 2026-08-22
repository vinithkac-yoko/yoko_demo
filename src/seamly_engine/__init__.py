"""Headless Seamly2D-compatible sewing-pattern engine.

A Seamly2D pattern is not a drawing — it is a program that draws itself. Nothing
in a ``.sm2d`` file stores "a line from (3.2, 8.1) to (9.7, 8.1)"; it stores
"point ``A13`` is ``14*#CM`` along the normal to ``A2``→``A12``". Coordinates are
outputs, recomputed whenever a measurement, a variable, or an upstream point
changes. This package implements that model: read the file, resolve the
formulas, evaluate the construction DAG to real geometry, mutate it safely, and
write it back.

Reading and evaluating::

    import seamly_engine as se

    pattern      = se.load_pattern("block.sm2d")
    measurements = se.load_measurements("client.vst")
    evaluated    = se.evaluate_pattern(pattern, measurements)

    state = se.export_state(pattern, evaluated, measurements)
    state["coverage"]        # {'total': 425, 'resolved': 425, 'fraction': 1.0}

Editing, via a session that re-evaluates and rolls back on breakage::

    from seamly_engine import PatternSession
    from seamly_engine.authoring import new_pattern

    session = PatternSession(new_pattern("Skirt block"))
    session.add_object("point", "endLine",
                       {"name": "B", "basePoint": "1", "angle": "270", "length": "60"})

Module map:

    model         Pattern / DraftBlock / PatternObject / Piece / Evaluated
    parser        .sm2d XML -> object model (lossless; keeps every raw attribute)
    writer        object model -> .sm2d (byte-identical round-trip)
    measurements  .vst / .smms multi-size tables
    formula       qmuparser-compatible expression engine
    geometry      points, arcs, cubic Beziers, Bezier paths, elliptical arcs
    evaluator     construction-DAG evaluation; formulas read live geometry
    operations    PatternSession — add / edit / delete, with rollback
    state         the semantically-tagged state representation
    pieces        piece grouping, outlines, piece construction
    render        SVG output (construction dimmed, final bold)
    authoring     blank-canvas pattern creation

See ``DESIGN.md`` for the file model, the formula language, and the pieces of
Seamly2D's behaviour this reproduces.
"""

from __future__ import annotations

from .evaluator import evaluate_pattern
from .measurements import MeasurementTable, parse_measurements
from .model import Evaluated, Pattern
from .operations import OpResult, PatternSession
from .parser import parse_pattern

__version__ = "0.1.0"

__all__ = [
    "Evaluated",
    "MeasurementTable",
    "OpResult",
    "Pattern",
    "PatternSession",
    "evaluate_pattern",
    "export_state",
    "load_measurements",
    "load_pattern",
    "parse_measurements",
    "parse_pattern",
]


def load_pattern(path: str) -> Pattern:
    """Parse a ``.sm2d`` / ``.val`` pattern file."""
    return parse_pattern(path)


def load_measurements(path: str) -> MeasurementTable:
    """Parse a ``.vst`` / ``.smms`` multi-size measurement table."""
    return parse_measurements(path)


def export_state(pattern, evaluated, measurements=None):
    """The semantically-tagged state representation (see :mod:`seamly_engine.state`)."""
    from .state import export_state as _export  # local import: state imports pieces

    return _export(pattern, evaluated, measurements)
