"""Headless Seamly2D-compatible pattern engine.

Public surface:

    from seamly_engine import load_pattern, load_measurements, evaluate_pattern
    from seamly_engine import export_state

See ``DESIGN.md`` for the architecture and the VLA state representation.
"""

from __future__ import annotations

from .evaluator import evaluate_pattern
from .measurements import MeasurementTable, parse_measurements
from .model import Evaluated, Pattern
from .parser import parse_pattern

__all__ = [
    "Pattern",
    "Evaluated",
    "MeasurementTable",
    "parse_pattern",
    "parse_measurements",
    "evaluate_pattern",
    "load_pattern",
    "load_measurements",
    "export_state",
]


def load_pattern(path: str) -> Pattern:
    return parse_pattern(path)


def load_measurements(path: str) -> MeasurementTable:
    return parse_measurements(path)


def export_state(pattern, evaluated, measurements=None):  # lazy import to avoid cycle
    from .state import export_state as _export

    return _export(pattern, evaluated, measurements)
