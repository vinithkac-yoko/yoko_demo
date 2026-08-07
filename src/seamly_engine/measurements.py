"""Measurement table (``.vst`` / ``.vit`` / ``.smms``) parsing and grading.

A multi-size table stores each measurement as ``base + f(size, height)``:

    <m name="bust_circ" base="76" size_increase="4" height_increase="0"/>

so a measurement's value at a chosen (size, height) is::

    base
    + size_increase   * (size   - base_size)   / size_step
    + height_increase * (height - base_height) / height_step

At the table's base size/height the increments vanish and the value is ``base``
— which is exactly the "base pattern" the engine evaluates by default.

The single-measurement (``.vit``) variant stores a literal ``value`` (which may
itself be a formula) instead of graded increments; both are supported.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class Measurement:
    name: str
    base: float
    size_increase: float = 0.0
    height_increase: float = 0.0
    value_formula: str | None = None  # for single-size .vit tables

    def value(
        self,
        size: float,
        height: float,
        base_size: float,
        base_height: float,
        size_step: float,
        height_step: float,
    ) -> float:
        if self.value_formula is not None:
            # Resolved later by the evaluator (may reference other measurements).
            raise NotImplementedError("formula-valued measurement resolved in evaluator")
        v = self.base
        if size_step:
            v += self.size_increase * (size - base_size) / size_step
        if height_step:
            v += self.height_increase * (height - base_height) / height_step
        return v


@dataclass
class MeasurementTable:
    unit: str = "cm"
    base_size: float = 0.0
    base_height: float = 0.0
    # Seamly multisize default steps (cm). Not stored per-file in older .vst, so
    # defaulted here and overridable; only matters away from the base size.
    size_step: float = 2.0
    height_step: float = 6.0
    measurements: dict[str, Measurement] = field(default_factory=dict)

    def resolve(self, size: float | None = None, height: float | None = None) -> dict[str, float]:
        """Return {name: value} at the given size/height (defaults to base)."""
        s = self.base_size if size is None else size
        h = self.base_height if height is None else height
        out: dict[str, float] = {}
        for name, m in self.measurements.items():
            if m.value_formula is not None:
                continue  # handled by evaluator's formula pass
            out[name] = m.value(
                s, h, self.base_size, self.base_height, self.size_step, self.height_step
            )
        return out


def parse_measurements(path_or_text: str, *, is_text: bool = False) -> MeasurementTable:
    root = ET.fromstring(path_or_text) if is_text else ET.parse(path_or_text).getroot()

    table = MeasurementTable()
    unit = root.findtext("unit")
    if unit:
        table.unit = unit.strip()

    size_el = root.find("size")
    if size_el is not None and size_el.get("base"):
        table.base_size = float(size_el.get("base"))
    height_el = root.find("height")
    if height_el is not None and height_el.get("base"):
        table.base_height = float(height_el.get("base"))

    body = root.find("body-measurements")
    if body is not None:
        for m in body.findall("m"):
            name = m.get("name")
            if not name:
                continue
            if m.get("value") is not None:  # single-size .vit form
                table.measurements[name] = Measurement(
                    name=name, base=0.0, value_formula=m.get("value")
                )
            else:
                table.measurements[name] = Measurement(
                    name=name,
                    base=float(m.get("base", "0") or 0),
                    size_increase=float(m.get("size_increase", "0") or 0),
                    height_increase=float(m.get("height_increase", "0") or 0),
                )
    return table
