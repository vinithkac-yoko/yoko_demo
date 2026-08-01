"""The drafting-action IR — the format between an instruction document and us.

Sewing-instruction documents (the PDFs a pattern-making school hands out) are
prose plus figures: *"A - C → Shoulder ÷ 2"*. An extraction pass turns each
sentence into one row of a JSONL file — a **drafting action**. This module
defines what a valid action looks like so the rest of the pipeline can be strict
about it.

Two things matter here:

* **The vocabulary is small and closed.** Ten verbs cover a whole garment
  document. Each verb declares which fields it needs, so a row that can't be
  compiled is caught at normalization time, with a precise reason.
* **Ambiguity is data, not failure.** An extracted row routinely under-determines
  the geometry — *"H - H₁ → 1.25\" seam allowance"* says how far but not which
  way, and only the figure knows. Those become :class:`Issue` records carrying a
  ``needs`` field naming exactly what a human (or a figure-reading pass) must
  supply. That list *is* the work queue for turning a document into a pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- vocabulary --------------------------------------------------------------
# Each verb maps to: the fields it must have, and a one-line meaning. "outputs"
# and "inputs" are always lists after normalization even when the source row
# used a bare string.
ACTIONS: dict[str, dict] = {
    "SET_ORIGIN": {
        "needs": (),
        "doc": "Establish the panel's origin point (top-left corner of the block).",
    },
    "DEFINE_POINT_DISTANCE": {
        "needs": ("inputs", "measure", "direction"),
        "doc": "New point at a measured distance from an existing point.",
    },
    "DEFINE_POINT_OFFSET": {
        "needs": ("inputs", "measure"),
        "doc": "New point offset from an existing one by a fixed amount "
               "(slope drops, cross marks, inward shifts).",
    },
    "DEFINE_POINT_MIDPOINT": {
        "needs": ("inputs2",),
        "doc": "New point halfway between two existing points.",
    },
    "DEFINE_POINT_RATIO_SPLIT": {
        "needs": ("inputs", "measure"),
        "doc": "Two points straddling a reference point, a measured span apart.",
    },
    "DRAW_CURVE": {
        "needs": ("inputs3",),
        "doc": "Fit a smooth curve through three or more existing points.",
    },
    "APPLY_SEAM_ALLOWANCE": {
        "needs": ("inputs", "measure"),
        "doc": "Offset point marking added seam allowance beyond a seam point.",
    },
    "TRACE_REFERENCE": {
        "needs": ("inputs",),
        "doc": "Start this panel from a copy of another panel's block.",
    },
    "TRANSFORM": {
        "needs": ("value_raw",),
        "doc": "Reorient a traced block (flip / rotate) before editing it.",
    },
    "ANNOTATE_REQUIREMENT": {
        "needs": (),
        "doc": "A required input measurement or a spec-driven value with no "
               "formula — becomes a pattern variable, not geometry.",
    },
}

#: ``cross`` is bearing-free: x comes from one point, y from another — how a
#: drafter finds where a horizontal guide meets a vertical one.
DIRECTIONS = {"vertical", "horizontal", "along_line", "cross", "offset",
              "inward", None}

#: Measurement names these documents use, with **placeholder** values in inches
#: for a mid-size Indian women's block. They exist so a compiled pattern
#: evaluates end-to-end; they are not a graded size table and must be replaced
#: with the wearer's real measurements before anything is cut.
MEASUREMENT_DEFAULTS: dict[str, float] = {
    "yoke_length": 15.0,
    "shoulder": 14.5,
    "neck_circumference": 14.0,
    "neck_depth": 6.0,
    "chest": 36.0,
    "ease": 2.0,
    "apex_to_apex": 7.5,
    "waist": 30.0,
    "full_length": 52.0,
    "reducted_shoulder_value": 3.0,
    "number_of_panels": 6.0,
    "flare_multiplier": 2.0,
}


@dataclass
class Issue:
    """Something wrong with, or missing from, an extracted row.

    ``blocking`` means the action cannot be compiled to geometry as-is;
    ``needs`` names the override key that would resolve it.
    """

    code: str
    detail: str
    needs: str = ""
    blocking: bool = True

    def as_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail,
                "needs": self.needs, "blocking": self.blocking}


@dataclass
class DraftAction:
    """One normalized drafting step."""

    garment: str
    panel: str
    step_index: float
    action: str
    outputs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    skipped: bool = False
    formula: str | None = None           # normalized to engine grammar
    formula_raw: str | None = None       # as extracted
    value: float | None = None           # numeric literal, source units
    value_raw: str | None = None
    direction: str | None = None
    notes: str = ""
    raw_text: str = ""
    source_image: str = ""
    issues: list[Issue] = field(default_factory=list)
    overrides: dict = field(default_factory=dict)
    duplicate_of: int | None = None

    @property
    def key(self) -> str:
        """Stable identifier used to attach overrides: ``panel:step_index``."""
        return f"{self.panel}:{self.step_index:g}"

    @property
    def blocked(self) -> bool:
        return any(i.blocking for i in self.issues)

    @property
    def measure(self) -> str | None:
        """The distance this step travels, as an engine expression."""
        if self.formula:
            return self.formula
        if self.value is not None:
            return _fmt(self.value)
        return None

    def instruction(self) -> str:
        """The natural-language form a user would type at the agent."""
        text = (self.raw_text or "").strip()
        if self.notes:
            text = f"{text} ({self.notes})" if text else self.notes
        return text

    def as_dict(self) -> dict:
        return {
            "garment": self.garment, "panel": self.panel,
            "step_index": self.step_index, "action": self.action,
            "skipped": self.skipped,
            "outputs": self.outputs, "inputs": self.inputs,
            "formula": self.formula, "formula_raw": self.formula_raw,
            "value": self.value, "value_raw": self.value_raw,
            "direction": self.direction, "notes": self.notes,
            "raw_text": self.raw_text, "source_image": self.source_image,
            "duplicate_of": self.duplicate_of,
            "issues": [i.as_dict() for i in self.issues],
        }


def _fmt(v: float) -> str:
    return f"{v:g}"


def required_fields(action: str) -> tuple[str, ...]:
    return ACTIONS.get(action, {}).get("needs", ())
