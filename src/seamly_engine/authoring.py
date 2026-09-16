"""Create patterns from scratch (blank canvas).

Drafting from nothing needs two things Seamly gives you implicitly: a draft block
to put objects in, and an **origin point** to measure from (every constructive
tool is relative to an existing point). ``new_pattern`` provides both, so the
agent can start with "point A is the origin" and build outward.
"""

from __future__ import annotations

from .model import DraftBlock, Increment, Pattern, PatternObject

# The cm-scale variable Aldrich-style drafts use so absolute numbers scale with
# the wearer's height. Included by default so formulas can reference #CM.
_DEFAULT_INCREMENTS = [
    Increment(
        name="#BaseHeight",
        formula="166",
        description="Height the pattern was drafted for; used by #CM.",
    ),
    Increment(
        name="#CM",
        formula="height/#BaseHeight",
        description="Scales absolute cm values to the wearer's height.",
    ),
]


def new_pattern(
    name: str = "Untitled pattern",
    *,
    unit: str = "cm",
    block_name: str = "Draft block 1",
    measurements_file: str = "",
    origin_name: str = "A",
    with_defaults: bool = True,
) -> Pattern:
    """A blank pattern containing one draft block and a single origin point."""
    pat = Pattern(unit=unit, pattern_name=name, measurements_file=measurements_file)
    if with_defaults:
        pat.increments = list(_DEFAULT_INCREMENTS)
    origin = PatternObject(
        id=1,
        tag="point",
        tool_type="single",
        raw={
            "id": "1",
            "type": "single",
            "name": origin_name,
            "x": "0",
            "y": "0",
            "mx": "0.13",
            "my": "0.26",
            "showPointName": "true",
        },
    )
    pat.draft_blocks.append(DraftBlock(name=block_name, objects=[origin]))
    return pat


def add_draft_block(pattern: Pattern, name: str) -> DraftBlock:
    db = DraftBlock(name=name)
    pattern.draft_blocks.append(db)
    return db
