"""Tests for the instruction-document pipeline.

These run against the real Angrakha Maxi source that motivated the pipeline, so
they double as a regression guard on the messiness of actual extracted files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "engine", ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset.build import build  # noqa: E402
from dataset.compile import compile_document  # noqa: E402
from dataset.normalize import (  # noqa: E402
    load_overrides, load_rows, normalize, normalize_formula, normalize_value,
)

SOURCE = ROOT / "dataset" / "sources" / "angrakha_maxi.jsonl"
OVERRIDES = ROOT / "dataset" / "sources" / "angrakha_maxi.overrides.json"


@pytest.fixture(scope="module")
def doc():
    steps, inserts = load_overrides(OVERRIDES)
    return normalize(load_rows(SOURCE), steps, inserts)


@pytest.fixture(scope="module")
def compiled(doc):
    return compile_document(doc, capture_state=False)


# --- formula / value normalization ------------------------------------------
def test_inch_suffixes_are_stripped():
    expr, used = normalize_formula('yoke_length + 1in', set())
    assert expr == "#yoke_length+1"
    assert used == {"yoke_length"}


def test_point_pair_becomes_a_live_length():
    """``A_to_B`` must become the engine pseudo-variable, not a frozen number —
    that is what keeps a flare parametric."""
    expr, used = normalize_formula("(A_to_B) * flare_multiplier", set())
    assert expr == "(Line_A_B)*#flare_multiplier"
    assert used == {"flare_multiplier"}


def test_known_point_names_are_not_mistaken_for_measurements():
    expr, used = normalize_formula("C1 + 2", {"C1"})
    assert expr == "C1+2" and used == set()


@pytest.mark.parametrize("raw,expected", [
    ("1.25in", 1.25), ('0.5"', 0.5), ("2in", 2.0), (None, None),
])
def test_values_parse(raw, expected):
    assert normalize_value(raw)[0] == expected


def test_a_choice_is_not_a_measurement():
    value, _raw, issue = normalize_value("2, 3, or 4 (as required)")
    assert value is None and issue is not None
    assert issue.needs == "measure"


# --- what the normalizer finds in the real file ------------------------------
def test_multi_output_row_is_split(doc):
    """``A - C = B - D`` is two steps, however the extractor wrote it."""
    bottom = [a for a in doc.actions if a.panel == "bottom" and a.step_index == 3]
    assert [a.outputs for a in bottom] == [["C"], ["D"]]
    assert [a.inputs for a in bottom] == [["A"], ["B"]]


def test_restated_step_is_flagged_not_duplicated(doc):
    restated = next(a for a in doc.actions
                    if a.panel == "back" and a.step_index == 12)
    assert restated.duplicate_of == 8


def test_undefined_reference_is_reported_when_unpatched():
    """Without the figure-derived insert, ``G - H`` has nothing to measure from."""
    bare = normalize(load_rows(SOURCE))
    step = next(a for a in bare.actions if a.panel == "back" and a.step_index == 7)
    assert any(i.code == "undefined_input" for i in step.issues)


def test_offsets_without_a_bearing_are_reported_when_unpatched():
    bare = normalize(load_rows(SOURCE))
    step = next(a for a in bare.actions if a.panel == "back" and a.step_index == 3)
    issue = next(i for i in step.issues if i.code == "missing_direction")
    assert issue.needs == "angle"


def test_overrides_close_the_open_questions(doc):
    """The ones the figures settle stop being reported; the rest still are."""
    open_keys = {a.key for a in doc.blocked}
    assert "back:3" not in open_keys and "back:7" not in open_keys
    assert "front_left:2" in open_keys        # "0.5 inward" — figure not decisive
    assert "front_right:2" in open_keys       # A1/B1 belong to a layout we don't have


# --- compilation --------------------------------------------------------------
def test_document_compiles_to_resolvable_geometry(compiled):
    assert compiled.session.evaluated.unresolved == {}
    assert len(compiled.session.pattern.all_objects()) > 80
    assert len(compiled.verified) >= 27


def test_every_panel_becomes_a_draft_block(compiled):
    names = [b.name for b in compiled.session.pattern.draft_blocks]
    assert names == ["back", "front_left", "front_right", "bottom"]


def test_measurements_became_variables(compiled):
    assert "#chest" in compiled.variables and "#waist" in compiled.variables
    # every increment evaluates
    assert set(compiled.variables) <= set(compiled.session.evaluated.increment_values)


def test_pattern_is_parametric_not_baked(compiled):
    """Changing one measurement must move the geometry it feeds."""
    session = compiled.session
    ids = {o.raw.get("name"): o.id for o in session.pattern.all_objects()}
    before = session.evaluated.points[ids["C"]].x
    assert session.set_variable("#shoulder", "20").ok
    after = session.evaluated.points[ids["C"]].x
    assert after > before
    session.set_variable("#shoulder", "14.5")


def test_traced_panel_is_rebuilt_not_frozen(compiled):
    """front_left traces the back, so it must carry the back's own points."""
    names = {o.raw.get("name") for o in compiled.session.pattern.all_objects()}
    assert {"A_L", "B_L", "C1_L"} <= names


def test_upside_down_panel_is_mirrored(compiled):
    """front_right is traced upside down: its yoke length runs the other way."""
    pts = compiled.session.evaluated.points
    ids = {o.raw.get("name"): o.id for o in compiled.session.pattern.all_objects()}
    assert pts[ids["B_L"]].y > pts[ids["A_L"]].y      # normal: down the page
    assert pts[ids["B_R"]].y < pts[ids["A_R"]].y      # flipped: up the page


def test_reused_label_keeps_both_points(compiled):
    """The front's "F" is not the "F" traced from the back — keep both."""
    names = {o.raw.get("name") for o in compiled.session.pattern.all_objects()}
    assert "F_L" in names and "F2_L" in names


def test_tool_calls_are_the_agents_own_vocabulary(compiled):
    from agent import TOOLS

    known = {t["name"] for t in TOOLS}
    emitted = {c["name"] for s in compiled.verified for c in s.tool_calls}
    assert emitted and emitted <= known


def test_dataset_records_carry_instruction_and_verified_calls(tmp_path):
    summary = build(SOURCE, tmp_path, OVERRIDES, with_state=True)
    assert summary["unresolved"] == 0
    records = [json.loads(line) for line in
               (tmp_path / "angrakha_maxi.dataset.jsonl").read_text().splitlines()]
    verified = [r for r in records if r["verified"]]
    assert len(verified) == summary["verified"]
    sample = next(r for r in verified if r["tool_calls"])
    assert sample["instruction"] and "state_before" in sample
    assert all(not k.startswith("__") for c in sample["tool_calls"] for k in c["input"])


def test_build_writes_a_reopenable_pattern(tmp_path):
    from seamly_engine.parser import parse_pattern

    build(SOURCE, tmp_path, OVERRIDES, with_state=False)
    reopened = parse_pattern(str(tmp_path / "angrakha_maxi.sm2d"))
    assert len(reopened.draft_blocks) == 4
    assert (tmp_path / "angrakha_maxi.report.md").read_text().startswith("# angrakha_maxi")


# --- the boundary -------------------------------------------------------------
def test_the_engine_does_not_know_this_package_exists():
    """`dataset/` builds *on* the engine and the agent; nothing flows back.

    The engine is the proven part of this system. Keeping the dependency
    one-directional is what lets it stay that way: a document pipeline that
    needed engine changes to work would be a pipeline that had started
    reshaping the thing it depends on.
    """
    import subprocess

    hits = subprocess.run(
        ["git", "grep", "-l", "-e", "dataset\\.", "-e", "import dataset",
         "--", "engine/", "backend/"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert hits == []


def test_only_the_engines_public_surface_is_used():
    """No reaching into private helpers — those are free to change."""
    import re

    offenders = []
    for path in (ROOT / "dataset").glob("*.py"):
        for line in path.read_text().splitlines():
            if re.search(r"from (seamly_engine|agent)[\w.]* import .*\b_\w", line):
                offenders.append(f"{path.name}: {line.strip()}")
    assert offenders == []
