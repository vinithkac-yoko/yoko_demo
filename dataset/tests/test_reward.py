"""Calibration tests for the reward function.

A reward you cannot calibrate is decoration, so these are the checks that give
the number meaning: the reference must score exactly 1, doing nothing must score
near 0, and each component must move only for the failure it claims to measure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "engine", ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset.evaluate import (  # noqa: E402
    LiteralPolicy, NoopPolicy, PerturbPolicy, ReferencePolicy, run_episode,
)
from dataset.reward import WEIGHTS  # noqa: E402

SOURCE = ROOT / "dataset" / "sources" / "angrakha_maxi.jsonl"


@pytest.fixture(scope="module")
def reference_run():
    return run_episode(SOURCE, ReferencePolicy())


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_the_document_scores_itself_perfectly(reference_run):
    """The ceiling. If this drifts below 1 the reward is punishing correctness."""
    assert reference_run.total == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in reference_run.components.values())


def test_the_document_scores_itself_perfectly_in_rollout():
    """Also true when the policy carries its own pattern forward — otherwise
    the two modes disagree about what 'correct' means."""
    run = run_episode(SOURCE, ReferencePolicy(), rollout=True)
    assert run.total == pytest.approx(1.0)
    assert run.detail["final"]["score"] == pytest.approx(1.0)


def test_doing_nothing_scores_near_zero(reference_run):
    run = run_episode(SOURCE, NoopPolicy())
    assert run.total < 0.1
    assert run.components["executed"] < 0.1
    # A policy that does nothing must not be paid for having broken nothing.
    assert run.components["nondestructive"] < 0.1
    assert run.total < reference_run.total


def test_baking_the_numbers_costs_only_parametricity():
    """The failure mode that renders perfectly and is worthless: identical
    geometry, every formula collapsed to a literal."""
    run = run_episode(SOURCE, LiteralPolicy())
    assert run.components["placement"] == pytest.approx(1.0)
    assert run.components["structure"] == pytest.approx(1.0)
    assert run.components["parametric"] < 0.6
    assert 0.85 < run.total < 0.97


def test_placement_degrades_smoothly_with_drift():
    scores = [run_episode(SOURCE, PerturbPolicy(scale=s)).components["placement"]
              for s in (1.02, 1.1, 1.3)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_drift_compounds_in_rollout_but_not_under_teacher_forcing():
    """The two modes are answering different questions, and the numbers should
    show it: teacher forcing isolates per-step skill, rollout lets errors pile
    up the way they would in the app."""
    forced = run_episode(SOURCE, PerturbPolicy(scale=1.1))
    rolled = run_episode(SOURCE, PerturbPolicy(scale=1.1), rollout=True)
    assert rolled.total < forced.total
    assert rolled.detail["final"]["score"] < 0.5


def test_a_finished_pattern_is_compared_by_name():
    run = run_episode(SOURCE, NoopPolicy(), rollout=True)
    final = run.detail["final"]
    assert final["matched"] == 0 and final["of"] > 40
    assert "A" in final["missing"]


def test_limit_shortens_the_episode():
    assert len(run_episode(SOURCE, ReferencePolicy(), limit=5).steps) == 5
