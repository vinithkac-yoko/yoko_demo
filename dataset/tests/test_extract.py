"""Tests for the PDF → drafting-actions front of the pipeline.

The model call itself needs a key and a network, so what's pinned here is
everything around it: the contract stays in step with the schema, the reply
parser survives the shapes a model actually returns, and the cache key changes
when it should.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "engine", ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import extract  # noqa: E402
from dataset.normalize import normalize  # noqa: E402
from dataset.schema import ACTIONS  # noqa: E402


def test_contract_lists_every_verb():
    """The prompt is generated from the schema so it can't drift from it."""
    text = extract.contract()
    for verb in ACTIONS:
        assert verb in text


def test_contract_pins_the_angle_convention():
    """Getting this wrong flips every pattern upside down."""
    text = extract.contract()
    assert "0 = right" in text and "270 = down" in text


@pytest.mark.parametrize("wrapper", [
    "{body}",
    "```json\n{body}\n```",
    "Here are the steps:\n\n{body}\n\nThat's the whole document.",
])
def test_reply_shapes_all_parse(wrapper):
    body = '[{"action": "SET_ORIGIN", "panel": "back", "step_index": 0, ' \
           '"output_ref": "A"}]'
    rows = extract.parse_rows(wrapper.format(body=body), "demo")
    assert len(rows) == 1 and rows[0]["output_ref"] == "A"
    assert rows[0]["garment"] == "demo"


def test_unknown_verbs_are_dropped_not_passed_on():
    reply = '[{"action": "SET_ORIGIN", "output_ref": "A", "panel": "back"},' \
            ' {"action": "DO_SOMETHING_ELSE", "output_ref": "B"}]'
    rows = extract.parse_rows(reply, "demo")
    assert [r["action"] for r in rows] == ["SET_ORIGIN"]


def test_missing_array_is_an_error_not_silent_emptiness():
    with pytest.raises(ValueError):
        extract.parse_rows("I could not read the figures.", "demo")


def test_extracted_bearing_reaches_the_compiler():
    """A bearing read off a figure must satisfy the same requirement an
    overrides file does — that is the whole point of extracting it."""
    reply = """[
      {"action": "SET_ORIGIN", "panel": "back", "step_index": 0, "output_ref": "A"},
      {"action": "DEFINE_POINT_OFFSET", "panel": "back", "step_index": 1,
       "output_ref": "A1", "input_refs": ["A"], "value": "1in",
       "direction": "offset", "angle": 270, "confidence": "high",
       "raw_text": "A - A1 -> 1\\" drop", "source_image": "fig1"}
    ]"""
    doc = normalize(extract.parse_rows(reply, "demo"))
    step = next(a for a in doc.actions if a.step_index == 1)
    assert step.angle == 270.0
    assert not step.blocked


def test_the_same_step_without_a_bearing_is_still_reported():
    reply = """[
      {"action": "SET_ORIGIN", "panel": "back", "step_index": 0, "output_ref": "A"},
      {"action": "DEFINE_POINT_OFFSET", "panel": "back", "step_index": 1,
       "output_ref": "A1", "input_refs": ["A"], "value": "1in",
       "direction": "offset", "raw_text": "A - A1 -> 1\\" drop"}
    ]"""
    doc = normalize(extract.parse_rows(reply, "demo"))
    step = next(a for a in doc.actions if a.step_index == 1)
    assert any(i.needs == "angle" for i in step.issues)


def test_cache_key_tracks_the_pdf_the_model_and_the_prompt():
    a = extract.cache_key(b"pdf-one", "model-x")
    assert a != extract.cache_key(b"pdf-two", "model-x")
    assert a != extract.cache_key(b"pdf-one", "model-y")
    assert a == extract.cache_key(b"pdf-one", "model-x")


def test_pages_go_to_the_model_as_figures_then_text():
    pages = [extract.Page(number=1, text="A - B -> yoke",
                          figures=[extract.Figure("fig1", b"\x89PNG...", "image/png")])]
    blocks = extract.build_content(pages, "demo")
    kinds = [b["type"] for b in blocks]
    assert "image" in kinds
    image_at = kinds.index("image")
    text_after = blocks[image_at + 1]["text"]
    assert "A - B -> yoke" in text_after
    assert any("fig1" in b.get("text", "") for b in blocks if b["type"] == "text")


def test_a_cold_run_without_a_key_says_so_plainly(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pdf = tmp_path / "thing.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        extract.extract(pdf, garment="thing", cache_dir=tmp_path / "cache")


def test_a_warm_cache_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pdf = tmp_path / "thing.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really")
    cache = tmp_path / "cache"
    cache.mkdir()
    key = extract.cache_key(pdf.read_bytes(), extract.DEFAULT_MODEL)
    (cache / f"thing-{key}.txt").write_text(
        '[{"action": "SET_ORIGIN", "panel": "back", "step_index": 0, "output_ref": "A"}]')
    rows, provenance = extract.extract(pdf, garment="thing", cache_dir=cache)
    assert len(rows) == 1 and provenance.startswith("cache:")
