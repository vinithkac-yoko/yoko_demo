"""Agent-loop tests with a fake model client (no API key / network needed).

These guard the failure mode that surfaced as an opaque HTTP 500 in the app: a
malformed tool call from the model, or an error inside the loop, must come back
as a readable message the model can correct — never an exception.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "engine"))

import agent  # noqa: E402
import seamly_engine as se  # noqa: E402
from seamly_engine.operations import PatternSession  # noqa: E402
from seamly_engine.pieces import pieces_for_key  # noqa: E402

FIX = ROOT / "engine" / "tests" / "fixtures"


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


@pytest.fixture
def session():
    return PatternSession(se.load_pattern(str(FIX / "aldrich_basic.sm2d")),
                          se.load_measurements(str(FIX / "aldrich_measurements.vst")))


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _names(session):
    return {o.raw.get("name"): o.id for o in session.pattern.all_objects() if o.raw.get("name")}


def test_tool_use_loop_applies_edit(monkeypatch, session):
    ids = _names(session)
    pieces = pieces_for_key(session.pattern, "A")
    calls = {"n": 0}

    def fake_create(client, messages, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp("tool_use", [_Block(type="tool_use", id="t1", name="add_dart", input={
                "baseLineP1": str(ids["A1"]), "baseLineP2": str(ids["A9"]),
                "dartP1": str(ids["A11"]), "dartP2": str(ids["A13"]),
                "dartP3": str(ids["A12"]), "name1": "X1", "name2": "X2"})])
        return _Resp("end_turn", [_Block(type="text", text="Added the dart.")])

    monkeypatch.setattr(agent, "_create", fake_create)
    out = agent.run_turn(session, "add a dart", pieces, "Skirt")
    assert out["reply"] == "Added the dart."
    assert out["tool_calls"] and out["tool_calls"][0]["ok"] is True
    assert len(session.added_ids) == 3  # the tool + its two leg points


def test_malformed_tool_call_does_not_raise(monkeypatch, session):
    """A tool call missing a required argument must return a correctable error."""
    pieces = pieces_for_key(session.pattern, "A")
    calls = {"n": 0}

    def fake_create(client, messages, model=None):
        calls["n"] += 1
        if calls["n"] == 1:  # missing 'tool_type'
            return _Resp("tool_use", [_Block(type="tool_use", id="t1",
                                             name="add_point", input={"attrs": {}})])
        return _Resp("end_turn", [_Block(type="text", text="Could not add that.")])

    monkeypatch.setattr(agent, "_create", fake_create)
    out = agent.run_turn(session, "add a point", pieces, "Skirt")
    assert out["tool_calls"][0]["ok"] is False
    assert "missing required argument" in out["tool_calls"][0]["message"]


def test_api_error_returns_message_not_exception(monkeypatch, session):
    pieces = pieces_for_key(session.pattern, "A")

    def boom(client, messages, model=None):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(agent, "_create", boom)
    out = agent.run_turn(session, "do something", pieces, "Skirt")
    assert "agent hit an error" in out["reply"]
    assert out["tool_calls"] == []


def test_dispatch_unknown_curve_kind(session):
    res = agent.dispatch_tool(session, "add_curve", {"kind": "banana"})
    assert res.ok is False and "unknown kind" in res.message


def test_model_picker_thinking_is_model_aware(monkeypatch):
    """Adaptive thinking only goes to models that accept it (Haiku rejects it)."""
    sent = {}

    class _Msgs:
        def create(self, **kw):
            sent.clear()
            sent.update(kw)
            return _Resp("end_turn", [])

    class _Client:
        messages = _Msgs()

    agent._create(_Client(), [], "claude-haiku-4-5")
    assert "thinking" not in sent and sent["model"] == "claude-haiku-4-5"

    agent._create(_Client(), [], "claude-opus-4-8")
    assert sent["thinking"] == {"type": "adaptive"}

    agent._create(_Client(), [], "claude-sonnet-5")
    assert sent["thinking"] == {"type": "adaptive"}


def test_run_turn_honours_requested_model(monkeypatch, session):
    used = []

    def fake_create(client, messages, model=None):
        used.append(model)
        return _Resp("end_turn", [_Block(type="text", text="done")])

    monkeypatch.setattr(agent, "_create", fake_create)
    agent.run_turn(session, "hello", None, "", model="claude-opus-4-8")
    assert used == ["claude-opus-4-8"]
