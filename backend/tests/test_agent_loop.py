"""Agent-loop tests with a fake model client (no API key / network needed).

Two things matter here: a malformed tool call or an error inside the loop
must come back as a readable ``RunResult`` — never an exception — and the
action log the loop produces (:class:`agent.Action` per tool call, carrying
the thinking text that preceded it) is what the UI is built on, so its shape
is pinned down directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

import agent

import seamly_engine as se
from seamly_engine.operations import PatternSession
from seamly_engine.pieces import pieces_for_key

FIX = ROOT / "tests" / "fixtures"


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


@pytest.fixture
def session():
    return PatternSession(
        se.load_pattern(str(FIX / "aldrich_basic.sm2d")),
        se.load_measurements(str(FIX / "aldrich_measurements.vst")),
    )


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _names(session):
    return {o.raw.get("name"): o.id for o in session.pattern.all_objects() if o.raw.get("name")}


# --- the action log --------------------------------------------------------
def test_tool_use_produces_one_action_with_its_reasoning(monkeypatch, session):
    ids = _names(session)
    pieces = pieces_for_key(session.pattern, "A")
    calls = {"n": 0}

    def fake_create(client, messages, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(
                "tool_use",
                [
                    _Block(type="thinking", thinking="This dart shapes the waist seam."),
                    _Block(
                        type="tool_use",
                        id="t1",
                        name="add_dart",
                        input={
                            "baseLineP1": str(ids["A1"]),
                            "baseLineP2": str(ids["A9"]),
                            "dartP1": str(ids["A11"]),
                            "dartP2": str(ids["A13"]),
                            "dartP3": str(ids["A12"]),
                            "name1": "X1",
                            "name2": "X2",
                        },
                    ),
                ],
            )
        return _Resp("end_turn", [_Block(type="text", text="Added the dart.")])

    monkeypatch.setattr(agent, "_create", fake_create)
    result = agent.run_instruction(session, "add a dart", pieces, "Skirt")

    assert result.stopped_reason == "done"
    assert result.final_text == "Added the dart."
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.ok is True and action.tool_name == "add_dart"
    assert action.reasoning == "This dart shapes the waist seam."
    assert action.step_index == 0
    assert len(session.added_ids) == 3  # the tool + its two leg points


def test_several_tool_calls_in_one_turn_share_the_preceding_thinking(monkeypatch, session):
    ids = _names(session)
    calls = {"n": 0}

    def fake_create(client, messages, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(
                "tool_use",
                [
                    _Block(type="thinking", thinking="Add two reference points."),
                    _Block(
                        type="tool_use",
                        id="t1",
                        name="add_point",
                        input={
                            "tool_type": "endLine",
                            "attrs": {
                                "name": "Z1",
                                "basePoint": str(ids["A1"]),
                                "angle": "0",
                                "length": "1",
                            },
                        },
                    ),
                    _Block(
                        type="tool_use",
                        id="t2",
                        name="add_point",
                        input={
                            "tool_type": "endLine",
                            "attrs": {
                                "name": "Z2",
                                "basePoint": str(ids["A1"]),
                                "angle": "90",
                                "length": "1",
                            },
                        },
                    ),
                ],
            )
        return _Resp("end_turn", [_Block(type="text", text="Done.")])

    monkeypatch.setattr(agent, "_create", fake_create)
    result = agent.run_instruction(session, "add two points", None, "")

    assert len(result.actions) == 2
    assert all(a.reasoning == "Add two reference points." for a in result.actions)
    assert [a.step_index for a in result.actions] == [0, 1]


def test_a_later_thinking_block_updates_reasoning_for_the_next_action(monkeypatch, session):
    ids = _names(session)
    calls = {"n": 0}

    def fake_create(client, messages, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(
                "tool_use",
                [
                    _Block(type="thinking", thinking="First reason."),
                    _Block(
                        type="tool_use",
                        id="t1",
                        name="add_point",
                        input={
                            "tool_type": "endLine",
                            "attrs": {
                                "name": "Z1",
                                "basePoint": str(ids["A1"]),
                                "angle": "0",
                                "length": "1",
                            },
                        },
                    ),
                    _Block(type="thinking", thinking="Second reason."),
                    _Block(
                        type="tool_use",
                        id="t2",
                        name="add_point",
                        input={
                            "tool_type": "endLine",
                            "attrs": {
                                "name": "Z2",
                                "basePoint": str(ids["A1"]),
                                "angle": "90",
                                "length": "1",
                            },
                        },
                    ),
                ],
            )
        return _Resp("end_turn", [_Block(type="text", text="Done.")])

    monkeypatch.setattr(agent, "_create", fake_create)
    result = agent.run_instruction(session, "add two points", None, "")

    assert [a.reasoning for a in result.actions] == ["First reason.", "Second reason."]


def test_a_failed_action_is_recorded_not_dropped(monkeypatch, session):
    calls = {"n": 0}

    def fake_create(client, messages, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(
                "tool_use",
                [_Block(type="tool_use", id="t1", name="add_point", input={"attrs": {}})],
            )
        return _Resp("end_turn", [_Block(type="text", text="Could not add that.")])

    monkeypatch.setattr(agent, "_create", fake_create)
    result = agent.run_instruction(session, "add a point", None, "")

    assert len(result.actions) == 1
    assert result.actions[0].ok is False
    assert "missing required argument" in result.actions[0].message


def test_api_error_returns_a_result_not_an_exception(monkeypatch, session):
    def boom(client, messages, model=None):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(agent, "_create", boom)
    result = agent.run_instruction(session, "do something", None, "")

    assert result.stopped_reason == "error"
    assert "agent hit an error" in result.final_text
    assert result.actions == []


def test_no_api_key_is_reported_not_attempted(monkeypatch, session):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = agent.run_instruction(session, "add a dart", None, "")
    assert result.stopped_reason == "no_key"
    assert result.actions == []


def test_max_steps_stops_the_run_and_keeps_what_happened(monkeypatch, session):
    def fake_create(client, messages, model=None):
        return _Resp(
            "tool_use",
            [
                _Block(
                    type="tool_use",
                    id="t",
                    name="add_variable",
                    input={
                        "name": f"#V{len(messages)}",
                        "formula": "1",
                    },
                )
            ],
        )

    monkeypatch.setattr(agent, "_create", fake_create)
    monkeypatch.setattr(agent, "MAX_ITERATIONS", 3)
    result = agent.run_instruction(session, "keep going forever", None, "")

    assert result.stopped_reason == "max_steps"
    assert len(result.actions) == 3


# --- dispatch: piece operations ---------------------------------------------
def test_dispatch_split_piece(session):
    from seamly_engine.pieces import piece_by_id

    ids = _names(session)
    back = next(p for p in session.pattern.pieces if p.name == "A - Skirt Back")
    res = agent.dispatch_tool(
        session,
        "split_piece",
        {
            "piece_id": back.id,
            "point_a_id": ids["A1"],
            "point_b_id": ids["A10"],
            "name_a": "Waistband",
            "name_b": "Skirt Body",
        },
    )
    assert res.ok, res.message
    assert piece_by_id(session.pattern, back.id) is None
    assert {p.name for p in session.pattern.pieces} & {"Waistband", "Skirt Body"} == {
        "Waistband",
        "Skirt Body",
    }


def test_dispatch_merge_piece_after_split(session):
    ids = _names(session)
    back = next(p for p in session.pattern.pieces if p.name == "A - Skirt Back")
    agent.dispatch_tool(
        session,
        "split_piece",
        {
            "piece_id": back.id,
            "point_a_id": ids["A1"],
            "point_b_id": ids["A10"],
            "name_a": "Waistband",
            "name_b": "Skirt Body",
        },
    )
    wb = next(p for p in session.pattern.pieces if p.name == "Waistband")
    sk = next(p for p in session.pattern.pieces if p.name == "Skirt Body")
    res = agent.dispatch_tool(
        session,
        "merge_piece",
        {
            "piece_a_id": wb.id,
            "piece_b_id": sk.id,
            "edge_point_1": ids["A1"],
            "edge_point_2": ids["A10"],
            "name": "Rejoined",
        },
    )
    assert res.ok, res.message
    assert {p.name for p in session.pattern.pieces} == {
        "A - Skirt Front",
        "B - Trousers Front",
        "B - Trousers Back",
        "C - Bodice Back",
        "C - Bodice Front",
        "D - 1 Piece Sleeve",
        "Rejoined",
    }


def test_dispatch_edit_and_delete_piece(session):
    back = next(p for p in session.pattern.pieces if p.name == "A - Skirt Back")
    res = agent.dispatch_tool(
        session, "edit_piece", {"piece_id": back.id, "name": "Renamed", "width": "2"}
    )
    assert res.ok, res.message
    assert next(p for p in session.pattern.pieces if p.id == back.id).name == "Renamed"

    res = agent.dispatch_tool(session, "delete_piece", {"piece_id": back.id})
    assert res.ok, res.message
    assert back.id not in {p.id for p in session.pattern.pieces}


def test_dispatch_unknown_curve_kind(session):
    res = agent.dispatch_tool(session, "add_curve", {"kind": "banana"})
    assert res.ok is False and "unknown kind" in res.message


def test_every_tool_schema_has_a_dispatch_branch(session):
    for name in agent.TOOL_NAMES:
        res = agent.dispatch_tool(session, name, {})
        assert "unknown tool" not in res.message, name


# --- model selection ---------------------------------------------------------
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

    # display=summarized is what makes reasoning non-empty: without it the API
    # returns thinking blocks with empty text, so the action log shows nothing.
    want = {"type": "adaptive", "display": "summarized"}
    agent._create(_Client(), [], "claude-opus-4-8")
    assert sent["thinking"] == want
    assert sent["output_config"] == {"effort": agent.EFFORT}

    agent._create(_Client(), [], "claude-sonnet-5")
    assert sent["thinking"] == want


def test_create_degrades_when_effort_rejected():
    """An SDK/model that rejects output_config still gets thinking; one that
    rejects thinking too still runs. Neither should fail the whole run."""
    sent = {}

    class _Msgs:
        def __init__(self, reject):
            self.reject = reject

        def create(self, **kw):
            if any(k in kw for k in self.reject):
                raise TypeError("unexpected keyword argument")
            sent.clear()
            sent.update(kw)
            return _Resp("end_turn", [])

    class _Client:
        def __init__(self, reject):
            self.messages = _Msgs(reject)

    agent._create(_Client({"output_config"}), [], "claude-sonnet-5")
    assert "output_config" not in sent
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}

    agent._create(_Client({"output_config", "thinking"}), [], "claude-sonnet-5")
    assert "thinking" not in sent and "output_config" not in sent


def test_run_instruction_honours_requested_model(monkeypatch, session):
    used = []

    def fake_create(client, messages, model=None):
        used.append(model)
        return _Resp("end_turn", [_Block(type="text", text="done")])

    monkeypatch.setattr(agent, "_create", fake_create)
    agent.run_instruction(session, "hello", None, "", model="claude-opus-4-8")
    assert used == ["claude-opus-4-8"]


# --- streaming + zero-action runs -------------------------------------------
def test_stream_yields_each_action_before_the_final_result(monkeypatch, session):
    """The stream must emit an action as soon as its edit is applied, so the
    UI can re-render mid-run — not batch them up at the end."""
    ids = _names(session)
    calls = []

    def fake_create(client, messages, model=None):
        if not calls:
            calls.append(1)
            return _Resp(
                "tool_use",
                [
                    _Block(type="thinking", thinking="move both points"),
                    _Block(
                        type="tool_use",
                        id="t1",
                        name="edit_object",
                        input={"object_id": ids["A1"], "attrs": {"x": "5"}},
                    ),
                    _Block(
                        type="tool_use",
                        id="t2",
                        name="edit_object",
                        input={"object_id": ids["A2"], "attrs": {"x": "6"}},
                    ),
                ],
            )
        return _Resp("end_turn", [_Block(type="text", text="all set")])

    monkeypatch.setattr(agent, "_create", fake_create)
    events = list(agent.stream_instruction(session, "move them", None, ""))

    kinds = [k for k, _ in events]
    assert kinds == ["action", "action", "done"], "actions must stream before the result"

    first, second = events[0][1], events[1][1]
    assert (first.step_index, second.step_index) == (0, 1)
    assert first.reasoning == "move both points"

    result = events[-1][1]
    assert result.stopped_reason == "done"
    assert result.final_text == "all set"
    assert len(result.actions) == 2


def test_run_instruction_matches_the_stream(monkeypatch, session):
    """The batch API is just the stream drained — same result either way."""

    def fake_create(client, messages, model=None):
        return _Resp("end_turn", [_Block(type="text", text="nothing to do")])

    monkeypatch.setattr(agent, "_create", fake_create)
    result = agent.run_instruction(session, "hello", None, "")
    assert result.stopped_reason == "done"
    assert result.final_text == "nothing to do"
    assert result.actions == []


def test_zero_action_run_surfaces_its_thinking(monkeypatch, session):
    """A run that spends its whole budget thinking and never acts should show
    what it was thinking, not a bare 'ran out of budget' with an empty log."""

    def fake_create(client, messages, model=None):
        return _Resp(
            "max_tokens",
            [_Block(type="thinking", thinking="I need to find the waist-to-hip points first")],
        )

    monkeypatch.setattr(agent, "_create", fake_create)
    result = agent.run_instruction(session, "add a belt", None, "")
    assert result.actions == []
    assert "waist-to-hip" in result.final_text


def test_zero_action_run_without_thinking_explains_the_budget(monkeypatch, session):
    """With no thinking text to show, the note should still be actionable."""

    def fake_create(client, messages, model=None):
        return _Resp("max_tokens", [])

    monkeypatch.setattr(agent, "_create", fake_create)
    result = agent.run_instruction(session, "add a belt", None, "")
    assert result.actions == []
    assert "select a block" in result.final_text


def test_stream_reports_errors_as_a_done_event(monkeypatch, session):
    """A crash mid-run must arrive as a terminal event, keeping any actions
    already applied — never propagate as an exception."""

    def fake_create(client, messages, model=None):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(agent, "_create", fake_create)
    events = list(agent.stream_instruction(session, "hi", None, ""))
    assert [k for k, _ in events] == ["done"]
    result = events[0][1]
    assert result.stopped_reason == "error"
    assert "model exploded" in result.final_text
