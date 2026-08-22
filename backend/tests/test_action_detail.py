"""Tests for the replay-based action-detail endpoint.

No per-action state is stored in the database (see the module docstring on
``store.py``) — a before/after view is derived by replaying a run's stored
actions against its starting version. These tests exercise that replay
directly against a real multi-step run, faking only the model call.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    import store as store_mod

    importlib.reload(store_mod)
    import agent as agent_mod
    import app as app_mod

    importlib.reload(app_mod)
    from fastapi.testclient import TestClient

    return TestClient(app_mod.app), app_mod, agent_mod


def test_action_detail_shows_reasoning_and_before_after(wired, monkeypatch):
    client, _app_mod, agent_mod = wired
    v = client.post("/api/patterns", json={"name": "Draft", "source": "blank"}).json()
    sid = v["session_id"]
    origin = 1  # the blank canvas's origin point, id 1

    calls = {"n": 0}

    def fake_create(client_, messages, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(
                "tool_use",
                [
                    _Block(type="thinking", thinking="Drop a point 10cm to the right."),
                    _Block(
                        type="tool_use",
                        id="t1",
                        name="add_point",
                        input={
                            "tool_type": "endLine",
                            "attrs": {
                                "name": "B",
                                "basePoint": str(origin),
                                "angle": "0",
                                "length": "10",
                            },
                        },
                    ),
                ],
            )
        return _Resp("end_turn", [_Block(type="text", text="Added point B.")])

    monkeypatch.setattr(agent_mod, "_create", fake_create)
    out = client.post(f"/api/sessions/{sid}/run", json={"text": "add point B"}).json()
    run = out["runs"][0]
    assert run["stopped_reason"] == "done"
    assert len(run["actions"]) == 1
    assert run["version_after_id"] or out["versions"]  # a version was created

    detail = client.get(f"/api/runs/{run['id']}/actions/0/detail").json()
    assert detail["action"]["reasoning"] == "Drop a point 10cm to the right."
    assert detail["action"]["ok"] is True
    assert detail["before"]["state_summary"]["objects"] == 1  # just the origin
    assert detail["after"]["state_summary"]["objects"] == 2  # origin + B
    assert "<svg" in detail["before"]["svg"] and "<svg" in detail["after"]["svg"]


def test_action_detail_replay_is_stable_across_multiple_calls(wired, monkeypatch):
    """Replay must be deterministic — asking for the same detail twice gives
    the same result, since nothing is mutated by looking at it."""
    client, _app_mod, agent_mod = wired
    v = client.post("/api/patterns", json={"name": "Draft", "source": "blank"}).json()
    sid = v["session_id"]

    calls = {"n": 0}

    def fake_create(client_, messages, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(
                "tool_use",
                [
                    _Block(
                        type="tool_use",
                        id="t1",
                        name="add_point",
                        input={
                            "tool_type": "endLine",
                            "attrs": {"name": "B", "basePoint": "1", "angle": "0", "length": "10"},
                        },
                    )
                ],
            )
        if calls["n"] == 2:
            return _Resp(
                "tool_use",
                [
                    _Block(
                        type="tool_use",
                        id="t2",
                        name="add_point",
                        input={
                            "tool_type": "endLine",
                            "attrs": {"name": "C", "basePoint": "1", "angle": "270", "length": "5"},
                        },
                    )
                ],
            )
        return _Resp("end_turn", [_Block(type="text", text="Done.")])

    monkeypatch.setattr(agent_mod, "_create", fake_create)
    out = client.post(f"/api/sessions/{sid}/run", json={"text": "add two points"}).json()
    run_id = out["runs"][0]["id"]
    assert len(out["runs"][0]["actions"]) == 2

    first = client.get(f"/api/runs/{run_id}/actions/1/detail").json()
    second = client.get(f"/api/runs/{run_id}/actions/1/detail").json()
    assert first == second
    # step 1's "before" already has B (step 0's result); "after" adds C too.
    assert first["before"]["state_summary"]["objects"] == 2
    assert first["after"]["state_summary"]["objects"] == 3


def test_action_detail_404s_for_an_unknown_run(wired):
    client, _app_mod, _agent_mod = wired
    res = client.get("/api/runs/run_doesnotexist/actions/0/detail")
    assert res.status_code == 404
