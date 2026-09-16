"""Store + API tests for the pattern-drafting workspace.

Covers the persistence model this pass introduced: version history as a
**branching DAG** (unchanged from before), and **runs + actions** replacing
chat history — one typed instruction is one run, its tool calls are its
actions, and "branch from here" means opening a new session on the version a
completed run produced.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # keep the agent offline
    import store as store_mod

    importlib.reload(store_mod)
    import app as app_mod

    importlib.reload(app_mod)
    from fastapi.testclient import TestClient

    return TestClient(app_mod.app)


# --- store -------------------------------------------------------------------
def test_version_branching(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_DB", str(tmp_path / "s.db"))
    import store as store_mod

    importlib.reload(store_mod)
    s = store_mod.Store()
    pid = s.create_pattern("P")
    root = s.save_version(pid, "<a/>", label="root")
    a = s.save_version(pid, "<b/>", parent_id=root, label="A")
    b = s.save_version(pid, "<c/>", parent_id=root, label="B")
    parents = {v["label"]: v["parent_id"] for v in s.list_versions(pid)}
    assert parents["root"] is None
    assert parents["A"] == root and parents["B"] == root  # two children => a branch
    assert a != b


def test_run_and_action_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_DB", str(tmp_path / "r.db"))
    import store as store_mod

    importlib.reload(store_mod)
    s = store_mod.Store()
    pid = s.create_pattern("P")
    before = s.save_version(pid, "<a/>", label="before")
    after = s.save_version(pid, "<b/>", parent_id=before, label="after")
    sid = s.create_session(pid, before)
    run_id = s.create_run(sid, "add a dart", "claude-sonnet-5", before)
    s.add_action(
        run_id,
        0,
        "add_dart",
        {"baseLineP1": "1"},
        reasoning="shapes the waist",
        ok=True,
        message="added trueDarts #9",
        touched_ids=[9, 10, 11],
    )
    s.add_action(
        run_id, 1, "add_point", {"attrs": {}}, ok=False, message="missing required argument"
    )
    s.finish_run(
        run_id, final_text="Added the dart.", stopped_reason="done", version_after_id=after
    )

    run = s.get_run(run_id)
    assert run["final_text"] == "Added the dart." and run["version_after_id"] == after
    actions = s.list_actions(run_id)
    assert [a["ok"] for a in actions] == [True, False]
    assert actions[0]["reasoning"] == "shapes the waist"
    assert actions[0]["touched_ids"] == [9, 10, 11]


# --- api -----------------------------------------------------------------------
def test_blank_canvas_flow(client):
    v = client.post("/api/patterns", json={"name": "Draft", "source": "blank"}).json()
    assert v["state_summary"]["objects"] == 1 and v["state_summary"]["unresolved"] == 0
    assert v["runs"] == [] and v["blocks"] == []  # nothing drafted or run yet
    sid = v["session_id"]

    saved = client.post(f"/api/sessions/{sid}/save", json={"label": "first"}).json()
    labels = [x["label"] for x in saved["versions"]]
    assert "initial" in labels and "first" in labels

    exp = client.get(f"/api/sessions/{sid}/export.sm2d")
    assert exp.status_code == 200 and exp.text.startswith("<?xml")
    assert "attachment" in exp.headers["content-disposition"]


def test_import_creates_pattern(client):
    v = client.post("/api/patterns", json={"name": "Src", "source": "blank"}).json()
    xml = client.get(f"/api/sessions/{v['session_id']}/export.sm2d").text
    imported = client.post(
        "/api/patterns/import", files={"file": ("mine.sm2d", xml, "application/xml")}
    ).json()
    assert imported["state_summary"]["objects"] == 1
    assert len(client.get("/api/patterns").json()["patterns"]) == 2


def test_open_version_branches(client):
    v = client.post("/api/patterns", json={"name": "Aldrich", "source": "sample"}).json()
    sid, pid = v["session_id"], v["pattern"]["id"]
    root = v["versions"][0]["id"]
    client.post(f"/api/sessions/{sid}/save", json={"label": "A"})
    opened = client.post(f"/api/patterns/{pid}/open", json={"version_id": root}).json()
    b = client.post(f"/api/sessions/{opened['session_id']}/save", json={"label": "B"}).json()
    parents = {x["label"]: x["parent_id"] for x in b["versions"]}
    assert parents["A"] == parents["B"] == root  # sibling branches


def test_block_selection_and_whole_pattern(client):
    v = client.post("/api/patterns", json={"name": "Aldrich", "source": "sample"}).json()
    sid = v["session_id"]
    assert [b["label"] for b in v["blocks"]] == ["Skirt", "Trousers", "Bodice", "1 Piece Sleeve"]
    picked = client.post(f"/api/sessions/{sid}/select_block", json={"block": "C"}).json()
    assert picked["block"]["label"] == "Bodice" and picked["svg"]
    whole = client.post(f"/api/sessions/{sid}/select_block", json={"block": ""}).json()
    assert whole["block"] is None


def test_run_without_a_key_is_recorded_and_makes_no_version(client):
    """No ANTHROPIC_API_KEY in the fixture — the run should come back as a
    readable no-op, not an error, and not fabricate a new version."""
    v = client.post("/api/patterns", json={"name": "Draft", "source": "blank"}).json()
    sid = v["session_id"]
    before_versions = len(v["versions"])

    out = client.post(f"/api/sessions/{sid}/run", json={"text": "add a dart"}).json()
    assert len(out["runs"]) == 1
    run = out["runs"][0]
    assert run["stopped_reason"] == "no_key"
    assert run["actions"] == []
    assert len(out["versions"]) == before_versions  # nothing to branch from


def test_run_history_persists_across_a_fresh_get(client):
    v = client.post("/api/patterns", json={"name": "Draft", "source": "blank"}).json()
    sid = v["session_id"]
    client.post(f"/api/sessions/{sid}/run", json={"text": "add a dart"})
    again = client.get(f"/api/sessions/{sid}").json()
    assert len(again["runs"]) == 1
    assert again["runs"][0]["instruction"] == "add a dart"


def test_branching_from_a_run_with_no_change_is_refused(client):
    v = client.post("/api/patterns", json={"name": "Draft", "source": "blank"}).json()
    sid = v["session_id"]
    out = client.post(f"/api/sessions/{sid}/run", json={"text": "add a dart"}).json()
    run_id = out["runs"][0]["id"]
    res = client.post(f"/api/sessions/{sid}/runs/{run_id}/branch")
    assert res.status_code == 400


def test_model_picker_api(client):
    v = client.post("/api/patterns", json={"name": "D", "source": "blank"}).json()
    sid = v["session_id"]
    assert v["model"] == "claude-sonnet-5"  # sensible default
    assert {m["id"] for m in v["models"]} == {
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-haiku-4-5",
    }

    out = client.post(f"/api/sessions/{sid}/model", json={"model": "claude-opus-4-8"}).json()
    assert out["model"] == "claude-opus-4-8"
    assert client.get(f"/api/sessions/{sid}").json()["model"] == "claude-opus-4-8"
    assert client.post(f"/api/sessions/{sid}/model", json={"model": "nope"}).status_code == 400


# --- streaming run -------------------------------------------------------------
def _sse_events(raw: str):
    """Parse an SSE body into [(event, data), ...]."""
    import json as _json

    out = []
    for chunk in raw.strip().split("\n\n"):
        if not chunk.strip():
            continue
        name, data = None, None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = _json.loads(line[len("data: ") :])
        out.append((name, data))
    return out


def test_run_stream_emits_actions_then_done(client, monkeypatch):
    """The SSE stream must deliver each action as it happens (with a render so
    the canvas can update mid-run) and end with the full session view."""
    import agent as agent_mod

    v = client.post("/api/patterns", json={"name": "S", "source": "blank"}).json()
    sid = v["session_id"]

    def fake_stream(session, instruction, pieces=None, label="", model=None, config=None):
        a = agent_mod.Action(
            step_index=0,
            tool_name="add_point",
            tool_input={"attrs": {"name": "Z1"}},
            reasoning="need a reference point",
            ok=True,
            message="added point #7",
            touched_ids=[7],
        )
        yield ("action", a)
        yield (
            "done",
            agent_mod.RunResult(final_text="Added it.", actions=[a], stopped_reason="done"),
        )

    monkeypatch.setattr(agent_mod, "stream_instruction", fake_stream)

    with client.stream(
        "GET", f"/api/sessions/{sid}/run/stream", params={"text": "add a point"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _sse_events("".join(r.iter_text()))

    kinds = [k for k, _ in events]
    assert kinds == ["start", "action", "done"]

    _, action_ev = events[1]
    assert action_ev["action"]["tool_name"] == "add_point"
    assert action_ev["action"]["reasoning"] == "need a reference point"
    assert action_ev["svg"].startswith("<svg"), "each action carries a fresh render"

    _, done_ev = events[2]
    assert done_ev["runs"][-1]["final_text"] == "Added it."


def test_streamed_run_is_persisted_like_a_batch_run(client, monkeypatch):
    """Streaming must not skip persistence — the run and its actions land in
    the store exactly as the POST endpoint would leave them."""
    import agent as agent_mod

    v = client.post("/api/patterns", json={"name": "S", "source": "blank"}).json()
    sid = v["session_id"]

    def fake_stream(session, instruction, pieces=None, label="", model=None, config=None):
        a = agent_mod.Action(
            step_index=0,
            tool_name="add_point",
            tool_input={},
            reasoning="because",
            ok=False,
            message="nope",
        )
        yield ("action", a)
        yield (
            "done",
            agent_mod.RunResult(final_text="Could not.", actions=[a], stopped_reason="done"),
        )

    monkeypatch.setattr(agent_mod, "stream_instruction", fake_stream)
    with client.stream("GET", f"/api/sessions/{sid}/run/stream", params={"text": "do it"}) as r:
        list(r.iter_text())

    view = client.get(f"/api/sessions/{sid}").json()
    run = view["runs"][-1]
    assert run["final_text"] == "Could not."
    assert [a["tool_name"] for a in run["actions"]] == ["add_point"]
    assert run["actions"][0]["reasoning"] == "because"


# --- tuning: config, presets, ratings, compare, export -------------------------
def test_config_defaults_expose_prompts_and_tools(client):
    d = client.get("/api/config/defaults").json()
    assert "You are a pattern-drafting agent" in d["config"]["system_prompt"]
    names = {t["name"] for t in d["tools"]}
    assert {"add_point", "split_piece", "merge_piece"} <= names
    assert "medium" in d["efforts"] and "auto" in d["state_scopes"]


def test_session_config_is_what_the_next_run_uses(client, monkeypatch):
    """Editing the prompt has to actually reach the model, or the playground
    is lying to you."""
    import agent as agent_mod

    v = client.post("/api/patterns", json={"name": "T", "source": "blank"}).json()
    sid = v["session_id"]

    r = client.post(
        f"/api/sessions/{sid}/config",
        json={"config": {"system_prompt": "TUNED PROMPT", "effort": "low", "max_steps": 3}},
    )
    assert r.status_code == 200
    assert client.get(f"/api/sessions/{sid}").json()["config"]["system_prompt"] == "TUNED PROMPT"

    seen = {}

    def fake_stream(session, instruction, pieces=None, label="", model=None, config=None):
        seen["system_prompt"] = config.system_prompt
        seen["effort"] = config.effort
        seen["max_steps"] = config.max_steps
        yield ("done", agent_mod.RunResult(final_text="ok", actions=[], stopped_reason="done"))

    monkeypatch.setattr(agent_mod, "stream_instruction", fake_stream)
    with client.stream("GET", f"/api/sessions/{sid}/run/stream", params={"text": "go"}) as r:
        list(r.iter_text())

    assert seen == {"system_prompt": "TUNED PROMPT", "effort": "low", "max_steps": 3}


def test_run_records_the_config_that_produced_it(client, monkeypatch):
    """Provenance: editing the config later must not rewrite history."""
    import agent as agent_mod

    v = client.post("/api/patterns", json={"name": "T", "source": "blank"}).json()
    sid = v["session_id"]
    client.post(f"/api/sessions/{sid}/config", json={"config": {"system_prompt": "FIRST"}})

    def fake_stream(session, instruction, pieces=None, label="", model=None, config=None):
        yield ("done", agent_mod.RunResult(final_text="ok", actions=[], stopped_reason="done"))

    monkeypatch.setattr(agent_mod, "stream_instruction", fake_stream)
    with client.stream("GET", f"/api/sessions/{sid}/run/stream", params={"text": "one"}) as r:
        list(r.iter_text())

    # change the config, run again
    client.post(f"/api/sessions/{sid}/config", json={"config": {"system_prompt": "SECOND"}})
    with client.stream("GET", f"/api/sessions/{sid}/run/stream", params={"text": "two"}) as r:
        list(r.iter_text())

    runs = client.get(f"/api/sessions/{sid}").json()["runs"]
    assert [r["config"]["system_prompt"] for r in runs] == ["FIRST", "SECOND"]


def test_preset_crud(client):
    created = client.post(
        "/api/presets", json={"name": "terse", "config": {"effort": "low", "max_tokens": 4000}}
    ).json()
    assert created["name"] == "terse" and created["config"]["effort"] == "low"

    pid = created["id"]
    updated = client.put(
        f"/api/presets/{pid}", json={"name": "terser", "config": {"effort": "low"}}
    ).json()
    assert updated["name"] == "terser"
    assert [p["id"] for p in client.get("/api/presets").json()["presets"]] == [pid]

    client.delete(f"/api/presets/{pid}")
    assert client.get("/api/presets").json()["presets"] == []


def test_preset_needs_a_name(client):
    assert client.post("/api/presets", json={"name": "  ", "config": {}}).status_code == 400


def test_rating_and_sample_export(client, monkeypatch):
    """Rated runs are the corpus — export carries the instruction, the config
    behind it, and the full action log."""
    import agent as agent_mod

    v = client.post("/api/patterns", json={"name": "T", "source": "blank"}).json()
    sid = v["session_id"]

    def fake_stream(session, instruction, pieces=None, label="", model=None, config=None):
        a = agent_mod.Action(
            step_index=0,
            tool_name="add_point",
            tool_input={"attrs": {"name": "Q"}},
            reasoning="needed a reference",
            ok=True,
            message="added point",
        )
        yield ("action", a)
        yield ("done", agent_mod.RunResult(final_text="done", actions=[a], stopped_reason="done"))

    monkeypatch.setattr(agent_mod, "stream_instruction", fake_stream)
    with client.stream("GET", f"/api/sessions/{sid}/run/stream", params={"text": "add one"}) as r:
        list(r.iter_text())

    run_id = client.get(f"/api/sessions/{sid}").json()["runs"][-1]["id"]
    assert client.get("/api/samples/export.jsonl").text == "", "unrated runs aren't samples"

    rated = client.post(f"/api/runs/{run_id}/rating", json={"rating": 1, "note": "clean"}).json()
    assert rated["rating"] == 1 and rated["note"] == "clean"

    import json as _json

    rows = [_json.loads(x) for x in client.get("/api/samples/export.jsonl").text.splitlines()]
    assert len(rows) == 1
    assert rows[0]["instruction"] == "add one"
    assert rows[0]["rating"] == 1
    assert rows[0]["actions"][0]["reasoning"] == "needed a reference"
    assert "system_prompt" in rows[0]["config"]

    # a thumbs-down is still a sample, and filtering works
    client.post(f"/api/runs/{run_id}/rating", json={"rating": -1})
    assert client.get("/api/samples/export.jsonl", params={"rating": 1}).text == ""
    assert (
        len(client.get("/api/samples/export.jsonl", params={"rating": -1}).text.splitlines()) == 1
    )


def test_rating_rejects_out_of_range(client, monkeypatch):
    import agent as agent_mod

    v = client.post("/api/patterns", json={"name": "T", "source": "blank"}).json()
    sid = v["session_id"]

    def fake_stream(session, instruction, pieces=None, label="", model=None, config=None):
        yield ("done", agent_mod.RunResult(final_text="ok", actions=[], stopped_reason="done"))

    monkeypatch.setattr(agent_mod, "stream_instruction", fake_stream)
    with client.stream("GET", f"/api/sessions/{sid}/run/stream", params={"text": "x"}) as r:
        list(r.iter_text())
    run_id = client.get(f"/api/sessions/{sid}").json()["runs"][-1]["id"]
    assert client.post(f"/api/runs/{run_id}/rating", json={"rating": 7}).status_code == 400


def test_compare_two_runs_shows_the_config_delta(client, monkeypatch):
    """The config diff is what attributes a behaviour change to a prompt or
    parameter change."""
    import agent as agent_mod

    v = client.post("/api/patterns", json={"name": "T", "source": "blank"}).json()
    sid = v["session_id"]

    def fake_stream(session, instruction, pieces=None, label="", model=None, config=None):
        yield ("done", agent_mod.RunResult(final_text="ok", actions=[], stopped_reason="done"))

    monkeypatch.setattr(agent_mod, "stream_instruction", fake_stream)

    client.post(f"/api/sessions/{sid}/config", json={"config": {"effort": "low"}})
    with client.stream("GET", f"/api/sessions/{sid}/run/stream", params={"text": "one"}) as r:
        list(r.iter_text())
    client.post(f"/api/sessions/{sid}/config", json={"config": {"effort": "high"}})
    with client.stream("GET", f"/api/sessions/{sid}/run/stream", params={"text": "two"}) as r:
        list(r.iter_text())

    runs = client.get(f"/api/sessions/{sid}").json()["runs"]
    a, b = runs[0]["id"], runs[1]["id"]
    cmp = client.get(f"/api/runs/{a}/compare", params={"against": b}).json()

    assert cmp["a"]["run"]["instruction"] == "one"
    assert cmp["b"]["run"]["instruction"] == "two"
    diff = {d["field"]: (d["a"], d["b"]) for d in cmp["config_diff"]}
    assert diff["effort"] == ("low", "high")
    assert "system_prompt" not in diff, "identical fields must not show as differences"
