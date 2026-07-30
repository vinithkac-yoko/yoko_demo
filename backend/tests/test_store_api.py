"""Store + API tests for the pattern-library flow.

Covers what the pivot added: blank-canvas creation, version history as a
**branching DAG**, chat history with **threaded branching**, save/export, and
import-as-a-new-version.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "engine"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")          # keep the agent offline
    import store as store_mod
    importlib.reload(store_mod)
    import app as app_mod
    importlib.reload(app_mod)
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


# --- store ------------------------------------------------------------------
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
    assert parents["A"] == root and parents["B"] == root   # two children => a branch


def test_message_branching(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_DB", str(tmp_path / "m.db"))
    import store as store_mod
    importlib.reload(store_mod)
    s = store_mod.Store()
    sid = s.create_session(None, None)
    m1 = s.add_message(sid, "user", "one")
    m2 = s.add_message(sid, "assistant", "two", parent_id=m1)
    m3 = s.add_message(sid, "user", "three", parent_id=m2)
    alt = s.add_message(sid, "user", "alt", parent_id=m1)     # branch off m1
    assert [m["text"] for m in s.message_chain(m3)] == ["one", "two", "three"]
    assert [m["text"] for m in s.message_chain(alt)] == ["one", "alt"]


# --- api --------------------------------------------------------------------
def test_blank_canvas_flow(client):
    v = client.post("/api/patterns", json={"name": "Draft", "source": "blank"}).json()
    assert v["state_summary"] == {"objects": 1, "points": 1, "curves": 0, "unresolved": 0}
    assert v["messages"] and v["blocks"] == []          # nothing drafted yet
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
    imported = client.post("/api/patterns/import",
                           files={"file": ("mine.sm2d", xml, "application/xml")}).json()
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
    assert parents["A"] == parents["B"] == root      # sibling branches


def test_block_selection_and_whole_pattern(client):
    v = client.post("/api/patterns", json={"name": "Aldrich", "source": "sample"}).json()
    sid = v["session_id"]
    assert [b["label"] for b in v["blocks"]] == ["Skirt", "Trousers", "Bodice", "1 Piece Sleeve"]
    picked = client.post(f"/api/sessions/{sid}/select_block", json={"block": "C"}).json()
    assert picked["block"]["label"] == "Bodice" and picked["svg"]
    whole = client.post(f"/api/sessions/{sid}/select_block", json={"block": ""}).json()
    assert whole["block"] is None


def test_message_persists_history(client):
    v = client.post("/api/patterns", json={"name": "Draft", "source": "blank"}).json()
    sid = v["session_id"]
    out = client.post(f"/api/sessions/{sid}/message", json={"text": "hello"}).json()
    roles = [m["role"] for m in out["messages"]]
    assert roles.count("user") == 1 and roles.count("assistant") >= 2
    # history survives a fresh GET of the session
    again = client.get(f"/api/sessions/{sid}").json()
    assert len(again["messages"]) == len(out["messages"])
