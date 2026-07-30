"""FastAPI backend for the VLA pattern-drafting system.

Flow: create or open a **pattern** (blank canvas, imported file, or a bundled
sample) → work in a **session** → save **versions** as you go. Versions form a
branching DAG (saving from an older version creates a branch), chat is persisted
and threaded so a conversation can branch from any earlier message, and a pattern
can be exported as a ``.sm2d`` file and re-imported as a new version.

Run:  PYTHONPATH=engine uvicorn app:app --app-dir backend --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the headless engine importable without installing it or relying on
# PYTHONPATH/start-command env handling (Railway/Nixpacks doesn't always apply an
# inline `PYTHONPATH=engine` prefix). The engine package lives at <repo>/engine.
_ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
if _ENGINE_DIR.is_dir() and str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import seamly_engine as se
from seamly_engine.authoring import new_pattern
from seamly_engine.operations import PatternSession
from seamly_engine.parser import parse_pattern
from seamly_engine.pieces import group_pieces, pieces_for_key
from seamly_engine.state import block_state, list_blocks
from seamly_engine.writer import pattern_to_xml

import agent
from store import Store

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "engine" / "tests" / "fixtures"
PWA_DIR = ROOT / "app"
DEFAULT_MEAS = "aldrich_measurements.vst"

app = FastAPI(title="VLA Pattern Drafting")
store = Store()

# Live working sessions (the evaluated pattern in memory), keyed by session id.
_LIVE: dict[str, PatternSession] = {}
_ACTIVE_BLOCK: dict[str, str] = {}


# --- models ------------------------------------------------------------------
class NewPattern(BaseModel):
    name: str = "Untitled pattern"
    source: str = "blank"          # blank | sample
    sample: str = "aldrich_basic.sm2d"
    measurements: str = DEFAULT_MEAS


class OpenVersion(BaseModel):
    version_id: str


class SelectBlock(BaseModel):
    block: str


class Message(BaseModel):
    text: str
    parent_id: str | None = None   # set to branch the conversation from a message


class SaveVersion(BaseModel):
    label: str = ""


class Rename(BaseModel):
    name: str


# --- helpers -----------------------------------------------------------------
def _measurements(name: str):
    path = FIXTURES / name if name else None
    return se.load_measurements(str(path)) if path and path.exists() else None


def _live(sid: str) -> PatternSession:
    sess = _LIVE.get(sid)
    if sess is None:
        # Rehydrate from the stored version so sessions survive a restart.
        row = store.get_session(sid)
        if row is None:
            raise HTTPException(404, "session not found")
        ver = store.get_version(row["version_id"]) if row["version_id"] else None
        if ver is None:
            raise HTTPException(410, "session has no saved version to restore")
        sess = PatternSession(parse_pattern(ver["xml"], is_text=True),
                              _measurements(ver["measurements"]))
        _LIVE[sid] = sess
    return sess


def _active_block(sid: str, sess: PatternSession):
    key = _ACTIVE_BLOCK.get(sid)
    return (key, pieces_for_key(sess.pattern, key)) if key else (None, [])


def _block_label(sess: PatternSession, key: str) -> str:
    for g in group_pieces(sess.pattern):
        if g["key"] == key:
            return g["label"]
    return key


def _view(sid: str, reply: str | None = None, tool_calls=None) -> dict:
    """The whole client-facing view of a session: pattern render, state, blocks,
    chat history, version history."""
    sess = _live(sid)
    row = store.get_session(sid) or {}
    key, pieces = _active_block(sid, sess)
    label = _block_label(sess, key) if key else ""
    state = (block_state(sess.pattern, sess.evaluated, pieces, sess.measurements, label=label,
                         extra_ids=sess.added_ids) if pieces
             else sess.state())
    pattern_id = row.get("pattern_id")
    return {
        "session_id": sid,
        "pattern": {"id": pattern_id,
                    "name": (store.get_pattern(pattern_id) or {}).get("name", "")
                    if pattern_id else sess.pattern.pattern_name},
        "version_id": row.get("version_id"),
        "reply": reply,
        "tool_calls": tool_calls or [],
        "block": {"key": key, "label": label,
                  "pieces": [p.name for p in pieces]} if key else None,
        "blocks": list_blocks(sess.pattern),
        "object_count": len(sess.pattern.all_objects()),
        "state_summary": {"objects": len(sess.pattern.all_objects()),
                          "points": len(sess.evaluated.points),
                          "curves": len(sess.evaluated.curves),
                          "unresolved": len(sess.evaluated.unresolved)},
        "svg": agent.render_svg_for(sess, pieces or None),
        "messages": store.list_messages(sid),
        "versions": store.list_versions(pattern_id) if pattern_id else [],
    }


# --- patterns ----------------------------------------------------------------
@app.get("/api/patterns")
def list_patterns() -> dict:
    return {"patterns": store.list_patterns(), "sessions": store.list_sessions()}


@app.post("/api/patterns")
def create_pattern(body: NewPattern) -> dict:
    """Start a new pattern — a blank canvas by default, or from a bundled sample."""
    if body.source == "sample":
        path = FIXTURES / body.sample
        if not path.exists():
            raise HTTPException(404, f"sample {body.sample!r} not found")
        pattern = se.load_pattern(str(path))
        name = body.name if body.name != "Untitled pattern" else (pattern.pattern_name or body.name)
    else:
        pattern = new_pattern(body.name, measurements_file=body.measurements)
        name = body.name

    pid = store.create_pattern(name)
    vid = store.save_version(pid, pattern_to_xml(pattern), label="initial",
                             measurements=body.measurements)
    sid = store.create_session(pid, vid, title=name)
    _LIVE[sid] = PatternSession(pattern, _measurements(body.measurements))
    store.add_message(sid, "assistant",
                      "Blank canvas ready — point A is the origin. Tell me what to draft."
                      if body.source == "blank" else
                      "Pattern loaded. What would you like to change?")
    return _view(sid)


@app.post("/api/patterns/import")
async def import_pattern(file: UploadFile = File(...), name: str = "",
                         measurements: str = DEFAULT_MEAS) -> dict:
    """Import a .sm2d/.val file as a new pattern (its first version)."""
    raw = (await file.read()).decode("utf-8", errors="replace")
    try:
        pattern = parse_pattern(raw, is_text=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not parse pattern: {e}")
    pid = store.create_pattern(name or pattern.pattern_name or (file.filename or "Imported"))
    vid = store.save_version(pid, pattern_to_xml(pattern), label=f"imported {file.filename}",
                             measurements=measurements)
    sid = store.create_session(pid, vid, title="import")
    _LIVE[sid] = PatternSession(pattern, _measurements(measurements))
    store.add_message(sid, "assistant", f"Imported {file.filename}. What would you like to change?")
    return _view(sid)


@app.patch("/api/patterns/{pattern_id}")
def rename_pattern(pattern_id: str, body: Rename) -> dict:
    store.rename_pattern(pattern_id, body.name)
    return {"ok": True}


@app.delete("/api/patterns/{pattern_id}")
def delete_pattern(pattern_id: str) -> dict:
    store.delete_pattern(pattern_id)
    return {"ok": True}


@app.get("/api/patterns/{pattern_id}/versions")
def list_versions(pattern_id: str) -> dict:
    return {"versions": store.list_versions(pattern_id),
            "sessions": store.list_sessions(pattern_id)}


# --- sessions ----------------------------------------------------------------
@app.post("/api/patterns/{pattern_id}/open")
def open_version(pattern_id: str, body: OpenVersion) -> dict:
    """Open a version in a new session. Saving from here **branches** the history."""
    ver = store.get_version(body.version_id)
    if ver is None or ver["pattern_id"] != pattern_id:
        raise HTTPException(404, "version not found")
    sid = store.create_session(pattern_id, ver["id"], title=ver["label"] or "session")
    _LIVE[sid] = PatternSession(parse_pattern(ver["xml"], is_text=True),
                                _measurements(ver["measurements"]))
    store.add_message(sid, "assistant",
                      f"Opened version “{ver['label'] or ver['id']}”. Edits here branch from it.")
    return _view(sid)


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict:
    return _view(sid)


@app.post("/api/sessions/{sid}/select_block")
def select_block(sid: str, body: SelectBlock) -> dict:
    sess = _live(sid)
    if body.block and not pieces_for_key(sess.pattern, body.block):
        raise HTTPException(404, f"no block {body.block!r}")
    if body.block:
        _ACTIVE_BLOCK[sid] = body.block
    else:
        _ACTIVE_BLOCK.pop(sid, None)   # empty string = whole pattern
    return _view(sid)


@app.post("/api/sessions/{sid}/message")
def post_message(sid: str, body: Message) -> dict:
    sess = _live(sid)
    key, pieces = _active_block(sid, sess)
    parent = body.parent_id or store.last_message_id(sid)
    user_msg = store.add_message(sid, "user", body.text, parent_id=parent)

    result = agent.run_turn(sess, body.text, pieces or None,
                            label=_block_label(sess, key) if key else "")
    store.add_message(sid, "assistant", result["reply"], parent_id=user_msg,
                      tool_calls=result["tool_calls"])
    store.touch_session(sid)
    return _view(sid, reply=result["reply"], tool_calls=result["tool_calls"])


@app.post("/api/sessions/{sid}/save")
def save_version(sid: str, body: SaveVersion) -> dict:
    """Snapshot the current pattern as a new version (child of the session's)."""
    sess = _live(sid)
    row = store.get_session(sid)
    if not row or not row["pattern_id"]:
        raise HTTPException(400, "session has no pattern to save to")
    meas = (store.get_version(row["version_id"]) or {}).get("measurements", DEFAULT_MEAS)
    vid = store.save_version(row["pattern_id"], pattern_to_xml(sess.pattern),
                             parent_id=row["version_id"],
                             label=body.label or "saved", measurements=meas)
    store.touch_session(sid, version_id=vid)
    store.add_message(sid, "note", f"Saved version: {body.label or 'saved'}")
    return _view(sid)


@app.get("/api/sessions/{sid}/export.sm2d")
def export_session(sid: str) -> Response:
    """Download the current pattern as a Seamly2D file."""
    sess = _live(sid)
    name = (sess.pattern.pattern_name or "pattern").replace(" ", "_")
    return Response(
        content=pattern_to_xml(sess.pattern), media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{name}.sm2d"'})


@app.get("/api/versions/{version_id}/export.sm2d")
def export_version(version_id: str) -> Response:
    ver = store.get_version(version_id)
    if ver is None:
        raise HTTPException(404, "version not found")
    return Response(content=ver["xml"], media_type="application/xml",
                    headers={"Content-Disposition": f'attachment; filename="{version_id}.sm2d"'})


@app.get("/api/sessions/{sid}/branch/{message_id}")
def branch_preview(sid: str, message_id: str) -> dict:
    """The conversation thread leading to a message — the branch you'd continue."""
    return {"chain": store.message_chain(message_id)}


@app.get("/api/sessions/{sid}/render.svg")
def get_render(sid: str) -> Response:
    sess = _live(sid)
    _, pieces = _active_block(sid, sess)
    return Response(content=agent.render_svg_for(sess, pieces or None),
                    media_type="image/svg+xml")


@app.get("/api/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = PWA_DIR / "index.html"
    return idx.read_text() if idx.exists() else "<h1>PWA not built</h1>"


if PWA_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(PWA_DIR), html=True), name="app")
