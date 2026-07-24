"""FastAPI backend for the VLA pattern-drafting system.

Exposes a persistent editing session over the headless engine and serves the
phone PWA. Sessions are kept in memory (swap for a store later). Each turn the
phone posts an instruction; the agent edits the pattern; the backend returns the
new structured state + SVG render so the phone view and the model stay in sync.

Run:  uvicorn app:app --reload --port 8000    (from the backend/ dir)
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Make the headless engine importable without installing it or relying on
# PYTHONPATH/start-command env handling (Railway/Nixpacks doesn't always apply an
# inline `PYTHONPATH=engine` prefix). The engine package lives at <repo>/engine.
_ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
if _ENGINE_DIR.is_dir() and str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import seamly_engine as se
from seamly_engine.operations import PatternSession
from seamly_engine.pieces import group_pieces, pieces_for_key
from seamly_engine.state import block_state, list_blocks

import agent

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "engine" / "tests" / "fixtures"
PWA_DIR = ROOT / "app"

app = FastAPI(title="VLA Pattern Drafting")

_SESSIONS: dict[str, PatternSession] = {}
_ACTIVE: dict[str, str] = {}  # session id -> active block key (e.g. "A")


class NewSession(BaseModel):
    pattern: str = "aldrich_basic.sm2d"
    measurements: str = "aldrich_measurements.vst"


class SelectBlock(BaseModel):
    block: str  # block key, e.g. "A"


class Message(BaseModel):
    text: str


def _blocks_payload(sid: str, sess: PatternSession, reply: str | None = None) -> dict:
    """Response for the block-picker: which blocks (garments) are available."""
    return {"session_id": sid, "reply": reply, "blocks": list_blocks(sess.pattern)}


def _block_label(sess: PatternSession, key: str) -> str:
    for g in group_pieces(sess.pattern):
        if g["key"] == key:
            return g["label"]
    return key


def _block_payload(sid: str, sess: PatternSession, key: str, pieces,
                   reply: str | None = None) -> dict:
    """Response for the active-block view: scoped state + rich scoped render."""
    label = _block_label(sess, key)
    return {
        "session_id": sid,
        "reply": reply,
        "block": {"key": key, "label": label, "pieces": [p.name for p in pieces]},
        "state": block_state(sess.pattern, sess.evaluated, pieces, sess.measurements,
                             label=label, extra_ids=getattr(sess, "added_ids", None)),
        "svg": agent.render_svg_for(sess, pieces),
    }


@app.post("/api/session")
def create_session(body: NewSession) -> dict:
    pat = FIXTURES / body.pattern
    meas = FIXTURES / body.measurements
    if not pat.exists():
        raise HTTPException(404, f"pattern {body.pattern!r} not found")
    session = PatternSession(se.load_pattern(str(pat)),
                             se.load_measurements(str(meas)) if meas.exists() else None)
    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = session
    return _blocks_payload(sid, session, reply="Which block would you like to work on?")


@app.post("/api/session/{sid}/select_block")
def select_block(sid: str, body: SelectBlock) -> dict:
    sess = _get(sid)
    pieces = pieces_for_key(sess.pattern, body.block)
    if not pieces:
        raise HTTPException(404, f"no block {body.block!r}")
    _ACTIVE[sid] = body.block
    label = _block_label(sess, body.block)
    return _block_payload(sid, sess, body.block, pieces,
                          reply=f"Working on the {label} block. What would you like to change?")


@app.get("/api/session/{sid}/render.svg")
def get_render(sid: str) -> Response:
    sess = _get(sid)
    _, pieces = _active_block(sid, sess)
    return Response(content=agent.render_svg_for(sess, pieces), media_type="image/svg+xml")


@app.post("/api/session/{sid}/message")
def post_message(sid: str, body: Message) -> dict:
    sess = _get(sid)
    key, pieces = _active_block(sid, sess)
    if not pieces:
        raise HTTPException(400, "select a block first")
    result = agent.run_turn(sess, body.text, pieces, label=_block_label(sess, key))
    return _block_payload(sid, sess, key, pieces, reply=result["reply"]) | {
        "tool_calls": result["tool_calls"]}


def _get(sid: str) -> PatternSession:
    sess = _SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    return sess


def _active_block(sid: str, sess: PatternSession):
    key = _ACTIVE.get(sid)
    return (key, pieces_for_key(sess.pattern, key)) if key else (None, [])


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = PWA_DIR / "index.html"
    return idx.read_text() if idx.exists() else "<h1>PWA not built</h1>"


if PWA_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(PWA_DIR), html=True), name="app")
