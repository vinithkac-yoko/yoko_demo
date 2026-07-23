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
from seamly_engine.pieces import piece_by_id
from seamly_engine.state import list_pieces, piece_state

import agent

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "engine" / "tests" / "fixtures"
PWA_DIR = ROOT / "app"

app = FastAPI(title="VLA Pattern Drafting")

_SESSIONS: dict[str, PatternSession] = {}
_ACTIVE: dict[str, int] = {}  # session id -> active piece (block) id


class NewSession(BaseModel):
    pattern: str = "aldrich_basic.sm2d"
    measurements: str = "aldrich_measurements.vst"


class SelectPiece(BaseModel):
    piece_id: int


class Message(BaseModel):
    text: str


def _pieces_payload(sid: str, sess: PatternSession, reply: str | None = None) -> dict:
    """Response for the block-picker: which blocks are available."""
    return {"session_id": sid, "reply": reply, "pieces": list_pieces(sess.pattern)}


def _block_payload(sid: str, sess: PatternSession, piece, reply: str | None = None) -> dict:
    """Response for the active-block view: scoped state + single-piece render."""
    return {
        "session_id": sid,
        "reply": reply,
        "piece": {"id": piece.id, "name": piece.name},
        "state": piece_state(sess.pattern, sess.evaluated, piece, sess.measurements),
        "svg": agent.render_svg_for(sess, piece),
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
    return _pieces_payload(sid, session,
                           reply="Which block would you like to work on?")


@app.post("/api/session/{sid}/select_piece")
def select_piece(sid: str, body: SelectPiece) -> dict:
    sess = _get(sid)
    piece = piece_by_id(sess.pattern, body.piece_id)
    if piece is None:
        raise HTTPException(404, f"no block with id {body.piece_id}")
    _ACTIVE[sid] = piece.id
    return _block_payload(sid, sess, piece,
                          reply=f"Working on “{piece.name}”. What would you like to change?")


@app.get("/api/session/{sid}/render.svg")
def get_render(sid: str) -> Response:
    sess = _get(sid)
    piece = _active_piece(sid, sess)
    return Response(content=agent.render_svg_for(sess, piece), media_type="image/svg+xml")


@app.post("/api/session/{sid}/message")
def post_message(sid: str, body: Message) -> dict:
    sess = _get(sid)
    piece = _active_piece(sid, sess)
    if piece is None:
        raise HTTPException(400, "select a block first")
    result = agent.run_turn(sess, body.text, piece)
    return _block_payload(sid, sess, piece, reply=result["reply"]) | {
        "tool_calls": result["tool_calls"]}


def _get(sid: str) -> PatternSession:
    sess = _SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    return sess


def _active_piece(sid: str, sess: PatternSession):
    pid = _ACTIVE.get(sid)
    return piece_by_id(sess.pattern, pid) if pid is not None else None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = PWA_DIR / "index.html"
    return idx.read_text() if idx.exists() else "<h1>PWA not built</h1>"


if PWA_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(PWA_DIR), html=True), name="app")
