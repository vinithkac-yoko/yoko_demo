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
from seamly_engine.render import render_svg

import agent

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "engine" / "tests" / "fixtures"
PWA_DIR = ROOT / "app"

app = FastAPI(title="VLA Pattern Drafting")

_SESSIONS: dict[str, PatternSession] = {}


class NewSession(BaseModel):
    pattern: str = "aldrich_basic.sm2d"
    measurements: str = "aldrich_measurements.vst"


class Message(BaseModel):
    text: str


def _payload(session_id: str, sess: PatternSession, reply: str | None = None) -> dict:
    return {
        "session_id": session_id,
        "reply": reply,
        "state": sess.state(),
        "svg": render_svg(sess.pattern, sess.evaluated),
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
    return _payload(sid, session, reply="Base pattern loaded. What would you like to change?")


@app.get("/api/session/{sid}/state")
def get_state(sid: str) -> dict:
    return _get(sid).state()


@app.get("/api/session/{sid}/render.svg")
def get_render(sid: str) -> Response:
    svg = render_svg(_get(sid).pattern, _get(sid).evaluated)
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/api/session/{sid}/message")
def post_message(sid: str, body: Message) -> dict:
    sess = _get(sid)
    result = agent.run_turn(sess, body.text)
    return _payload(sid, sess, reply=result["reply"]) | {"tool_calls": result["tool_calls"]}


def _get(sid: str) -> PatternSession:
    sess = _SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    return sess


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = PWA_DIR / "index.html"
    return idx.read_text() if idx.exists() else "<h1>PWA not built</h1>"


if PWA_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(PWA_DIR), html=True), name="app")
