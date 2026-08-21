"""FastAPI backend for the pattern-drafting workspace.

Flow: create or open a **pattern** → work in a **session** → type an
**instruction**, which runs to completion and produces an **action log**
(``run_instruction`` in ``agent.py``). A successful run auto-saves a new
**version** — versions form a branching DAG, so "branch from this run" just
means opening a new session on the version it produced. There is no chat
history to persist: the action log *is* the record of what happened, and
it's derived on demand (see ``replay_run`` below) rather than duplicated in
the database once per tool call.

Run:  PYTHONPATH=src uvicorn app:app --app-dir backend --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the headless engine importable without installing it or relying on
# PYTHONPATH/start-command env handling (Railway/Nixpacks doesn't always apply
# an inline `PYTHONPATH=src` prefix). The engine package lives at <repo>/src.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import agent
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel
from store import Store

import seamly_engine as se
from seamly_engine.authoring import new_pattern
from seamly_engine.operations import PatternSession
from seamly_engine.parser import parse_pattern
from seamly_engine.pieces import group_pieces, pieces_for_key
from seamly_engine.state import list_blocks
from seamly_engine.writer import pattern_to_xml

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
APP_DIR = ROOT / "app"
DEFAULT_MEAS = "aldrich_measurements.vst"

app = FastAPI(title="Pattern Drafting Workspace")
store = Store()

# Live working sessions (the evaluated pattern in memory), keyed by session id.
_LIVE: dict[str, PatternSession] = {}
_ACTIVE_BLOCK: dict[str, str] = {}


# --- request bodies ----------------------------------------------------------
class NewPattern(BaseModel):
    name: str = "Untitled pattern"
    source: str = "blank"  # blank | sample
    sample: str = "aldrich_basic.sm2d"
    measurements: str = DEFAULT_MEAS


class OpenVersion(BaseModel):
    version_id: str


class SelectBlock(BaseModel):
    block: str  # "" clears the filter -> whole pattern


class Instruction(BaseModel):
    text: str


class SaveVersion(BaseModel):
    label: str = ""


class Rename(BaseModel):
    name: str


class SetModel(BaseModel):
    model: str


# --- helpers -----------------------------------------------------------------
def _measurements(name: str):
    path = FIXTURES / name if name else None
    return se.load_measurements(str(path)) if path and path.exists() else None


def _live(sid: str) -> PatternSession:
    sess = _LIVE.get(sid)
    if sess is None:
        row = store.get_session(sid)
        if row is None:
            raise HTTPException(404, "session not found")
        ver = store.get_version(row["version_id"]) if row["version_id"] else None
        if ver is None:
            raise HTTPException(410, "session has no saved version to restore")
        sess = PatternSession(
            parse_pattern(ver["xml"], is_text=True), _measurements(ver["measurements"])
        )
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


def _view(sid: str) -> dict:
    """The whole client-facing view of a session: pattern render, state,
    blocks, run/action history, version history."""
    sess = _live(sid)
    row = store.get_session(sid) or {}
    key, pieces = _active_block(sid, sess)
    label = _block_label(sess, key) if key else ""
    pattern_id = row.get("pattern_id")
    runs = store.list_runs(sid)
    for r in runs:
        r["actions"] = store.list_actions(r["id"])
    return {
        "session_id": sid,
        "pattern": {
            "id": pattern_id,
            "name": (store.get_pattern(pattern_id) or {}).get("name", "")
            if pattern_id
            else sess.pattern.pattern_name,
        },
        "version_id": row.get("version_id"),
        "model": row.get("model") or agent.MODEL,
        "models": agent.MODELS,
        "block": {"key": key, "label": label, "pieces": [p.name for p in pieces]} if key else None,
        "blocks": list_blocks(sess.pattern),
        "state_summary": {
            "objects": len(sess.pattern.all_objects()),
            "pieces": len(sess.pattern.pieces),
            "points": len(sess.evaluated.points),
            "curves": len(sess.evaluated.curves),
            "unresolved": len(sess.evaluated.unresolved),
        },
        "svg": agent.render_svg_for(sess, pieces or None),
        "runs": runs,
        "versions": store.list_versions(pattern_id) if pattern_id else [],
    }


def _run_and_persist(sid: str, instruction: str) -> dict:
    """Run an instruction to completion, persist the run + its actions, and
    auto-save a new version if it made any successful change."""
    sess = _live(sid)
    row = store.get_session(sid) or {}
    key, pieces = _active_block(sid, sess)
    model = row.get("model") or None

    run_id = store.create_run(sid, instruction, model or agent.MODEL, row.get("version_id"))
    result = agent.run_instruction(
        sess, instruction, pieces or None, label=_block_label(sess, key) if key else "", model=model
    )
    for a in result.actions:
        store.add_action(
            run_id,
            a.step_index,
            a.tool_name,
            a.tool_input,
            reasoning=a.reasoning,
            ok=a.ok,
            message=a.message,
            blocked_by=a.blocked_by,
            touched_ids=a.touched_ids,
        )

    version_after = None
    if any(a.ok for a in result.actions) and row.get("pattern_id"):
        meas = (
            (store.get_version(row["version_id"]) or {}).get("measurements", DEFAULT_MEAS)
            if row.get("version_id")
            else DEFAULT_MEAS
        )
        version_after = store.save_version(
            row["pattern_id"],
            pattern_to_xml(sess.pattern),
            parent_id=row.get("version_id"),
            label=instruction[:80],
            measurements=meas,
        )
        store.touch_session(sid, version_id=version_after)
    else:
        store.touch_session(sid)

    store.finish_run(
        run_id,
        final_text=result.final_text,
        stopped_reason=result.stopped_reason,
        version_after_id=version_after,
    )
    return {"run_id": run_id, **_view(sid)}


# --- patterns ------------------------------------------------------------------
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
    vid = store.save_version(
        pid, pattern_to_xml(pattern), label="initial", measurements=body.measurements
    )
    sid = store.create_session(pid, vid, title=name)
    _LIVE[sid] = PatternSession(pattern, _measurements(body.measurements))
    return _view(sid)


@app.post("/api/patterns/import")
async def import_pattern(
    file: UploadFile = File(...), name: str = "", measurements: str = DEFAULT_MEAS
) -> dict:
    """Import a .sm2d/.val file as a new pattern (its first version)."""
    raw = (await file.read()).decode("utf-8", errors="replace")
    try:
        pattern = parse_pattern(raw, is_text=True)
    except Exception as e:
        raise HTTPException(400, f"could not parse pattern: {e}") from e
    pid = store.create_pattern(name or pattern.pattern_name or (file.filename or "Imported"))
    vid = store.save_version(
        pid, pattern_to_xml(pattern), label=f"imported {file.filename}", measurements=measurements
    )
    sid = store.create_session(pid, vid, title="import")
    _LIVE[sid] = PatternSession(pattern, _measurements(measurements))
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
    return {
        "versions": store.list_versions(pattern_id),
        "sessions": store.list_sessions(pattern_id),
    }


# --- sessions ------------------------------------------------------------------
@app.post("/api/patterns/{pattern_id}/open")
def open_version(pattern_id: str, body: OpenVersion) -> dict:
    """Open a version in a new session. Running an instruction here **branches**
    the version history from it."""
    ver = store.get_version(body.version_id)
    if ver is None or ver["pattern_id"] != pattern_id:
        raise HTTPException(404, "version not found")
    sid = store.create_session(pattern_id, ver["id"], title=ver["label"] or "session")
    _LIVE[sid] = PatternSession(
        parse_pattern(ver["xml"], is_text=True), _measurements(ver["measurements"])
    )
    return _view(sid)


@app.post("/api/sessions/{sid}/runs/{run_id}/branch")
def branch_from_run(sid: str, run_id: str) -> dict:
    """Open a fresh session on the version a completed run produced — the
    "branch from here" action in the run log."""
    row = store.get_session(sid)
    run = store.get_run(run_id)
    if row is None or run is None or run["session_id"] != sid:
        raise HTTPException(404, "run not found")
    if not run["version_after_id"]:
        raise HTTPException(400, "this run made no change to branch from")
    return open_version(row["pattern_id"], OpenVersion(version_id=run["version_after_id"]))


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict:
    return _view(sid)


@app.post("/api/sessions/{sid}/select_block")
def select_block(sid: str, body: SelectBlock) -> dict:
    """Narrow the agent's view (and the render) to one garment block, or clear
    the filter (empty string) to work across the whole pattern — the default,
    since split/merge often need to see more than one piece at once."""
    sess = _live(sid)
    if body.block and not pieces_for_key(sess.pattern, body.block):
        raise HTTPException(404, f"no block {body.block!r}")
    if body.block:
        _ACTIVE_BLOCK[sid] = body.block
    else:
        _ACTIVE_BLOCK.pop(sid, None)
    return _view(sid)


@app.post("/api/sessions/{sid}/model")
def set_model(sid: str, body: SetModel) -> dict:
    """Switch the model for this session — cheap for tweaks, strong for drafting."""
    if body.model not in agent.MODEL_IDS:
        raise HTTPException(400, f"unknown model {body.model!r}")
    store.set_session_model(sid, body.model)
    return _view(sid)


@app.post("/api/sessions/{sid}/run")
def run_instruction(sid: str, body: Instruction) -> dict:
    """Run one typed instruction to completion. Returns the full session view,
    including the new run and its action log; a successful run auto-saves a
    new version."""
    return _run_and_persist(sid, body.text)


@app.post("/api/sessions/{sid}/save")
def save_version(sid: str, body: SaveVersion) -> dict:
    """Snapshot the current pattern as a new version without running an
    instruction (e.g. after a manual tweak elsewhere)."""
    sess = _live(sid)
    row = store.get_session(sid)
    if not row or not row["pattern_id"]:
        raise HTTPException(400, "session has no pattern to save to")
    meas = (store.get_version(row["version_id"]) or {}).get("measurements", DEFAULT_MEAS)
    vid = store.save_version(
        row["pattern_id"],
        pattern_to_xml(sess.pattern),
        parent_id=row["version_id"],
        label=body.label or "saved",
        measurements=meas,
    )
    store.touch_session(sid, version_id=vid)
    return _view(sid)


@app.get("/api/sessions/{sid}/export.sm2d")
def export_session(sid: str) -> Response:
    sess = _live(sid)
    name = (sess.pattern.pattern_name or "pattern").replace(" ", "_")
    return Response(
        content=pattern_to_xml(sess.pattern),
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{name}.sm2d"'},
    )


@app.get("/api/versions/{version_id}/export.sm2d")
def export_version(version_id: str) -> Response:
    ver = store.get_version(version_id)
    if ver is None:
        raise HTTPException(404, "version not found")
    return Response(
        content=ver["xml"],
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{version_id}.sm2d"'},
    )


@app.get("/api/sessions/{sid}/render.svg")
def get_render(sid: str) -> Response:
    sess = _live(sid)
    _, pieces = _active_block(sid, sess)
    return Response(content=agent.render_svg_for(sess, pieces or None), media_type="image/svg+xml")


# --- action detail (click an action -> reasoning + before/after) --------------
def replay_run(run_id: str, upto_step: int | None = None) -> tuple[PatternSession, list[dict]]:
    """Rebuild a run's pattern session from its starting version by replaying
    its stored actions in order, up to (and including, if given) ``upto_step``.

    No per-action state is stored in the database — this is how a before/after
    view is derived on demand instead. Replay is safe to repeat: a failed
    action rolled back when it first ran and rolls back identically here, and
    tool inputs reference ids that come out deterministically identical to the
    original live run, because they're assigned the same way (highest existing
    id + 1) from the same starting pattern through the same sequence of prior
    actions.
    """
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    ver = store.get_version(run["version_before_id"]) if run["version_before_id"] else None
    if ver is None:
        raise HTTPException(410, "this run's starting version is no longer available")
    sess = PatternSession(
        parse_pattern(ver["xml"], is_text=True), _measurements(ver["measurements"])
    )
    actions = store.list_actions(run_id)
    applied = []
    for a in actions:
        if upto_step is not None and a["step_index"] > upto_step:
            break
        agent.dispatch_tool(sess, a["tool_name"], a["tool_input"])
        applied.append(a)
    return sess, applied


@app.get("/api/runs/{run_id}/actions/{step_index}/detail")
def action_detail(run_id: str, step_index: int) -> dict:
    """Reasoning plus a before/after render for one action in a run's log —
    what clicking an action in the UI shows."""
    actions = store.list_actions(run_id)
    action = next((a for a in actions if a["step_index"] == step_index), None)
    if action is None:
        raise HTTPException(404, "no such action in this run")

    before_sess, _ = replay_run(run_id, upto_step=step_index - 1)
    before_svg = agent.render_svg_for(before_sess)
    after_sess, _ = replay_run(run_id, upto_step=step_index)
    touched = set(action["touched_ids"])
    after_svg = se_render_highlighted(after_sess, touched)

    return {
        "action": action,
        "before": {"svg": before_svg, "state_summary": _summary(before_sess)},
        "after": {"svg": after_svg, "state_summary": _summary(after_sess)},
    }


def se_render_highlighted(sess: PatternSession, touched: set[int]) -> str:
    from seamly_engine.render import render_svg

    return render_svg(sess.pattern, sess.evaluated, width=1000, highlight_ids=touched)


def _summary(sess: PatternSession) -> dict:
    return {
        "objects": len(sess.pattern.all_objects()),
        "pieces": len(sess.pattern.pieces),
        "unresolved": len(sess.evaluated.unresolved),
    }


@app.get("/api/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = APP_DIR / "index.html"
    return idx.read_text() if idx.exists() else "<h1>App not built</h1>"
