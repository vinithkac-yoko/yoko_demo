"""VLA agent loop — Anthropic Messages API with per-operation tools.

Each turn the model receives: the user's instruction, the **structured pattern
state** (the semantically-tagged representation — the primary input), and, when a
raster renderer is available, the **rendered image** of the pattern. It then
calls one tool per Seamly2D operation to edit the pattern via
:class:`~seamly_engine.operations.PatternSession`, which re-evaluates and rolls
back on breakage. After each tool call the fresh state is fed back in the
tool_result so the model always acts on current geometry.

This is the Anthropic SDK tool-use loop (not the separate "Claude Agent SDK"
product, which is a filesystem/coding agent). The model call is gated behind
``ANTHROPIC_API_KEY`` so the engine/render/edit paths stay runnable offline.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from seamly_engine.operations import OpResult, PatternSession
from seamly_engine.pieces import scoped_ids
from seamly_engine.render import render_svg
from seamly_engine.state import block_state, compact_state

MODEL = os.getenv("VLA_MODEL", "claude-opus-4-8")
MAX_ITERATIONS = 12

SYSTEM_PROMPT = """\
You are a VLA (vision-language-action) agent that edits sewing patterns in a \
headless Seamly2D-compatible engine. You perform **instructed edits** on ONE \
block (pattern piece) at a time — the user has already chosen which block.

You reason over TWO representations of the current state, not a screenshot alone:
1. A structured JSON state where every object (point/line/arc/spline) is \
semantically tagged: `role` (seamline/dart/grainline/drill_hole/construction/…), \
`construction` vs `final_outline`, `built_from` (dependencies), `dependents` \
(what breaks if it changes), and `formula` (the raw Seamly expression plus its \
resolved numeric value). Point objects include resolved `xy` coordinates (cm).
2. When present, a rendered image: bold strokes are final piece outlines, dimmed \
strokes are construction geometry, darts are magenta, drill holes teal.

To make an edit, call the operation tools. Guidance:
- Formulas are Seamly expressions. Measurements (e.g. `waist_circ`) and variables \
(e.g. `#CM`, the cm-scale factor) can be referenced directly. To "let out the \
waist 2cm", find the object whose length formula controls that dimension and add \
to it (e.g. `(waist_circ/4)+4*#CM` becomes `(waist_circ/4)+4*#CM+2`).
- Prefer editing the construction object that drives a dimension over a leaf.
- DELETE is block-and-report: if an object has dependents the delete is refused \
and returns the dependent chain. Delete those first or pick another edit.
- Every edit auto-re-evaluates and rolls back if it makes the pattern invalid.

Work step by step. After making the change(s), briefly explain what you did and \
why in plain language. Keep it concise."""

# --- tool schemas: one per operation (grows toward full parity) --------------
TOOLS: list[dict[str, Any]] = [
    {
        "name": "edit_formula",
        "description": (
            "Change a formula attribute of an object to realize an instructed "
            "edit. attr is 'length', 'angle', or 'radius'. new_formula is a "
            "Seamly expression (may reference measurements like waist_circ and "
            "variables like #CM). Auto re-evaluates; rolls back if it breaks the "
            "pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "object_id": {"type": "integer", "description": "id of the object to edit"},
                "attr": {"type": "string", "enum": ["length", "angle", "radius"]},
                "new_formula": {"type": "string"},
            },
            "required": ["object_id", "attr", "new_formula"],
        },
    },
    {
        "name": "delete_object",
        "description": (
            "Delete a construction object by id. Refused (block-and-report) if "
            "other objects depend on it; the response lists the dependent ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"object_id": {"type": "integer"}},
            "required": ["object_id"],
        },
    },
    # Roadmap: add_point_endline, add_point_alongline, add_dart, add_line, ...
]


def dispatch_tool(session: PatternSession, name: str, args: dict) -> OpResult:
    if name == "delete_object":
        return session.delete_object(int(args["object_id"]))
    if name == "edit_formula":
        return session.edit_formula(int(args["object_id"]), args["attr"], args["new_formula"])
    return OpResult(False, f"unknown tool {name!r}")


def render_svg_for(session: PatternSession, pieces=None) -> str:
    """Rich construction SVG scoped to the active block (its pieces + drivers), or
    the whole pattern if no block is selected."""
    plist = list(pieces) if pieces else None
    ids = scoped_ids(session.pattern, plist) if plist else None
    return render_svg(session.pattern, session.evaluated, width=1000,
                      object_ids=ids, pieces=plist)


def _render_png(session: PatternSession, pieces=None) -> bytes | None:
    """Rasterize the block to PNG for the vision input. Returns None if no raster
    backend is installed — the agent then runs state-only (by design)."""
    try:
        import cairosvg  # optional; needs libcairo at runtime
    except Exception:
        return None
    try:
        return cairosvg.svg2png(bytestring=render_svg_for(session, pieces).encode(),
                                output_width=1000)
    except Exception:
        return None


def _state_block(session: PatternSession, pieces=None, label: str = "") -> dict:
    if pieces:
        state = block_state(session.pattern, session.evaluated, list(pieces),
                            session.measurements, label=label)
    else:
        state = compact_state(session.pattern, session.evaluated, session.measurements)
    return {"type": "text", "text": "BLOCK STATE (JSON):\n" + json.dumps(state)}


def _user_turn(session: PatternSession, user_text: str, pieces=None, label: str = "") -> list[dict]:
    content: list[dict] = [{"type": "text", "text": user_text},
                           _state_block(session, pieces, label)]
    png = _render_png(session, pieces)
    if png is not None:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png).decode(),
            },
        })
    return content


def run_turn(session: PatternSession, user_text: str, pieces=None, label: str = "") -> dict:
    """Run one chat turn on the active block: model reasons over image+state,
    calls operation tools, returns its final text plus the tool calls it made."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {
            "reply": (
                "⚙️ Agent model not wired yet (set ANTHROPIC_API_KEY). The engine, "
                "state export, render, and edit/delete operations are live — your "
                f"message was: “{user_text}”."
            ),
            "tool_calls": [],
        }

    import anthropic

    client = anthropic.Anthropic()
    messages: list[dict] = [
        {"role": "user", "content": _user_turn(session, user_text, pieces, label)}]
    tool_calls: list[dict] = []

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return {"reply": text.strip() or "(no response)", "tool_calls": tool_calls}

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            op = dispatch_tool(session, block.name, block.input)
            tool_calls.append({"tool": block.name, "input": block.input, "ok": op.ok,
                               "message": op.message})
            # Feed the outcome AND the refreshed state back to the model.
            payload = {"ok": op.ok, "message": op.message}
            if op.blocked_by:
                payload["blocked_by"] = op.blocked_by
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                    _state_block(session, pieces, label),
                ],
                "is_error": not op.ok,
            })
        messages.append({"role": "user", "content": results})

    return {"reply": "Stopped after reaching the maximum number of edit steps.",
            "tool_calls": tool_calls}
