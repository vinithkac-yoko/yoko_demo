"""VLA agent loop (Claude Agent SDK integration point).

The agent receives, each turn: the user's instruction, the rendered pattern
(image) and the structured VLA state, then calls **one tool per Seamly2D
operation** to edit the pattern. Tools mutate the pattern via
:class:`seamly_engine.operations.PatternSession`, which re-evaluates and can roll
back — so the model always acts on valid, fresh geometry, and delete is
block-and-report.

This module defines the tool schemas and the turn driver. The actual model call
is gated behind ``ANTHROPIC_API_KEY``; without it, ``run_turn`` returns a stub so
the whole HTTP + engine path is runnable offline during development.
"""

from __future__ import annotations

import os
from typing import Any

from seamly_engine.operations import OpResult, PatternSession

# --- tool schemas: one per operation (grows toward full parity) --------------
# These are the actions the model may take. Each maps to a PatternSession method.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "delete_object",
        "description": (
            "Delete a construction object by id. Refused if other objects depend "
            "on it; the response lists the dependent ids so you can delete those "
            "first or choose another edit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"object_id": {"type": "integer"}},
            "required": ["object_id"],
        },
    },
    {
        "name": "edit_formula",
        "description": (
            "Change a formula attribute (e.g. 'length', 'angle', 'radius') of an "
            "object. Automatically re-evaluates the whole pattern and rolls the "
            "edit back if it makes any previously-valid object unresolvable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "object_id": {"type": "integer"},
                "attr": {"type": "string", "enum": ["length", "angle", "radius"]},
                "new_formula": {"type": "string"},
            },
            "required": ["object_id", "attr", "new_formula"],
        },
    },
    # Roadmap: add_point_endline, add_point_alongline, add_dart, add_line,
    # add_spline, ... — the full parity add_* set.
]


def dispatch_tool(session: PatternSession, name: str, args: dict) -> OpResult:
    """Execute one tool call against the session."""
    if name == "delete_object":
        return session.delete_object(int(args["object_id"]))
    if name == "edit_formula":
        return session.edit_formula(int(args["object_id"]), args["attr"], args["new_formula"])
    return OpResult(False, f"unknown tool {name!r}")


def run_turn(session: PatternSession, user_text: str) -> dict:
    """Run one chat turn. Returns {reply, tool_calls}.

    With ANTHROPIC_API_KEY set this drives the Claude Agent SDK loop (image +
    state + instruction -> tool calls). Without it, returns a stub so the rest of
    the system is exercisable offline.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {
            "reply": (
                "⚙️ Agent model not wired yet (set ANTHROPIC_API_KEY). "
                "The engine, state export, render, and edit/delete operations are "
                f"live — your message was: “{user_text}”."
            ),
            "tool_calls": [],
        }

    # --- Claude Agent SDK loop (integration point) ---------------------------
    # from claude_agent_sdk import ClaudeSDKClient, ...
    # 1. Build the prompt: user_text + session.state() + a rendered PNG.
    # 2. Provide TOOLS; on each tool_use, call dispatch_tool(session, name, args)
    #    and feed the OpResult back as the tool_result.
    # 3. Loop until the model stops calling tools; return its final text.
    raise NotImplementedError("Agent SDK loop wiring is the next roadmap item.")
