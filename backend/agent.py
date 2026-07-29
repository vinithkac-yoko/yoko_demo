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
import logging
import os
from typing import Any

log = logging.getLogger("vla.agent")

from seamly_engine.operations import OpResult, PatternSession
from seamly_engine.pieces import scoped_ids
from seamly_engine.render import render_svg
from seamly_engine.state import block_state, compact_state

# Cheapest capable default. Override with the VLA_MODEL env var (e.g.
# claude-sonnet-5 for more capability, claude-opus-4-8 for the most).
MODEL = os.getenv("VLA_MODEL", "claude-haiku-4-5")
MAX_ITERATIONS = int(os.getenv("VLA_MAX_STEPS", "8"))
# Width of the PNG sent to the model. Image tokens scale with pixel area, so this
# is a direct cost lever; the structured state is the primary input regardless.
VISION_WIDTH = int(os.getenv("VLA_VISION_WIDTH", "700"))
SEND_IMAGE = os.getenv("VLA_SEND_IMAGE", "1") != "0"

# Models that accept adaptive thinking. Older/cheaper models (e.g. Haiku 4.5)
# reject it, so we simply omit the parameter for them.
_ADAPTIVE_THINKING = ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6", "sonnet-5", "fable")

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

You can do anything the Seamly object model allows, via the tools:
- ADD geometry: `add_point` (any point tool_type — its schema lists the attrs \
each needs), `add_line`, `add_curve` (arc / arcWithLength / elArc / spline), \
`add_dart` (a true dart). Reference existing objects by their integer id from \
the state; give new points a `name`.
- EDIT: `edit_object` changes any attribute(s) of an object (a length, angle, \
radius, or a reference point).
- DELETE: `delete_object` — block-and-report; if it has dependents the delete is \
refused and returns the dependent chain (delete those first, or pick another way).

Guidance:
- Formulas are Seamly expressions. Measurements (e.g. `waist_circ`) and variables \
(e.g. `#CM`, the cm-scale factor) can be referenced directly. To "let out the \
waist 2cm", edit the object whose length drives that dimension (e.g. \
`(waist_circ/4)+4*#CM` → `(waist_circ/4)+4*#CM+2`).
- To build new structure, add construction points first, then lines/curves/darts \
between them. Prefer editing the construction object that drives a dimension.
- Every add/edit auto-re-evaluates and rolls back if the result is invalid; the \
tool result says whether it failed and why, plus the fresh state.

IMPORTANT: only a successful tool call changes the pattern. Never say you \
changed something unless you actually called a tool and it returned ok. If the \
request is a change, you must call a tool; if you can't identify which object to \
edit from the state, say so and ask rather than pretending.

Work step by step. After the change(s), briefly explain what you did in plain \
language. Keep it concise."""

# --- tool schemas: the full Seamly object model (add / edit / delete) --------
# Attribute values are Seamly expressions or object-id strings; ids reference
# objects by their integer id as shown in the state.
_ATTRS = {"type": "object", "additionalProperties": {"type": "string"},
          "description": "Seamly attributes as strings (formulas or object ids)"}

POINT_TYPES = [
    "single", "endLine", "alongLine", "normal", "bisector", "intersectXY",
    "lineIntersect", "height", "shoulder", "pointOfContact",
    "lineIntersectAxis", "curveIntersectAxis",
    "cutSpline", "cutArc", "cutSplinePath",
    "pointOfIntersectionCircles", "pointOfIntersectionArcs",
]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "add_point",
        "description": (
            "Create a new point. Required attrs by tool_type: "
            "single{x,y}; endLine{basePoint,angle,length}; "
            "alongLine{firstPoint,secondPoint,length}; "
            "normal{firstPoint,secondPoint,length,angle?}; "
            "bisector{firstPoint,secondPoint,thirdPoint,length}; "
            "intersectXY{firstPoint,secondPoint} (x from first, y from second); "
            "lineIntersect{p1Line1,p2Line1,p1Line2,p2Line2}; "
            "height{basePoint,p1Line,p2Line} (foot of perpendicular); "
            "shoulder{p1Line,p2Line,pShoulder,length}; "
            "lineIntersectAxis{basePoint,angle,p1Line,p2Line}; "
            "curveIntersectAxis{basePoint,angle,curve}; "
            "cutSpline|cutArc|cutSplinePath{curve,length}; "
            "pointOfIntersectionCircles{c1Center,c2Center,c1Radius,c2Radius,crossPoint}. "
            "Always include a 'name'. Optional: lineType,lineColor,lineWeight."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tool_type": {"type": "string", "enum": POINT_TYPES},
                "attrs": _ATTRS,
            },
            "required": ["tool_type", "attrs"],
        },
    },
    {
        "name": "add_line",
        "description": "Draw a line between two existing points.",
        "input_schema": {
            "type": "object",
            "properties": {
                "firstPoint": {"type": "string"}, "secondPoint": {"type": "string"},
                "lineType": {"type": "string"}, "lineColor": {"type": "string"},
                "lineWeight": {"type": "string"},
            },
            "required": ["firstPoint", "secondPoint"],
        },
    },
    {
        "name": "add_curve",
        "description": (
            "Create a curve. kind: 'arc'{center,radius,angle1,angle2}, "
            "'arcWithLength'{center,radius,angle1,length}, "
            "'elArc'{center,radius1,radius2,angle1,angle2,rotationAngle}, "
            "'spline'. For a spline set spline_type='cubicBezier' with "
            "attrs{point1,point2,point3,point4}, or 'cubicBezierPath' with "
            "path_points=[ids] (on-curve, ctrl, ctrl, on-curve, …)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["arc", "arcWithLength", "elArc", "spline"]},
                "spline_type": {"type": "string", "enum": ["cubicBezier", "cubicBezierPath"]},
                "attrs": _ATTRS,
                "path_points": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "add_dart",
        "description": (
            "Add a true dart on a base line. baseLineP1/baseLineP2 define the seam "
            "the dart sits on; dartP1/dartP2(apex)/dartP3 are the dart points. "
            "Produces two leg points named name1/name2."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "baseLineP1": {"type": "string"}, "baseLineP2": {"type": "string"},
                "dartP1": {"type": "string"}, "dartP2": {"type": "string"},
                "dartP3": {"type": "string"},
                "name1": {"type": "string"}, "name2": {"type": "string"},
            },
            "required": ["baseLineP1", "baseLineP2", "dartP1", "dartP2", "dartP3"],
        },
    },
    {
        "name": "edit_object",
        "description": (
            "Change one or more attributes of an existing object (e.g. a length, "
            "angle, radius, or a reference point). Auto re-evaluates; rolls back "
            "if it makes the pattern invalid."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"object_id": {"type": "integer"}, "attrs": _ATTRS},
            "required": ["object_id", "attrs"],
        },
    },
    {
        "name": "delete_object",
        "description": (
            "Delete an object by id. Refused (block-and-report) if other objects "
            "depend on it; the response lists the dependent ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"object_id": {"type": "integer"}},
            "required": ["object_id"],
        },
    },
]

_ARC_KIND = {"arc": ("arc", "simple"), "arcWithLength": ("arc", "arcWithLength"),
             "elArc": ("elArc", "simple")}


def dispatch_tool(session: PatternSession, name: str, args: dict) -> OpResult:
    """Execute one tool call. Never raises — a malformed call from the model comes
    back as a failed OpResult so the model can correct itself."""
    try:
        return _dispatch(session, name, args)
    except KeyError as e:
        return OpResult(False, f"{name}: missing required argument {e}")
    except Exception as e:  # noqa: BLE001 - surface any tool error to the model
        log.exception("tool %s failed", name)
        return OpResult(False, f"{name} failed: {type(e).__name__}: {e}")


def _dispatch(session: PatternSession, name: str, args: dict) -> OpResult:
    if name == "add_point":
        return session.add_object("point", args["tool_type"], args.get("attrs", {}))
    if name == "add_line":
        attrs = {"firstPoint": args["firstPoint"], "secondPoint": args["secondPoint"]}
        for k in ("lineType", "lineColor", "lineWeight"):
            if args.get(k):
                attrs[k] = args[k]
        return session.add_object("line", "", attrs)
    if name == "add_curve":
        kind = args["kind"]
        attrs = dict(args.get("attrs", {}))
        if kind == "spline":
            tag, typ = "spline", args.get("spline_type", "cubicBezier")
        elif kind in _ARC_KIND:
            tag, typ = _ARC_KIND[kind]
        else:
            return OpResult(False, f"add_curve: unknown kind {kind!r}")
        kids = None
        if args.get("path_points"):
            kids = [{"__tag__": "pathPoint", "pSpline": str(i)} for i in args["path_points"]]
        return session.add_object(tag, typ, attrs, children=kids)
    if name == "add_dart":
        attrs = {k: args[k] for k in ("baseLineP1", "baseLineP2", "dartP1", "dartP2", "dartP3")}
        for k in ("name1", "name2"):
            if args.get(k):
                attrs[k] = args[k]
        return session.add_object("point", "trueDarts", attrs)
    if name == "edit_object":
        return session.edit_object(int(args["object_id"]), args.get("attrs", {}))
    if name == "delete_object":
        return session.delete_object(int(args["object_id"]))
    return OpResult(False, f"unknown tool {name!r}")


def render_svg_for(session: PatternSession, pieces=None) -> str:
    """Rich construction SVG scoped to the active block (its pieces + drivers), or
    the whole pattern if no block is selected."""
    plist = list(pieces) if pieces else None
    ids = scoped_ids(session.pattern, plist) if plist else None
    added = getattr(session, "added_ids", None) or set()
    if ids is not None and added:
        ids = ids | added  # keep newly-created geometry visible
    return render_svg(session.pattern, session.evaluated, width=1000,
                      object_ids=ids, pieces=plist, highlight_ids=added)


def _render_png(session: PatternSession, pieces=None) -> bytes | None:
    """Rasterize the block to PNG for the vision input. Returns None if no raster
    backend is installed — the agent then runs state-only (by design)."""
    if not SEND_IMAGE:
        return None
    try:
        import cairosvg  # optional; needs libcairo at runtime
    except Exception:
        return None
    try:
        return cairosvg.svg2png(bytestring=render_svg_for(session, pieces).encode(),
                                output_width=VISION_WIDTH)
    except Exception:
        return None


def _state_delta(session: PatternSession, pieces, label: str, ids: set[int]) -> dict:
    """Just the objects touched by a tool call, instead of resending the whole
    block state after every action (a large per-turn cost)."""
    state = block_state(session.pattern, session.evaluated, list(pieces or []),
                        session.measurements, label=label,
                        extra_ids=getattr(session, "added_ids", None))
    touched = [o for o in state["objects"] if o["id"] in ids]
    return {"type": "text",
            "text": "CHANGED OBJECTS:\n" + json.dumps(touched) if touched
                    else "No object changes."}


def _state_block(session: PatternSession, pieces=None, label: str = "") -> dict:
    if pieces:
        state = block_state(session.pattern, session.evaluated, list(pieces),
                            session.measurements, label=label,
                            extra_ids=getattr(session, "added_ids", None))
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

    try:
        return _run_turn(session, user_text, pieces, label)
    except Exception as e:  # noqa: BLE001 — never 500 the request
        log.exception("agent turn failed")
        return {"reply": f"⚠️ The agent hit an error: {type(e).__name__}: {e}",
                "tool_calls": []}


def _create(client, messages: list[dict]):
    """One model call. Adaptive thinking is only sent to models that support it;
    if a model rejects it anyway we retry without it rather than failing."""
    kwargs = dict(model=MODEL, max_tokens=8000, system=SYSTEM_PROMPT,
                  tools=TOOLS, messages=messages)
    if not any(m in MODEL for m in _ADAPTIVE_THINKING):
        return client.messages.create(**kwargs)
    try:
        return client.messages.create(thinking={"type": "adaptive"}, **kwargs)
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if "BadRequest" not in name and "TypeError" not in name:
            raise
        log.warning("adaptive thinking rejected (%s: %s) — retrying without it", name, e)
        return client.messages.create(**kwargs)


def _run_turn(session: PatternSession, user_text: str, pieces=None, label: str = "") -> dict:
    import anthropic

    client = anthropic.Anthropic()
    messages: list[dict] = [
        {"role": "user", "content": _user_turn(session, user_text, pieces, label)}]
    tool_calls: list[dict] = []

    for _ in range(MAX_ITERATIONS):
        response = _create(client, messages)
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "max_tokens" and not text.strip():
                text = "(ran out of output budget before replying — try a simpler request)"
            return {"reply": text.strip() or "(no response)", "tool_calls": tool_calls}

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            before_added = set(getattr(session, "added_ids", set()))
            op = dispatch_tool(session, block.name, block.input)
            tool_calls.append({"tool": block.name, "input": block.input, "ok": op.ok,
                               "message": op.message})
            # Feed back the outcome plus ONLY the objects it touched (resending
            # the whole block state every step is a large, avoidable cost).
            payload = {"ok": op.ok, "message": op.message}
            if op.blocked_by:
                payload["blocked_by"] = op.blocked_by
            touched = set(getattr(session, "added_ids", set())) - before_added
            oid = block.input.get("object_id")
            if isinstance(oid, int):
                touched.add(oid)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                    _state_delta(session, pieces, label, touched),
                ],
                "is_error": not op.ok,
            })
        messages.append({"role": "user", "content": results})

    return {"reply": "Stopped after reaching the maximum number of edit steps.",
            "tool_calls": tool_calls}
