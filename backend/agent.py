"""The agent loop — one typed instruction in, an ordered list of actions out.

This is a from-scratch redesign of the earlier chat-turn loop, not a port of
it. The workspace is no longer a conversation: you type one instruction, the
model runs it to completion against the pattern, and what comes back is a
**build log** — an ordered list of :class:`Action`, each one real tool call
the model made, with the reasoning that led to it and whether it succeeded.
That log is the UI (see ``app/index.html``); there is no chat reply to
render, only the actions and a short closing note.

Each turn the model receives the structured pattern state (the primary
input) and, when a raster backend is available, a rendered image. It calls
one tool per Seamly2D operation — construction geometry *and* whole-piece
operations (split, merge, create, edit, delete) — via
:func:`dispatch_tool`, which drives a
:class:`~seamly_engine.operations.PatternSession`. Every tool re-evaluates
and rolls back on breakage; dispatch never raises.

Reasoning capture: adaptive thinking is enabled for models that support it,
and the thinking block(s) the model writes before a tool call become that
:class:`Action`'s ``reasoning``. When one thinking block precedes several
tool calls in the same turn, they share its text — the model deliberated
once and acted several times; splitting the reasoning per call would
invent a granularity the model never had.

This is the Anthropic Messages API tool-use loop (not the separate "Claude
Agent SDK" product, which is a filesystem/coding agent). The model call is
gated behind ``ANTHROPIC_API_KEY`` so the engine/render/edit paths stay
runnable offline.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from seamly_engine.operations import OpResult, PatternSession
from seamly_engine.pieces import scoped_ids
from seamly_engine.render import render_svg
from seamly_engine.state import block_state, compact_state

log = logging.getLogger("vla.agent")

# Models the UI offers. Sonnet is the default: adaptive thinking, near-Opus
# quality on agentic/tool work at a fraction of the cost — including on
# split/merge, which need more spatial reasoning than a single-point edit.
# Haiku is the budget option but has no thinking, so it's markedly weaker at
# multi-step drafting; Opus is the strongest for the hardest construction.
MODELS: list[dict[str, Any]] = [
    {
        "id": "claude-sonnet-5",
        "label": "Sonnet 5",
        "note": "Balanced — recommended",
        "thinking": True,
    },
    {
        "id": "claude-opus-4-8",
        "label": "Opus 4.8",
        "note": "Strongest for hard drafting",
        "thinking": True,
    },
    {
        "id": "claude-haiku-4-5",
        "label": "Haiku 4.5",
        "note": "Cheapest — simple edits only",
        "thinking": False,
    },
]
MODEL_IDS = {m["id"] for m in MODELS}

MODEL = os.getenv("VLA_MODEL", "claude-sonnet-5")
MAX_ITERATIONS = int(os.getenv("VLA_MAX_STEPS", "12"))
# Width of the PNG sent to the model. Image tokens scale with pixel area, so
# this is a direct cost lever; the structured state is the primary input.
VISION_WIDTH = int(os.getenv("VLA_VISION_WIDTH", "900"))
SEND_IMAGE = os.getenv("VLA_SEND_IMAGE", "1") != "0"
# Ceiling for one model response — thinking tokens count against this too, not
# just the visible reply. An unscoped instruction can hand the model the whole
# pattern's state, and adaptive thinking over that can burn most of a small
# budget before it ever writes a closing summary (the tool call itself still
# lands first, so the edit isn't lost — only the wrap-up sentence is).
MAX_TOKENS = int(os.getenv("VLA_MAX_TOKENS", "16000"))

# Models that accept adaptive thinking. Older/cheaper models (e.g. Haiku 4.5)
# reject it, so it's simply omitted for them.
_ADAPTIVE_THINKING = ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6", "sonnet-5", "fable")

SYSTEM_PROMPT = """\
You are a pattern-drafting agent. You take ONE typed instruction at a time \
("split the bodice front along the princess line from B4 to the hem", "merge \
the yoke back into the body", "let out the waist 2cm") and carry it out \
completely against a Seamly2D-compatible pattern by calling tools — construction \
geometry AND whole-piece operations. There is no back-and-forth in a turn: \
work the instruction to completion, then stop.

You reason over TWO representations of the current state, not a screenshot alone:
1. A structured JSON state where every object (point/line/arc/spline) is \
semantically tagged: `role` (seamline/dart/grainline/drill_hole/construction/…), \
`construction` vs `final_outline`, `built_from` (dependencies), `dependents` \
(what breaks if it changes), and `formula` (the raw Seamly expression plus its \
resolved numeric value). Point objects include resolved `xy` coordinates (cm). \
Pieces are listed with their outline node ids and internal paths.
2. When present, a rendered image: bold strokes are final piece outlines, dimmed \
strokes are construction geometry, darts are magenta, drill holes teal.

Construction tools: `add_point` (any point tool_type — its schema lists the \
attrs each needs), `add_line`, `add_curve` (arc / arcWithLength / elArc / \
spline), `add_dart` (a true dart), `add_operation` (rotate/move/mirror a set of \
objects into copies), `add_variable`, `edit_object` (any attribute), \
`delete_object` (block-and-report — refused with the dependent chain if \
anything depends on it).

Piece tools — this is how you reshape a garment from its basic blocks:
- `split_piece` cuts one piece into two along the straight line between two \
points. Each point is either an existing outline vertex, or a point you build \
first with the ordinary point tools that lands exactly on one of the piece's \
straight edges — construct it there before calling split_piece. A point on a \
*curved* edge isn't supported. Internal paths that straddle the cut are dropped \
and named in the result — re-add them on the correct side if that happens.
- `merge_piece` combines two pieces along a seam they already share, named by \
its two endpoints. It only works when both pieces reference the *same* \
construction points along that edge (this is exactly what you get by merging \
two halves that came from `split_piece`, or two pieces deliberately drafted to \
share a seam). If it's refused, the pieces don't actually share that edge.
- `create_piece` turns construction geometry into a brand-new cut piece: give \
node_ids in outline order (points and curves mixed as needed), it closes \
automatically.
- `edit_piece` renames a piece or changes its seam allowance / width. \
`delete_piece` removes a piece; its construction geometry is untouched (a \
piece is a view onto construction objects, not their owner).

Guidance:
- Formulas are Seamly expressions. Measurements (e.g. `waist_circ`) and variables \
(e.g. `#CM`, the cm-scale factor) can be referenced directly. To "let out the \
waist 2cm", edit the object whose length drives that dimension (e.g. \
`(waist_circ/4)+4*#CM` → `(waist_circ/4)+4*#CM+2`).
- To build new structure, add construction points first, then lines/curves/darts \
between them, then turn them into a piece if the instruction calls for one. \
Prefer editing the construction object that already drives a dimension over \
adding a parallel one.
- Every tool call auto-re-evaluates and rolls back if the result would be \
invalid; the tool result says whether it failed and why. If a call fails, read \
the reason and either fix the call or take a different approach — don't repeat \
the same failing call.
- Before you split or merge, check the piece's outline and internal paths in \
the state so you're not guessing which points bound it.

IMPORTANT: only a successful tool call changes the pattern. Never claim you \
changed something unless you actually called a tool and it returned ok. If you \
can't identify which object or piece to act on from the state, say so rather \
than guessing.

When the instruction is fully carried out (or you've determined it can't be, and \
said why), stop. Keep your final note brief — the action log is the record of \
what happened; the note is just a one-line summary."""

# --- tool schemas: the full Seamly object model, plus whole-piece operations --
_ATTRS = {
    "type": "object",
    "additionalProperties": {"type": "string"},
    "description": "Seamly attributes as strings (formulas or object ids)",
}

POINT_TYPES = [
    "single",
    "endLine",
    "alongLine",
    "normal",
    "bisector",
    "intersectXY",
    "lineIntersect",
    "height",
    "shoulder",
    "pointOfContact",
    "lineIntersectAxis",
    "curveIntersectAxis",
    "cutSpline",
    "cutArc",
    "cutSplinePath",
    "pointOfIntersectionCircles",
    "pointOfIntersectionArcs",
    "pointOfIntersectionCurves",
    "triangle",
    "pointFromCircleAndTangent",
    "pointFromArcAndTangent",
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
            "pointOfIntersectionCircles{c1Center,c2Center,c1Radius,c2Radius,crossPoint}; "
            "pointOfIntersectionArcs{firstArc,secondArc,crossPoint}; "
            "pointOfIntersectionCurves{curve1,curve2,vCrossPoint,hCrossPoint} "
            "(cross points: 1=highest/leftmost, 2=lowest/rightmost); "
            "triangle{axisP1,axisP2,firstPoint,secondPoint}; "
            "pointFromCircleAndTangent{cCenter,cRadius,tangent,crossPoint}; "
            "pointFromArcAndTangent{arc,tangent,crossPoint}. "
            "Always include a 'name'. Optional: lineType,lineColor,lineWeight."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"tool_type": {"type": "string", "enum": POINT_TYPES}, "attrs": _ATTRS},
            "required": ["tool_type", "attrs"],
        },
    },
    {
        "name": "add_line",
        "description": "Draw a line between two existing points.",
        "input_schema": {
            "type": "object",
            "properties": {
                "firstPoint": {"type": "string"},
                "secondPoint": {"type": "string"},
                "lineType": {"type": "string"},
                "lineColor": {"type": "string"},
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
                "baseLineP1": {"type": "string"},
                "baseLineP2": {"type": "string"},
                "dartP1": {"type": "string"},
                "dartP2": {"type": "string"},
                "dartP3": {"type": "string"},
                "name1": {"type": "string"},
                "name2": {"type": "string"},
            },
            "required": ["baseLineP1", "baseLineP2", "dartP1", "dartP2", "dartP3"],
        },
    },
    {
        "name": "add_operation",
        "description": (
            "Transform a set of existing objects, creating transformed copies "
            "(exactly like Seamly's Operations tools). kind: "
            "'flippingByLine' — mirror across the line p1Line→p2Line; "
            "'flippingByAxis' — mirror across a vertical|horizontal axis through "
            "center; 'rotation' — rotate about center by angle (degrees, "
            "counter-clockwise on screen); 'moving' — translate by length at "
            "angle. source_ids are the objects to copy (points and curves). "
            "Use this for 'mirror the front to make the back', 'rotate this "
            "dart 10°', 'move this piece 5cm right'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["flippingByLine", "flippingByAxis", "rotation", "moving"],
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "ids of the objects to transform",
                },
                "p1Line": {
                    "type": "string",
                    "description": "flippingByLine: mirror-axis start point id",
                },
                "p2Line": {
                    "type": "string",
                    "description": "flippingByLine: mirror-axis end point id",
                },
                "center": {
                    "type": "string",
                    "description": "flippingByAxis/rotation: centre point id",
                },
                "axisType": {"type": "string", "enum": ["vertical", "horizontal"]},
                "angle": {"type": "string", "description": "rotation/moving: angle expression"},
                "length": {"type": "string", "description": "moving: distance expression"},
                "suffix": {
                    "type": "string",
                    "description": "suffix for the copies' names, e.g. '_m'",
                },
            },
            "required": ["kind", "source_ids"],
        },
    },
    {
        "name": "create_piece",
        "description": (
            "Turn construction geometry into a real cut piece (a Seamly detail). "
            "node_ids are the objects forming the seam outline **in order around "
            "the piece** — mix points and curves (splines/arcs) as the outline "
            "requires; the piece closes automatically. Optionally add internal "
            "paths (darts, fold/guide lines, drill holes) and a grainline anchor. "
            "Use this once the outline points exist, e.g. 'make this into a front "
            "bodice piece'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "node_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "outline objects in order (min 3)",
                },
                "seam_allowance": {"type": "boolean"},
                "width": {"type": "string", "description": "seam allowance width, e.g. '1'"},
                "grainline_anchor": {
                    "type": "integer",
                    "description": "point id the grainline is centred on",
                },
                "internal_paths": {
                    "type": "array",
                    "description": "darts / guide lines inside the piece",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "line_type": {"type": "string"},
                            "node_ids": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["name", "node_ids"],
                    },
                },
            },
            "required": ["name", "node_ids"],
        },
    },
    {
        "name": "split_piece",
        "description": (
            "Cut a piece into two along the straight line between two points on "
            "its outline. Each point is either an existing outline vertex, or a "
            "point you've just constructed that sits exactly on one of the "
            "piece's straight edges (build it first with add_point). A point on "
            "a curved edge isn't supported — splitting the curve itself isn't "
            "implemented. Internal paths (darts, drill holes) follow whichever "
            "new piece contains them; ones that straddle the cut are dropped and "
            "named in the result. The original piece is removed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "piece_id": {"type": "integer"},
                "point_a_id": {"type": "integer"},
                "point_b_id": {"type": "integer"},
                "name_a": {"type": "string", "description": "name for the piece on point_a's side"},
                "name_b": {"type": "string", "description": "name for the piece on point_b's side"},
                "seam_allowance": {
                    "type": "boolean",
                    "description": "default: same as the source piece",
                },
                "width": {"type": "string", "description": "default: same as the source piece"},
            },
            "required": ["piece_id", "point_a_id", "point_b_id", "name_a", "name_b"],
        },
    },
    {
        "name": "merge_piece",
        "description": (
            "Combine two pieces into one along a seam they share, named by its "
            "two endpoints. Both pieces must reference the same construction "
            "points along that edge — this is exactly the case for two pieces "
            "produced by split_piece, or two pieces deliberately drafted to "
            "share a seam. Refused if the pieces don't actually share that edge "
            "(check the state for which points each piece's outline uses). Both "
            "source pieces are removed; their internal paths carry over."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "piece_a_id": {"type": "integer"},
                "piece_b_id": {"type": "integer"},
                "edge_point_1": {"type": "integer"},
                "edge_point_2": {"type": "integer"},
                "name": {"type": "string"},
                "seam_allowance": {"type": "boolean", "description": "default: same as piece_a"},
                "width": {"type": "string", "description": "default: same as piece_a"},
            },
            "required": ["piece_a_id", "piece_b_id", "edge_point_1", "edge_point_2", "name"],
        },
    },
    {
        "name": "edit_piece",
        "description": "Rename a piece, or change its seam allowance on/off or width.",
        "input_schema": {
            "type": "object",
            "properties": {
                "piece_id": {"type": "integer"},
                "name": {"type": "string"},
                "seam_allowance": {"type": "boolean"},
                "width": {"type": "string"},
            },
            "required": ["piece_id"],
        },
    },
    {
        "name": "delete_piece",
        "description": (
            "Remove a piece. Its construction geometry is untouched — a piece is "
            "a view onto construction objects, not their owner — so this never "
            "cascades to anything else."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"piece_id": {"type": "integer"}},
            "required": ["piece_id"],
        },
    },
    {
        "name": "add_variable",
        "description": (
            "Add or update a pattern variable (a Seamly 'increment'), e.g. "
            "#Ease_Bust = 5. Variables can be referenced in any formula and are "
            "the clean way to make a value adjustable in one place."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "e.g. #Ease_Waist"},
                "formula": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "formula"],
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
            "Delete a construction object by id. Refused (block-and-report) if "
            "other objects depend on it; the response lists the dependent ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"object_id": {"type": "integer"}},
            "required": ["object_id"],
        },
    },
]
TOOL_NAMES = {t["name"] for t in TOOLS}

_ARC_KIND = {
    "arc": ("arc", "simple"),
    "arcWithLength": ("arc", "arcWithLength"),
    "elArc": ("elArc", "simple"),
}


def dispatch_tool(session: PatternSession, name: str, args: dict) -> OpResult:
    """Execute one tool call. Never raises — a malformed call from the model
    comes back as a failed OpResult so the model can correct itself."""
    try:
        return _dispatch(session, name, args)
    except KeyError as e:
        return OpResult(False, f"{name}: missing required argument {e}")
    except Exception as e:
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
    if name == "add_operation":
        srcs = args.get("source_ids") or []
        if not srcs:
            return OpResult(False, "add_operation: source_ids is empty")
        attrs = {"suffix": args.get("suffix", "_c")}
        for k in ("p1Line", "p2Line", "center", "axisType", "angle", "length"):
            if args.get(k):
                attrs[k] = args[k]
        kids = [{"src": str(s)} for s in srcs]  # dst ids are reserved by the engine
        return session.add_object("operation", args["kind"], attrs, children=kids)
    if name == "create_piece":
        return session.create_piece(
            args["name"],
            [int(i) for i in args.get("node_ids", [])],
            seam_allowance=bool(args.get("seam_allowance", True)),
            width=str(args.get("width", "1")),
            internal_paths=args.get("internal_paths"),
            grainline_anchor=(
                int(args["grainline_anchor"]) if args.get("grainline_anchor") is not None else None
            ),
        )
    if name == "split_piece":
        return session.split_piece(
            int(args["piece_id"]),
            int(args["point_a_id"]),
            int(args["point_b_id"]),
            name_a=args["name_a"],
            name_b=args["name_b"],
            seam_allowance=args.get("seam_allowance"),
            width=args.get("width"),
        )
    if name == "merge_piece":
        return session.merge_piece(
            int(args["piece_a_id"]),
            int(args["piece_b_id"]),
            int(args["edge_point_1"]),
            int(args["edge_point_2"]),
            name=args["name"],
            seam_allowance=args.get("seam_allowance"),
            width=args.get("width"),
        )
    if name == "edit_piece":
        return session.edit_piece(
            int(args["piece_id"]),
            name=args.get("name"),
            seam_allowance=args.get("seam_allowance"),
            width=args.get("width"),
        )
    if name == "delete_piece":
        return session.delete_piece(int(args["piece_id"]))
    if name == "add_variable":
        return session.set_variable(args["name"], args["formula"], args.get("description", ""))
    if name == "edit_object":
        return session.edit_object(int(args["object_id"]), args.get("attrs", {}))
    if name == "delete_object":
        return session.delete_object(int(args["object_id"]))
    return OpResult(False, f"unknown tool {name!r}")


# --- rendering -----------------------------------------------------------------
def render_svg_for(session: PatternSession, pieces=None) -> str:
    """Rich construction SVG scoped to the active block (its pieces + drivers), or
    the whole pattern if no block is selected."""
    plist = list(pieces) if pieces else None
    ids = scoped_ids(session.pattern, plist) if plist else None
    added = getattr(session, "added_ids", None) or set()
    if ids is not None and added:
        ids = ids | added
    return render_svg(
        session.pattern,
        session.evaluated,
        width=1000,
        object_ids=ids,
        pieces=plist,
        highlight_ids=added,
    )


def _render_png(session: PatternSession, pieces=None) -> bytes | None:
    """Rasterize to PNG for the vision input. Returns None if no raster backend
    is installed — the agent then runs state-only (by design)."""
    if not SEND_IMAGE:
        return None
    try:
        import cairosvg
    except Exception:
        return None
    try:
        return cairosvg.svg2png(
            bytestring=render_svg_for(session, pieces).encode(), output_width=VISION_WIDTH
        )
    except Exception:
        return None


# --- the action log --------------------------------------------------------
@dataclass
class Action:
    """One tool call the model made, in order. This is the unit the UI is
    built around: the action log is a list of these, and clicking one shows
    its ``reasoning`` plus a before/after render (derived by replay, not
    stored here — see ``backend/store.py``)."""

    step_index: int
    tool_name: str
    tool_input: dict
    reasoning: str
    ok: bool
    message: str
    blocked_by: list[int] = field(default_factory=list)
    touched_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "reasoning": self.reasoning,
            "ok": self.ok,
            "message": self.message,
            "blocked_by": self.blocked_by,
            "touched_ids": self.touched_ids,
        }


@dataclass
class RunResult:
    final_text: str
    actions: list[Action]
    stopped_reason: str  # "done" | "max_steps" | "error" | "no_key"

    def as_dict(self) -> dict:
        return {
            "final_text": self.final_text,
            "actions": [a.as_dict() for a in self.actions],
            "stopped_reason": self.stopped_reason,
        }


def _state_block(session: PatternSession, pieces=None, label: str = "") -> dict:
    if pieces:
        state = block_state(
            session.pattern,
            session.evaluated,
            list(pieces),
            session.measurements,
            label=label,
            extra_ids=getattr(session, "added_ids", None),
        )
    else:
        state = compact_state(session.pattern, session.evaluated, session.measurements)
    return {"type": "text", "text": "PATTERN STATE (JSON):\n" + json.dumps(state)}


def _state_delta(session: PatternSession, pieces, label: str, ids: set[int]) -> dict:
    """Just the objects touched by a tool call, instead of resending the whole
    state after every action (a large, avoidable per-turn cost)."""
    state = (
        block_state(
            session.pattern,
            session.evaluated,
            list(pieces or []),
            session.measurements,
            label=label,
            extra_ids=getattr(session, "added_ids", None),
        )
        if pieces
        else compact_state(session.pattern, session.evaluated, session.measurements)
    )
    touched = [o for o in state["objects"] if o["id"] in ids]
    return {
        "type": "text",
        "text": "CHANGED OBJECTS:\n" + json.dumps(touched) if touched else "No object changes.",
    }


def _instruction_turn(
    session: PatternSession, instruction: str, pieces=None, label: str = ""
) -> list[dict]:
    content: list[dict] = [
        {"type": "text", "text": instruction},
        _state_block(session, pieces, label),
    ]
    png = _render_png(session, pieces)
    if png is not None:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(png).decode(),
                },
            }
        )
    return content


def _create(client, messages: list[dict], model: str | None = None):
    """One model call. Adaptive thinking is only sent to models that support
    it; if a model rejects it anyway, retry without it rather than failing."""
    model = model or MODEL
    kwargs = dict(
        model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages
    )
    if not any(m in model for m in _ADAPTIVE_THINKING):
        return client.messages.create(**kwargs)
    try:
        return client.messages.create(thinking={"type": "adaptive"}, **kwargs)
    except Exception as e:
        name = type(e).__name__
        if "BadRequest" not in name and "TypeError" not in name:
            raise
        log.warning("adaptive thinking rejected (%s: %s) — retrying without it", name, e)
        return client.messages.create(**kwargs)


def run_instruction(
    session: PatternSession,
    instruction: str,
    pieces=None,
    label: str = "",
    model: str | None = None,
) -> RunResult:
    """Run one typed instruction to completion. Never raises — any failure
    comes back as a ``RunResult`` with ``stopped_reason`` set, so the caller
    always has something to persist and show."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return RunResult(
            final_text=(
                "Agent model not wired yet (set ANTHROPIC_API_KEY). The engine, state "
                f"export, render, and every edit/piece operation are live — your "
                f"instruction was: “{instruction}”."
            ),
            actions=[],
            stopped_reason="no_key",
        )
    try:
        return _run_instruction(session, instruction, pieces, label, model)
    except Exception as e:
        log.exception("run failed")
        return RunResult(
            final_text=f"The agent hit an error: {type(e).__name__}: {e}",
            actions=[],
            stopped_reason="error",
        )


def _run_instruction(
    session: PatternSession, instruction: str, pieces, label: str, model: str | None
) -> RunResult:
    import anthropic

    client = anthropic.Anthropic()
    messages: list[dict] = [
        {"role": "user", "content": _instruction_turn(session, instruction, pieces, label)}
    ]
    actions: list[Action] = []
    step = 0

    for _ in range(MAX_ITERATIONS):
        response = _create(client, messages, model)
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            if response.stop_reason == "max_tokens" and not text.strip():
                text = "(ran out of output budget mid-run — try a smaller instruction)"
            return RunResult(
                final_text=text.strip() or "(done)", actions=actions, stopped_reason="done"
            )

        messages.append({"role": "assistant", "content": response.content})
        results = []
        reasoning = ""  # the most recent thinking block seen this turn
        for block in response.content:
            kind = getattr(block, "type", "")
            if kind == "thinking":
                reasoning = getattr(block, "thinking", "") or reasoning
                continue
            if kind != "tool_use":
                continue

            before_added = set(getattr(session, "added_ids", set()))
            op = dispatch_tool(session, block.name, block.input)
            touched = set(getattr(session, "added_ids", set())) - before_added
            oid = block.input.get("object_id") or block.input.get("piece_id")
            if isinstance(oid, int):
                touched.add(oid)
            actions.append(
                Action(
                    step_index=step,
                    tool_name=block.name,
                    tool_input=dict(block.input),
                    reasoning=reasoning,
                    ok=op.ok,
                    message=op.message,
                    blocked_by=list(op.blocked_by or []),
                    touched_ids=sorted(touched),
                )
            )
            step += 1

            payload = {"ok": op.ok, "message": op.message}
            if op.blocked_by:
                payload["blocked_by"] = op.blocked_by
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": [
                        {"type": "text", "text": json.dumps(payload)},
                        _state_delta(session, pieces, label, touched),
                    ],
                    "is_error": not op.ok,
                }
            )
        messages.append({"role": "user", "content": results})

    return RunResult(
        final_text="Stopped after reaching the maximum number of steps.",
        actions=actions,
        stopped_reason="max_steps",
    )
