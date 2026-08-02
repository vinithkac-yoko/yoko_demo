"""The action space: one tool per Seamly2D operation.

This is the environment's half of the agent contract. :data:`TOOLS` is the
JSON-schema description a model is given; :func:`dispatch_tool` is the
transition function that turns one of its calls into a real mutation on a
:class:`~seamly_engine.operations.PatternSession`.

It lives in the engine rather than alongside a policy because it belongs to the
environment: what the actions *are*, and what they do to the pattern, is a
property of Seamly2D — not of whichever model happens to be driving. Any policy
(a prompted agent, a fine-tuned model, a replay of a document, a human) drives
the pattern through exactly this surface, so they are all measured on equal
terms.

Two invariants the rest of the system relies on:

* **Dispatch never raises.** A malformed call comes back as a failed
  :class:`~seamly_engine.operations.OpResult` carrying the reason, so a policy
  can read the error and correct itself instead of crashing the episode.
* **The engine decides what is legal.** Every mutation re-evaluates the pattern
  and rolls back if the result cannot be computed, and delete is
  block-and-report. Tools cannot leave the pattern in a broken state.
"""

from __future__ import annotations

import logging
from typing import Any

from .operations import OpResult, PatternSession

log = logging.getLogger("seamly_engine.actions")

__all__ = ["TOOLS", "POINT_TYPES", "dispatch_tool"]

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
    "pointOfIntersectionCurves", "triangle",
    "pointFromCircleAndTangent", "pointFromArcAndTangent",
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
                "kind": {"type": "string",
                         "enum": ["flippingByLine", "flippingByAxis", "rotation", "moving"]},
                "source_ids": {"type": "array", "items": {"type": "integer"},
                               "description": "ids of the objects to transform"},
                "p1Line": {"type": "string", "description": "flippingByLine: mirror-axis start point id"},
                "p2Line": {"type": "string", "description": "flippingByLine: mirror-axis end point id"},
                "center": {"type": "string", "description": "flippingByAxis/rotation: centre point id"},
                "axisType": {"type": "string", "enum": ["vertical", "horizontal"]},
                "angle": {"type": "string", "description": "rotation/moving: angle expression"},
                "length": {"type": "string", "description": "moving: distance expression"},
                "suffix": {"type": "string", "description": "suffix for the copies' names, e.g. '_m'"},
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
                "node_ids": {"type": "array", "items": {"type": "integer"},
                             "description": "outline objects in order (min 3)"},
                "seam_allowance": {"type": "boolean"},
                "width": {"type": "string", "description": "seam allowance width, e.g. '1'"},
                "grainline_anchor": {"type": "integer",
                                     "description": "point id the grainline is centred on"},
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
    if name == "add_operation":
        srcs = args.get("source_ids") or []
        if not srcs:
            return OpResult(False, "add_operation: source_ids is empty")
        attrs = {"suffix": args.get("suffix", "_c")}
        for k in ("p1Line", "p2Line", "center", "axisType", "angle", "length"):
            if args.get(k):
                attrs[k] = args[k]
        kids = [{"src": str(s)} for s in srcs]   # dst ids are reserved by the engine
        return session.add_object("operation", args["kind"], attrs, children=kids)
    if name == "create_piece":
        return session.create_piece(
            args["name"], [int(i) for i in args.get("node_ids", [])],
            seam_allowance=bool(args.get("seam_allowance", True)),
            width=str(args.get("width", "1")),
            internal_paths=args.get("internal_paths"),
            grainline_anchor=(int(args["grainline_anchor"])
                              if args.get("grainline_anchor") is not None else None))
    if name == "add_variable":
        return session.set_variable(args["name"], args["formula"], args.get("description", ""))
    if name == "edit_object":
        return session.edit_object(int(args["object_id"]), args.get("attrs", {}))
    if name == "delete_object":
        return session.delete_object(int(args["object_id"]))
    return OpResult(False, f"unknown tool {name!r}")

