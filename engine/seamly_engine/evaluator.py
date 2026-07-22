"""Evaluate a pattern DAG into concrete geometry.

A ``<calculation>`` block is already in dependency order (every object refers
only to earlier ids), so a single forward pass suffices. For each object we:

1. build a :class:`~seamly_engine.formula.Scope` whose resolver is bound to the
   geometry computed so far (plus measurements, increments, and the tool's
   ``CurrentLength``), then
2. evaluate the tool's formula attributes and compute its point/curve.

Anything we cannot yet compute (unimplemented tool types, or objects whose
references are themselves unresolved) is recorded in ``Evaluated.unresolved``
with a reason instead of aborting the whole pattern — so the engine always
returns a partial, inspectable result and an honest coverage number.

The pseudo-variable resolver is the subtle part: ``Line_A_B`` /
``AngleLine_A_B`` read the live coordinates of points *A* and *B* by name, and
because point names themselves contain underscores (``A8_Back``) the split is
disambiguated against the set of known names.
"""

from __future__ import annotations

from . import geometry as geo
from .formula import FormulaError, Scope, evaluate, identifiers
from .measurements import MeasurementTable
from .model import Evaluated, Pattern, PatternObject


class _Unresolved(Exception):
    """Internal: raised when an object cannot be computed this pass."""


def evaluate_pattern(pattern: Pattern, measurements: MeasurementTable | None = None,
                     *, size: float | None = None, height: float | None = None) -> Evaluated:
    ev = Evaluated()
    meas = measurements.resolve(size, height) if measurements else {}

    # id -> object and name -> id (names are unique within a pattern).
    objs = pattern.object_by_id()
    name_of: dict[int, str] = {}
    id_of_name: dict[str, int] = {}
    for o in pattern.all_objects():
        nm = o.raw.get("name")
        if nm:
            name_of[o.id] = nm
            id_of_name[nm] = o.id
        # Double-point tools (trueDarts) declare two output points inline via
        # point1/point2 + name1/name2; register those names so pseudo-vars and
        # downstream refs resolve.
        if o.tool_type == "trueDarts":
            for pid_attr, nm_attr in (("point1", "name1"), ("point2", "name2")):
                pid, pnm = o.raw.get(pid_attr), o.raw.get(nm_attr)
                if pid and pid.isdigit() and pnm:
                    name_of[int(pid)] = pnm
                    id_of_name[pnm] = int(pid)

    _resolve_increments(pattern, meas, ev)

    for obj in pattern.all_objects():
        try:
            _eval_object(obj, objs, meas, ev, id_of_name)
        except _Unresolved as e:
            ev.unresolved[obj.id] = str(e)
        except FormulaError as e:
            ev.unresolved[obj.id] = f"formula error: {e}"
        except Exception as e:  # geometry degeneracy etc. — never crash the pass
            ev.unresolved[obj.id] = f"{type(e).__name__}: {e}"

    return ev


# --- increments --------------------------------------------------------------
def _resolve_increments(pattern: Pattern, meas: dict[str, float], ev: Evaluated) -> None:
    """Resolve custom variables, which may reference measurements and each
    other. Done with a memoized recursive resolver so declaration order and
    forward references don't matter."""
    by_name = {inc.name: inc for inc in pattern.increments}
    resolving: set[str] = set()

    def resolve_name(name: str) -> float:
        if name in ev.increment_values:
            return ev.increment_values[name]
        if name in meas:
            return meas[name]
        if name in by_name:
            if name in resolving:
                raise FormulaError(f"cyclic increment {name}")
            resolving.add(name)
            scope = Scope(resolve=resolve_name)
            try:
                val = evaluate(by_name[name].formula, scope)
            finally:
                resolving.discard(name)
            ev.increment_values[name] = val
            return val
        raise FormulaError(f"unknown variable {name!r}")

    for inc in pattern.increments:
        try:
            resolve_name(inc.name)
        except FormulaError:
            # Placeholder variables like '##_USER_VARIABLES_##' aren't real;
            # leave them out of the resolved table rather than failing.
            pass


# --- per-object evaluation ---------------------------------------------------
def _eval_object(obj: PatternObject, objs: dict[int, PatternObject],
                 meas: dict[str, float], ev: Evaluated, id_of_name: dict[str, int]) -> None:
    tag = obj.tag
    if tag == "point":
        _eval_point(obj, objs, meas, ev, id_of_name)
    elif tag == "line":
        _require_points(obj, ev, ["firstPoint", "secondPoint"])  # nothing to store
    elif tag == "arc":
        _eval_arc(obj, meas, ev, id_of_name)
    elif tag == "spline":
        _eval_spline(obj, ev)
    elif tag == "operation":
        _eval_operation(obj, ev)
    # unknown tags are simply ignored (not construction geometry)


def _point(ev: Evaluated, oid: int) -> geo.Point:
    p = ev.points.get(oid)
    if p is None:
        raise _Unresolved(f"depends on unresolved object {oid}")
    return p


def _require_points(obj: PatternObject, ev: Evaluated, attrs: list[str]) -> None:
    for a in attrs:
        v = obj.raw.get(a)
        if v and v.lstrip("-").isdigit():
            _point(ev, int(v))


def _scope_for(obj: PatternObject, ev: Evaluated, meas: dict[str, float],
               id_of_name: dict[str, int], current_length: float | None) -> Scope:
    names = set(id_of_name)

    def resolve(name: str) -> float:
        if name == "CurrentLength":
            if current_length is None:
                raise FormulaError("CurrentLength not available for this tool")
            return current_length
        if name.startswith("#"):
            if name in ev.increment_values:
                return ev.increment_values[name]
            raise FormulaError(f"unknown increment {name!r}")
        if name in meas:
            return meas[name]
        if name in ev.increment_values:
            return ev.increment_values[name]
        # pseudo-variables reading live geometry
        if name.startswith("Line_"):
            a, b = _split_two_names(name[len("Line_"):], names)
            return _point(ev, id_of_name[a]).dist(_point(ev, id_of_name[b]))
        if name.startswith("AngleLine_"):
            a, b = _split_two_names(name[len("AngleLine_"):], names)
            return geo.line_angle(_point(ev, id_of_name[a]), _point(ev, id_of_name[b]))
        if name.startswith("RadiusArc_"):
            arc_id = int(name.rsplit("_", 1)[1])
            r = ev.values.get(arc_id, {}).get("radius")
            if r is None:
                raise FormulaError(f"unresolved arc radius {name!r}")
            return r
        if name.startswith("SplPath_") or name.startswith("Spl_"):
            prefix = "SplPath_" if name.startswith("SplPath_") else "Spl_"
            a, b = _split_two_names(name[len(prefix):], names)
            return _curve_partial_length(ev, id_of_name[a], id_of_name[b])
        raise FormulaError(f"unresolvable variable {name!r}")

    return Scope(resolve=resolve)


def _curve_partial_length(ev: Evaluated, id_a: int, id_b: int) -> float:
    """Arc length of a spline/spline-path between two points that lie on it.

    Answers ``SplPath_A_B`` / ``Spl_A_B``. The two points may be defining nodes
    of the curve *or* points constructed to lie on it (e.g. a
    ``curveIntersectAxis`` result), so resolution is geometric: among all
    resolved curves, pick the one both points sit closest to, then return the
    arc-length distance between their positions along that curve.
    """
    pa, pb = ev.points.get(id_a), ev.points.get(id_b)
    if pa is None or pb is None:
        raise FormulaError(f"SplPath endpoints {id_a},{id_b} unresolved")

    best_len: float | None = None
    best_score = float("inf")
    for curve in ev.curves.values():
        poly = getattr(curve, "polyline", None)
        if poly is None:
            continue
        pts = curve.polyline(48)
        if len(pts) < 2:
            continue
        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + pts[i - 1].dist(pts[i]))
        (da, sa) = min(((p.dist(pa), cum[i]) for i, p in enumerate(pts)))
        (db, sb) = min(((p.dist(pb), cum[i]) for i, p in enumerate(pts)))
        score = da + db
        if score < best_score:
            best_score = score
            best_len = abs(sa - sb)

    # Require both endpoints to actually sit on the chosen curve.
    if best_len is None or best_score > 1.0:
        raise FormulaError(f"no resolved curve joins points {id_a} and {id_b}")
    return best_len


def _split_two_names(rest: str, names: set[str]) -> tuple[str, str]:
    """Split ``A_B`` into two known point names, where names may contain '_'.

    Prefers the split that makes both halves known names; falls back to the last
    underscore if ambiguous."""
    positions = [i for i, c in enumerate(rest) if c == "_"]
    for i in positions:
        a, b = rest[:i], rest[i + 1:]
        if a in names and b in names:
            return a, b
    if positions:
        i = positions[-1]
        return rest[:i], rest[i + 1:]
    raise FormulaError(f"cannot split pseudo-var operands {rest!r}")


def _eval_point(obj: PatternObject, objs: dict[int, PatternObject],
                meas: dict[str, float], ev: Evaluated, id_of_name: dict[str, int]) -> None:
    t = obj.tool_type

    def scope(cl: float | None = None) -> Scope:
        return _scope_for(obj, ev, meas, id_of_name, cl)

    def ref(attr: str) -> geo.Point:
        v = obj.raw.get(attr)
        if not (v and v.lstrip("-").isdigit()):
            raise _Unresolved(f"{obj.id} missing ref {attr}")
        return _point(ev, int(v))

    if t == "single":
        p = geo.Point(float(obj.raw["x"]), float(obj.raw["y"]))

    elif t == "endLine":
        base = ref("basePoint")
        angle = evaluate(obj.raw["angle"], scope())
        length = evaluate(obj.raw["length"], scope())
        p = geo.from_polar(base, angle, length)
        ev.values[obj.id] = {"angle": angle, "length": length}

    elif t == "alongLine":
        p1, p2 = ref("firstPoint"), ref("secondPoint")
        length = evaluate(obj.raw["length"], scope(p1.dist(p2)))
        p = geo.along(p1, p2, length)
        ev.values[obj.id] = {"length": length}

    elif t == "normal":
        p1, p2 = ref("firstPoint"), ref("secondPoint")
        cl = p1.dist(p2)
        base_angle = geo.line_angle(p1, p2) + 90.0
        offset = evaluate(obj.raw.get("angle", "0"), scope(cl))
        length = evaluate(obj.raw["length"], scope(cl))
        p = geo.from_polar(p1, base_angle + offset, length)
        ev.values[obj.id] = {"length": length, "angle": base_angle + offset}

    elif t == "bisector":
        p1, p2, p3 = ref("firstPoint"), ref("secondPoint"), ref("thirdPoint")
        d1 = geo.unit(p2, p1)
        d2 = geo.unit(p2, p3)
        bis = geo.Point(d1.x + d2.x, d1.y + d2.y)
        norm = (bis.x ** 2 + bis.y ** 2) ** 0.5
        if norm < 1e-12:
            raise _Unresolved("degenerate bisector")
        bis = geo.Point(bis.x / norm, bis.y / norm)
        length = evaluate(obj.raw["length"], scope(p2.dist(p1)))
        p = geo.Point(p2.x + bis.x * length, p2.y + bis.y * length)
        ev.values[obj.id] = {"length": length}

    elif t == "intersectXY":
        p1, p2 = ref("firstPoint"), ref("secondPoint")
        p = geo.Point(p1.x, p2.y)

    elif t == "lineIntersectAxis":
        base = ref("basePoint")
        p1, p2 = ref("p1Line"), ref("p2Line")
        angle = evaluate(obj.raw["angle"], scope())
        far = geo.from_polar(base, angle, 1000.0)
        hit = geo.line_intersection(base, far, p1, p2)
        if hit is None:
            raise _Unresolved("axis parallel to line")
        p = hit
        ev.values[obj.id] = {"angle": angle}

    elif t == "pointOfContact":
        center = ref("center")
        p1, p2 = ref("firstPoint"), ref("secondPoint")
        radius = evaluate(obj.raw["radius"], scope())
        pts = geo.circle_line_intersections(center, radius, p1, p2)
        if not pts:
            raise _Unresolved("circle does not meet line")
        p = min(pts, key=lambda q: q.dist(p2))  # nearest to secondPoint
        ev.values[obj.id] = {"radius": radius}

    elif t == "curveIntersectAxis":
        base = ref("basePoint")
        curve = ev.curves.get(int(obj.raw["curve"]))
        if curve is None:
            raise _Unresolved("curve not resolved")
        angle = evaluate(obj.raw["angle"], scope())
        hit = geo.polyline_axis_intersection(curve.polyline(), base, angle)
        if hit is None:
            raise _Unresolved("axis does not meet curve")
        p = hit
        ev.values[obj.id] = {"angle": angle}

    elif t == "trueDarts":
        base1, base2 = ref("baseLineP1"), ref("baseLineP2")
        d1, d2, d3 = ref("dartP1"), ref("dartP2"), ref("dartP3")
        q1, q2 = geo.true_darts(base1, base2, d1, d2, d3)
        # This element produces two *separate* output points; it has no geometry
        # of its own. Register the outputs by their own ids and return early.
        ev.points[int(obj.raw["point1"])] = q1
        ev.points[int(obj.raw["point2"])] = q2
        return

    else:
        raise _Unresolved(f"point tool {t!r} not implemented")

    ev.points[obj.id] = p


def _eval_arc(obj: PatternObject, meas: dict[str, float], ev: Evaluated,
              id_of_name: dict[str, int]) -> None:
    scope = _scope_for(obj, ev, meas, id_of_name, None)
    center_id = int(obj.raw["center"])
    center = _point(ev, center_id)
    radius = evaluate(obj.raw["radius"], scope)
    a1 = evaluate(obj.raw.get("angle1", "0"), scope)
    a2 = evaluate(obj.raw.get("angle2", "0"), scope)
    ev.curves[obj.id] = geo.Arc(center, radius, a1, a2)
    ev.values[obj.id] = {"radius": radius, "angle1": a1, "angle2": a2}


def _eval_spline(obj: PatternObject, ev: Evaluated) -> None:
    t = obj.tool_type
    if t == "cubicBezier":
        ids = [int(obj.raw[f"point{i}"]) for i in range(1, 5)]
        pts = [_point(ev, i) for i in ids]
        ev.curves[obj.id] = geo.CubicBezier(*pts)
        ev.curve_nodes[obj.id] = [ids[0], ids[3]]  # on-curve endpoints
    elif t == "cubicBezierPath":
        ids = [int(ch["pSpline"]) for ch in obj.children if "pSpline" in ch]
        pts = [_point(ev, i) for i in ids]
        ev.curves[obj.id] = geo.BezierPath(tuple(pts))
        ev.curve_nodes[obj.id] = ids[0::3]  # on-curve nodes at 0,3,6,...
    else:
        raise _Unresolved(f"spline tool {t!r} not implemented")


def _eval_operation(obj: PatternObject, ev: Evaluated) -> None:
    """Apply an operation tool, producing destination objects from sources.

    Currently mirrors *points* (flippingByLine / flippingByAxis). Rotation and
    moving, and mirroring of curves/arcs, are on the parity roadmap; unsupported
    sources are left unresolved rather than producing wrong geometry."""
    t = obj.tool_type
    if t in ("flippingByLine", "flippingByAxis"):
        p1 = ev.points.get(int(obj.raw["p1Line"]))
        p2 = ev.points.get(int(obj.raw["p2Line"]))
        if p1 is None or p2 is None:
            raise _Unresolved("mirror axis endpoints unresolved")
        for pair in obj.children:
            src, dst = pair.get("src", ""), pair.get("dst", "")
            if not (src.isdigit() and dst.isdigit()):
                continue
            src_pt = ev.points.get(int(src))
            if src_pt is not None:
                ev.points[int(dst)] = geo.reflect_point(src_pt, p1, p2)
            else:
                ev.unresolved[int(dst)] = f"operation source {src} not a resolved point"
    else:
        raise _Unresolved(f"operation {t!r} not implemented")
