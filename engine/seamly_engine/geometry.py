"""Geometry kernel.

All coordinates are in the pattern's own unit (cm for the sample files) and use
Seamly2D's screen convention: **x grows right, y grows DOWN**. Angles are in
degrees, measured counter-clockwise in the *visual* sense, which — because y is
flipped — means a point at distance ``L`` and angle ``a`` from base ``b`` is::

    x = b.x + L*cos(a)
    y = b.y - L*sin(a)

so angle 0 points right, 90 points up on screen, 270 points down. This matches
Seamly's ``VPointF`` / line-angle behaviour and is the single source of truth
for every tool that consumes an angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, o: "Point") -> "Point":
        return Point(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Point") -> "Point":
        return Point(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> "Point":
        return Point(self.x * k, self.y * k)

    __rmul__ = __mul__

    def dist(self, o: "Point") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


def from_polar(base: Point, angle_deg: float, length: float) -> Point:
    """Point at ``length``/``angle_deg`` from ``base`` (screen convention)."""
    a = math.radians(angle_deg)
    return Point(base.x + length * math.cos(a), base.y - length * math.sin(a))


def line_angle(p1: Point, p2: Point) -> float:
    """Visual angle in degrees of the directed line p1->p2, in [0, 360)."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    # y is flipped, so negate dy to get the visual angle.
    a = math.degrees(math.atan2(-dy, dx))
    return a % 360.0


def unit(p1: Point, p2: Point) -> Point:
    d = p1.dist(p2)
    if d == 0:
        return Point(0.0, 0.0)
    return Point((p2.x - p1.x) / d, (p2.y - p1.y) / d)


def along(p1: Point, p2: Point, length: float) -> Point:
    """Point at signed ``length`` from ``p1`` toward ``p2``."""
    return p1 + unit(p1, p2) * length


def line_intersection(a1: Point, a2: Point, b1: Point, b2: Point) -> Point | None:
    """Intersection of the infinite lines a1a2 and b1b2, or None if parallel."""
    x1, y1 = a1.x, a1.y
    x2, y2 = a2.x, a2.y
    x3, y3 = b1.x, b1.y
    x4, y4 = b2.x, b2.y
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return Point(x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def circle_line_intersections(center: Point, radius: float, p1: Point, p2: Point) -> list[Point]:
    """Intersections of the infinite line p1p2 with a circle."""
    d = p2 - p1
    fx, fy = p1.x - center.x, p1.y - center.y
    a = d.x * d.x + d.y * d.y
    if a == 0:
        return []
    b = 2 * (fx * d.x + fy * d.y)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    pts = [Point(p1.x + t1 * d.x, p1.y + t1 * d.y)]
    if abs(t1 - t2) > 1e-12:
        pts.append(Point(p1.x + t2 * d.x, p1.y + t2 * d.y))
    return pts


# --- arcs --------------------------------------------------------------------
@dataclass(frozen=True)
class Arc:
    center: Point
    radius: float
    angle1: float  # start angle, degrees (screen convention)
    angle2: float  # end angle, degrees

    def point_at(self, angle_deg: float) -> Point:
        return from_polar(self.center, angle_deg, self.radius)

    def polyline(self, steps: int = 64) -> list[Point]:
        a1, a2 = self.angle1, self.angle2
        # Sweep the short way that respects the stored ordering; full circle if equal.
        if abs((a2 - a1) % 360.0) < 1e-9 and a2 != a1:
            span = a2 - a1
        else:
            span = (a2 - a1)
        pts = []
        for i in range(steps + 1):
            pts.append(self.point_at(a1 + span * i / steps))
        return pts

    def length(self) -> float:
        span = abs(self.angle2 - self.angle1)
        return math.radians(span) * self.radius


# --- cubic bezier ------------------------------------------------------------
@dataclass(frozen=True)
class CubicBezier:
    p0: Point
    p1: Point
    p2: Point
    p3: Point

    def point_at(self, t: float) -> Point:
        mt = 1 - t
        a = mt * mt * mt
        b = 3 * mt * mt * t
        c = 3 * mt * t * t
        d = t * t * t
        return Point(
            a * self.p0.x + b * self.p1.x + c * self.p2.x + d * self.p3.x,
            a * self.p0.y + b * self.p1.y + c * self.p2.y + d * self.p3.y,
        )

    def polyline(self, steps: int = 32) -> list[Point]:
        return [self.point_at(i / steps) for i in range(steps + 1)]

    def length(self, steps: int = 64) -> float:
        pts = self.polyline(steps)
        return sum(pts[i].dist(pts[i + 1]) for i in range(len(pts) - 1))


@dataclass(frozen=True)
class BezierPath:
    """A chain of cubic segments. ``on_and_controls`` is the flattened Seamly
    ``cubicBezierPath`` point list: on-curve, ctrl, ctrl, on-curve, ctrl, ctrl,
    on-curve, ... Consecutive segments share their joining on-curve point."""

    on_and_controls: tuple[Point, ...]

    def segments(self) -> list[CubicBezier]:
        pts = self.on_and_controls
        segs = []
        i = 0
        while i + 3 < len(pts):
            segs.append(CubicBezier(pts[i], pts[i + 1], pts[i + 2], pts[i + 3]))
            i += 3
        return segs

    def polyline(self, steps: int = 24) -> list[Point]:
        out: list[Point] = []
        for seg in self.segments():
            poly = seg.polyline(steps)
            out.extend(poly if not out else poly[1:])
        return out

    def length(self) -> float:
        return sum(seg.length() for seg in self.segments())


@dataclass(frozen=True)
class EllipticalArc:
    center: Point
    radius1: float  # semi-axis along the (rotated) x
    radius2: float  # semi-axis along the (rotated) y
    angle1: float
    angle2: float
    rotation: float = 0.0

    def point_at(self, angle_deg: float) -> Point:
        a = math.radians(angle_deg)
        # ellipse point before rotation (screen convention: y down)
        ex = self.radius1 * math.cos(a)
        ey = -self.radius2 * math.sin(a)
        rot = math.radians(-self.rotation)
        rx = ex * math.cos(rot) - ey * math.sin(rot)
        ry = ex * math.sin(rot) + ey * math.cos(rot)
        return Point(self.center.x + rx, self.center.y + ry)

    def polyline(self, steps: int = 64) -> list[Point]:
        span = self.angle2 - self.angle1
        return [self.point_at(self.angle1 + span * i / steps) for i in range(steps + 1)]

    def length(self) -> float:
        pts = self.polyline()
        return sum(pts[i].dist(pts[i + 1]) for i in range(len(pts) - 1))


def point_at_arclength(poly: list[Point], s: float) -> Point:
    """Point at arc length ``s`` along a polyline (used by the cut tools)."""
    if not poly:
        raise ValueError("empty polyline")
    if s <= 0:
        return poly[0]
    acc = 0.0
    for i in range(len(poly) - 1):
        seg = poly[i].dist(poly[i + 1])
        if acc + seg >= s:
            t = (s - acc) / seg if seg else 0.0
            return Point(poly[i].x + t * (poly[i + 1].x - poly[i].x),
                         poly[i].y + t * (poly[i + 1].y - poly[i].y))
        acc += seg
    return poly[-1]


def polyline_axis_intersection(poly: list[Point], base: Point, angle_deg: float) -> Point | None:
    """First intersection of a polyline with the infinite axis through ``base``
    at ``angle_deg`` (used by curveIntersectAxis). Returns the intersection
    nearest to ``base``."""
    axis_far = from_polar(base, angle_deg, 1e6)
    axis_near = from_polar(base, angle_deg, -1e6)
    best: Point | None = None
    best_d = math.inf
    for i in range(len(poly) - 1):
        hit = _segment_intersection(poly[i], poly[i + 1], axis_near, axis_far)
        if hit is not None:
            d = hit.dist(base)
            if d < best_d:
                best, best_d = hit, d
    return best


def foot_of_perpendicular(p: Point, a: Point, b: Point) -> Point:
    """Foot of the perpendicular from ``p`` onto the infinite line a-b (the
    'height' tool)."""
    dx, dy = b.x - a.x, b.y - a.y
    denom = dx * dx + dy * dy
    if denom == 0:
        return a
    t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / denom
    return Point(a.x + t * dx, a.y + t * dy)


def circle_circle_intersections(c1: Point, r1: float, c2: Point, r2: float) -> list[Point]:
    d = c1.dist(c2)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h2 = r1 * r1 - a * a
    h = math.sqrt(h2) if h2 > 0 else 0.0
    xm = c1.x + a * (c2.x - c1.x) / d
    ym = c1.y + a * (c2.y - c1.y) / d
    rx = -(c2.y - c1.y) * (h / d)
    ry = (c2.x - c1.x) * (h / d)
    if h == 0:
        return [Point(xm, ym)]
    return [Point(xm + rx, ym + ry), Point(xm - rx, ym - ry)]


def contact_points(p: Point, center: Point, radius: float) -> list[Point]:
    """Tangent points on a circle from an external point ``p``.

    Port of Seamly's ``VGObject::ContactPoints`` — the tangent line from ``p``
    touches the circle where the radius is perpendicular to it, which puts the
    touch points on a circle of diameter ``|p-center|`` (Thales). Used by the
    point-from-arc/circle-and-tangent tools.
    """
    d = p.dist(center)
    if d < radius:          # inside the circle: no tangent
        return []
    if abs(d - radius) < 1e-12:
        return [p]          # on the circle: it is its own tangent point
    mid = Point((p.x + center.x) / 2.0, (p.y + center.y) / 2.0)
    return circle_circle_intersections(mid, d / 2.0, center, radius)


def triangle_point(axis_p1: Point, axis_p2: Point, first: Point, second: Point) -> Point:
    """Port of Seamly's ``VToolTriangle::FindPoint``.

    Walks along the axis from where it crosses the hypotenuse (first-second)
    until the triangle becomes right-angled — i.e. until ``c² <= a² + b²``.
    """
    start = line_intersection(axis_p1, axis_p2, first, second)
    if start is None:
        raise ValueError("triangle: axis is parallel to the hypotenuse")
    c = math.floor(first.dist(second))
    angle = line_angle(axis_p1, axis_p2)
    step = 1.0
    length = 0.0
    for _ in range(100000):                      # bounded; Seamly loops unbounded
        length += step
        candidate = from_polar(start, angle, length)
        a = math.floor(candidate.dist(first))
        b = math.floor(candidate.dist(second))
        if c * c <= a * a + b * b:
            return candidate
    raise ValueError("triangle: no right-angle point found")


def polyline_intersections(a: list[Point], b: list[Point]) -> list[Point]:
    """All crossing points between two polylines (curve-curve intersection)."""
    out: list[Point] = []
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            hit = _segments_cross(a[i], a[i + 1], b[j], b[j + 1])
            if hit is not None:
                out.append(hit)
    return out


def _segments_cross(a1: Point, a2: Point, b1: Point, b2: Point) -> Point | None:
    r = a2 - a1
    s = b2 - b1
    denom = r.x * s.y - r.y * s.x
    if abs(denom) < 1e-12:
        return None
    qp = b1 - a1
    t = (qp.x * s.y - qp.y * s.x) / denom
    u = (qp.x * r.y - qp.y * r.x) / denom
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return Point(a1.x + t * r.x, a1.y + t * r.y)
    return None


def pick_cross(pts: list[Point], vertical: str = "1", horizontal: str = "1") -> Point:
    """Seamly's V/H cross-point selection: 1 = highest/leftmost, 2 = lowest/rightmost."""
    if not pts:
        raise ValueError("no intersection")
    # y grows downward, so "highest" is the smallest y
    ys = sorted(pts, key=lambda p: p.y)
    chosen = ys[0] if str(vertical) == "1" else ys[-1]
    same = [p for p in pts if abs(p.y - chosen.y) < 1e-9]
    if len(same) > 1:
        xs = sorted(same, key=lambda p: p.x)
        chosen = xs[0] if str(horizontal) == "1" else xs[-1]
    return chosen


def rotate_point(p: Point, center: Point, angle_deg: float) -> Point:
    """Rotate ``p`` about ``center`` by ``angle_deg`` (visual CCW, screen coords)."""
    r = center.dist(p)
    if r == 0:
        return p
    return from_polar(center, line_angle(center, p) + angle_deg, r)


def reflect_point(p: Point, a: Point, b: Point) -> Point:
    """Mirror point ``p`` across the infinite line through ``a`` and ``b``."""
    dx, dy = b.x - a.x, b.y - a.y
    denom = dx * dx + dy * dy
    if denom == 0:
        return p
    t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / denom
    foot = Point(a.x + t * dx, a.y + t * dy)
    return Point(2 * foot.x - p.x, 2 * foot.y - p.y)


def true_darts(base_p1: Point, base_p2: Point, dart_p1: Point, dart_p2: Point,
               dart_p3: Point) -> tuple[Point, Point]:
    """Port of Seamly2D ``VToolTrueDarts::FindPoint``.

    Given a base line (seam the dart interrupts) and the three dart points
    (dart_p1/left, dart_p2/apex, dart_p3/right), returns the two "true" dart
    leg points that keep the base line continuous once the dart is folded
    closed. Qt's ``QLineF`` angle/setAngle convention is the same screen
    convention this module uses, so the port is direct.
    """
    # degrees = d2d3.angleTo(d2d1)  == angle(d2->d1) - angle(d2->d3), in [0,360)
    degrees = (line_angle(dart_p2, dart_p1) - line_angle(dart_p2, dart_p3)) % 360.0

    # Rotate the line d2->baseP2 by +degrees, keeping length; take its endpoint.
    len_d2blp2 = dart_p2.dist(base_p2)
    ang_d2blp2 = line_angle(dart_p2, base_p2) + degrees
    d2blp2_end = from_polar(dart_p2, ang_d2blp2, len_d2blp2)

    # p1 = intersection of line(baseP1, d2blp2_end) with line d2->d1.
    p1 = line_intersection(base_p1, d2blp2_end, dart_p2, dart_p1)
    if p1 is None:
        raise ValueError("true darts: base line parallel to dart leg")

    # p2 = endpoint of line d2->p1 rotated by -degrees, keeping length.
    len_d2p1 = dart_p2.dist(p1)
    ang_d2p1 = line_angle(dart_p2, p1) - degrees
    p2 = from_polar(dart_p2, ang_d2p1, len_d2p1)
    return p1, p2


def _segment_intersection(a1: Point, a2: Point, b1: Point, b2: Point) -> Point | None:
    """Intersection point if segment a1a2 crosses infinite line b1b2 within a1a2."""
    r = a2 - a1
    s = b2 - b1
    denom = r.x * s.y - r.y * s.x
    if abs(denom) < 1e-12:
        return None
    qp = b1 - a1
    t = (qp.x * s.y - qp.y * s.x) / denom
    if -1e-9 <= t <= 1 + 1e-9:
        return Point(a1.x + t * r.x, a1.y + t * r.y)
    return None
