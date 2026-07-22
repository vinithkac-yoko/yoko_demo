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
