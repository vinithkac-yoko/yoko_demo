"""A reward function for pattern-drafting actions.

Given the pattern **before** a step, what the reference document did, and what a
policy did instead, produce a scalar in [0, 1] plus the components it came from.
Nothing here needs a human or a model: every component is computed from the
engine's own geometry, so the signal is available for evaluation, best-of-n
selection, and rejection sampling alike.

Seven components, and the reason each exists:

``executed``
    Did the policy actually act, and did every call dispatch? Doing nothing
    scores zero here — an unchanged pattern is not a successful step.
``valid``
    Does the pattern still evaluate? An edit that leaves objects unresolvable is
    worthless however good it looks.
``nondestructive``
    Did anything that already existed move or disappear? Drafting is additive;
    silently shifting an earlier point is a serious error that geometry-only
    scoring would miss. Only credited to a policy that actually acted —
    "broke nothing" is not an achievement when you did nothing.
``placement``
    Did the new points land where the reference put them? Distance-shaped, not
    binary: within ``tolerance`` is full marks, beyond ``slack`` is zero, linear
    between. This is the component that actually measures drafting skill.
``structure``
    Did it use the same kind of tool, built from the same points? Two ways of
    reaching the same coordinate are not equally good — one of them survives a
    change of measurements.
``parametric``
    Did it write ``#chest/6`` or did it write ``6.0``? A pattern of baked
    numbers is not a pattern, it is a drawing. Scored against the reference's
    own parametricity so a step that genuinely is a constant isn't punished.
``economy``
    Did it get there without strewing scaffolding everywhere?

Weights are a starting point, not a claim — they are a plain dict, and
``dataset/evaluate.py`` exists partly so they can be tuned against outcomes you
care about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from seamly_engine.model import Pattern
from seamly_engine.operations import PatternSession

#: Tolerances are in pattern units (inches for these documents). A twentieth of
#: an inch is well inside what a drafter would call the same point; an inch out
#: is a different point entirely.
TOLERANCE = 0.05
SLACK = 1.0

WEIGHTS: dict[str, float] = {
    "executed": 0.18,
    "valid": 0.14,
    "nondestructive": 0.10,
    "placement": 0.28,
    "structure": 0.10,
    "parametric": 0.15,
    "economy": 0.05,
}

_FORMULA_ATTRS = ("length", "angle", "radius", "radius1", "radius2",
                  "angle1", "angle2", "rotationAngle")


@dataclass
class StepReward:
    total: float
    components: dict[str, float]
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"total": round(self.total, 4),
                "components": {k: round(v, 4) for k, v in self.components.items()},
                "detail": self.detail}


@dataclass
class EpisodeReward:
    total: float
    steps: list[StepReward]
    components: dict[str, float]
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"total": round(self.total, 4),
                "steps": len(self.steps),
                "components": {k: round(v, 4) for k, v in self.components.items()},
                "detail": self.detail}


# --- the step reward ---------------------------------------------------------
def score_step(before: PatternSession, reference: PatternSession,
               candidate: PatternSession | None, *,
               candidate_before: PatternSession | None = None,
               executed: bool = True, weights: dict[str, float] | None = None,
               tolerance: float = TOLERANCE, slack: float = SLACK) -> StepReward:
    """Score one drafting step.

    ``candidate`` may be ``None`` when the policy produced nothing usable; the
    components are still reported so a failure is legible rather than just zero.

    ``candidate_before`` matters in rollout mode, where the policy is carrying
    its own drifting pattern forward: each side's *delta* is then measured
    against its own prior state, so the comparison stays meaningful after the
    two patterns have diverged. Left unset, both come from ``before``
    (teacher forcing).
    """
    weights = weights or WEIGHTS
    cand_before = candidate_before or before
    pre_ids = _ids(cand_before.pattern)
    ref_new = _new_objects(before, reference)
    cand_new = _new_objects(cand_before, candidate) if candidate else {}

    # Acting is the point: an unchanged pattern is not a successful step. But
    # some steps legitimately build no geometry (declaring a variable, flipping
    # a traced block), so "did something" means objects *or* variables, and a
    # step the reference itself left empty is satisfied by leaving it empty.
    acted = bool(cand_new) or _changed_variables(cand_before, candidate)
    expected = bool(ref_new) or _changed_variables(before, reference)
    did_the_work = acted or not expected

    components = {
        "executed": 1.0 if (executed and candidate is not None and did_the_work) else 0.0,
        "valid": _valid(cand_before, candidate, did_the_work),
        "nondestructive": (_nondestructive(cand_before, candidate, pre_ids)
                           if did_the_work else 0.0),
        "placement": _placement(reference, candidate, ref_new, cand_new,
                                tolerance, slack),
        "structure": _structure(reference, candidate, ref_new, cand_new),
        "parametric": _parametric(reference, candidate, ref_new, cand_new),
        "economy": _economy(ref_new, cand_new),
    }
    total = sum(weights.get(k, 0.0) * v for k, v in components.items())
    detail = {
        "reference_objects": len(ref_new),
        "candidate_objects": len(cand_new),
        "reference_names": sorted(_names(reference.pattern, ref_new)),
        "candidate_names": sorted(_names(candidate.pattern, cand_new)) if candidate else [],
    }
    return StepReward(total=total, components=components, detail=detail)


def score_episode(steps: list[StepReward], *,
                  reference: PatternSession | None = None,
                  candidate: PatternSession | None = None,
                  final_weight: float = 0.2) -> EpisodeReward:
    """Mean step reward, plus how the finished patterns compare.

    The end state only means something when the policy carried its own pattern
    forward (rollout). Under teacher forcing every step starts from the
    reference's state, so the final patterns are near-identical whatever the
    policy did — pass ``final_weight=0`` there, which is what
    :func:`dataset.evaluate.run_episode` does.
    """
    if not steps:
        return EpisodeReward(0.0, [], {k: 0.0 for k in WEIGHTS})
    keys = steps[0].components.keys()
    components = {k: sum(s.components[k] for s in steps) / len(steps) for k in keys}
    total = sum(s.total for s in steps) / len(steps)
    detail: dict = {}
    if reference is not None and candidate is not None:
        detail["final"] = compare_patterns(reference, candidate)
        if final_weight:
            total = (1 - final_weight) * total + final_weight * detail["final"]["score"]
    return EpisodeReward(total=total, steps=steps, components=components, detail=detail)


def compare_patterns(reference: PatternSession, candidate: PatternSession, *,
                     tolerance: float = TOLERANCE) -> dict:
    """How close two finished patterns are, by named point."""
    ref_pts = _named_points(reference)
    cand_pts = _named_points(candidate)
    if not ref_pts:
        return {"score": 1.0, "matched": 0, "missing": [], "mean_error": 0.0}
    matched, errors, missing = 0, [], []
    for name, p in ref_pts.items():
        q = cand_pts.get(name)
        if q is None:
            missing.append(name)
            continue
        d = _dist(p, q)
        errors.append(d)
        if d <= tolerance:
            matched += 1
    coverage = matched / len(ref_pts)
    resolved = 1.0 if not candidate.evaluated.unresolved else 0.0
    return {
        "score": round(0.8 * coverage + 0.2 * resolved, 4),
        "matched": matched,
        "of": len(ref_pts),
        "missing": sorted(missing)[:12],
        "mean_error": round(sum(errors) / len(errors), 4) if errors else None,
    }


# --- components --------------------------------------------------------------
def _valid(before: PatternSession, candidate: PatternSession | None,
           did_the_work: bool = True) -> float:
    """Validity is only credit-worthy if the step was actually taken; an
    untouched pattern is trivially valid and must not be paid for it."""
    if candidate is None or not did_the_work:
        return 0.0
    newly = set(candidate.evaluated.unresolved) - set(before.evaluated.unresolved)
    return 0.0 if newly else 1.0


def _nondestructive(before: PatternSession, candidate: PatternSession | None,
                    pre_ids: set[int]) -> float:
    """Everything that existed before must still be there, unmoved."""
    if candidate is None:
        return 0.0
    if not pre_ids:
        return 1.0
    kept = 0
    for oid in pre_ids:
        was = before.evaluated.points.get(oid)
        now = candidate.evaluated.points.get(oid)
        if oid not in _ids(candidate.pattern):
            continue
        if was is None or now is None:
            kept += 1 if (was is None and now is None) else 0
        elif _dist(was, now) <= 1e-6:
            kept += 1
    return kept / len(pre_ids)


def _placement(reference: PatternSession, candidate: PatternSession | None,
               ref_new: dict[int, object], cand_new: dict[int, object],
               tolerance: float, slack: float) -> float:
    """Distance-shaped agreement between the new points, matched by name first.


    Name-first matching matters: a policy that names its point ``C1`` like the
    document did is claiming to have built *that* point, and should be scored
    against it even if it landed somewhere else. Unnamed or renamed points fall
    back to nearest-neighbour so a policy isn't punished for its labelling.
    """
    ref_pts = {oid: reference.evaluated.points[oid] for oid in ref_new
               if oid in reference.evaluated.points}
    if not ref_pts:
        # The step built no points (a variable, a flip). Credit only a policy
        # that also refrained from inventing geometry.
        return 1.0 if (candidate is not None and not cand_new) else 0.0
    if candidate is None or not cand_new:
        return 0.0

    cand_pts = {oid: candidate.evaluated.points[oid] for oid in cand_new
                if oid in candidate.evaluated.points}
    if not cand_pts:
        return 0.0
    ref_names = _names(reference.pattern, ref_new)
    cand_by_name = {n: oid for oid, n in _names(candidate.pattern, cand_new).items() if n}

    scores, used = [], set()
    for oid, p in ref_pts.items():
        name = ref_names.get(oid)
        match = cand_by_name.get(name) if name else None
        if match is None or match not in cand_pts:
            free = [(cid, _dist(p, q)) for cid, q in cand_pts.items() if cid not in used]
            if not free:
                scores.append(0.0)
                continue
            match = min(free, key=lambda kv: kv[1])[0]
        used.add(match)
        scores.append(_shape(_dist(p, cand_pts[match]), tolerance, slack))
    return sum(scores) / len(scores)


def _structure(reference: PatternSession, candidate: PatternSession | None,
               ref_new: dict, cand_new: dict) -> float:
    """Same kinds of tool, built from the same named points."""
    if candidate is None:
        return 0.0
    if not ref_new:
        return 1.0 if not cand_new else 0.0
    if not cand_new:
        return 0.0
    ref_kinds = _counter((o.tag, o.tool_type) for o in ref_new.values())
    cand_kinds = _counter((o.tag, o.tool_type) for o in cand_new.values())
    kinds = _overlap(ref_kinds, cand_kinds)

    ref_deps = {n for o in ref_new.values() for n in _dep_names(reference.pattern, o)}
    cand_deps = {n for o in cand_new.values() for n in _dep_names(candidate.pattern, o)}
    deps = len(ref_deps & cand_deps) / len(ref_deps | cand_deps) if (ref_deps | cand_deps) else 1.0
    return 0.6 * kinds + 0.4 * deps


def _parametric(reference: PatternSession, candidate: PatternSession | None,
                ref_new: dict, cand_new: dict) -> float:  # noqa: D401
    """Fraction of formulas that reference something, not just a number.

    Measured relative to the reference: if the document's own step is a literal
    (*"1 inch seam allowance"*), a literal is the right answer.
    """
    if candidate is None or (ref_new and not cand_new):
        return 0.0
    ref_frac = _symbolic_fraction(ref_new.values())
    cand_frac = _symbolic_fraction(cand_new.values())
    if ref_frac is None:
        return 1.0
    if cand_frac is None:
        return 0.0
    return 1.0 if cand_frac >= ref_frac - 1e-9 else cand_frac / ref_frac


def _economy(ref_new: dict, cand_new: dict) -> float:
    """Mild penalty for building far more than the step needed."""
    if not cand_new:
        return 1.0 if not ref_new else 0.0

    ratio = len(cand_new) / max(len(ref_new), 1)
    if ratio <= 1.5:
        return 1.0
    return max(0.0, 1.0 - (ratio - 1.5) / 3.0)


# --- helpers -----------------------------------------------------------------
def _ids(pattern: Pattern) -> set[int]:
    return {o.id for o in pattern.all_objects()}


def _new_objects(before: PatternSession, after: PatternSession | None) -> dict[int, object]:
    if after is None:
        return {}
    pre = _ids(before.pattern)
    return {o.id: o for o in after.pattern.all_objects() if o.id not in pre}


def _names(pattern: Pattern, objs: dict[int, object]) -> dict[int, str]:
    return {oid: (o.raw.get("name") or "") for oid, o in objs.items()}


def _changed_variables(before: PatternSession, after: PatternSession | None) -> bool:
    if after is None:
        return False
    was = before.evaluated.increment_values
    now = after.evaluated.increment_values
    return any(k not in was or was[k] != v for k, v in now.items())


def _named_points(session: PatternSession) -> dict[str, object]:
    out = {}
    for o in session.pattern.all_objects():
        name = o.raw.get("name")
        pt = session.evaluated.points.get(o.id)
        if name and pt is not None:
            out[name] = pt
    return out


def _dep_names(pattern: Pattern, obj) -> set[str]:
    by_id = pattern.object_by_id()
    return {by_id[r].raw.get("name", str(r)) for r in obj.refs if r in by_id}


def _symbolic_fraction(objs) -> float | None:
    total = symbolic = 0
    for o in objs:
        for attr in _FORMULA_ATTRS:
            raw = o.raw.get(attr)
            if raw is None:
                continue
            total += 1
            if any(c.isalpha() or c == "#" for c in str(raw)):
                symbolic += 1
    return None if total == 0 else symbolic / total


def _counter(items) -> dict:
    out: dict = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def _overlap(a: dict, b: dict) -> float:
    if not a and not b:
        return 1.0
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))
    return inter / union if union else 1.0


def _shape(distance: float, tolerance: float, slack: float) -> float:
    if distance <= tolerance:
        return 1.0
    if distance >= slack:
        return 0.0
    return 1.0 - (distance - tolerance) / (slack - tolerance)


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
