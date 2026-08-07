# seamly-engine

A **headless, Seamly2D-compatible sewing-pattern engine** in Python. It reads
`.sm2d` / `.val` pattern files, resolves their formulas against a measurement
table, evaluates the construction graph to real geometry, lets you edit that
geometry safely, and writes the result back out.

Standard library only. No Qt, no GUI, no network.

```bash
pip install -e '.[dev]'
pytest                      # 53 tests, against a real production pattern
```

## A pattern is a program, not a drawing

Nothing in a `.sm2d` file says "a line from (3.2, 8.1) to (9.7, 8.1)". It says
*"point `A13` is `14*#CM` along the normal to `A2`→`A12`"*. Coordinates are
**outputs**, recomputed whenever a measurement, a variable, or an upstream point
changes.

In the 425-object test fixture, exactly one coordinate pair is stored — the
origin. Everything else is derived:

```xml
<point id="1"  type="single"    name="A"   x="0.79375" y="1.05833"/>
<point id="2"  type="endLine"   name="A1"  basePoint="1" angle="270" length="5*#CM"/>
<point id="16" type="normal"    name="A13" firstPoint="14" secondPoint="2"
       angle="0" length="14*#CM"/>
```

That single fact explains the rest of the design: the strict id ordering, the
formula language, the three-layer piece model, and why deleting a point is a
graph operation rather than a delete.

## Formulas are coupled to geometry

`length` / `angle` / `radius` are expressions in Seamly's qmuparser grammar —
arithmetic, C-style ternaries (`size>22?4.75:4`), degree trig (`cosD`) — over
three kinds of variable: measurement names, `#`-prefixed increments, and
**pseudo-variables that read live geometry**:

| Pseudo-variable | Resolves to |
|---|---|
| `Line_A_B` | current distance between the points named *A* and *B* |
| `AngleLine_A_B` | that segment's on-screen angle |
| `RadiusArc_<id>` | an arc's current radius |
| `Spl_A_B` / `SplPath_A_B` | arc length along a spline between two points on it |
| `CurrentLength` | the current tool's natural base length |

Because those read geometry, evaluation is a single interleaved pass — resolve
dependencies, compute geometry, publish it back into the namespace for later
formulas. You cannot compute a point whose `angle="AngleLine_A13_A13a+90"`
without real coordinates for `A13`.

## Coordinates

**x rightward, y downward**, but angles are **counter-clockwise**. So `0` is
right, `90` is **up**, `180` is left, `270` is **down**. Getting this backwards
produces a pattern that evaluates perfectly and is mirrored vertically.

## The state representation

`export_state()` returns a semantically-tagged view of the whole pattern — the
reason this engine exists, and the part that is not in Seamly2D:

```jsonc
{"id": 386, "name": "C1", "kind": "point", "tool_type": "endLine",
 "role": "seamline", "is_construction": false, "is_final_outline": true,
 "built_from": [377], "dependents": [387, 390, 397, 467],
 "geometry": {"type": "point", "x": 10.79375, "y": 2.55833},
 "formula": {"length": {"raw": "1.5*#CM", "resolved": 1.5},
             "angle":  {"raw": "270", "resolved": 270.0}}}
```

**Construction-vs-real is derived from piece membership, never guessed.**
Seamly's `lineType` is cosmetic — a dotted line can be a real dart leg and a
solid one pure scaffolding — so an object counts as part of the finished garment
only if some piece's outline or internal path references it. On the fixture that
split is 119 final-outline against 341 construction.

`compact_state()` is the same tagging without sampled curve polylines, for when
the state has to fit in a context window.

## Editing

`PatternSession` is the mutation API, and it makes two guarantees:

**Mutations re-evaluate and roll back.** If an add or edit leaves any
previously-resolved object unresolvable, the whole change is reverted and the
ids that would have broken come back with the refusal.

**Delete is block-and-report, never cascade.** Deleting an object others depend
on is refused, with the full dependent chain returned; the caller decides
whether to remove those first. This mirrors desktop Seamly2D and keeps the
pattern always-valid.

```python
import seamly_engine as se
from seamly_engine import PatternSession
from seamly_engine.authoring import new_pattern
from seamly_engine.writer import pattern_to_xml

session = PatternSession(new_pattern("Skirt block", with_defaults=False))

res = session.add_object("point", "endLine",
                         {"name": "B", "basePoint": "1", "angle": "270", "length": "60"})
print(res.ok, res.message)          # True  'added endLine B (#2)'

res = session.delete_object(1)      # the origin everything hangs off
print(res.ok, res.blocked_by)       # False [2]

open("skirt.sm2d", "w").write(pattern_to_xml(session.pattern))   # opens in Seamly2D
```

Reading an existing pattern:

```python
pattern      = se.load_pattern("tests/fixtures/aldrich_basic.sm2d")
measurements = se.load_measurements("tests/fixtures/aldrich_measurements.vst")
evaluated    = se.evaluate_pattern(pattern, measurements)

se.export_state(pattern, evaluated, measurements)["coverage"]
# {'total': 425, 'resolved': 425, 'fraction': 1.0}
```

## What's implemented

| | |
|---|---|
| **Point tools** | all 21 — `single`, `endLine`, `alongLine`, `normal`, `bisector`, `intersectXY`, `lineIntersect`, `height`, `shoulder`, `triangle`, `pointOfContact`, `lineIntersectAxis`, `curveIntersectAxis`, `cutSpline`, `cutArc`, `cutSplinePath`, `pointOfIntersectionCircles`, `pointOfIntersectionArcs`, `pointOfIntersectionCurves`, `pointFromCircleAndTangent`, `pointFromArcAndTangent`, plus `trueDarts` |
| **Curves** | `arc`, `arcWithLength`, `elArc`, `cubicBezier`, `cubicBezierPath` |
| **Operations** | `rotation`, `moving`, `flippingByLine`, `flippingByAxis` — over points **and** curves |
| **Pieces** | grouping, outline resolution, and construction (modeling copies + `<piece>`, with seam allowance, internal paths and a grainline) |
| **I/O** | lossless read, byte-identical write, `.vst`/`.smms` multi-size measurement tables |
| **Output** | SVG render, construction dimmed and final outlines bold |

## Verification

The suite runs against the **Aldrich 6th-ed. basic set** — a real production
file (skirt, trousers, bodice, one-piece sleeve), not a toy.

* 425 objects parsed, **425 evaluated to finite geometry, 0 unresolved** —
  including `trueDarts`, `flippingByLine`, `SplPath` arc-length-along-path,
  `pointOfContact`, `curveIntersectAxis` and `bisector`.
* Checked numerically against the file: the origin matches exactly, and
  `A1→A9 = 19.00 cm` matches `waist_circ/4 + 4·#CM`.
* Writer round-trip: 425 objects, 7 pieces, 302 points, **0 coordinate
  differences**, byte-identical re-write.
* The mutation guarantees are tested directly: rollback on a breaking edit,
  refusal-with-dependents on delete, and edits applied to the production file.

## What's missing

Honest list, in [DESIGN.md](DESIGN.md#whats-missing). The short version: seam
allowance is stored as a width but the widened cut line is not computed; there
are no notches and no union tool; nothing has yet been opened in desktop
Seamly2D to confirm it agrees with our geometry; and there is no size-range
export.

## Scope

**In:** parsing, formulas, geometry, evaluation, mutation, the state
representation, rendering, writing.

**Out:** anything that decides *which* edit to make, and anything that serves a
UI. Applications and agents depend on this package; nothing here depends on
them. That direction is deliberate — this is the part that has to stay correct,
and it must not drift to suit a consumer.

An agent-facing action space (JSON tool schemas plus a dispatcher over
`PatternSession`) used to live here and was moved out. It is preserved at the
`pre-split` tag:

```bash
git show pre-split:src/seamly_engine/actions.py
```

## Licence and relationship to Seamly2D

**GPL-3.0-or-later** — see [LICENSE](LICENSE).

This is an independent reimplementation of the Seamly2D file format and
evaluation model, written from the format itself. Two geometry routines
(`true_darts`, `triangle_point`) are ports of the corresponding Seamly2D C++
tools, which is why the licence matches upstream's.

It is **not affiliated with or endorsed by the Seamly2D project**. "Seamly2D" is
the name of that project; this package is named for the format it reads.

The test fixture is the Aldrich 6th-edition basic block set, a Seamly2D
community sample.
