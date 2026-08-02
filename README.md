# seamly-engine

A **headless, Seamly2D-compatible sewing-pattern engine** in Python, with a
state representation built for machine agents.

This repository is the **environment**. It parses, evaluates, mutates, renders
and writes parametric sewing patterns, and it exposes a small, closed action
space that any policy — a prompted model, a fine-tuned one, a replayed
instruction document, a person — can drive. What those policies *are* lives
elsewhere; this repo is the world they act in, and its job is to stay correct.

## What makes it a usable environment

Three things, and they map to the three parts of any agent contract.

**Observation — `state.py`.** Not a screenshot and not raw XML. Every object is
semantically tagged: what tool made it, what it is *for* (seamline, dart,
grainline, drill hole, construction), whether it belongs to a real cut piece,
what it was built from, what depends on it, and each formula in both its raw and
its resolved form.

```jsonc
{"id": 386, "name": "C1", "kind": "point", "tool_type": "endLine",
 "role": "seamline", "construction": false, "final_outline": true,
 "built_from": [377], "dependents": [387, 390, 397, 467],
 "xy": [10.79375, 2.55833],
 "formula": {"length": {"raw": "1.5*#CM", "value": 1.5},
             "angle":  {"raw": "270", "value": 270.0}}}
```

Construction-vs-real is **derived from piece membership, never guessed**.
Seamly's `lineType` is cosmetic — a dotted line can be a real dart leg and a
solid one pure scaffolding — so an object is real iff a piece's outline or one
of its internal paths references it.

**Action space — `actions.py`.** Nine tools covering the Seamly object model:
`add_point` (all 21 point tools), `add_line`, `add_curve`, `add_dart`,
`add_operation`, `create_piece`, `add_variable`, `edit_object`, `delete_object`.
`TOOLS` is the JSON schema a model is shown; `dispatch_tool` is the transition
function.

**Transition guarantees — `operations.py`.** Every mutation re-evaluates the
whole pattern and **rolls back** if the result can't be computed. Delete is
**block-and-report**: deleting something others depend on is refused and the
dependent chain comes back, so the pattern is never silently broken. Dispatch
**never raises** — a malformed call returns a readable reason a policy can
correct from.

## Formulas are not numbers

`length`/`angle`/`radius` are expressions in Seamly's qmuparser grammar:
arithmetic, C-style ternaries (`size>22?4.75:4`), degree trig (`cosD`), and
three kinds of variable — measurement names, `#`-prefixed increments, and
**pseudo-variables that read live geometry**:

| Pseudo-variable | Resolves to |
|---|---|
| `Line_A_B` | current distance between points *A* and *B* |
| `AngleLine_A_B` | that segment's visual angle |
| `RadiusArc_…` | an arc's radius |
| `Spl_A_B` / `SplPath_A_B` | arc length along a spline between two points on it |
| `CurrentLength` | the current tool's natural base length |

Because these read geometry, formula evaluation is **coupled to the DAG
evaluator**: resolve dependencies → compute geometry → expose it back into the
namespace for later formulas. You cannot compute a point whose angle is
`AngleLine_A13_A13a+90` without real coordinates for `A13`.

## Layout

```
seamly_engine/
  model.py         Pattern / DraftBlock / PatternObject / Piece / Evaluated
  parser.py        .sm2d XML -> object model (lossless; keeps every raw attribute)
  writer.py        object model -> .sm2d (byte-identical round-trip)
  measurements.py  .vst / .smms multi-size tables
  formula.py       qmuparser-compatible expression engine
  geometry.py      points, arcs, cubic Beziers, Bezier paths, elliptical arcs
  evaluator.py     construction-DAG evaluation
  operations.py    PatternSession — the mutation API, with rollback
  actions.py       the action space: TOOLS + dispatch_tool
  state.py         the semantically-tagged state representation
  pieces.py        piece grouping, outlines, piece construction
  render.py        SVG (construction dimmed, final bold)
  authoring.py     blank-canvas pattern creation
tests/             57 tests + real fixtures (Aldrich 6th-ed. basic set)
```

## Status

Proven on a real production file — the Aldrich 6th-ed. basic set (skirt,
trousers, bodice, one-piece sleeve):

* Parses all **425 objects** and the multi-size `.vst`.
* Evaluates **425 / 425 to finite geometry, 0 unresolved** — including
  `trueDarts` (ported from the Seamly C++ source), `flippingByLine`,
  `SplPath` arc-length-along-path, `pointOfContact`, `curveIntersectAxis`,
  `bisector`.
* Verified numerically against the file: `A1→A9 = 19.00 cm`, matching
  `waist_circ/4 + 4·#CM`.
* **Lossless writer round-trip**: 425 objects, 7 pieces, 302 points, 0
  coordinate differences, byte-identical re-write.
* All 21 Seamly point tools, 5 curve types, 4 operation tools, and piece
  creation are implemented and exercised through the action space.

Known gaps are in [DESIGN.md](DESIGN.md#whats-missing).

## Quick start

```bash
pip install -e '.[dev]'
pytest -q                                  # 57 passing
```

Read a pattern and inspect the state a policy would see:

```python
import seamly_engine as se

pat  = se.load_pattern("tests/fixtures/aldrich_basic.sm2d")
meas = se.load_measurements("tests/fixtures/aldrich_measurements.vst")
ev   = se.evaluate_pattern(pat, meas)

print(se.export_state(pat, ev, meas)["coverage"])
# {'total': 425, 'resolved': 425, 'fraction': 1.0, ...}
```

Drive it as an environment:

```python
from seamly_engine import TOOLS, dispatch_tool
from seamly_engine.authoring import new_pattern
from seamly_engine.operations import PatternSession
from seamly_engine.state import compact_state
from seamly_engine.writer import pattern_to_xml

session = PatternSession(new_pattern("Skirt block", with_defaults=False))

observation = compact_state(session.pattern, session.evaluated)   # what a policy sees
result = dispatch_tool(session, "add_point", {                    # what it does
    "tool_type": "endLine",
    "attrs": {"name": "B", "basePoint": "1", "angle": "270", "length": "60"},
})
print(result.ok, result.message)                                  # -> True 'added endLine B (#2)'

open("skirt.sm2d", "w").write(pattern_to_xml(session.pattern))    # opens in Seamly2D
```

`TOOLS` is already in Anthropic tool-use schema form, so wiring a model to it is
a `messages.create(tools=TOOLS, ...)` call plus a loop over the `tool_use`
blocks.

## Scope

**In:** parsing, formulas, geometry, evaluation, mutation, the action space,
state export, rendering, writing.

**Out:** anything that chooses *which* action to take, and anything that serves
a UI. Policies, prompts, training, instruction-document ingestion and evaluation
harnesses live in their own repositories and depend on this one. The dependency
runs one way, on purpose: this is the part that has to stay correct, and it
can't be allowed to drift to suit a model.
