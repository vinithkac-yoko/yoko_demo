# VLA Pattern-Drafting System — Design

## Context

We are building a system where an AI model that behaves like a **VLA
(vision-language-action) model** takes a **base sewing pattern** and performs
**instructed edits** on it ("let out the waist 2cm", "add a bust dart", "raise
the neckline"). The environment is **Seamly2D**, an open-source parametric
pattern-drafting CAD tool.

The key requirement that shapes everything: the model must **not reason from a
screenshot alone**. It also consumes a **structured representation of the
pattern state** in which *every point, line, and curve is semantically tagged* —
construction vs. final, dart vs. seamline vs. grainline, what each object is
built from, what depends on it, and the resolved value of every formula. That
representation is what lets the model understand and safely manipulate the
current state.

Everything is designed for a **phone**: the phone is a thin chat/control client;
the engine and agent run server-side.

### Decisions locked in (from requirements discussion)

| Decision | Choice |
|---|---|
| What the model does | **Instructed edits** on an existing base pattern |
| Model approach | **Prompted multimodal agent** (Claude) via the **Claude Agent SDK** |
| Action execution | **Direct file manipulation** via our own **headless Seamly2D engine** (all ops incl. delete) |
| Engine language | **Python** |
| Phone client | **Mobile-responsive web app / PWA** |
| Operation scope | **Full Seamly2D parity** (sequenced, see roadmap) |
| Formula engine | **Full qmuparser-compatible engine** from the start |
| Ground truth | **Validate against real Seamly2D** as an oracle |
| Session model | **Multi-turn persistent** editing session |
| Delete semantics | **Block + report dependents** (never silently cascade) |
| Build sequence | Engine parity and phone UI **in parallel** |

### Why we built the engine from scratch

The only existing Python library (**Patro**, formerly PyValentina) is dormant
(last release 2020), implements ~6 of ~35 tool types, has no working writer, and
its formula engine **stubs curve-length/angle math to `0`** — which real
patterns depend on heavily. It is a design reference, not a foundation.

## Architecture

```
Phone PWA (chat + live pattern view)            app/
        │  HTTPS / WebSocket
FastAPI backend ── Claude Agent SDK loop         backend/
        │            (one tool per Seamly2D operation)
Headless engine (Python)                         engine/seamly_engine/
  parser.py      .sm2d XML  → object model (lossless, keeps raw attrs)
  measurements.py .vst/.smms → graded measurement table
  formula.py     qmuparser-compatible expression engine
  geometry.py    points, lines, arcs, cubic Bézier + Bézier paths
  evaluator.py   topological DAG eval; formulas read *live* geometry
  operations.py  add / edit / DELETE-with-dependents  (roadmap: mutation API)
  state.py       the VLA state representation (semantic tagging)
  render.py      SVG (construction dimmed, final bold) — image for VLA + phone
  writer.py      object model → .sm2d XML  (roadmap)
```

### The Seamly2D file model (what we reimplement)

A `.sm2d` pattern is XML: `<pattern>` → `<increments>` (user variables) → one or
more `<draftBlock>`, each containing:

* `<calculation>` — the ordered **construction DAG**: `<point>` (types
  `single`, `endLine`, `alongLine`, `normal`, `bisector`, `intersectXY`,
  `pointOfContact`, `lineIntersectAxis`, `curveIntersectAxis`, `trueDarts`, …),
  `<line>`, `<arc>`, `<spline>` (`cubicBezier` / `cubicBezierPath`), and
  `<operation>` (rotation / moving / flippingByLine / flippingByAxis). Every
  object has an integer `id` and refers only to earlier ids.
* `<modeling>` — piece-local **copies** of calculation objects (`idObject` maps
  back), plus internal `<path>`s (darts, drill holes, guide lines).
* `<pieces>` — the **final pattern pieces**: a main seam/cut outline whose
  `<nodes>` reference modeling ids, plus `<iPaths>`, `<grainline>`, `<anchors>`.

Measurements (`.vst`/`.smms`) are a multi-size table: each `<m>` is
`base + size_increase·Δsize + height_increase·Δheight`.

### Formulas are not numbers

`length`/`angle`/`radius` are **expressions** in qmuparser grammar:
arithmetic, C-style ternary (`size>22?4.75:4`), degree trig (`cosD`), and three
kinds of variable — measurement names, `#`-prefixed increments, and
**pseudo-variables that read live geometry**:

* `Line_A_B` — length of the segment between points *A* and *B*
* `AngleLine_A_B` — visual angle of that segment
* `RadiusArc_A_id` — an arc's radius
* `SplPath_A_B` / `Spl_A_B` — arc length along a spline between two points on it
* `CurrentLength` — the current tool's natural base length

Because these read geometry, **formula evaluation is coupled to the DAG
evaluator**: resolve dependencies → compute geometry → expose it back into the
variable namespace for later formulas. (This is exactly why "literals-first" was
rejected: you cannot compute point `A13c`'s `angle="AngleLine_A13_A13a+90"`
without real coordinates for `A13`/`A13a`.)

## The VLA state representation

`state.export_state(pattern, evaluated)` returns JSON. Per object:

```jsonc
{
  "id": 16, "name": "A13", "kind": "point", "tool_type": "normal",
  "role": "dart",                    // anchor | seamline | dart | drill_hole |
                                     // grainline | notch | guide | curve_control |
                                     // construction | construction_line | curve
  "is_construction": false,
  "is_final_outline": true,          // ∈ a piece's cut/sew outline (derived, not guessed)
  "built_from": [14, 2],             // DAG deps
  "dependents": [20, 21, 153],       // what breaks if deleted → drives delete UX
  "visual": {"line_type": "dotLine", "color": "black", "weight": "0.35"},
  "piece_membership": [110],
  "geometry": {"type": "point", "x": 8.046, "y": 19.611},
  "formula": {
    "length": {"raw": "14*#CM", "resolved": 14.0, "references": ["#CM"]},
    "angle":  {"raw": "0", "resolved": 273.764}
  }
}
```

Plus pattern-level context: measurements (resolved at current size/height),
variables (formula + value), and the piece list. **Construction vs. real is
decided by piece membership, not line style.** `lineType` (solid/dotted/dashed)
is cosmetic in Seamly — a dotted line can be a real dart leg, a solid line can be
pure scaffolding. An object is *real* iff it (or its modeling copy) is referenced
by a piece's outline `<nodes>` or one of its internal paths (darts/drill-holes/
grainline); everything else is construction. Roles are derived the same way,
never inferred by the model.

### Block-scoped workflow

The pattern is organized into **blocks** (garments: Skirt / Trousers / Bodice /
Sleeve). Piece names encode this — `"A - Skirt Back"` / `"A - Skirt Front"` are
the front/back pieces of block `A`; `pieces.group_pieces` groups them. The UI
asks which block to work on first, then shows *only that block* in the rich
construction view (all lines + points, construction dimmed, final bold), scoped
to the block's objects + their construction drivers (`scoped_ids`). The connected
seam outline of each piece is overlaid bold (`render_svg(pieces=…)`) so straight
seam segments that aren't standalone line objects still read. The agent's state
and edits are scoped to the block (`block_state`), cutting each turn from ~114KB
to ~17–27KB and preventing edits leaking into another block.

`render.py` produces the matching SVG (construction dimmed, final bold, darts and
drill-holes styled, final points labeled) — the image the VLA sees and the phone
shows, guaranteed consistent with the structured state.

## Delete semantics

Delete is **block + report dependents**, mirroring real Seamly2D and keeping the
pattern always-valid: deleting an object that others depend on is refused and the
full dependent chain is returned to the model, which then decides whether to
delete those first. `dependents` in the state export is precisely this chain.

## Agent loop (backend)

The Claude Agent SDK drives a multi-turn session. Each turn the model receives
the rendered SVG/PNG **and** the structured state, plus the user instruction, and
calls **one tool per Seamly2D operation** (`add_point_endline`,
`add_point_alongline`, `edit_formula`, `delete_object`, `add_dart`, …). Each tool
mutates the in-memory pattern via the engine, which re-evaluates, auto-validates,
and returns the new state + render. Invalid results roll back.

## Current status (this milestone)

The **vertical-slice engine is complete and proven on a real production file**
(the Aldrich 6th-ed. basic set: skirt, trousers, bodice, one-piece sleeve):

* Parses the full `.sm2d` (425 objects) and the multi-size `.vst`.
* Full formula engine (ternary, degree trig, increments, all pseudo-variables).
* DAG evaluator computes **all 425 objects to finite geometry, 0 unresolved**,
  including `trueDarts` (ported from Seamly source), `flippingByLine`,
  `SplPath` arc-length-along-path, `pointOfContact`, `curveIntersectAxis`,
  `bisector`.
* Validated numerically: origin point matches the file exactly; `A1→A9 = 19.0cm`
  matches `waist_circ/4 + 4·#CM`.
* State exporter tags every object (119 final-outline vs. 341 construction; darts,
  grainlines, drill-holes, curve controls identified).
* SVG renderer produces a legible construction-and-pieces view.
* 17 passing tests (`engine/tests/test_engine.py`).

## Action set (the agent's tools = the Seamly object model)

`add_point`, `add_line`, `add_curve`, `add_dart`, `add_operation`,
`create_piece`, `add_variable`, `edit_object` (any attribute), `delete_object`
(block-and-report). Every mutation assigns ids, re-evaluates, and **rolls back**
if the result can't be computed or breaks the pattern, so the agent can build
geometry safely.

**Pieces** (`create_piece`): turns construction geometry into a real cut piece —
Seamly's two-layer form (a `<modeling>` copy per object + a `<piece>` whose
`<nodes>` reference them in outline order), with seam allowance, internal paths
(darts/guides) and a grainline. Created pieces render, export, and reopen; the
writer appends them without disturbing pieces that came from an imported file.

**Point tools — all 21 of Seamly's are supported:** `single`, `endLine`,
`alongLine`, `normal`, `bisector`, `intersectXY`, `lineIntersect`, `height`,
`shoulder`, `triangle`, `pointOfContact`, `lineIntersectAxis`,
`curveIntersectAxis`, `cutSpline`, `cutArc`, `cutSplinePath`,
`pointOfIntersectionCircles`, `pointOfIntersectionArcs`,
`pointOfIntersectionCurves`, `pointFromCircleAndTangent`,
`pointFromArcAndTangent`, plus `trueDarts` via `add_dart`.

**Curves:** `arc`, `arcWithLength`, `elArc`, `cubicBezier`, `cubicBezierPath`.

**Operations** (`add_operation`): `flippingByLine`, `flippingByAxis`, `rotation`,
`moving` — each transforms a set of source objects into copies, and works on
**points *and* curves** (Bézier, Bézier path, arc).

The trickier algorithms (`trueDarts`, `triangle`, tangent points) are ports of
the corresponding Seamly2D C++ tools, verified numerically in the tests.

## Instruction documents → patterns and datasets

A pattern-making school's handout (prose + hand-drawn figures) can be turned into
a working parametric pattern *and* into evaluation data for the agent. `dataset/`
does both: a closed ten-verb IR for extracted drafting steps, a normalizer that
reports precisely what the prose leaves undetermined, and a compiler that
executes each step as **real agent tool calls** on a live `PatternSession` — so a
step only enters the dataset if the engine actually built its geometry.

The first document (Angrakha Maxi, 4 panels) compiles 27 of 31 steps into a
101-object pattern with 0 unresolved objects. See `docs/DATASETS.md`.

## Roadmap

1. **Remaining piece features** — seam-allowance *offsetting* (the widened cut
   line is stored as a width, not yet computed as geometry), notches, and the
   union tool.
2. **Real-Seamly2D oracle** — automated cross-check (open engine output in the
   Seamly2D CLI, diff geometry) in CI.
3. **Grading** — evaluate/export across the size range.
```
