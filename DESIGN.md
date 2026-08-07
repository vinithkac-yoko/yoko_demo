# seamly-engine — design

## What this is

A headless reimplementation of Seamly2D's pattern model, evaluation and file
format, plus the thing that is not in Seamly2D: a **semantically-tagged state
representation** that a program — or a model — can reason over.

The originating requirement shapes everything: an agent editing a sewing pattern
must **not reason from a screenshot alone**. It needs a structured view in which
every point, line and curve is tagged — construction or final, dart or seamline
or grainline, what it was built from, what depends on it, and the resolved value
of every formula. That representation is the reason this repository exists; the
rest is what's needed to make it true and keep it true.

### Scope boundary

| In | Out |
|---|---|
| Parse / evaluate / mutate / render / write patterns | Deciding *which* edit to make |
| The state representation | Prompts, model calls, tool schemas, training loops |
| The mutation API and its guarantees | Serving a UI or an API |
| Correctness against real Seamly2D files | Ingesting instruction documents; eval harnesses |

Consumers depend on this package; nothing here depends on them. That direction
is deliberate — this is the part that has to stay correct, and a component that
bends to suit whichever model is being tried this month stops being ground truth.

An agent-facing action space (JSON tool schemas plus a dispatcher) lived here
briefly and was moved out to the consumer that needs it. It is preserved at the
`pre-split` tag: `git show pre-split:src/seamly_engine/actions.py`.

### Why it was written from scratch

The only existing Python library, **Patro** (formerly PyValentina), is dormant
(last release 2020), implements roughly 6 of ~35 tool types, has no working
writer, and **stubs curve-length and angle math to `0`** — which real patterns
depend on heavily. It is a design reference, not a foundation.

## The file model

A `.sm2d` pattern is XML: `<pattern>` → `<increments>` (user variables) → one or
more `<draftBlock>`, each containing:

* **`<calculation>`** — the ordered **construction DAG**. `<point>` (types
  `single`, `endLine`, `alongLine`, `normal`, `bisector`, `intersectXY`,
  `pointOfContact`, `lineIntersectAxis`, `curveIntersectAxis`, `trueDarts`, …),
  `<line>`, `<arc>`, `<spline>` (`cubicBezier` / `cubicBezierPath`), and
  `<operation>` (rotation / moving / flippingByLine / flippingByAxis). Every
  object has an integer `id` and refers only to earlier ids.
* **`<modeling>`** — piece-local **copies** of calculation objects (`idObject`
  maps back), plus internal `<path>`s (darts, drill holes, guide lines).
* **`<pieces>`** — the **final cut pieces**: a seam outline whose `<nodes>`
  reference modeling ids, plus `<iPaths>`, `<grainline>`, `<anchors>`.

Measurements (`.vst` / `.smms`) are a multi-size table: each `<m>` resolves to
`base + size_increase·Δsize + height_increase·Δheight`.

Fidelity strategy in the writer: `<calculation>` is rebuilt from the object model
(the part we mutate — every object keeps its original attributes in `raw`, so
nothing is lost), while `modeling`, `pieces`, `groups` and root-level sections we
parse but never mutate are re-emitted **verbatim** from the XML captured at parse
time. Importing and re-saving does not degrade a file.

## Formulas are coupled to geometry

`length`/`angle`/`radius` are qmuparser expressions, not numbers: arithmetic,
C-style ternaries (`size>22?4.75:4`), degree trig (`sinD`, `cosD`), and three
kinds of variable — measurement names, `#`-prefixed increments, and
**pseudo-variables that read live geometry** (`Line_A_B`, `AngleLine_A_B`,
`RadiusArc_…`, `Spl_A_B`, `SplPath_A_B`, `CurrentLength`).

This is why evaluation is a single ordered pass rather than two phases: resolve
dependencies → compute geometry → expose it back into the variable namespace for
later formulas. A "literals-first" design was rejected early because you cannot
compute a point whose `angle="AngleLine_A13_A13a+90"` without real coordinates
for `A13` and `A13a`.

`formula.py` keeps a parse cache and takes a `Scope(resolve=callable)` so the
evaluator — which owns geometry — answers the pseudo-variables. `_split_two_names`
disambiguates `Line_A8_Back_C2`, where the point names themselves contain
underscores.

## The state representation

`state.export_state(pattern, evaluated)` returns JSON. Per object:

```jsonc
{
  "id": 16, "name": "A13", "kind": "point", "tool_type": "normal",
  "role": "dart",                    // anchor | seamline | dart | drill_hole |
                                     // grainline | notch | guide | curve_control |
                                     // construction | construction_line | curve
  "is_construction": false,
  "is_final_outline": true,          // derived from piece membership, not guessed
  "built_from": [14, 2],
  "dependents": [20, 21, 153],       // what breaks if deleted -> drives delete UX
  "visual": {"line_type": "dotLine", "color": "black", "weight": "0.35"},
  "piece_membership": [110],
  "geometry": {"type": "point", "x": 8.046, "y": 19.611},
  "formula": {
    "length": {"raw": "14*#CM", "resolved": 14.0, "references": ["#CM"]},
    "angle":  {"raw": "0", "resolved": 273.764}
  }
}
```

Plus pattern-level context: measurements resolved at the current size/height,
variables with formula and value, and the piece list. `compact_state` is the
token-lean variant — same tagging, without sampled curve polylines — for when
the state goes into a model's context every turn.

**Construction vs. real is decided by piece membership, never by line style.**
`lineType` is cosmetic in Seamly: a dotted line can be a real dart leg, a solid
line pure scaffolding. An object is *real* iff it (or its modeling copy) is
referenced by a piece's outline `<nodes>` or one of its internal paths. Roles are
derived the same way. Nothing here is inferred by a model.

`block_state` and `pieces.scoped_ids` narrow all of this to one garment block
plus its transitive construction drivers, which is what makes a per-turn payload
affordable.

## Mutation

`operations.PatternSession` wraps a parsed pattern and a measurement table, and
is the only supported way to change one. Every method returns an `OpResult`
rather than raising, so a caller driving the engine in a loop can read the
failure and try something else.

`add_object(tag, tool_type, attrs, children=...)` covers the whole object model:
all **21** Seamly point tools, five curve types (`arc`, `arcWithLength`,
`elArc`, `cubicBezier`, `cubicBezierPath`), lines, `trueDarts`, and the four
operation tools (`rotation`, `moving`, `flippingByLine`, `flippingByAxis`, over
points **and** curves). `create_piece` turns construction geometry into a real
cut piece; `set_variable`, `edit_object` and `delete_object` do the rest.

### Guarantees

1. **Mutations re-evaluate and roll back.** If an add or edit leaves any
   previously-resolved object unresolvable, the whole change is reverted and the
   ids that would have broken come back with the refusal. There is no
   half-applied state.
2. **Delete is block-and-report, never cascade.** Deleting an object others
   depend on is refused and the full dependent chain is returned; the caller
   decides whether to delete those first. This mirrors desktop Seamly2D and keeps
   the pattern always-valid.
3. **Ids are allocated correctly, including the awkward case.** `trueDarts`
   declares two output points inline as `point1`/`point2` attributes; those ids
   never appear as elements of their own but are referenceable, so the allocator
   has to account for them.

The trickier algorithms — `trueDarts`, `triangle`, tangent contact points — are
ports of the corresponding Seamly2D C++ tools, verified numerically in the tests.
That is also why this package is GPL-3.0: it matches upstream, removing any
derivative-work question about those two routines.

## Rendering

`render.py` produces SVG from the same state the agent sees, so image and
structure can never disagree: construction dimmed, final outlines bold, darts and
drill holes styled, final points labelled. `object_ids` scopes it to a block;
`pieces` overlays connected seam outlines so straight seam segments that aren't
standalone line objects still read; `highlight_ids` accents geometry just created.

## Verification

The suite (53 tests) runs against the **Aldrich 6th-ed. basic set** — a real
production file, not a toy.

* 425 objects parsed, **425 evaluated to finite geometry, 0 unresolved**.
* Numerically checked against the file: origin matches exactly;
  `A1→A9 = 19.00 cm` matches `waist_circ/4 + 4·#CM`.
* Writer round-trip: 425 objects, 7 pieces, 302 points, **0 coordinate
  differences**, byte-identical re-write.
* State export tags 119 final-outline vs. 341 construction objects, identifying
  darts, grainlines, drill holes and curve controls.
* The mutation guarantees tested directly: an edit that breaks dependents is
  rolled back with coordinates unchanged, an object that cannot be computed is
  never added, delete refuses with its dependent chain, a leaf deletes cleanly,
  and edits apply to the production file with 0 unresolved afterwards.

## What's missing

Honest list, roughly in order of how much it costs a real garment.

1. **Seam-allowance offsetting.** A piece stores `width` but the widened cut line
   is never computed as geometry — you get a seam line and a number, not an outer
   edge. Offsetting a closed outline of mixed lines and curves means handling
   self-intersection at concave corners.
2. **Notches and the union tool.** Neither exists. Notches matter for anything
   that gets sewn; union matters for merging pieces (e.g. princess lines).
3. **No real-Seamly2D oracle.** Everything is verified against our own arithmetic
   and a round-trip. The round-trip proves we don't *corrupt* files; it does not
   prove desktop Seamly2D agrees with our geometry. Opening generated output in
   the real application, and eventually diffing it in CI, is the highest-value
   cheap check available.
4. **Grading.** The evaluator accepts `size`/`height`, but nothing walks a size
   range and exports the set.
5. **Curve shape is under-constrained by the tools.** `cubicBezierPath` requires
   its control points to exist as real point objects. That is faithful to Seamly,
   but it means "draw a smooth curve through these four points" is a decision the
   caller has to make, not something the engine offers.
6. **Full-circle arcs.** `Arc.polyline` sweeps from `angle1` to `angle2` in the
   stored direction; equal angles give a degenerate zero-length arc rather than a
   full circle. No pattern seen so far stores one that way, and there is no
   fixture to validate a change against, so the behaviour is documented rather
   than guessed at.

## Repository conventions

* `src/` layout — tests import the installed package, so a broken packaging
  config fails loudly instead of silently working through the source tree.
* **Standard library only.** No runtime dependencies, so the engine imports
  anywhere: a notebook, a training loop, a serverless function.
* `ruff` for both linting and formatting, configured in `pyproject.toml`.
  `ruff check . && ruff format --check . && pytest` is the whole gate.
* Python 3.11+.
