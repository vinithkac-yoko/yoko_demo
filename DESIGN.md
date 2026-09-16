# Pattern Drafting Workspace — design

## What this is

A system for turning typed instructions into sewing-pattern edits — split a
piece, merge two pieces along a seam, let out a dart, reshape a garment — with
every edit computed by a real headless Seamly2D-compatible engine, not
guessed at by a model. Three layers, each with a clean job:

```
app/index.html          desktop workspace: instruction bar + action log
        │  HTTP
backend/app.py           FastAPI: patterns, versions, sessions, runs
backend/agent.py         the loop: instruction -> ordered tool calls
backend/store.py         SQLite: the branching version DAG, run/action history
        │  PatternSession
src/seamly_engine/       the engine (see "The engine" below)
```

The dependency runs one way: the engine has no idea the backend or the UI
exist. That boundary used to be a separate repository; it's a single
repository again (kept together on purpose, rather than split across
several), but the boundary itself is still real and still enforced — nothing
in `src/seamly_engine/` imports from `backend/`.

## Why a typed instruction, not a chat

The workspace is not a conversation. You type one instruction — *"split the
bodice front along the princess line from B4 to the hem"* — it runs to
completion, and what comes back is a **build log**: the ordered list of real
tool calls the model made, each one inspectable. There is no back-and-forth
within a turn, and no chat history to scroll through — the action log *is*
the record of what happened.

Two things follow from that:

* **Every action carries its reasoning.** Adaptive thinking is on for models
  that support it; the thinking text that preceded a tool call becomes that
  action's `reasoning`. Clicking an action in the log shows it alongside a
  rendered **before/after** of the pattern (the geometry that action touched,
  highlighted) and the raw tool call.
* **A completed instruction is a save point.** Every run that changed
  anything auto-saves a new pattern version, and "branch from here" on any
  past run opens a fresh session starting from what that run produced. This
  reuses the same version DAG that already handled branching — a run doesn't
  need its own branching structure.

## The agent loop (`backend/agent.py`)

One call to `run_instruction(session, instruction, ...)` drives a
`PatternSession` (see below) through the Anthropic Messages API tool-use loop:
the model receives the structured pattern state (the primary input) and,
when a raster backend is available, a rendered image, then calls tools until
the instruction is done or it hits the step limit. Every tool call becomes
one `Action`:

```python
Action(step_index=0, tool_name="split_piece",
       tool_input={"piece_id": 538, "point_a_id": 12, "point_b_id": 47, ...},
       reasoning="B4 and the hem point already bound the princess line...",
       ok=True, message="split into “Front Panel” (#812) and “Side Panel” (#813)",
       touched_ids=[812, 813, ...])
```

**13 tools**, mirroring the engine's mutation API one-to-one: the construction
set (`add_point` — all 21 point tools, `add_line`, `add_curve`, `add_dart`,
`add_operation`, `add_variable`, `edit_object`, `delete_object`) plus the
**piece** set (`create_piece`, `split_piece`, `merge_piece`, `edit_piece`,
`delete_piece`). `dispatch_tool` maps a call onto the matching
`PatternSession` method; it never raises, so a malformed call comes back as a
failed `Action` the model can read and correct from, same as every other
mutation.

Reasoning attribution: when one thinking block precedes several tool calls in
the same turn, they share its text. Splitting it per call would invent a
granularity the model's own deliberation never had.

## Persistence (`backend/store.py`)

```
patterns  →  versions (parent_id DAG)  ←  sessions.version_id
                    ↑ version_after_id
                 runs (one per instruction)  →  actions (one per tool call)
```

`versions` is unchanged from the very first version of this project: an
immutable XML snapshot per save, `parent_id` making it a DAG rather than a
line, so opening an old version and saving branches instead of overwriting.

What's new is `runs` + `actions`, replacing what used to be a threaded chat
table. A `run` names the version it started from and the version it produced;
its `actions` are thin by design — no per-action pattern snapshot is stored.
A before/after view is instead **derived by replay**: rebuild the run's
starting `PatternSession`, re-dispatch its stored actions in order up to the
requested step. This is safe to repeat (a failed action rolls back identically
on replay) and exact (ids are assigned the same way — highest existing id + 1
— from the same starting pattern through the same sequence of prior actions,
so a stored tool call's integer ids resolve identically every time).

## The desktop workspace (`app/index.html`)

Single-file, no build step, desktop-only by design — the earlier phone-first
direction was explicitly dropped in favour of a bigger canvas and an
inspectable action log, which doesn't fit a phone screen. Library (patterns
list, new/import) → workspace (resizable pattern canvas + instruction bar +
action log) → click an action for its detail modal (reasoning, before/after
render, raw diff) → versions modal for the DAG and branching.

---

# The engine (`src/seamly_engine/`)

## What this is

A headless reimplementation of Seamly2D's pattern model, evaluation and file
format, plus the thing that is not in Seamly2D: a **semantically-tagged state
representation** that a program — or a model — can reason over.

The originating requirement shapes everything: an agent editing a sewing
pattern must **not reason from a screenshot alone**. It needs a structured
view in which every point, line and curve is tagged — construction or final,
dart or seamline or grainline, what it was built from, what depends on it,
and the resolved value of every formula. That representation is the reason
this package exists; the rest is what's needed to make it true and keep it
true.

### Scope boundary

| In | Out |
|---|---|
| Parse / evaluate / mutate / render / write patterns | Deciding *which* edit to make |
| The state representation | Prompts, model calls, tool schemas |
| The mutation API and its guarantees | Sessions, persistence, HTTP, the UI |
| Correctness against real Seamly2D files | Ingesting instruction documents; eval harnesses |

`backend/` depends on `src/seamly_engine/`; nothing in `src/seamly_engine/`
imports from `backend/`. That direction is deliberate — this is the part that
has to stay correct, and a component that bends to suit whichever model is
being tried this month stops being ground truth. It's also what would let the
engine be published or reused on its own without dragging the app along.

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
affordable. Block-scoping is optional now rather than a forced first step —
`select_block` narrows the view when you want it, but split/merge routinely
need to see more than one piece at once, so the default is the whole pattern.

## Mutation

`operations.PatternSession` wraps a parsed pattern and a measurement table, and
is the only supported way to change one. Every method returns an `OpResult`
rather than raising, so a caller driving the engine in a loop can read the
failure and try something else.

`add_object(tag, tool_type, attrs, children=...)` covers the whole construction
object model: all **21** Seamly point tools, five curve types (`arc`,
`arcWithLength`, `elArc`, `cubicBezier`, `cubicBezierPath`), lines, `trueDarts`,
and the four operation tools (`rotation`, `moving`, `flippingByLine`,
`flippingByAxis`, over points **and** curves). `set_variable`, `edit_object`
and `delete_object` do the rest of construction editing.

Pieces get the full complement too, not just creation:

* **`create_piece`** turns construction geometry into a real cut piece
  (modeling copies + a `<piece>`, with seam allowance, internal paths, a
  grainline).
* **`split_piece`** cuts one piece into two along the straight line between
  two of its outline points. Each point is either an existing vertex or one
  you've just built that lands exactly on one of the piece's *straight*
  edges — it gets spliced into the outline as part of the split, so a real
  yoke or princess cut doesn't need the vertex to have pre-existed. A point
  on a curved edge isn't accepted (splitting the curve object itself is a
  bigger operation this doesn't attempt). Internal paths that straddle the
  cut are dropped and named in the result rather than silently misassigned.
* **`merge_piece`** is close to split's exact inverse: name the shared edge
  by its two endpoints, and the identical construction-id sequence between
  them (in either winding direction) is dissolved, stitching the remaining
  two arcs into one outline. Refuses cleanly if the two pieces don't actually
  reference the same points along that edge.
* **`edit_piece`** / **`delete_piece`** rename, reconfigure (seam allowance,
  width) or remove a piece. Deleting a piece never cascades to its
  construction geometry — a piece is a view onto construction objects, not
  their owner.

### Guarantees

1. **Mutations re-evaluate and roll back.** If an add or edit leaves any
   previously-resolved object unresolvable, the whole change is reverted and the
   ids that would have broken come back with the refusal. There is no
   half-applied state. Piece operations validate the same way — a split or
   merge whose result doesn't resolve to a real outline rolls back too.
2. **Delete is block-and-report, never cascade.** Deleting a construction object
   others depend on is refused and the full dependent chain is returned; the
   caller decides whether to delete those first. This mirrors desktop Seamly2D
   and keeps the pattern always-valid.
3. **Ids are allocated correctly, including the awkward cases.** `trueDarts`
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
standalone line objects still read; `highlight_ids` accents geometry just created
or touched by one action — this is what the action-detail before/after view
in the UI is built from.

## Verification

The suite (98 tests: engine + backend) runs against the **Aldrich 6th-ed. basic
set** — a real production file, not a toy.

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
* **Split/merge round-tripped against a real piece**: splitting the Skirt Back
  (16 outline nodes, 5 internal paths — two darts, a reference line, two
  drill-hole arcs) and merging the two halves back recovers geometry
  identical to the original, all five internal paths intact.
* The agent loop and API tested with a fake model client (no network): the
  action log's shape, reasoning attribution across multi-tool-call turns, and
  the replay-based action-detail endpoint against a real multi-step run.

## What's missing

Honest list, roughly in order of how much it costs a real garment.

1. **Seam-allowance offsetting.** A piece stores `width` but the widened cut line
   is never computed as geometry — you get a seam line and a number, not an outer
   edge. Offsetting a closed outline of mixed lines and curves means handling
   self-intersection at concave corners.
2. **Notches.** Don't exist yet — they matter for anything that gets sewn.
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
7. **Splitting along a curved edge.** `split_piece` only accepts endpoints on
   straight edges (existing vertices, or a fresh point that lands exactly on
   one). A cut that needs to start or end partway along an existing curve
   would mean splitting the curve object itself — not attempted.

## Repository conventions

* `src/` layout for the engine — tests import the installed package, so a
  broken packaging config fails loudly instead of silently working through
  the source tree. `backend/` isn't an installed package; it puts `src/` on
  `PYTHONPATH` itself (`app.py`) so it's importable without a separate install
  step, matching how it runs on Railway.
* **The engine has zero runtime dependencies** (standard library only), so it
  imports anywhere: a notebook, a training loop, a serverless function. The
  backend's dependencies (FastAPI, the Anthropic SDK, cairosvg) are isolated
  to `requirements.txt` / `backend/requirements.txt`.
* `ruff` for both linting and formatting, configured in `pyproject.toml`.
  `ruff check . && ruff format --check . && pytest tests backend/tests` is
  the whole gate.
* Python 3.11+.
