# Pattern Drafting Workspace

Type an instruction — *"split the bodice front along the princess line from
B4 to the hem"*, *"merge the yoke back into the body"*, *"let out the waist
2cm"* — and a model carries it out against a real, headless
Seamly2D-compatible pattern engine. Every edit is computed geometry, not a
guess: points, curves, and whole pieces (split, merge, create, edit, delete)
all go through the same mutation API that re-evaluates and rolls back if the
result would be invalid.

The workspace is a desktop build log, not a chat: you type one instruction,
it runs to completion, and what comes back is the ordered list of tool calls
the model made — each one showing the reasoning behind it and a before/after
render when you click it. Every completed instruction is a save point you can
branch from.

## Layout

```
src/seamly_engine/   the engine — parse, evaluate, mutate, render, write
                      patterns. Standard library only; see its own docs.
backend/              FastAPI app, the agent loop, SQLite persistence
app/                  the desktop workspace (single-file, no build step)
tests/                engine tests (98 total; backend/tests/ has the rest)
```

See **[DESIGN.md](DESIGN.md)** for the full architecture — the agent loop,
the action log and reasoning capture, the version/run/action persistence
model, and (in its second half) the engine internals: the file format,
the formula language, the state representation, and the mutation
guarantees.

## Quick start

```bash
pip install -e '.[dev]'          # the engine
pip install -r requirements.txt  # the backend (FastAPI, Anthropic SDK, cairosvg)

export ANTHROPIC_API_KEY=sk-ant-...   # enables the agent; the app still runs without it
PYTHONPATH=src uvicorn app:app --app-dir backend --port 8000
# open http://localhost:8000
```

Without `ANTHROPIC_API_KEY` the engine, state export, render, save/export,
and version branching are all fully usable — running an instruction just
comes back with a plain notice instead of calling the model.

## Tests

```bash
pytest tests backend/tests -q       # 98 passing
ruff check . && ruff format --check .
```

## What you can do

Everything Seamly2D's construction tools support (all 21 point tools, 5 curve
types, the 4 transform operations, true darts, variables), plus the full
piece lifecycle: create a piece from construction geometry, **split** one
piece into two along a straight cut, **merge** two pieces along a seam they
share, edit a piece's seam allowance/width/name, or delete one — all through
typed instructions, and all through the same API a script could call
directly (`seamly_engine.operations.PatternSession`).

Known gaps — seam-allowance offset geometry, notches, a real-Seamly2D
oracle, grading, curved split edges — are listed honestly in
[DESIGN.md](DESIGN.md#whats-missing).

## Deploy (Railway)

The repo is Railway-ready (Nixpacks):

* `requirements.txt` (root) has the backend's PyPI deps; `Procfile` /
  `nixpacks.toml` start `uvicorn` bound to `$PORT`. The engine
  (`src/seamly_engine`) is **not** pip-installed during the build — Nixpacks
  copies only `requirements.txt` in its install phase, so an editable install
  of the repo root would fail there — it goes on `PYTHONPATH` at start
  instead (`nixpacks.toml`, and `backend/app.py` inserts it itself too, as a
  fallback for when Railway doesn't apply the inline env prefix).
* `nixpacks.toml` installs `cairo` so the pattern rasterizes for the vision
  input, and pins the install phase to `pip install -r requirements.txt`
  explicitly — the repo root now also has a `pyproject.toml` (the engine's
  own package manifest), and pinning the install command keeps Nixpacks'
  Python-project auto-detection from picking that instead and skipping
  FastAPI/the Anthropic SDK.
* Set the `ANTHROPIC_API_KEY` variable in the Railway service.

### Persisting your patterns

Saved patterns, versions, and run history live in SQLite at `VLA_DB` (default
`./data/vla.db`). **Railway's container filesystem is ephemeral**, so to keep
history across deploys, attach a Volume and point the DB at it:

1. Railway service → **Volumes** → add a volume mounted at `/data`.
2. Set the variable **`VLA_DB=/data/vla.db`**.

Without a volume the app still works, but the library resets on each deploy.

### Cost controls (Railway variables)

Pick the model per session in the app (header dropdown) — cheap for simple
tweaks, strong for hard drafting (split/merge lean more on spatial reasoning
than a single-point edit, so the default favours a model with thinking on):

| Model | Cost /1M (in-out) | When to use |
|---|---|---|
| **Sonnet 5** (default) | $3/$15 | Balanced — adaptive thinking, near-Opus on agentic/tool work |
| **Opus 4.8** | $5/$25 | Hardest construction/drafting |
| **Haiku 4.5** | $1/$5 | Cheapest, but **no thinking** — simple edits only |

| Variable | Default | Effect |
|---|---|---|
| `VLA_MODEL` | `claude-sonnet-5` | Default model for *new* sessions (the picker overrides per session). |
| `VLA_VISION_WIDTH` | `900` | Width of the PNG sent to the model. Image tokens scale with pixel area — lower is cheaper. |
| `VLA_SEND_IMAGE` | `1` | Set to `0` to run **state-only** (no image) for the cheapest turns; the structured state is the primary input regardless. |
| `VLA_MAX_STEPS` | `12` | Max tool-call rounds per instruction — caps the worst-case cost of one run. |

Per-turn payload is also kept small by sending only the objects a tool call
actually touched back to the model, rather than the whole pattern state every
step.
