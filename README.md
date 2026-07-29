# VLA Pattern Drafting

A system where an AI **VLA (vision-language-action) model** performs **instructed
edits** on a base sewing pattern inside a **headless, Seamly2D-compatible
engine**. The model reasons over both a **rendered image** and a **structured,
semantically-tagged representation** of the pattern state — every point, line,
and curve is tagged (construction vs. final, dart vs. seamline vs. grainline),
with its formula, its resolved value, what it's built from, and what depends on
it. The phone is a thin chat/control client; the engine and agent run
server-side.

See **[DESIGN.md](DESIGN.md)** for the full architecture, the state-representation
schema, and the roadmap.

## Layout

```
engine/     Headless Seamly2D engine (Python) — the core
  seamly_engine/
    parser.py        .sm2d XML → object model
    measurements.py  .vst/.smms multi-size table + grading
    formula.py       qmuparser-compatible expression engine
    geometry.py      points, lines, arcs, cubic Bézier + paths
    evaluator.py     DAG evaluation (formulas read live geometry)
    operations.py    edit / delete (block + report dependents)
    state.py         the VLA state representation (semantic tagging)
    render.py        SVG (construction dimmed, final bold)
  tests/             pytest suite + real fixtures (Aldrich basic set)
backend/    FastAPI + Anthropic tool-use agent loop (vision + per-operation tools)
app/        Mobile-responsive PWA chat client
```

## Status

The **vertical-slice engine is complete and proven on a real production
pattern** — the Aldrich 6th-ed. basic set (skirt, trousers, bodice, one-piece
sleeve, 425 objects): parses the file, runs the full formula engine, and
evaluates **all 425 objects to finite geometry with 0 unresolved**, including
`trueDarts`, `flippingByLine`, `SplPath` arc-length-along-path, `pointOfContact`,
`curveIntersectAxis`, and `bisector`. Delete/edit operations, the semantic state
export, the SVG render, and the backend + PWA skeletons are in place. The agent
model call and full tool-parity are the next roadmap items.

## Quick start

```bash
# Engine + tests
cd engine
pip install -e '.[dev]'
pytest -q                      # 21 passing

# Evaluate + inspect a pattern
python -c "
import seamly_engine as se
pat = se.load_pattern('tests/fixtures/aldrich_basic.sm2d')
meas = se.load_measurements('tests/fixtures/aldrich_measurements.vst')
ev  = se.evaluate_pattern(pat, meas)
state = se.export_state(pat, ev, meas)
print(state['coverage'])         # {'total': 425, 'resolved': 425, 'fraction': 1.0, ...}
"

# Backend + PWA  (from the repo root)
pip install -r requirements.txt              # PyPI deps (engine runs from PYTHONPATH)
export ANTHROPIC_API_KEY=sk-ant-...          # enables the VLA agent loop
PYTHONPATH=engine uvicorn app:app --app-dir backend --port 8000
# open http://localhost:8000 on your phone/browser
```

Without `ANTHROPIC_API_KEY` the engine, state export, render, and edit/delete
paths are still fully exercisable; the chat just returns a stub. The agent uses
`claude-opus-4-8` with adaptive thinking, reasoning over the **structured state**
(primary) plus the **rendered image** (when a raster backend is present), and
calls one tool per operation (`edit_formula`, `delete_object`, …).

## Deploy on Railway

The repo is Railway-ready (Nixpacks). Push it to a Railway service:

- `requirements.txt` (root) installs the engine + backend; `Procfile` / `nixpacks.toml`
  start `uvicorn` bound to `$PORT`.
- `nixpacks.toml` installs `cairo` so the pattern rasterizes for the vision input.
  If cairo is ever unavailable the agent falls back to state-only reasoning, so
  the app still runs.
- Set the `ANTHROPIC_API_KEY` variable in the Railway service.

### Cost controls (Railway variables)

| Variable | Default | Effect |
|---|---|---|
| `VLA_MODEL` | `claude-haiku-4-5` | Cheapest capable model ($1/$5 per 1M). Use `claude-sonnet-5` ($3/$15) or `claude-opus-4-8` ($5/$25) for more capability. |
| `VLA_VISION_WIDTH` | `700` | Width of the PNG sent to the model. Image tokens scale with pixel area — lower is cheaper. |
| `VLA_SEND_IMAGE` | `1` | Set to `0` to run **state-only** (no image at all) for the cheapest turns; the structured state is the primary input regardless. |
| `VLA_MAX_STEPS` | `8` | Max tool-call rounds per turn — caps the worst-case cost of one message. |

Per-turn payload is also kept small by sending only the *changed objects* after
each tool call rather than the whole block state.

The PWA is served at `/` and talks to the same origin, so no separate frontend
deploy is needed.
