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
backend/    FastAPI + Claude Agent SDK loop (per-operation tools)
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

# Backend + PWA
cd ../backend
pip install -r requirements.txt
uvicorn app:app --port 8000      # open http://localhost:8000 on your phone/browser
```

Set `ANTHROPIC_API_KEY` to enable the agent loop; without it the engine, state
export, render, and edit/delete paths are still fully exercisable.
