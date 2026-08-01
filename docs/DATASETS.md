# Instruction documents → patterns and training data

## The question

*"I will just give you the PDFs. Will you be able to interpret them and turn
them into Seamly2D actions? And build a reward system around it?"*

**Yes to both.** The input is the PDF a pattern-making school hands out — prose
plus hand-drawn figures. The output is a working parametric pattern, a dataset
of verified drafting actions, and a scalar reward that can score any policy
against the document without a human in the loop.

Not training data in the naive sense, though: one document is ~30 steps, nowhere
near enough to train anything, and fine-tuning isn't the bottleneck. What these
files buy you first is an **eval** — the thing the project most lacks.

## The pipeline

```
   PDF ──▶ extract.py ──▶ actions.jsonl ──▶ normalize.py ──▶ compile.py ──▶ .sm2d
            (vision)        (the IR)         (find gaps)      (execute)     dataset
                                                                  │           report
                                                                  ▼
                                                            evaluate.py ──▶ reward
```

Four commands, each usable on its own:

```bash
python -m dataset.extract  Angrakha_Maxi.pdf --garment angrakha_maxi
python -m dataset.build    dataset/sources/angrakha_maxi.jsonl -o build/
python -m dataset.evaluate dataset/sources/angrakha_maxi.jsonl --policy agent
python -m dataset.evaluate dataset/sources/angrakha_maxi.jsonl --policy agent --rollout
```

## Reading the PDF

`dataset/extract.py` splits each page into its text and its embedded figures
(dropping the letterhead that repeats on every page), sends both to the model,
and gets back rows in the IR. Two things make it work rather than merely run:

* **The extraction contract is generated from `schema.py`**, so the prompt
  cannot drift from what the compiler accepts. It also pins the angle convention
  — degrees counter-clockwise with y down, `0` right and `270` down — which is
  the single easiest thing to get backwards.
* **A bearing is required, not optional.** For every offset, slope drop, and
  seam allowance the extractor must give an `angle` read off the drawing, and
  may **insert** steps the prose omits but the figure shows. Each row carries its
  own `confidence`.

Every reply is cached by content hash, so a document is read once, can be
reviewed and hand-corrected, and then committed. Re-running costs nothing.

## What one document actually yields

```bash
python -m dataset.build dataset/sources/angrakha_maxi.jsonl -o build/
```

Four artefacts come out of a single pass:

| File | What it is |
|---|---|
| `angrakha_maxi.sm2d` | A **real parametric pattern** — 101 objects, 4 draft blocks, 0 unresolved. Opens in this tool and in desktop Seamly2D. |
| `angrakha_maxi.svg` | A render, to eyeball against the source figures. |
| `angrakha_maxi.dataset.jsonl` | One record per drafting step: the instruction in the document's own words, the **pattern state the agent would have seen**, and the **tool calls that produce the next step**. |
| `angrakha_maxi.report.md` | Everything the document doesn't say, with the exact question and the figure that answers it. |

On the Angrakha Maxi: **27 of 31 steps compile and verify.**

### Why the tool calls in the dataset can be trusted

Every step is executed through `backend/agent.py`'s own `dispatch_tool`, against
a live `PatternSession`. A record only says `"verified": true` if the engine
actually built the geometry — the tool name, the argument shape and the
resulting coordinates all held up. Nothing in the dataset is a guess about what
*would* work.

A record looks like this:

```jsonc
{
  "instruction": "A - C -> Shoulder / 2",
  "panel": "back",
  "source_image": "page1_back_block1",
  "state_before": { "objects": [...], "variables": {...} },
  "tool_calls": [
    {"name": "add_point", "input": {"tool_type": "endLine",
       "attrs": {"name": "C", "basePoint": "1", "angle": "0",
                 "length": "#shoulder/2"}}},
    {"name": "add_line", "input": {"firstPoint": "1", "secondPoint": "4"}}
  ],
  "verified": true
}
```

That shape is directly usable three ways, in increasing order of payoff:

1. **Few-shot examples.** Real (instruction → correct call) pairs in the agent's
   own vocabulary, which is far more useful in the system prompt than invented
   ones.
2. **An eval set.** Replay the instruction, compare the agent's calls to the
   verified ones, score. This is the thing the project currently lacks most:
   there is no way to tell whether Opus is worth its cost over Sonnet without
   one, and no way to tell whether a prompt change helped.
3. **Fine-tuning data,** eventually — but only at a few thousand steps, i.e.
   ~100+ documents. Not the reason to start.

## What the first document taught us

The pipeline reports problems rather than papering over them. In 27 extracted
rows it found, unprompted:

* **`output_ref` is sometimes a string, sometimes a list.** `A - C = B - D` is
  one row with two outputs; it's really two steps. Split automatically.
* **A point that's measured from but never defined.** Step 7 is
  `G - H → (Chest + ease) ÷ 4`. No sentence defines `G`. The *figure* does — it
  labels `G` on the centre-back line, level with the arm-depth line.
* **A step stated twice.** `H - H₁ → 1.25" seam allowance` appears under two
  different figures. Detected as a restatement, not compiled twice.
* **Distances with no bearing.** `C - C₁ → shoulder slope 1"` says how far, not
  which way. Ten of these; only the figures resolve them.
* **A free-text value that's really a parameter.** `× 2, 3 or 4 (as per
  required flare)` becomes the variable `#flare_multiplier`.
* **Labels that restart per figure.** The front panel's `F` is not the `F`
  traced from the back. Both points are kept; the new one is renamed and the
  reuse is recorded.
* **A mis-extraction.** Step 24's `raw_text` says `C - D → (A - B) × 2,3,4` but
  the row records output `D` from input `B` with a vertical direction. The
  figure shows `A-B-C-D` is a rectangle, so `C-D` can't be a vertical distance
  from `B`.

**The load-bearing finding: the figures are not decoration.** Roughly a third of
the steps cannot be turned into geometry from the text alone. Any pipeline built
on these documents has to read the figures, or have a person read them once.

### How that's handled

Each document gets an optional `*.overrides.json` beside it. It answers the open
questions once, **with provenance**, and can insert steps the prose omits but the
figure shows:

```jsonc
"back:3": {
  "angle": 270,
  "why": "page1_back_block1: C1 sits directly below C. A shoulder slope is
          always dropped square from the shoulder-width point."
}
```

Anything the figures *don't* settle is deliberately left out, so it keeps
appearing in the report as an open question. Two remain on this document —
`D - D1 → 0.5" inward` (which way is "inward"?) and `A1 - B1` (points belonging
to a cutting layout the text never draws).

## The reward system

`dataset/reward.py` scores a drafting step from the engine's own geometry — no
human, no judge model. Seven components, each earning its place by catching a
failure the others miss:

| Component | Weight | Catches |
|---|---|---|
| `executed` | 0.18 | malformed calls, and doing nothing |
| `valid` | 0.14 | edits that leave the pattern unresolvable |
| `nondestructive` | 0.10 | silently moving geometry that already existed |
| `placement` | 0.28 | did the new points land where the document put them |
| `structure` | 0.10 | same tool, built from the same points |
| `parametric` | 0.15 | `#chest/6` versus a baked `6.0` |
| `economy` | 0.05 | scaffolding strewn everywhere |

`placement` is distance-shaped rather than binary: within 0.05" is full marks,
beyond 1" is zero, linear between. `parametric` is scored *relative to the
reference*, so a step that genuinely is a constant ("1 inch seam allowance")
isn't punished for being one.

### Calibrating it

A metric nobody has calibrated is decoration, so `dataset/evaluate.py` ships
policies whose scores are known in advance. Measured on the Angrakha Maxi:

| Policy | What it does | Teacher-forced | Rollout |
|---|---|---|---|
| `reference` | replays the document's own calls | **1.000** | **1.000** |
| `noop` | nothing | 0.061 | 0.089 |
| `literal` | identical geometry, every formula collapsed to its number | 0.915 | 0.932 |
| `perturb` | same construction, distances ×1.1 | 0.885 | 0.706 |
| `agent` | the real model, one turn per step | — | — |

The two that matter: `reference` scores exactly 1.0 (the reward never punishes
correctness), and `literal` loses **only** `parametric` — 0.43 against 1.00,
everything else untouched. That is the component doing precisely the job it
claims, on the failure mode that renders perfectly and is worthless.

### Two ways to run an episode

*Teacher forcing* (default) hands the policy the reference's pattern at every
step, so one bad step can't poison the rest. It isolates per-step skill — the
right mode for comparing models or prompt changes.

*Rollout* (`--rollout`) lets the policy carry its own pattern forward. Errors
compound exactly as they would in the app, and it is the only mode where
comparing the finished patterns means anything: `perturb` scores 0.885 forced
but 0.706 rolled out, finishing with 4 of 53 points in the right place.

### What it is not

This rewards **agreement with one document's construction**, not garment
quality. Two drafters can reach a correct block by different routes; the second
one scores lower here. That is the right trade for evaluation and for
rejection sampling, and the wrong one for judging whether a pattern fits — which
still needs a person. The weights are a plain dict and are meant to be tuned
against outcomes you actually care about.

## Is it worth collecting more?

Yes — and the PDFs are all that's needed now; the JSONL is generated. Concretely:

* **Text alone caps out at ~two-thirds of a garment.** The figures carry the
  rest, which is why extraction is multimodal and why every row records the
  `source_image` it came from.
* **Value per document is high and front-loaded.** One document produced a
  complete four-panel parametric block and 27 verified examples. Ten
  documents is a usable eval set. A hundred is a fine-tuning corpus and,
  separately, a library of Indian-ethnic-wear blocks — which desktop Seamly2D
  does not ship and cannot be bought.
* **Cost per document falls fast.** The vocabulary is small and closed: ten
  verbs covered a whole garment, and the second document from the same school
  will mostly reuse them. Extraction is one cached model call per PDF.
* **Review is where the remaining human time goes.** The report names every
  step the document under-determines, and the overrides file answers each one
  once with provenance. Rows the extractor marks `confidence: "low"` are the
  ones to read first.

The honest limitation: the placeholder measurements in `schema.py` are made up.
The compiled pattern is parametric and evaluates, but it is drafted for nobody
until real measurements go in — these documents state which measurements a
garment needs, never what they are.

## Layout

```
dataset/
  schema.py     the drafting-action IR — ten verbs, each declaring what it needs
  extract.py    PDF (text + figures) -> drafting actions, cached by content hash
  normalize.py  clean the rows; say precisely what the prose leaves out
  compile.py    execute each step as real agent tool calls on a live pattern
  build.py      CLI: pattern + render + dataset + open-questions report
  reward.py     score a step from geometry alone — seven components
  evaluate.py   replay a document through a policy; calibration policies
  sources/      the documents, each with its optional overrides file
```

Extraction needs `ANTHROPIC_API_KEY` for a cold run; everything downstream —
build, compile, reward, evaluate — runs offline.
