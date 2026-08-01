# Instruction documents → patterns and training data

## The question

*"I have these kinds of files. Can we create a dataset for our model with
them?"* — a two-page PDF of drafting instructions (Angrakha Maxi), plus a JSONL
where each sentence has been extracted into a structured row.

**Yes, and they're worth collecting.** But not as training data in the naive
sense — 27 rows is nowhere near enough to train anything, and fine-tuning isn't
the bottleneck anyway. What these files are worth is different and, for this
system, more valuable. This document explains what the pipeline in `dataset/`
does with them and what it found in the first one.

## What one document actually yields

Run it:

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

## Is it worth collecting more?

Yes — but collect the **PDFs**, not just the extracted JSONL. Concretely:

* **The JSONL alone caps out at ~two-thirds of a garment.** The figures carry
  the rest. Keep both, and keep the `source_image` link between them (the
  extraction already does — that field is what makes the report actionable).
* **Value per document is high and front-loaded.** One document produced a
  complete four-panel parametric block and 27 verified examples. Ten
  documents is a usable eval set. A hundred is a fine-tuning corpus and,
  separately, a library of Indian-ethnic-wear blocks — which desktop Seamly2D
  does not ship and cannot be bought.
* **Cost per document falls fast.** The vocabulary is small and closed: ten
  verbs covered a whole garment, and the second document from the same school
  will mostly reuse them. What doesn't amortise is the figure-reading, which is
  currently a person with the overrides file — and is the obvious next thing to
  automate, since reading a hand-drawn figure and saying "C₁ is below C" is
  exactly what a vision model is good at.
* **The extractor needs tightening at the source.** Every issue in the list
  above is cheap to fix in extraction and expensive to fix downstream. The
  schema in `dataset/schema.py` is the contract to extract against.

The honest limitation: the placeholder measurements in `schema.py` are made up.
The compiled pattern is parametric and evaluates, but it is drafted for nobody
until real measurements go in — these documents state which measurements a
garment needs, never what they are.

## Layout

```
dataset/
  schema.py     the drafting-action IR — ten verbs, each declaring what it needs
  normalize.py  clean the rows; say precisely what the prose leaves out
  compile.py    execute each step as real agent tool calls on a live pattern
  build.py      CLI: pattern + render + dataset + open-questions report
  sources/      the documents, each with its optional overrides file
```
