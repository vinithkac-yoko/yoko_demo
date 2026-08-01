"""Read a drafting-instruction PDF and emit drafting actions.

    python -m dataset.extract Angrakha_Maxi.pdf --garment angrakha_maxi

This is the front of the pipeline: the input is the PDF a pattern-making school
hands out, the output is the JSONL that `dataset.normalize` and
`dataset.compile` already consume. No hand-written intermediate.

**The figures are the point.** Roughly a third of the steps in a real document
cannot be turned into geometry from the prose alone — *"C - C₁ → shoulder slope
1\""* says how far, never which way. So each page goes to the model as its text
**and** its embedded figures, and the extraction contract demands a bearing for
every offset, read off the drawing. Rows may also be *inserted*: a figure often
labels a point (``G``) that no sentence defines, and the extractor is told to
emit it rather than leave a dangling reference.

Everything is cached by content hash next to the output, so re-running is free
and deterministic, and a document's extraction can be reviewed, corrected by
hand, and committed — the model reads each PDF once, not once per build.

Requires ``ANTHROPIC_API_KEY`` for a cold run; a warm cache needs nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .schema import ACTIONS, DIRECTIONS

#: Bump when the contract below changes — it is part of the cache key, so an
#: edited prompt re-extracts instead of silently reusing stale rows.
PROMPT_VERSION = 1

DEFAULT_MODEL = os.getenv("VLA_EXTRACT_MODEL", "claude-opus-4-8")

_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".gif": "image/gif", ".webp": "image/webp"}


@dataclass
class Figure:
    name: str
    data: bytes
    media_type: str


@dataclass
class Page:
    number: int
    text: str
    figures: list[Figure] = field(default_factory=list)


# --- reading the PDF ---------------------------------------------------------
def read_pdf(path: str | Path, *, min_figure_bytes: int = 4000) -> list[Page]:
    """Split a PDF into pages of text plus their embedded figures.

    Logos and letterheads repeat on every page; they are dropped, both to save
    tokens and because a model shown the academy's logo four times starts
    describing it.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw: list[Page] = []
    seen: dict[str, int] = {}
    for i, page in enumerate(reader.pages, start=1):
        figures = []
        for image in page.images:
            data = image.data
            if len(data) < min_figure_bytes:
                continue
            digest = hashlib.sha256(data).hexdigest()
            seen[digest] = seen.get(digest, 0) + 1
            suffix = Path(image.name).suffix.lower()
            figures.append(Figure(name=f"page{i}_{image.name}", data=data,
                                  media_type=_MEDIA.get(suffix, "image/png")))
        raw.append(Page(number=i, text=(page.extract_text() or "").strip(),
                        figures=figures))

    if len(raw) > 1:
        repeated = {d for d, n in seen.items() if n == len(raw)}
        for page in raw:
            page.figures = [f for f in page.figures
                            if hashlib.sha256(f.data).hexdigest() not in repeated]
    return raw


# --- the extraction contract -------------------------------------------------
def contract() -> str:
    """The rules the extractor must follow, generated from the live schema so
    the prompt cannot drift from what the compiler accepts."""
    verbs = "\n".join(f"  {name} — {spec['doc']}" for name, spec in ACTIONS.items())
    directions = ", ".join(sorted(d for d in DIRECTIONS if d))
    return f"""\
You are reading a sewing-pattern drafting document and converting it into a \
sequence of machine-executable drafting actions.

Emit a JSON array. One object per drafting step, in the order a drafter would \
work. No prose outside the array.

VERBS (use only these):
{verbs}

FIELDS
  garment       slug for the garment, same on every row
  panel         which pattern piece this step belongs to (e.g. "back",
                "front_left", "bottom"). Documents restart their point labels
                per panel; keep panels separate.
  step_index    integer, restarting at 0 for each panel. Use a fractional index
                (e.g. 6.5) for a step you are inserting between two printed ones.
  action        one of the verbs above
  output_ref    the point (or curve) this step creates, e.g. "C1". A string.
  input_refs    the points it is measured from, in order, e.g. ["A"]
  formula       the measurement expression exactly as written, e.g.
                "(chest + ease) / 4". Keep measurement names as words. Write
                inches as 1in. If the step is a plain number, leave null and use
                `value`.
  value         a literal distance as a string, e.g. "1.25in"
  direction     one of: {directions}
                  vertical    — straight down the page (or up, on a flipped panel)
                  horizontal  — straight across
                  along_line  — measured along an existing line; also fill `along`
                  cross       — x from the first input, y from the second; used
                                where a horizontal guide meets a vertical one
  angle         REQUIRED whenever the direction is an offset, a slope drop, a
                seam allowance, or anything the prose does not pin down.
                Degrees, counter-clockwise, y pointing DOWN the page:
                  0 = right, 90 = up, 180 = left, 270 = down, 45 = up-and-right.
                Read it off the figure. Do not guess from the wording.
  along         ["A", "C"] — the two points whose line an along_line step follows
  confidence    "high" | "medium" | "low" — how sure you are of this row
  notes         anything a drafter would need; say "figure-derived" if the step
                is not stated in the prose
  raw_text      the sentence this came from, verbatim (or your description of
                the figure, for an inserted step)
  source_image  which figure you used, if any
  page          page number

RULES
1. The figures are not decoration. Read them. Every bearing, every point the
   prose omits, and every panel layout comes from them.
2. If the prose measures from a point no earlier step defines, the figure
   almost certainly shows it. Insert a step that defines it, with
   notes "figure-derived" and a fractional step_index placing it before its use.
3. If a section says only "changes to be done", the panel still needs the block
   it starts from. Emit TRACE_REFERENCE with input_refs naming the source panel.
4. One step per row. If a sentence says "A - C = B - D", that is two rows.
5. Do not invent measurements. If the document says "as per requirement", use
   ANNOTATE_REQUIREMENT and leave the formula null.
6. If a step is restated under a later figure, emit it once.
7. Set confidence "low" rather than guessing, and say what is unclear in notes.
"""


def build_content(pages: list[Page], garment: str) -> list[dict]:
    """The user-message content blocks: each page's figures then its text."""
    import base64

    blocks: list[dict] = [{"type": "text", "text":
                           f"Garment: {garment}\nThe document has "
                           f"{len(pages)} page(s)."}]
    for page in pages:
        blocks.append({"type": "text", "text": f"\n=== PAGE {page.number} ==="})
        for fig in page.figures:
            blocks.append({"type": "text", "text": f"figure `{fig.name}`:"})
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": fig.media_type,
                "data": base64.standard_b64encode(fig.data).decode()}})
        blocks.append({"type": "text",
                       "text": f"page {page.number} text:\n{page.text}"})
    blocks.append({"type": "text", "text":
                   "Now emit the JSON array of drafting actions."})
    return blocks


# --- parsing the reply -------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_rows(reply: str, garment: str) -> list[dict]:
    """Pull the JSON array out of a reply and coerce it into schema shape.

    Tolerant on the way in (fenced or bare, missing optional fields), strict on
    the way out: unknown verbs are dropped rather than passed to the compiler.
    """
    text = reply.strip()
    match = _FENCE.search(text)
    if match:
        text = match.group(1).strip()
    if not text.startswith("["):
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("no JSON array in the extraction reply")
        text = text[start:end + 1]

    rows = []
    for i, item in enumerate(json.loads(text)):
        if not isinstance(item, dict) or item.get("action") not in ACTIONS:
            continue
        rows.append({
            "garment": item.get("garment") or garment,
            "panel": item.get("panel") or "main",
            "step_index": _num(item.get("step_index"), i),
            "action": item["action"],
            "output_ref": item.get("output_ref"),
            "input_refs": list(item.get("input_refs") or []),
            "formula": item.get("formula"),
            "value": item.get("value"),
            "direction": item.get("direction"),
            "angle": _num(item.get("angle"), None),
            "along": list(item.get("along") or []),
            "confidence": item.get("confidence") or "",
            "notes": item.get("notes") or "",
            "raw_text": item.get("raw_text") or "",
            "source_image": item.get("source_image") or "",
            "page": _num(item.get("page"), None),
        })
    return rows


def _num(v, default):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# --- the pass ----------------------------------------------------------------
def cache_key(pdf_bytes: bytes, model: str) -> str:
    h = hashlib.sha256()
    h.update(pdf_bytes)
    h.update(f"|{model}|v{PROMPT_VERSION}".encode())
    return h.hexdigest()[:16]


def extract(pdf_path: str | Path, *, garment: str | None = None,
            model: str = DEFAULT_MODEL, cache_dir: str | Path | None = None,
            max_tokens: int = 16000, refresh: bool = False) -> tuple[list[dict], str]:
    """Extract drafting actions from a PDF. Returns (rows, provenance)."""
    pdf_path = Path(pdf_path)
    garment = garment or _slug(pdf_path.stem)
    data = pdf_path.read_bytes()
    key = cache_key(data, model)

    cache = Path(cache_dir) if cache_dir else pdf_path.parent / ".extract-cache"
    cached = cache / f"{garment}-{key}.txt"
    if cached.exists() and not refresh:
        return parse_rows(cached.read_text(encoding="utf-8"), garment), f"cache:{cached}"

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            f"no cached extraction for {pdf_path.name} and ANTHROPIC_API_KEY is "
            f"not set. Set the key to read the PDF, or place a cached reply at "
            f"{cached}.")

    import anthropic

    client = anthropic.Anthropic()
    reply = client.messages.create(
        model=model, max_tokens=max_tokens, system=contract(),
        messages=[{"role": "user", "content": build_content(read_pdf(pdf_path), garment)}],
    )
    text = "".join(b.text for b in reply.content if getattr(b, "type", "") == "text")
    cache.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return parse_rows(text, garment), f"{model}:{key}"


def write_jsonl(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c == "_") else " " for c in text]
    return "_".join("".join(keep).split()).lower() or "garment"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf")
    ap.add_argument("--garment", default=None, help="slug (default: from filename)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-o", "--out", default="dataset/sources",
                    help="directory for <garment>.jsonl")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)

    garment = args.garment or _slug(Path(args.pdf).stem)
    rows, provenance = extract(args.pdf, garment=garment, model=args.model,
                               refresh=args.refresh)
    out = write_jsonl(rows, Path(args.out) / f"{garment}.jsonl")
    low = [r for r in rows if r.get("confidence") == "low"]
    print(json.dumps({"rows": len(rows), "low_confidence": len(low),
                      "panels": sorted({r["panel"] for r in rows}),
                      "source": provenance, "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
