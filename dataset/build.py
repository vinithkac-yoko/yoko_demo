"""Turn an instruction document into a pattern, a dataset, and a work queue.

    python -m dataset.build dataset/sources/angrakha_maxi.jsonl -o build/

Writes four files next to each other:

``<garment>.sm2d``
    The compiled parametric pattern — openable in this tool or in desktop
    Seamly2D. Measurements are ``#`` variables, so changing one re-drafts it.
``<garment>.svg``
    A render of that pattern, for eyeballing it against the source figures.
``<garment>.dataset.jsonl``
    One record per drafting step: the instruction in the document's own words,
    the pattern state the agent would have seen at that moment, and the tool
    calls that are known to produce the next step — because they were executed.
    Few-shot examples and an eval set in one.
``<garment>.report.md``
    What the document doesn't say. Every unresolved step with the exact
    question to answer and the figure to answer it from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compile import compile_document
from .normalize import Document, load_overrides, load_rows, normalize


def build(source: str | Path, out_dir: str | Path = "build",
          overrides_path: str | Path | None = None,
          *, with_state: bool = True) -> dict:
    source = Path(source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if overrides_path is None:
        guess = source.with_suffix(".overrides.json")
        overrides_path = guess if guess.exists() else None

    steps, inserts = load_overrides(overrides_path)
    doc = normalize(load_rows(source), steps, inserts)
    result = compile_document(doc, capture_state=with_state)

    stem = doc.garment
    from seamly_engine.render import render_svg
    from seamly_engine.writer import pattern_to_xml

    (out / f"{stem}.sm2d").write_text(
        pattern_to_xml(result.session.pattern), encoding="utf-8")
    (out / f"{stem}.svg").write_text(
        render_svg(result.session.pattern, result.session.evaluated, width=1400),
        encoding="utf-8")

    with (out / f"{stem}.dataset.jsonl").open("w", encoding="utf-8") as fh:
        for step in result.steps:
            record = step.as_dict()
            if with_state and step.state_before is not None:
                record["state_before"] = step.state_before
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    (out / f"{stem}.report.md").write_text(_report(doc, result), encoding="utf-8")

    ev = result.session.evaluated
    return {
        "garment": stem,
        "steps": len(result.steps),
        "verified": len(result.verified),
        "unverified": len(result.unverified),
        "objects": len(result.session.pattern.all_objects()),
        "unresolved": len(ev.unresolved),
        "variables": len(result.variables),
        "out": str(out),
    }


def _report(doc: Document, result) -> str:
    lines = [f"# {doc.garment} — extraction report", ""]
    ev = result.session.evaluated
    lines += [
        f"* Panels: {', '.join(doc.panels)}",
        f"* Steps: {len(result.steps)} — **{len(result.verified)} compiled and "
        f"verified**, {len(result.unverified)} not",
        f"* Geometry: {len(result.session.pattern.all_objects())} objects, "
        f"{len(ev.unresolved)} unresolved",
        f"* Measurements promoted to variables: "
        f"{', '.join(sorted(result.variables)) or 'none'}",
        "",
        "## Open questions",
        "",
        "Each of these is a step the prose under-determines. Answer it once in "
        "`<source>.overrides.json` and the document compiles from then on.",
        "",
    ]
    open_rows = [s for s in result.unverified if s.action.blocked]
    if not open_rows:
        lines.append("_None — every step is pinned down._")
    for step in open_rows:
        a = step.action
        lines.append(f"### `{a.key}` — {a.raw_text or a.action}")
        lines.append("")
        lines.append(f"* figure: `{a.source_image or 'n/a'}`")
        for issue in a.issues:
            if issue.blocking:
                lines.append(f"* **{issue.code}** — {issue.detail}"
                             + (f"  → supply `{issue.needs}`" if issue.needs else ""))
        lines.append("")

    noted = [s for s in result.unverified if not s.action.blocked]
    if noted:
        lines += ["## Steps intentionally not compiled", ""]
        for step in noted:
            lines.append(f"* `{step.action.key}` — {step.skipped or step.message}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="extracted drafting-action .jsonl")
    ap.add_argument("-o", "--out", default="build", help="output directory")
    ap.add_argument("--overrides", default=None,
                    help="resolutions file (defaults to <source>.overrides.json)")
    ap.add_argument("--no-state", action="store_true",
                    help="omit per-step pattern state (much smaller output)")
    args = ap.parse_args(argv)
    summary = build(args.source, args.out, args.overrides, with_state=not args.no_state)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
