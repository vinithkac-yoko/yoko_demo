"""Turn sewing-instruction documents into patterns, datasets and rewards.

A pattern-making school's handout is prose plus hand-drawn figures. This package
takes it from the PDF all the way to a scored policy:

    schema.py     the drafting-action IR — a closed vocabulary of ten verbs
    extract.py    PDF (text + figures) -> drafting actions, cached by content hash
    normalize.py  clean the rows, and say precisely what the prose leaves out
    compile.py    execute each step as real agent tool calls on a live pattern
    build.py      CLI: pattern + render + dataset + open-questions report
    reward.py     score a step from the engine's geometry alone
    evaluate.py   replay a document through a policy; calibration policies

    python -m dataset.extract  garment.pdf
    python -m dataset.build    dataset/sources/garment.jsonl
    python -m dataset.evaluate dataset/sources/garment.jsonl --policy agent

Only extraction needs ``ANTHROPIC_API_KEY``; everything downstream runs offline.
"""
