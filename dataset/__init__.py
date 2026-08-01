"""Turn sewing-instruction documents into patterns and training data.

A pattern-making school's handout is prose plus figures. An extraction pass
gives one JSONL row per sentence; this package takes it from there:

    schema.py     the drafting-action IR — a closed vocabulary of ten verbs
    normalize.py  clean the rows, and say precisely what the prose leaves out
    compile.py    execute each step as real agent tool calls on a live pattern
    build.py      CLI: pattern + render + dataset + open-questions report

Run it with ``python -m dataset.build <source.jsonl>``.
"""
