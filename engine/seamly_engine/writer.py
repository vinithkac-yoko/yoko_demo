"""Serialize a :class:`~seamly_engine.model.Pattern` back to Seamly2D XML.

This is what makes saving, versioning, and exporting possible: an edited pattern
can be written to a ``.sm2d`` file and reopened (in this tool or in desktop
Seamly2D).

Fidelity strategy: the ``<calculation>`` section is rebuilt from the object model
(that's the part we mutate — every object keeps its original attributes in
``raw``, so nothing is lost). Sections we parse but never mutate (``modeling``,
``pieces``, ``groups``, and root-level elements like ``gradation`` /
``patternLabel``) are re-emitted **verbatim** from the XML captured at parse
time, so importing and re-saving a pattern doesn't degrade it.
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from .model import Pattern

SM2D_VERSION = "0.6.8"


def _attrs(raw: dict[str, str]) -> str:
    """Render attributes in a stable (sorted) order, id first — deterministic
    output means version diffs show real changes only."""
    items = sorted((k, v) for k, v in raw.items() if not k.startswith("__"))
    items.sort(key=lambda kv: (kv[0] != "id", kv[0]))
    return "".join(f" {k}={quoteattr(str(v))}" for k, v in items)


def _object_xml(obj, indent: str) -> str:
    kids = [c for c in obj.children if c.get("__tag__") or "pSpline" in c or "src" in c]
    if not kids:
        return f"{indent}<{obj.tag}{_attrs(obj.raw)}/>"

    lines = [f"{indent}<{obj.tag}{_attrs(obj.raw)}>"]
    if obj.tag == "operation":
        # children are src/dst pairs -> <source>/<destination> item lists
        srcs = [c["src"] for c in obj.children if c.get("src")]
        dsts = [c["dst"] for c in obj.children if c.get("dst")]
        lines.append(f"{indent}    <source>")
        for s in srcs:
            lines.append(f'{indent}        <item idObject="{s}"/>')
        lines.append(f"{indent}    </source>")
        lines.append(f"{indent}    <destination>")
        for d in dsts:
            lines.append(f'{indent}        <item idObject="{d}"/>')
        lines.append(f"{indent}    </destination>")
    else:
        for c in kids:
            tag = c.get("__tag__", "pathPoint")
            attrs = {k: v for k, v in c.items() if k != "__tag__"}
            lines.append(f"{indent}    <{tag}{_attrs(attrs)}/>")
    lines.append(f"{indent}</{obj.tag}>")
    return "\n".join(lines)


def pattern_to_xml(pattern: Pattern) -> str:
    """Return the pattern as a Seamly2D ``.sm2d`` XML document."""
    out: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>', "<pattern>"]
    out.append(f"    <!--Pattern saved by the VLA pattern engine.-->")
    out.append(f"    <version>{escape(pattern.version or SM2D_VERSION)}</version>")
    out.append(f"    <unit>{escape(pattern.unit or 'cm')}</unit>")
    out.append(f"    <description>{escape(pattern.description)}</description>")
    out.append(f"    <notes>{escape(pattern.notes)}</notes>")

    for raw in pattern.raw_root_sections:
        out.append("    " + raw)

    out.append(f"    <patternName>{escape(pattern.pattern_name)}</patternName>")
    if pattern.pattern_number:
        out.append(f"    <patternNumber>{escape(pattern.pattern_number)}</patternNumber>")
    if pattern.measurements_file:
        out.append(f"    <measurements>{escape(pattern.measurements_file)}</measurements>")

    out.append("    <increments>")
    for inc in pattern.increments:
        out.append("        <increment"
                   f" description={quoteattr(inc.description)}"
                   f" formula={quoteattr(inc.formula)}"
                   f" name={quoteattr(inc.name)}/>")
    out.append("    </increments>")

    for db in pattern.draft_blocks:
        out.append(f"    <draftBlock name={quoteattr(db.name)}>")
        out.append("        <calculation>")
        for obj in db.objects:
            out.append(_object_xml(obj, "            "))
        out.append("        </calculation>")
        for name in ("modeling", "pieces", "groups"):
            raw = db.raw_sections.get(name)
            if raw:
                out.append("        " + raw)
        out.append("    </draftBlock>")

    out.append("</pattern>")
    return "\n".join(out) + "\n"


def write_pattern(pattern: Pattern, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(pattern_to_xml(pattern))
