"""Run a policy through a document's drafting steps and score it.

    python -m dataset.evaluate dataset/sources/angrakha_maxi.jsonl --policy reference
    python -m dataset.evaluate dataset/sources/angrakha_maxi.jsonl --policy agent

An *episode* is a compiled document replayed one step at a time. At each step
the policy is handed the pattern exactly as the reference had it, plus the
instruction in the document's own words, and whatever it does is scored by
:mod:`dataset.reward` against what the document did.

The built-in policies exist to keep the reward function honest — a metric you
cannot calibrate is decoration:

``reference``  replays the document's own calls; **must** score 1.0
``noop``       does nothing; the floor
``literal``    same geometry, but every formula replaced by its number — should
               lose only ``parametric``, which is how you check that component
               is measuring what it claims
``perturb``    same shape, distances scaled; should lose ``placement`` smoothly
``agent``      the real model, one turn per step (needs ``ANTHROPIC_API_KEY``)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "engine", _ROOT / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from seamly_engine.operations import PatternSession  # noqa: E402

from .compile import CompiledStep, compile_document  # noqa: E402
from .normalize import load_overrides, load_rows, normalize  # noqa: E402
from .reward import EpisodeReward, score_episode, score_step  # noqa: E402


# --- policies ----------------------------------------------------------------
class Policy:
    """Given the pattern so far and the instruction, emit tool calls."""

    name = "policy"

    def act(self, session: PatternSession, step: CompiledStep) -> list[dict]:
        raise NotImplementedError


class ReferencePolicy(Policy):
    name = "reference"

    def act(self, session, step):
        return copy.deepcopy(step.tool_calls)


class NoopPolicy(Policy):
    name = "noop"

    def act(self, session, step):
        return []


class LiteralPolicy(Policy):
    """Right geometry, wrong craft: every formula collapsed to its value.

    This is the failure mode that looks perfect in a render and is useless in
    practice — the pattern stops responding to measurements.
    """

    name = "literal"

    def act(self, session, step):
        from seamly_engine.formula import Scope, evaluate

        calls = copy.deepcopy(step.tool_calls)
        scope = Scope(resolve=_incremental_resolver(session))
        for call in calls:
            attrs = call["input"].get("attrs")
            if not attrs:
                continue
            for key in ("length", "angle", "radius"):
                raw = attrs.get(key)
                if raw is None or not _is_symbolic(raw):
                    continue
                try:
                    attrs[key] = f"{evaluate(str(raw), scope):g}"
                except Exception:  # noqa: BLE001 — a formula we can't fold stays as is
                    pass
        return calls


def _incremental_resolver(session: PatternSession):
    """Resolve ``#vars`` and ``Line_A_B`` from the session's current geometry."""
    values = dict(session.evaluated.increment_values)
    points = {o.raw.get("name"): session.evaluated.points.get(o.id)
              for o in session.pattern.all_objects() if o.raw.get("name")}

    def resolve(name: str) -> float:
        if name in values:
            return values[name]
        if name.lstrip("#") in values:
            return values[name.lstrip("#")]
        if name.startswith("Line_"):
            rest = name[5:]
            for i in range(1, len(rest)):
                a, b = points.get(rest[:i]), points.get(rest[i + 1:])
                if rest[i] == "_" and a is not None and b is not None:
                    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
        raise KeyError(name)

    return resolve


def _is_symbolic(raw) -> bool:
    return any(c.isalpha() or c == "#" for c in str(raw))


class PerturbPolicy(Policy):
    """Same construction, distances scaled — a policy that drafts sloppily."""

    name = "perturb"

    def __init__(self, scale: float = 1.1):
        self.scale = scale

    def act(self, session, step):
        calls = copy.deepcopy(step.tool_calls)
        for call in calls:
            attrs = call["input"].get("attrs") or {}
            raw = attrs.get("length")
            if raw is None:
                continue
            attrs["length"] = f"({raw})*{self.scale:g}"
        return calls


class AgentPolicy(Policy):
    """The real thing: one model turn per step, with the tools it has in the app."""

    name = "agent"

    def __init__(self, model: str | None = None, max_steps: int = 4):
        self.model = model or os.getenv("VLA_MODEL", "claude-sonnet-5")
        self.max_steps = max_steps

    def act(self, session, step):
        import anthropic

        from agent import TOOLS, dispatch_tool  # noqa: F401 — dispatch happens in run_step
        from seamly_engine.state import compact_state

        client = anthropic.Anthropic()
        state = json.dumps(compact_state(session.pattern, session.evaluated))
        prompt = (
            "You are drafting a sewing pattern step by step. Here is the pattern "
            "state so far:\n\n" + state +
            "\n\nDo exactly this one drafting step, and nothing more:\n"
            f"{step.instruction}\n\n"
            "Angles are degrees counter-clockwise with y pointing down: "
            "0 = right, 270 = down. Reference existing objects by their integer "
            "id. Keep measurements symbolic (use the # variables) rather than "
            "typing numbers."
        )
        reply = client.messages.create(
            model=self.model, max_tokens=2000, tools=TOOLS,
            messages=[{"role": "user", "content": prompt}])
        return [{"name": b.name, "input": b.input} for b in reply.content
                if getattr(b, "type", "") == "tool_use"]


POLICIES = {p.name: p for p in (ReferencePolicy, NoopPolicy, LiteralPolicy,
                                PerturbPolicy, AgentPolicy)}


# --- the episode -------------------------------------------------------------
def load_reference(source: str | Path, overrides: str | Path | None = None):
    source = Path(source)
    if overrides is None:
        guess = source.with_suffix(".overrides.json")
        overrides = guess if guess.exists() else None
    steps, inserts = load_overrides(overrides)
    doc = normalize(load_rows(source), steps, inserts)
    return compile_document(doc, capture_state=False, keep_sessions=True)


def run_episode(source: str | Path, policy: Policy, *,
                overrides: str | Path | None = None, rollout: bool = False,
                limit: int | None = None, verbose: bool = False) -> EpisodeReward:
    """Replay a document step by step and score the policy at each one.

    Two modes, and the difference is the whole argument about how to evaluate an
    agent that builds on its own work:

    *Teacher forcing* (default) hands the policy the reference's pattern at
    every step, so one bad step can't poison the rest. It isolates per-step
    skill, which is what you want when comparing models or prompts.

    *Rollout* lets the policy carry its own pattern forward. It is the honest
    end-to-end measure — errors compound, exactly as they would in the app — and
    the only mode where the finished-pattern comparison means anything.
    """
    from agent import dispatch_tool

    result = load_reference(source, overrides)
    scorable = [s for s in result.steps if s.ok and s.tool_calls and s.session_after]
    if limit:
        scorable = scorable[:limit]

    rewards: list = []
    carried: PatternSession | None = None
    previous: PatternSession | None = None   # the reference state before this step
    for step in result.steps:
        if step not in scorable:
            previous = step.session_after or previous
            continue
        before = previous if previous is not None else _empty_session(result)
        cand_before = carried if (rollout and carried is not None) else before
        candidate = copy.deepcopy(cand_before)
        executed = True
        for call in policy.act(candidate, step):
            res = dispatch_tool(candidate, call["name"], dict(call.get("input", {})))
            if not res.ok:
                executed = False
        reward = score_step(before, step.session_after, candidate,
                            candidate_before=cand_before, executed=executed)
        rewards.append(reward)
        carried = candidate
        previous = step.session_after
        if verbose:
            print(f"  {step.action.key:<16} {reward.total:.3f}  {step.instruction[:60]}")
    return score_episode(rewards, reference=result.session, candidate=carried,
                         final_weight=0.2 if rollout else 0.0)


def _empty_session(result) -> PatternSession:
    """The pattern before any step ran: the blocks and the variables declared
    up front, and no geometry. Taken from the compiler rather than reconstructed
    — stripping the finished pattern would leave behind every variable a later
    step went on to create."""
    if result.initial_session is not None:
        return copy.deepcopy(result.initial_session)
    empty = copy.deepcopy(result.session)
    for block in empty.pattern.draft_blocks:
        block.objects = []
    empty.evaluated = empty._evaluate()  # noqa: SLF001 — re-evaluate the emptied pattern
    return empty


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="extracted drafting-action .jsonl")
    ap.add_argument("--policy", default="reference", choices=sorted(POLICIES))
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rollout", action="store_true",
                    help="let the policy carry its own pattern forward "
                         "(end-to-end; errors compound)")
    ap.add_argument("--model", default=None, help="agent policy: model id")
    ap.add_argument("--scale", type=float, default=1.1, help="perturb policy: factor")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    cls = POLICIES[args.policy]
    if cls is AgentPolicy:
        policy = cls(model=args.model)
    elif cls is PerturbPolicy:
        policy = cls(scale=args.scale)
    else:
        policy = cls()

    reward = run_episode(args.source, policy, overrides=args.overrides,
                         rollout=args.rollout, limit=args.limit,
                         verbose=args.verbose)
    print(json.dumps({"policy": policy.name,
                      "mode": "rollout" if args.rollout else "teacher_forced",
                      **reward.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
