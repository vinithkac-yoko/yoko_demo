"""qmuparser-compatible expression engine.

Seamly2D formulas (the ``length``/``angle``/``radius`` attributes and the
``<increment>`` formulas) are strings written in the grammar of QMuParser, a
Qt fork of muparser. They are NOT plain floats: they contain arithmetic,
C-style ternary conditionals (``size>22?4.75:4``), function calls
(``sinD(...)``), ``#``-prefixed custom increments, plain measurement names,
and *pseudo-variables* that reference the live geometry of already-computed
objects (``Line_A_B``, ``AngleLine_A_B``, ``RadiusArc_A_id``, ``CurrentLength``).

Because pseudo-variables read geometry, evaluation cannot be standalone: the
evaluator passes in a :class:`Scope` whose ``resolve`` callback is bound to the
current geometry state. This module owns only the grammar (tokenizer + Pratt
parser + evaluator); it knows nothing about geometry itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# --- identifier characters ---------------------------------------------------
# Seamly identifiers include letters, digits, '_', and '#' (increment prefix,
# e.g. '#CM', '##_USER_VARIABLES_##'). '_' also appears mid-name in point names
# baked into pseudo-vars (e.g. 'Line_A8_A8_Back').
_ID_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_#")
_ID_BODY = _ID_START | set("0123456789")


class FormulaError(ValueError):
    """Raised when a formula cannot be tokenized, parsed, or evaluated."""


# --- built-in functions ------------------------------------------------------
# Mirrors QMuParser's function table (qmuparser.cpp). Angles are in degrees for
# the ``*D`` variants, radians for the bare trig functions, matching Seamly.
def _deg(fn: Callable[[float], float]) -> Callable[[float], float]:
    return lambda x: fn(math.radians(x))


def _adeg(fn: Callable[[float], float]) -> Callable[[float], float]:
    return lambda x: math.degrees(fn(x))


FUNCTIONS: dict[str, tuple[int, Callable[..., float]]] = {
    "sin": (1, math.sin), "cos": (1, math.cos), "tan": (1, math.tan),
    "asin": (1, math.asin), "acos": (1, math.acos), "atan": (1, math.atan),
    "sinh": (1, math.sinh), "cosh": (1, math.cosh), "tanh": (1, math.tanh),
    "asinh": (1, math.asinh), "acosh": (1, math.acosh), "atanh": (1, math.atanh),
    "sinD": (1, _deg(math.sin)), "cosD": (1, _deg(math.cos)), "tanD": (1, _deg(math.tan)),
    "asinD": (1, _adeg(math.asin)), "acosD": (1, _adeg(math.acos)), "atanD": (1, _adeg(math.atan)),
    "log": (1, math.log10), "log10": (1, math.log10), "log2": (1, math.log2),
    "ln": (1, math.log), "exp": (1, math.exp), "sqrt": (1, math.sqrt),
    "abs": (1, abs), "sign": (1, lambda x: (x > 0) - (x < 0)), "rint": (1, lambda x: float(round(x))),
    "fmod": (2, math.fmod), "atan2": (2, lambda y, x: math.atan2(y, x)),
    "degTorad": (1, math.radians), "radTodeg": (1, math.degrees),
    "min": (-1, min), "max": (-1, max),
    "sum": (-1, lambda *a: sum(a)),
    "avg": (-1, lambda *a: sum(a) / len(a) if a else 0.0),
}

CONSTANTS: dict[str, float] = {"_pi": math.pi, "_e": math.e, "pi": math.pi}


# --- tokenizer ---------------------------------------------------------------
@dataclass
class _Tok:
    kind: str  # 'num' | 'id' | 'op' | 'lparen' | 'rparen' | 'comma'
    value: str


def tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            # scientific notation (rare in Seamly but valid muparser)
            if j < n and src[j] in "eE":
                j += 1
                if j < n and src[j] in "+-":
                    j += 1
                while j < n and src[j].isdigit():
                    j += 1
            toks.append(_Tok("num", src[i:j]))
            i = j
            continue
        if c in _ID_START:
            j = i
            while j < n and src[j] in _ID_BODY:
                j += 1
            toks.append(_Tok("id", src[i:j]))
            i = j
            continue
        # multi-char operators first
        two = src[i:i + 2]
        if two in ("<=", ">=", "==", "!=", "&&", "||"):
            toks.append(_Tok("op", two))
            i += 2
            continue
        if c in "+-*/^<>?:":
            toks.append(_Tok("op", c))
            i += 1
            continue
        if c == "(":
            toks.append(_Tok("lparen", c)); i += 1; continue
        if c == ")":
            toks.append(_Tok("rparen", c)); i += 1; continue
        if c == ",":
            toks.append(_Tok("comma", c)); i += 1; continue
        raise FormulaError(f"unexpected character {c!r} in formula {src!r}")
    return toks


# --- AST ---------------------------------------------------------------------
class Node:
    pass


@dataclass
class Num(Node):
    value: float


@dataclass
class Var(Node):
    name: str


@dataclass
class Unary(Node):
    op: str
    operand: Node


@dataclass
class Binary(Node):
    op: str
    left: Node
    right: Node


@dataclass
class Ternary(Node):
    cond: Node
    if_true: Node
    if_false: Node


@dataclass
class Call(Node):
    name: str
    args: list[Node]


# Binary operator precedence (higher binds tighter). Ternary handled separately.
_PRECEDENCE = {
    "||": 1, "&&": 2,
    "==": 3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5,
    "^": 6,
}
_RIGHT_ASSOC = {"^"}


class _Parser:
    def __init__(self, toks: list[_Tok], src: str):
        self.toks = toks
        self.pos = 0
        self.src = src

    def _peek(self) -> _Tok | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _next(self) -> _Tok:
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def parse(self) -> Node:
        node = self._parse_ternary()
        if self.pos != len(self.toks):
            raise FormulaError(f"trailing tokens in formula {self.src!r}")
        return node

    def _parse_ternary(self) -> Node:
        cond = self._parse_binary(0)
        t = self._peek()
        if t and t.kind == "op" and t.value == "?":
            self._next()
            if_true = self._parse_ternary()
            sep = self._peek()
            if not (sep and sep.kind == "op" and sep.value == ":"):
                raise FormulaError(f"expected ':' in ternary of {self.src!r}")
            self._next()
            if_false = self._parse_ternary()
            return Ternary(cond, if_true, if_false)
        return cond

    def _parse_binary(self, min_prec: int) -> Node:
        left = self._parse_unary()
        while True:
            t = self._peek()
            if not (t and t.kind == "op" and t.value in _PRECEDENCE):
                break
            prec = _PRECEDENCE[t.value]
            if prec < min_prec:
                break
            op = self._next().value
            next_min = prec if op in _RIGHT_ASSOC else prec + 1
            right = self._parse_binary(next_min)
            left = Binary(op, left, right)
        return left

    def _parse_unary(self) -> Node:
        t = self._peek()
        if t and t.kind == "op" and t.value in ("+", "-"):
            op = self._next().value
            return Unary(op, self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> Node:
        t = self._peek()
        if t is None:
            raise FormulaError(f"unexpected end of formula {self.src!r}")
        if t.kind == "num":
            self._next()
            return Num(float(t.value))
        if t.kind == "lparen":
            self._next()
            node = self._parse_ternary()
            if not (self._peek() and self._peek().kind == "rparen"):
                raise FormulaError(f"missing ')' in {self.src!r}")
            self._next()
            return node
        if t.kind == "id":
            name = self._next().value
            nxt = self._peek()
            if nxt and nxt.kind == "lparen" and name in FUNCTIONS:
                self._next()  # consume '('
                args: list[Node] = []
                if not (self._peek() and self._peek().kind == "rparen"):
                    args.append(self._parse_ternary())
                    while self._peek() and self._peek().kind == "comma":
                        self._next()
                        args.append(self._parse_ternary())
                if not (self._peek() and self._peek().kind == "rparen"):
                    raise FormulaError(f"missing ')' after call {name} in {self.src!r}")
                self._next()
                return Call(name, args)
            return Var(name)
        raise FormulaError(f"unexpected token {t.value!r} in {self.src!r}")


# --- evaluation --------------------------------------------------------------
Resolver = Callable[[str], float]


@dataclass
class Scope:
    """Variable resolution context for one formula evaluation.

    ``resolve`` receives an identifier that is neither a numeric literal nor a
    known constant/function and must return its value. The evaluator (which
    owns geometry) supplies this so that pseudo-variables such as ``Line_A_B``
    and ``CurrentLength`` can be answered from live state.
    """

    resolve: Resolver


def _eval(node: Node, scope: Scope) -> float:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Var):
        if node.name in CONSTANTS:
            return CONSTANTS[node.name]
        return scope.resolve(node.name)
    if isinstance(node, Unary):
        v = _eval(node.operand, scope)
        return -v if node.op == "-" else v
    if isinstance(node, Binary):
        l = _eval(node.left, scope)
        r = _eval(node.right, scope)
        op = node.op
        if op == "+": return l + r
        if op == "-": return l - r
        if op == "*": return l * r
        if op == "/":
            if r == 0:
                raise FormulaError("division by zero")
            return l / r
        if op == "^": return math.pow(l, r)
        if op == "<": return 1.0 if l < r else 0.0
        if op == ">": return 1.0 if l > r else 0.0
        if op == "<=": return 1.0 if l <= r else 0.0
        if op == ">=": return 1.0 if l >= r else 0.0
        if op == "==": return 1.0 if l == r else 0.0
        if op == "!=": return 1.0 if l != r else 0.0
        if op == "&&": return 1.0 if (l != 0 and r != 0) else 0.0
        if op == "||": return 1.0 if (l != 0 or r != 0) else 0.0
        raise FormulaError(f"unknown operator {op!r}")
    if isinstance(node, Ternary):
        return _eval(node.if_true, scope) if _eval(node.cond, scope) != 0 else _eval(node.if_false, scope)
    if isinstance(node, Call):
        arity, fn = FUNCTIONS[node.name]
        args = [_eval(a, scope) for a in node.args]
        if arity != -1 and len(args) != arity:
            raise FormulaError(f"{node.name} expects {arity} args, got {len(args)}")
        return float(fn(*args))
    raise FormulaError(f"cannot evaluate node {node!r}")


# Small AST cache: the same formula string is evaluated repeatedly (e.g. across
# sizes) so parsing once pays off.
_AST_CACHE: dict[str, Node] = {}


def parse(src: str) -> Node:
    ast = _AST_CACHE.get(src)
    if ast is None:
        ast = _Parser(tokenize(src), src).parse()
        _AST_CACHE[src] = ast
    return ast


def evaluate(src: str, scope: Scope) -> float:
    """Parse (cached) and evaluate ``src`` against ``scope``."""
    if src is None or src.strip() == "":
        raise FormulaError("empty formula")
    return _eval(parse(src), scope)


def identifiers(src: str) -> set[str]:
    """Return the set of variable identifiers referenced by ``src``.

    Used for dependency ordering of increments and for surfacing which
    measurements/variables a formula touches in the VLA state export.
    """
    out: set[str] = set()

    def walk(node: Node) -> None:
        if isinstance(node, Var):
            if node.name not in CONSTANTS:
                out.add(node.name)
        elif isinstance(node, Unary):
            walk(node.operand)
        elif isinstance(node, Binary):
            walk(node.left); walk(node.right)
        elif isinstance(node, Ternary):
            walk(node.cond); walk(node.if_true); walk(node.if_false)
        elif isinstance(node, Call):
            for a in node.args:
                walk(a)

    walk(parse(src))
    return out
