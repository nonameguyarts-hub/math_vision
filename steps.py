"""
steps.py - step-by-step working for MathVision.

    from steps import steps_for
    out = steps_for("2 x + 3 = 7")
    out -> {"ok": True, "answer": "x = 2",
            "steps": [{"latex": "2x + 3 = 7", "why": "Start"}, ...], "note": None}

Scope: arithmetic with brackets, powers, exact square roots and fractions;
equality checks ("7 \\times 8 = 56"); linear equations in one variable.
Anything else returns ok=False and the UI shows the answer only.

Safety rule: the finished working is compared with solver.solve_latex. If the two
disagree, or anything unexpected happens, ok=False is returned (never wrong working).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import Any, Optional

import sympy as sp

from solver import SINGLE, normalize_latex, read_group, solve_latex, split_eq, split_relation

MAX_STEPS = 40


class Unsupported(Exception):
    """The expression is outside what the step generator handles."""


class NeedVars(Unsupported):
    """A letter was found where a number was expected."""


# --------------------------------------------------------------------------- nodes
@dataclass(frozen=True)
class Num:
    v: Fraction
    lit: Optional[str] = None      # original decimal text, e.g. "1.5"


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Neg:
    x: Any


@dataclass(frozen=True)
class Frac:
    n: Any
    d: Any
    keep: bool = False             # True = written over a common denominator, do not reduce yet


@dataclass(frozen=True)
class Sqrt:
    x: Any


@dataclass(frozen=True)
class Pow:
    b: Any
    e: Any


@dataclass(frozen=True)
class Group:
    x: Any
    open: str = "("


@dataclass(frozen=True)
class Sum:
    terms: tuple                   # ((sign, node), ...)


@dataclass(frozen=True)
class Prod:
    factors: tuple                 # ((op, node), ...) op in "*", "/", "i" (implicit)


# --------------------------------------------------------------------------- numbers
def mk(v: Fraction):
    """Canonical node for a rational value."""
    v = Fraction(v)
    if v.denominator == 1:
        return Num(v)
    f = Frac(Num(Fraction(abs(v.numerator))), Num(Fraction(v.denominator)))
    return f if v > 0 else Neg(f)


def exact_sqrt(v: Fraction) -> Fraction:
    if v < 0:
        raise Unsupported("square root of a negative number")
    a, b = isqrt(v.numerator), isqrt(v.denominator)
    if a * a == v.numerator and b * b == v.denominator:
        return Fraction(a, b)
    raise Unsupported("irrational square root")


def rpow(b: Fraction, e: Fraction) -> Fraction:
    if e.denominator != 1:
        raise Unsupported("fractional exponent")
    k = int(e)
    if abs(k) > 64:
        raise Unsupported("exponent too large")
    if b == 0 and k < 0:
        raise Unsupported("division by zero")
    r = b ** k
    if len(str(abs(r.numerator))) > 40:
        raise Unsupported("number too large")
    return Fraction(r)


def value(n, subs=None) -> Fraction:
    if isinstance(n, Num):
        return n.v
    if isinstance(n, Var):
        if subs and n.name in subs:
            return subs[n.name]
        raise Unsupported("variable")
    if isinstance(n, Neg):
        return -value(n.x, subs)
    if isinstance(n, Group):
        return value(n.x, subs)
    if isinstance(n, Frac):
        d = value(n.d, subs)
        if d == 0:
            raise Unsupported("division by zero")
        return value(n.n, subs) / d
    if isinstance(n, Sqrt):
        return exact_sqrt(value(n.x, subs))
    if isinstance(n, Pow):
        return rpow(value(n.b, subs), value(n.e, subs))
    if isinstance(n, Sum):
        r = Fraction(0)
        for s, t in n.terms:
            r += s * value(t, subs)
        return r
    if isinstance(n, Prod):
        r = Fraction(1)
        for op, t in n.factors:
            x = value(t, subs)
            if op == "/":
                if x == 0:
                    raise Unsupported("division by zero")
                r = r / x
            else:
                r = r * x
        return r
    raise Unsupported("unknown node")


def _int_num(n) -> bool:
    return isinstance(n, Num) and n.v.denominator == 1 and n.lit is None


def canon_frac(f: Frac) -> bool:
    if f.keep:
        return True
    if not (_int_num(f.n) and _int_num(f.d)):
        return False
    return f.n.v > 0 and f.d.v > 1 and gcd(int(f.n.v), int(f.d.v)) == 1


def is_final(n) -> bool:
    if isinstance(n, Num):
        return True
    if isinstance(n, Frac):
        return canon_frac(n)
    if isinstance(n, Neg):
        return isinstance(n.x, Frac) and canon_frac(n.x)
    if isinstance(n, Group):
        return is_final(n.x) and value(n.x) < 0
    return False


def has_var(n) -> bool:
    if isinstance(n, Var):
        return True
    for f in getattr(n, "__dataclass_fields__", {}):
        c = getattr(n, f)
        if isinstance(c, tuple):
            for item in c:
                sub = item[1] if isinstance(item, tuple) else item
                if has_var(sub):
                    return True
        elif hasattr(c, "__dataclass_fields__") and has_var(c):
            return True
    return False


def norm(node):
    """Tidy a freshly built Sum of one term."""
    if isinstance(node, Sum) and len(node.terms) == 1:
        s, t = node.terms[0]
        if s == 1:
            return t
        if is_final(t) and not has_var(t):
            return mk(-value(t))
    return node


# --------------------------------------------------------------------------- parser
class Parser:
    def __init__(self, toks, allow_vars):
        self.t, self.i, self.allow_vars = list(toks), 0, allow_vars

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def parse(self):
        node = self.sum()
        if self.i != len(self.t):
            raise Unsupported(f"unexpected token {self.peek()}")
        return node

    @staticmethod
    def starts_operand(tok):
        return tok is not None and (tok.isdigit() or tok in ("(", "[", "\\frac", "\\sqrt", ".") or tok in SINGLE)

    def sum(self):
        terms, sign = [], 1
        if self.peek() in ("+", "-"):
            sign = -1 if self.peek() == "-" else 1
            self.i += 1
        terms.append((sign, self.prod()))
        while self.peek() in ("+", "-"):
            s = -1 if self.peek() == "-" else 1
            self.i += 1
            terms.append((s, self.prod()))
        return norm(Sum(tuple(terms)))

    def times_x(self, last):
        nxt = self.peek(1)
        return (isinstance(last, (Num, Group, Frac, Sqrt, Pow))
                and nxt is not None
                and (nxt.isdigit() or nxt in ("(", "[", "\\frac", "\\sqrt")))

    def prod(self):
        factors = [("*", self.factor())]
        while True:
            t = self.peek()
            if t in ("\\times", "\\cdot", "\\ast"):
                self.i += 1
                factors.append(("*", self.factor()))
            elif t in ("\\div", "/"):
                self.i += 1
                factors.append(("/", self.factor()))
            elif t in ("x", "X") and self.times_x(factors[-1][1]):
                self.i += 1
                factors.append(("*", self.factor()))
            elif self.starts_operand(t):
                factors.append(("i", self.factor()))
            else:
                break
        return factors[0][1] if len(factors) == 1 else Prod(tuple(factors))

    def factor(self):
        t = self.peek()
        if t in ("+", "-"):
            self.i += 1
            inner = self.factor()
            return norm(Sum((((-1 if t == "-" else 1), inner),)))
        return self.power()

    def var(self, t):
        if not self.allow_vars:
            raise NeedVars("letter")
        return Var(t)

    def power(self):
        base = self.atom()
        if self.peek() == "^":
            self.i += 1
            t = self.peek()
            if t == "{":
                grp, k = read_group(self.t, self.i)
                exp = Parser(grp, self.allow_vars).parse()
                self.i = k
            elif t is not None and t.isdigit():
                self.i += 1
                exp = Num(Fraction(int(t)))
            elif t is not None and t in SINGLE:
                self.i += 1
                exp = self.var(t)
            else:
                raise Unsupported("bad exponent")
            return Pow(base, exp)
        return base

    def atom(self):
        t = self.peek()
        if t is None:
            raise Unsupported("unexpected end")
        if t.isdigit() or t == ".":
            j = self.i
            while j < len(self.t) and (self.t[j].isdigit() or self.t[j] == "."):
                j += 1
            lit = "".join(self.t[self.i:j])
            self.i = j
            if not re.fullmatch(r"\d+(\.\d+)?", lit):
                raise Unsupported("bad number")
            return Num(Fraction(lit), lit if "." in lit else None)
        if t in ("(", "["):
            self.i += 1
            inner = self.sum()
            if self.peek() not in (")", "]"):
                raise Unsupported("missing bracket")
            self.i += 1
            return Group(inner, t)
        if t == "\\frac":
            a, k = read_group(self.t, self.i + 1)
            b, k2 = read_group(self.t, k)
            self.i = k2
            return Frac(Parser(a, self.allow_vars).parse(), Parser(b, self.allow_vars).parse())
        if t == "\\sqrt":
            if self.peek(1) == "[":
                raise Unsupported("nth root")
            a, k = read_group(self.t, self.i + 1)
            self.i = k
            return Sqrt(Parser(a, self.allow_vars).parse())
        if t in SINGLE:
            self.i += 1
            return self.var(t)
        raise Unsupported(f"unsupported token {t}")


# --------------------------------------------------------------------------- rendering
def fmt_int(v: Fraction) -> str:
    return str(int(v))


def R(n, lead=True, subs=None, wrap=False) -> str:
    """Render a node as LaTeX. lead=False wraps negative numbers in brackets;
    wrap=True brackets a substituted variable (used for factors and power bases)."""
    if isinstance(n, Num):
        s = n.lit if n.lit else (fmt_int(n.v) if n.v.denominator == 1 else str(n.v))
        return f"({s})" if (n.v < 0 and not lead) else s
    if isinstance(n, Var):
        if subs and n.name in subs:
            body = R(mk(subs[n.name]))
            return "(" + body + ")" if (wrap or (subs[n.name] < 0 and not lead)) else body
        return n.name
    if isinstance(n, Neg):
        s = "-" + R(n.x, True, subs)
        return s if lead else f"({s})"
    if isinstance(n, Frac):
        return "\\frac{" + R(n.n, True, subs) + "}{" + R(n.d, True, subs) + "}"
    if isinstance(n, Sqrt):
        return "\\sqrt{" + R(n.x, True, subs) + "}"
    if isinstance(n, Pow):
        b = R(n.b, False, subs, wrap=True)
        if isinstance(n.b, (Frac, Neg, Pow)):
            b = "(" + R(n.b, True, subs) + ")"
        return b + "^{" + R(n.e, True, subs) + "}"
    if isinstance(n, Group):
        close = ")" if n.open == "(" else "]"
        return n.open + R(n.x, True, subs) + close
    if isinstance(n, Sum):
        out = ""
        for i, (s, t) in enumerate(n.terms):
            if i == 0:
                out += ("-" + R(t, False, subs)) if s == -1 else R(t, True, subs)
            else:
                out += (" - " if s == -1 else " + ") + R(t, False, subs)
        return out
    if isinstance(n, Prod):
        out = ""
        for i, (op, t) in enumerate(n.factors):
            if i == 0:
                out += R(t, True, subs)
            elif op == "/":
                out += " \\div " + R(t, False, subs, wrap=True)
            elif op == "i" and isinstance(t, (Group, Sqrt, Var)):
                out += R(t, False, subs, wrap=True)
            else:
                out += " \\times " + R(t, False, subs, wrap=True)
        return out
    raise Unsupported("cannot render")


def plain(v: Fraction) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


# --------------------------------------------------------------------------- stepping
def _par(items):
    """Step every non-final item once; return (new items, reasons)."""
    out, why = [], []
    for it in items:
        if is_final(it):
            out.append(it)
        else:
            r = step(it)
            if r is None:
                raise Unsupported("stuck")
            out.append(r[0])
            why += r[1]
    return out, why


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def step(n):
    """One step of working: returns (new node, [reasons]) or None when finished."""
    if is_final(n) or isinstance(n, (Num, Var)):
        return None

    if isinstance(n, Neg):
        r = step(n.x)
        if r is None:
            return None
        new, why = r
        return (Num(-new.v), why) if isinstance(new, Num) else (Neg(new), why)

    if isinstance(n, Group):
        x = n.x
        if is_final(x):
            return x, ["Remove the brackets"]
        r = step(x)
        if r is None:
            raise Unsupported("stuck in brackets")
        new, why = r
        if is_final(new) and value(new) >= 0:
            return new, why
        return Group(new, n.open), why

    if isinstance(n, Frac):
        if not (is_final(n.n) and is_final(n.d)):
            (a, b), why = _par([n.n, n.d])
            return Frac(a, b, n.keep), why
        v = value(n)
        simple = _int_num(n.n) and _int_num(n.d)
        return mk(v), ["Simplify the fraction" if simple else "Divide the numerator by the denominator"]

    if isinstance(n, Sqrt):
        if not is_final(n.x):
            (a,), why = _par([n.x])
            return Sqrt(a), why
        return mk(exact_sqrt(value(n.x))), ["Take the square root"]

    if isinstance(n, Pow):
        if not (is_final(n.b) and is_final(n.e)):
            (a, b), why = _par([n.b, n.e])
            return Pow(a, b), why
        return mk(rpow(value(n.b), value(n.e))), ["Evaluate the power"]

    if isinstance(n, Prod):
        nodes = [f for _, f in n.factors]
        if not all(is_final(f) for f in nodes):
            new, why = _par(nodes)
            return Prod(tuple((op, f) for (op, _), f in zip(n.factors, new))), why
        ops = [op for op, _ in n.factors[1:]]
        vals = [value(f) for f in nodes]
        if ("/" not in ops and len(nodes) >= 2 and any(v.denominator != 1 for v in vals)
                and all(v > 0 for v in vals)):
            num = den = 1
            for v in vals:
                num *= v.numerator
                den *= v.denominator
            return Frac(Num(Fraction(num)), Num(Fraction(den))), ["Multiply the numerators and the denominators"]
        has_mul = any(op in ("*", "i") for op in ops)
        has_div = "/" in ops
        word = "Multiply and divide from left to right" if (has_mul and has_div) else ("Divide" if has_div else "Multiply")
        return mk(value(n)), [word]

    if isinstance(n, Sum):
        nodes = [t for _, t in n.terms]
        if not all(is_final(t) for t in nodes):
            new, why = _par(nodes)
            return Sum(tuple((s, t) for (s, _), t in zip(n.terms, new))), why
        vals = [s * value(t) for s, t in n.terms]
        if all(v.denominator == 1 for v in vals):
            ops = set()
            for (s, t), v in list(zip(n.terms, vals))[1:]:
                ops.add("Add" if v >= 0 else "Subtract")
            word = " and ".join(sorted(ops)) if ops else "Add"
            word = "Add and subtract" if len(ops) > 1 else word
            return mk(sum(vals, Fraction(0))), [word]
        L = 1
        for v in vals:
            L = _lcm(L, v.denominator)
        if all(isinstance(t, Frac) and t.keep and int(value(t.d)) == L for t in nodes):
            N = sum(s * int(t.n.v) for s, t in n.terms)
            ops = {("Add" if s * int(t.n.v) >= 0 else "Subtract") for (s, t) in n.terms[1:]}
            word = ("Add and subtract" if len(ops) > 1 else (ops.pop() if ops else "Add")) + " the numerators"
            if N == 0:
                return Num(Fraction(0)), [word]
            res = Frac(Num(Fraction(abs(N))), Num(Fraction(L)))
            if N < 0:
                res = Neg(res)
            return res, [word]
        new_terms = []
        for i, v in enumerate(vals):
            sign = -1 if v < 0 else 1
            new_terms.append((sign, Frac(Num(Fraction(abs(v) * L)), Num(Fraction(L)), keep=True)))
        return Sum(tuple(new_terms)), [f"Rewrite the terms over the common denominator {L}"]

    raise Unsupported("cannot step")


def _why(reasons) -> str:
    return "; ".join(dict.fromkeys(reasons))


def _run(nodes, join, rows):
    """Reduce a list of nodes (the sides of an equality, or a single expression) in parallel."""
    cur = list(nodes)
    for _ in range(MAX_STEPS):
        if all(is_final(c) for c in cur):
            return cur
        new, why = _par(cur)
        cur = new
        text = join(cur)
        if text != rows[-1]["latex"]:
            rows.append({"latex": text, "why": _why(why)})
    raise Unsupported("did not finish")


def _expr(tree):
    rows = [{"latex": R(tree), "why": "Start"}]
    cur = _run([tree], lambda c: R(c[0]), rows)[0]
    if isinstance(cur, Group):
        cur = cur.x
    return rows, value(cur)


def _check(sides):
    join = lambda c: " = ".join(R(x) for x in c)
    rows = [{"latex": join(sides), "why": "Start"}]
    cur = _run(sides, join, rows)
    vals = [value(c) for c in cur]
    equal = all(vals[k] == vals[k + 1] for k in range(len(vals) - 1))
    if equal:
        rows.append({"latex": join(cur) + " \\;\\checkmark", "why": "Both sides are equal"})
    else:
        rows.append({"latex": " \\neq ".join(R(c) for c in cur), "why": "The sides are not equal"})
    return rows, equal


# --------------------------------------------------------------------------- linear equations
def _vars(n, acc):
    if isinstance(n, Var):
        acc.add(n.name)
    for f in getattr(n, "__dataclass_fields__", {}):
        c = getattr(n, f)
        if isinstance(c, tuple):
            for item in c:
                _vars(item[1] if isinstance(item, tuple) else item, acc)
        elif hasattr(c, "__dataclass_fields__"):
            _vars(c, acc)
    return acc


def _sym(n, x):
    if isinstance(n, Num):
        return sp.Rational(n.v.numerator, n.v.denominator)
    if isinstance(n, Var):
        return x
    if isinstance(n, Neg):
        return -_sym(n.x, x)
    if isinstance(n, Group):
        return _sym(n.x, x)
    if isinstance(n, Frac):
        return _sym(n.n, x) / _sym(n.d, x)
    if isinstance(n, Sqrt):
        return sp.sqrt(_sym(n.x, x))
    if isinstance(n, Pow):
        return _sym(n.b, x) ** _sym(n.e, x)
    if isinstance(n, Sum):
        return sum((s * _sym(t, x) for s, t in n.terms), sp.Integer(0))
    if isinstance(n, Prod):
        r = sp.Integer(1)
        for op, t in n.factors:
            r = r / _sym(t, x) if op == "/" else r * _sym(t, x)
        return r
    raise Unsupported("cannot convert")


def _lin(expr, x):
    try:
        p = sp.Poly(sp.expand(expr), x)
    except Exception:
        raise Unsupported("not a polynomial")
    if p.degree() > 1:
        raise Unsupported("not linear")
    a, b = p.coeff_monomial(x), p.coeff_monomial(1)
    if not (a.is_Rational and b.is_Rational):
        raise Unsupported("irrational coefficients")
    return Fraction(int(a.p), int(a.q)), Fraction(int(b.p), int(b.q))


def lin_latex(a: Fraction, b: Fraction, v: str) -> str:
    out = ""
    if a != 0:
        coef = "" if a == 1 else ("-" if a == -1 else R(mk(a)))
        out = f"{coef}{v}"
    if b != 0:
        bs = R(mk(abs(b)))
        out = (out + (" + " if b > 0 else " - ") + bs) if out else (("" if b > 0 else "-") + bs)
    return out or "0"


def _linear(left, right, name):
    x = sp.Symbol(name)
    a1, b1 = _lin(_sym(left, x), x)
    a2, b2 = _lin(_sym(right, x), x)
    A, C = a1 - a2, b2 - b1
    if A == 0:
        raise Unsupported("no unique solution")
    sol = C / A

    rows = [{"latex": R(left) + " = " + R(right), "why": "Start"}]

    def add(latex, why):
        if latex.replace(" ", "") != rows[-1]["latex"].replace(" ", ""):
            rows.append({"latex": latex, "why": why})

    add(f"{lin_latex(a1, b1, name)} = {lin_latex(a2, b2, name)}", "Simplify both sides")
    if a2 != 0:
        verb = "Subtract" if a2 > 0 else "Add"
        amt = ("" if abs(a2) == 1 else plain(abs(a2))) + name
        add(f"{lin_latex(A, b1, name)} = {lin_latex(0, b2, name)}", f"{verb} {amt} on both sides")
    if b1 != 0:
        verb = "Subtract" if b1 > 0 else "Add"
        add(f"{lin_latex(A, 0, name)} = {lin_latex(0, C, name)}", f"{verb} {plain(abs(b1))} on both sides")
    if A != 1:
        add(f"{name} = {R(mk(sol))}", f"Divide both sides by {plain(A)}")

    subs = {name: sol}
    lv, rv = value(left, subs), value(right, subs)
    if lv != rv:
        raise Unsupported("check failed")
    rows.append({
        "latex": f"{R(left, True, subs)} = {R(right, True, subs)} \\;\\Rightarrow\\; {R(mk(lv))} = {R(mk(rv))} \\;\\checkmark",
        "why": f"Check by substituting {name} = {plain(sol)}",
    })
    return rows, name, sol


def _linear_ineq(left, right, op: str, name: str):
    x = sp.Symbol(name)
    a1, b1 = _lin(_sym(left, x), x)
    a2, b2 = _lin(_sym(right, x), x)
    A, C = a1 - a2, b2 - b1
    if A == 0:
        raise Unsupported("no unique solution")
    sol = C / A

    op_latex_map = {"<": "<", ">": ">", "<=": "\\leq", ">=": "\\geq"}
    op_str_map = {"<": "<", ">": ">", "<=": "<=", ">=": ">="}
    flip_map = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}

    cur_latex = op_latex_map[op]
    rows = [{"latex": f"{R(left)} {cur_latex} {R(right)}", "why": "Start"}]

    def add(latex, why):
        if latex.replace(" ", "") != rows[-1]["latex"].replace(" ", ""):
            rows.append({"latex": latex, "why": why})

    add(f"{lin_latex(a1, b1, name)} {cur_latex} {lin_latex(a2, b2, name)}", "Simplify both sides")
    if a2 != 0:
        verb = "Subtract" if a2 > 0 else "Add"
        amt = ("" if abs(a2) == 1 else plain(abs(a2))) + name
        add(f"{lin_latex(A, b1, name)} {cur_latex} {lin_latex(0, b2, name)}", f"{verb} {amt} on both sides")
    if b1 != 0:
        verb = "Subtract" if b1 > 0 else "Add"
        add(f"{lin_latex(A, 0, name)} {cur_latex} {lin_latex(0, C, name)}", f"{verb} {plain(abs(b1))} on both sides")

    if A == 1:
        final_op = op
    elif A > 0:
        final_op = op
        final_latex = op_latex_map[final_op]
        add(f"{name} {final_latex} {R(mk(sol))}", f"Divide both sides by {plain(A)}")
    else:  # A < 0, multiplying/dividing by negative flips the inequality
        final_op = flip_map[op]
        final_latex = op_latex_map[final_op]
        add(f"{name} {final_latex} {R(mk(sol))}", f"Divide both sides by {plain(A)} and flip the inequality sign")

    final_latex = op_latex_map[final_op]
    answer_latex = f"{name} {final_latex} {R(mk(sol))}"
    expected_result = f"{name} {op_str_map[final_op]} {plain(sol)}"
    return rows, name, answer_latex, expected_result


# --------------------------------------------------------------------------- public API
FAIL_NOTE = "Step-by-step working isn't available for this expression."


def steps_for(latex: str) -> dict:
    out = {"ok": False, "answer": None, "steps": [], "note": FAIL_NOTE}
    try:
        ref = solve_latex(latex)
        if not ref["ok"]:
            out["note"] = ref["error"]
            return out
        toks = normalize_latex(latex).split()
        parts, ops = split_relation(toks)
        if len(parts) > 1 and parts[-1] == []:
            parts = parts[:-1]
            ops = ops[:-1]

        def build(allow):
            return [Parser(p, allow).parse() for p in parts]

        try:
            sides, vars_mode = build(False), False
        except NeedVars:
            sides, vars_mode = build(True), True

        if len(sides) == 1:
            if vars_mode:
                raise Unsupported("symbolic expression")
            rows, v = _expr(sides[0])
            if ref["kind"] != "value" or sp.Rational(ref["result"]) != sp.Rational(v.numerator, v.denominator):
                raise Unsupported("mismatch with solver")
            answer = R(mk(v))
        elif len(ops) == 1 and ops[0] in ("<", ">", "<=", ">="):
            op = ops[0]
            if not vars_mode:
                raise Unsupported("numeric inequality")
            names = set()
            for s in sides:
                _vars(s, names)
            if len(sides) != 2 or len(names) != 1:
                raise Unsupported("only one inequality in one unknown")
            rows, name, answer, exp_res = _linear_ineq(sides[0], sides[1], op, next(iter(names)))
            if ref["kind"] != "solve" or ref["result"].replace(" ", "") != exp_res.replace(" ", ""):
                raise Unsupported("mismatch with solver")
        elif not vars_mode:
            rows, equal = _check(sides)
            if ref["kind"] != "check" or ref["result"] != str(equal):
                raise Unsupported("mismatch with solver")
            answer = "\\text{True}" if equal else "\\text{False}"
        else:
            names = set()
            for s in sides:
                _vars(s, names)
            if len(sides) != 2 or len(names) != 1:
                raise Unsupported("only one equation in one unknown")
            rows, name, sol = _linear(sides[0], sides[1], next(iter(names)))
            if ref["kind"] != "solve" or ref["result"].replace(" ", "") != f"{name}={plain(sol)}":
                raise Unsupported("mismatch with solver")
            answer = f"{name} = {R(mk(sol))}"
        out.update(ok=True, answer=answer, steps=rows, note=None)
    except Exception as e:                      # never let working crash the app
        note = (ref.get("note") if ('ref' in locals() and isinstance(ref, dict) and ref.get("note")) else FAIL_NOTE)
        out.update(ok=False, answer=None, steps=[], note=note)
        out["_debug"] = f"{type(e).__name__}: {e}"
    return out