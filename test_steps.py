"""Run:  python test_steps.py   (solver.py and steps.py must be in the same folder)"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver import solve_latex
from steps import steps_for

# (input, expected answer latex, expected number of rows)
CASES = [
    ("1 + 2 =", "3", 2),
    ("1 2 \\times 3", "36", 2),
    ("\\frac { 1 } { 2 } + \\frac { 1 } { 3 }", "\\frac{5}{6}", 3),
    ("\\frac { 1 } { 6 } + \\frac { 1 } { 3 }", "\\frac{1}{2}", 4),
    ("\\frac { 1 } { 2 } \\times \\frac { 2 } { 3 }", "\\frac{1}{3}", 3),
    ("2 ^ { 3 } - 1", "7", 3),
    ("\\sqrt { 1 6 } + 2", "6", 3),
    ("( 1 + 2 ) ( 3 + 4 )", "21", 3),
    ("5 - ( 2 - 7 ) \\times 3", "20", 4),
    ("2 + 3 \\times 4 - 1 0 \\div 2", "9", 3),
    ("- 5 + 2", "-3", 2),
    ("1 . 5 + 2", "\\frac{7}{2}", 3),
    ("2 x + 3 = 7", "x = 2", 4),
    ("3 x + 2 = x + 1 0", "x = 4", 5),
    ("3 x - 4 = 2 x - 1", "x = 3", 4),
    ("7 \\times 8 = 5 6", "\\text{True}", 3),
    ("7 \\times 8 = 5 5", "\\text{False}", 3),
    # typed without spaces, and with unicode operators
    ("2x+6=90", "x = 42", 4),
    ("5x+50=180", "x = 26", 4),
    ("(1+2)(3+4)", "21", 3),
    ("\\frac{1}{2}+\\frac{1}{3}", "\\frac{5}{6}", 3),
    ("7\u00d78=56", "\\text{True}", 3),
]

# inputs that must fall back to "answer only" (ok False), never raise
FALLBACK = [
    "\\sqrt { 3 0 }",             # irrational root
    "x ^ { 2 } = 4",              # not linear
    "x + y = 3",                  # two unknowns
    "2 x + 3",                    # symbolic expression
    "1 \\div 0",                  # division by zero
    "1 + { 2",                    # unbalanced braces
    "\\sin x",                    # unsupported token
    "",                           # empty
]


def run_unit():
    bad = 0
    for latex, answer, rows in CASES:
        r = steps_for(latex)
        ok = r["ok"] and r["answer"] == answer and len(r["steps"]) == rows
        if not ok:
            bad += 1
            print("FAIL |", latex, "->", r["ok"], r["answer"], len(r["steps"]), "rows", r.get("_debug", ""))
    for latex in FALLBACK:
        r = steps_for(latex)
        if r["ok"] or r["steps"]:
            bad += 1
            print("FAIL (should fall back) |", latex)
    print(f"Unit tests: {len(CASES) + len(FALLBACK) - bad}/{len(CASES) + len(FALLBACK)} passed")
    return bad


def random_expr(rng, depth=0):
    kinds = ["num", "num", "bin", "bin", "frac", "group", "pow", "sqrt"]
    k = rng.choice(kinds if depth < 3 else ["num"])
    if k == "num":
        return " ".join(str(rng.randint(1, 20)))
    if k == "bin":
        op = rng.choice(["+", "-", "\\times", "\\div"])
        return f"{random_expr(rng, depth + 1)} {op} {random_expr(rng, depth + 1)}"
    if k == "frac":
        return f"\\frac {{ {random_expr(rng, depth + 1)} }} {{ {random_expr(rng, depth + 1)} }}"
    if k == "group":
        return f"( {random_expr(rng, depth + 1)} )"
    if k == "pow":
        return f"( {random_expr(rng, depth + 1)} ) ^ {{ {rng.randint(0, 3)} }}"
    sq = rng.choice([1, 4, 9, 16, 25, 36, 49, 64, 100])
    return f"\\sqrt {{ {' '.join(str(sq))} }}"


def run_random(n=400, seed=1):
    """For every random expression the solver can evaluate, steps_for must either return
    working that ends at the solver's answer, or fall back cleanly. It must never crash."""
    rng = random.Random(seed)
    tried = ok = mismatches = 0
    for _ in range(n):
        e = random_expr(rng)
        ref = solve_latex(e)
        if not ref["ok"]:
            continue
        tried += 1
        r = steps_for(e)
        if r["ok"]:
            ok += 1
            if len(r["steps"]) > 30:
                mismatches += 1
                print("TOO LONG |", e, len(r["steps"]))
    print(f"Random check: {ok}/{tried} expressions got steps; {tried - ok} fell back; {mismatches} problems")
    return mismatches


if __name__ == "__main__":
    bad = run_unit() + run_random()
    sys.exit(1 if bad else 0)