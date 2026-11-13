import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import solver
from solver import solve_latex, normalize_latex
from steps import steps_for
from ui_checks import review

test_cases = [
    # Recognizer real outputs
    ("2 x + 3 y = 74", "Equation", "Recognizer Photo 1 output (two unknowns)"),
    ("5 x + 50 = 160", "Equation", "Recognizer Photo 2 (5x+50=160)"),
    ("500 - 50 = 160", "Arithmetic", "Recognizer Photo 2 alternative (500-50=160)"),
    ("5 x + 50 = 170", "Equation", "Ground truth Photo 2 (5x+50=170)"),

    # Multi-digit check
    ("74 + 170", "Arithmetic", "Multi-digit preservation check"),

    # Strings requested by user
    (r"\frac x y", "Arithmetic", "Fraction with bare variables"),
    (r"\frac { 1 } { 2 } + 3 = x", "Equation", "Fraction arithmetic linear equation"),
    ("x ^ 2 = 9", "Equation", "Quadratic equation"),
    (r"\sqrt { 16 }", "Arithmetic", "Exact square root"),
    (r"3 \times 4", "Arithmetic", "Times symbol"),
    ("2 x + 3 = 11", "Equation", "Standard linear equation"),
    ("205 = x ^ 4 + 9 x ^ 3 + 13 x ^ 2 - x", "Auto", "Quartic equation from red-ink photo"),
]

print("=" * 70)
print("DOWNSTREAM PIPELINE VERIFICATION")
print("=" * 70)

for expr, mode, note in test_cases:
    print(f"\nExpression: {repr(expr)} | Mode: {mode} ({note})")
    
    # 1. Check normalize_latex
    norm_toks = normalize_latex(expr).split()
    print(f"   Tokens in normalize_latex: {norm_toks}")

    # 2. Check solver.solve_latex
    sol = solve_latex(expr)
    print(f"   solve_latex() -> ok={sol['ok']}, kind={sol['kind']}, result={repr(sol['result'])}, error={repr(sol['error'])}")

    # 3. Check steps.steps_for
    stp = steps_for(expr)
    print(f"   steps_for()   -> ok={stp['ok']}, answer={repr(stp['answer'])}, note={repr(stp['note'])}, steps_count={len(stp.get('steps', []))}")

    # 4. Check ui_checks.review
    rev = review(expr, mode)
    print(f"   review()      -> ok={rev['ok']}, warnings={rev['warnings']}, needs_confirm={rev['needs_confirm']}")

print("\n" + "=" * 70)
