import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui_checks import review

cases = [
    ("1 2 \\times 3", "Arithmetic", True, False, False),            # plain arithmetic: no warning
    ("2 x 3", "Arithmetic", True, False, False),                    # 'x' as times: no warning
    ("2 q + 3", "Arithmetic", True, True, True),                    # letter in arithmetic mode: warn + confirm
    ("2 x + 3 = 7", "Arithmetic", True, True, True),
    ("2 x + 3 = 7", "Equation", True, False, False),
    ("1 + 2", "Equation", True, True, False),                       # warning, no confirm needed
    ("2 ^ { 7 0 }", "Arithmetic", True, True, True),                # huge result
    ("2x+3=7", "Equation", True, False, False),                     # typed without spaces
    ("1 + { 2", "Arithmetic", False, False, False),                 # unbalanced braces
    ("", "Arithmetic", False, False, False),
]
bad = 0
for latex, mode, ok, warned, confirm in cases:
    r = review(latex, mode)
    got = (r["ok"], bool(r["warnings"]), r["needs_confirm"])
    if got != (ok, warned, confirm):
        bad += 1
        print("FAIL |", repr(latex), mode, got, "expected", (ok, warned, confirm))
print(f"UI check tests: {len(cases) - bad}/{len(cases)} passed")
from ui_checks import friendly
for err, must in [("unsupported token _", "Subscripts"), ("unbalanced braces", "brackets"), ("empty", "Nothing to solve")]:
    if must.lower() not in friendly(err).lower():
        bad += 1; print("FAIL friendly |", err, "->", friendly(err))
print("friendly messages checked")
sys.exit(1 if bad else 0)