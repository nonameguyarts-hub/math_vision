import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui_checks import detect_mode, has_nested_radicals, review

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
    # Auto mode cases (no letters -> Arithmetic, letters present -> Equation)
    ("2 x + 3 = 7", "Auto", True, False, False),
    ("1 + 2", "Auto", True, False, False),
    ("205 = x ^ 4 + 9 x ^ 3 + 13 x ^ 2 - x", "Auto", True, False, False),
    ("205 = x ^ 4 + 9 x ^ 3 + 13 x ^ 2 - x", "Equation", True, False, False),
    ("2 ^ { 7 0 }", "Auto", True, True, True),                      # huge result in Auto mode
]
bad = 0
for latex, mode, ok, warned, confirm in cases:
    r = review(latex, mode)
    got = (r["ok"], bool(r["warnings"]), r["needs_confirm"])
    if got != (ok, warned, confirm):
        bad += 1
        print("FAIL |", repr(latex), mode, got, "expected", (ok, warned, confirm))

# Check default argument for review() is Auto
r_def = review("205 = x ^ 4 + 9 x ^ 3 + 13 x ^ 2 - x")
if (r_def["ok"], bool(r_def["warnings"]), r_def["needs_confirm"]) != (True, False, False):
    bad += 1
    print("FAIL default mode for review() | got", (r_def["ok"], bool(r_def["warnings"]), r_def["needs_confirm"]))

# Test detect_mode
detect_cases = [
    ("1 2 \\times 3", "Arithmetic"),
    ("2 x 3", "Arithmetic"),
    (r"\frac{1}{2} + 3", "Arithmetic"),
    ("2 x + 3 = 7", "Equation"),
    ("205 = x ^ 4 + 9 x ^ 3 + 13 x ^ 2 - x", "Equation"),
]
for s, exp in detect_cases:
    if detect_mode(s) != exp:
        bad += 1
        print("FAIL detect_mode |", repr(s), "got", detect_mode(s), "expected", exp)

# Test has_nested_radicals
radical_cases = [
    (r"\sqrt{2}", False),
    (r"\frac{-1 + \sqrt{5}}{2}", False),
    (r"\sqrt{1 + \sqrt{2}}", True),
    (r"\sqrt{\frac{1}{\sqrt{2}}}", True),
]
for s, exp in radical_cases:
    if has_nested_radicals(s) != exp:
        bad += 1
        print("FAIL has_nested_radicals |", repr(s), "got", has_nested_radicals(s), "expected", exp)

print(f"UI check tests: {len(cases) + 1 + len(detect_cases) + len(radical_cases) - bad}/{len(cases) + 1 + len(detect_cases) + len(radical_cases)} passed")
from ui_checks import friendly
for err, must in [("unsupported token _", "Subscripts"), ("unbalanced braces", "brackets"), ("empty", "Nothing to solve")]:
    if must.lower() not in friendly(err).lower():
        bad += 1; print("FAIL friendly |", err, "->", friendly(err))
print("friendly messages checked")
sys.exit(1 if bad else 0)