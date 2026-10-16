import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver import solve_latex
from steps import steps_for
from ui_checks import review


def test_phase1_identity():
    # a) Identity bug: (lhs - rhs) == 0 -> kind "identity", true for all real x
    r1 = solve_latex("2 x = 2 x")
    assert r1["ok"] is True, f"Expected ok=True, got {r1}"
    assert r1["kind"] == "identity", f"Expected kind='identity', got {r1['kind']}"
    assert r1["result"] == "true for all real x", f"Expected 'true for all real x', got {r1['result']}"

    # Contradiction x = x + 1 must remain "no solution"
    r2 = solve_latex("x = x + 1")
    assert r2["ok"] is True, f"Expected ok=True, got {r2}"
    assert r2["kind"] == "solve", f"Expected kind='solve', got {r2['kind']}"
    assert r2["result"] == "x = no solution", f"Expected 'x = no solution', got {r2['result']}"

    # steps.py and ui_checks handling
    st = steps_for("2 x = 2 x")
    assert isinstance(st, dict) and not st["ok"], "steps_for should handle identity cleanly without crashing"

    rev_eq = review("2 x = 2 x", "Equation")
    assert rev_eq["ok"] is True
    assert not rev_eq["warnings"], f"Equation mode should not warn about identity: {rev_eq['warnings']}"

    rev_ar = review("2 x = 2 x", "Arithmetic")
    assert rev_ar["needs_confirm"] is True, "Arithmetic mode should warn when variables are present in identity"
    print("PASS: Phase 1a (identity & contradiction)")


def test_phase1_frac_aliases():
    # b) Alias \dfrac and \tfrac to \frac
    r1 = solve_latex(r"\dfrac { 1 } { 2 }")
    assert r1["ok"] is True, f"Expected ok=True, got {r1}"
    assert r1["result"] == "1/2", f"Expected 1/2, got {r1['result']}"

    r2 = solve_latex(r"\tfrac { 3 } { 4 } + \frac { 1 } { 4 }")
    assert r2["ok"] is True, f"Expected ok=True, got {r2}"
    assert r2["result"] == "1", f"Expected 1, got {r2['result']}"

    # Steps support
    st = steps_for(r"\dfrac { 1 } { 2 } + \tfrac { 1 } { 3 }")
    assert st["ok"] is True
    assert st["answer"] == r"\frac{5}{6}"
    print("PASS: Phase 1b (dfrac and tfrac aliases)")


def test_phase1_trailing_equals():
    # c) Trailing "="
    r1 = solve_latex("5 x + 50 =")
    assert r1["ok"] is False, f"Expected ok=False for '5 x + 50 =', got {r1}"
    assert r1["error"] == "the equation is incomplete: nothing after the equals sign", f"Got: {r1['error']}"

    r2 = solve_latex("x =")
    assert r2["ok"] is False
    assert r2["error"] == "the equation is incomplete: nothing after the equals sign"

    # Legitimate trailing "=" in arithmetic must NOT break
    r3 = solve_latex("1 + 2 =")
    assert r3["ok"] is True, f"Expected ok=True for '1 + 2 =', got {r3}"
    assert r3["result"] == "3"

    # Leading "=" remains syntax error
    r4 = solve_latex("= 160")
    assert r4["ok"] is False
    assert r4["error"] == "empty side of '='"
    print("PASS: Phase 1c (trailing '=' handling)")


def test_phase1_complex_roots():
    # d) Complex roots
    r1 = solve_latex("x ^ 2 + 1 = 0")
    assert r1["ok"] is True, f"Expected ok=True, got {r1}"
    assert "no real solutions" in r1["result"], f"Expected 'no real solutions' in result, got {r1['result']}"
    assert r1.get("note") == "Complex roots: -I, I", f"Expected complex roots note, got {r1.get('note')}"

    st = steps_for("x ^ 2 + 1 = 0")
    assert st["ok"] is False
    assert st["note"] == "Complex roots: -I, I"

    # Real roots should not have complex roots note
    r2 = solve_latex("x ^ 2 = 9")
    assert r2["ok"] is True
    assert r2["result"] == "x = -3, 3"
    assert r2.get("note") is None
    print("PASS: Phase 1d (complex roots)")


def test_phase1_false_statement():
    # e) False numeric statement
    r = solve_latex("500 - 50 = 160")
    assert r["ok"] is True
    assert r["kind"] == "check"
    assert r["result"] == "False"

    rev = review("500 - 50 = 160", "Arithmetic")
    assert rev["ok"] is True
    assert rev["needs_confirm"] is True
    assert any("false" in w.lower() for w in rev["warnings"]), f"Expected false warning in {rev['warnings']}"
    print("PASS: Phase 1e (false numeric statement)")


def test_phase1_nth_root():
    # f) \sqrt [ n ] { x } for integer n
    r1 = solve_latex(r"\sqrt [ 3 ] { 8 }")
    assert r1["ok"] is True, f"Expected ok=True, got {r1}"
    assert r1["result"] == "2", f"Expected 2, got {r1['result']}"

    r2 = solve_latex(r"\sqrt [ 4 ] { 8 1 }")
    assert r2["ok"] is True
    assert r2["result"] == "3"

    # Non-integer n must produce error
    r3 = solve_latex(r"\sqrt [ 2 . 5 ] { 8 }")
    assert r3["ok"] is False
    assert r3["error"] == "root index must be an integer"

    # Zero root index must produce error
    r4 = solve_latex(r"\sqrt [ 0 ] { 8 }")
    assert r4["ok"] is False
    assert r4["error"] == "root index cannot be zero"

    # Normal square root without brackets remains working
    r5 = solve_latex(r"\sqrt { 4 9 } + 3")
    assert r5["ok"] is True
    assert r5["result"] == "10"
    print("PASS: Phase 1f (nth root)")


if __name__ == "__main__":
    test_phase1_identity()
    test_phase1_frac_aliases()
    test_phase1_trailing_equals()
    test_phase1_complex_roots()
    test_phase1_false_statement()
    test_phase1_nth_root()
    print("\nALL PHASE 1 TESTS PASSED!")
