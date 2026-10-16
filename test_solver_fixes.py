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


def test_followup_negative_base_roots():
    # Negative base roots: odd roots evaluate via real_root, even roots reject as non-real
    r1 = solve_latex(r"\sqrt [ 3 ] { - 8 }")
    assert r1["ok"] is True, f"Expected ok=True for \sqrt[3]{-8}, got {r1}"
    assert r1["result"] == "-2", f"Expected -2, got {r1['result']}"

    r2 = solve_latex(r"\sqrt [ 3 ] { 2 7 }")
    assert r2["ok"] is True
    assert r2["result"] == "3"

    r3 = solve_latex(r"\sqrt [ 4 ] { 1 6 }")
    assert r3["ok"] is True
    assert r3["result"] == "2"

    r4 = solve_latex(r"\sqrt [ 4 ] { - 1 6 }")
    assert r4["ok"] is False, f"Expected ok=False for \sqrt[4]{-16}, got {r4}"
    assert "not a real number" in r4["error"], f"Expected non-real error message, got {r4['error']}"

    r5 = solve_latex(r"\sqrt { - 4 }")
    assert r5["ok"] is False, f"Expected ok=False for \sqrt{-4}, got {r5}"
    assert "not a real number" in r5["error"], f"Expected non-real error message, got {r5['error']}"
    print("PASS: Follow-up (negative base roots)")


def test_phase2_inequalities():
    # 1. Linear inequalities with steps
    cases_with_steps = [
        ("x + 3 < 5", "x < 2", False),
        (r"2 x - 1 \geq 7", "x >= 4", False),
        ("- x > 4", "x < -4", True),  # sign flip
        (r"3 x \leq 2 x + 5", "x <= 5", False),
        (r"3 x + 2 \leq 2 x + 7", "x <= 5", False),
        (r"\frac { x } { - 2 } > 3", "x < -6", True),  # sign flip
        ("2 ( x - 1 ) < x + 4", "x < 6", False),
    ]

    for latex, expected_sol, expect_flip in cases_with_steps:
        r = solve_latex(latex)
        assert r["ok"] is True, f"solve_latex failed for '{latex}': {r}"
        assert r["result"] == expected_sol, f"For '{latex}': expected '{expected_sol}', got '{r['result']}'"

        st = steps_for(latex)
        assert st["ok"] is True, f"steps_for failed for '{latex}': {st}"
        if expect_flip:
            assert any("flip" in step["why"].lower() for step in st["steps"]), \
                f"Expected sign flip explanation in steps for '{latex}': {st['steps']}"

    # 2. Compound / Quadratic inequalities (no steps, just answer)
    quad_cases = [
        ("x ^ 2 < 9", "-3 < x < 3"),
        ("x ^ 2 > 9", "x < -3 or x > 3"),
        (r"x ^ 2 - 4 \geq 0", "x <= -2 or x >= 2"),
    ]
    for latex, expected_sol in quad_cases:
        r = solve_latex(latex)
        assert r["ok"] is True, f"solve_latex failed for '{latex}': {r}"
        assert r["result"] == expected_sol, f"For '{latex}': expected '{expected_sol}', got '{r['result']}'"
        st = steps_for(latex)
        assert st["ok"] is False, f"Quadratic inequality '{latex}' should have no steps"

    # 3. Identities and contradictions
    r_id = solve_latex("2 x + 1 > 2 x")
    assert r_id["ok"] is True
    assert r_id["result"] == "true for all real x"
    st_id = steps_for("2 x + 1 > 2 x")
    assert st_id["ok"] is False

    r_contra = solve_latex("x < x - 1")
    assert r_contra["ok"] is True
    assert r_contra["result"] == "no solution"
    st_contra = steps_for("x < x - 1")
    assert st_contra["ok"] is False

    # 4. UI checks for inequalities
    # Equation mode accepts inequalities without warnings
    rev_eq = review("x + 3 < 5", "Equation")
    assert rev_eq["ok"] is True
    assert not rev_eq["warnings"], f"Equation mode should accept inequality without warning: {rev_eq['warnings']}"

    # Mixed = and inequality sign should warn
    rev_mixed = review("x = 2 < 3", "Equation")
    assert any("both" in w.lower() for w in rev_mixed["warnings"]), f"Expected mixed = and inequality warning: {rev_mixed['warnings']}"
    assert rev_mixed["needs_confirm"] is True

    print("PASS: Phase 2 (inequalities)")


def test_phase2_fixes_followup():
    # 1. ASCII <= and >= mapping
    r_le = solve_latex("x <= 5")
    assert r_le["ok"] is True and r_le["result"] == "x <= 5"
    st_le = steps_for("x <= 5")
    assert st_le["ok"] is True

    r_ge = solve_latex("x >= 5")
    assert r_ge["ok"] is True and r_ge["result"] == "x >= 5"
    st_ge = steps_for("x >= 5")
    assert st_ge["ok"] is True

    # 2. !=, \neq, \ne mapped to friendly not supported yet message
    for ineq_str in ("x != 5", r"x \neq 5", r"x \ne 5"):
        r = solve_latex(ineq_str)
        assert r["ok"] is False
        rev = review(ineq_str, "Equation")
        assert rev["error"] is not None
        assert "supported yet" in rev["error"].lower(), f"Expected not supported yet message for '{ineq_str}': {rev['error']}"

    # 3. Regression test: \left( and \right) with = must NOT warn about both = and ineq sign
    rev_left1 = review(r"\left( x \right) = 5", "Equation")
    assert rev_left1["ok"] is True
    assert not any("both" in w.lower() for w in rev_left1["warnings"]), f"False positive mixed warning: {rev_left1['warnings']}"

    rev_left2 = review(r"\left ( 2 x + 1 \right ) = 7", "Equation")
    assert rev_left2["ok"] is True
    assert not any("both" in w.lower() for w in rev_left2["warnings"]), f"False positive mixed warning: {rev_left2['warnings']}"

    # 4. Single-point solutions
    r_pt1 = solve_latex(r"x ^ 2 \leq 0")
    assert r_pt1["ok"] is True
    assert r_pt1["result"] == "x = 0", f"Expected 'x = 0', got {r_pt1['result']}"

    r_pt2 = solve_latex(r"( x - 1 ) ^ 2 \leq 0")
    assert r_pt2["ok"] is True
    assert r_pt2["result"] == "x = 1", f"Expected 'x = 1', got {r_pt2['result']}"

    r_empty = solve_latex("x ^ 2 < 0")
    assert r_empty["ok"] is True
    assert r_empty["result"] == "no solution"

    # 5. Chained inequalities
    r_chained = solve_latex("- 3 < x < 3")
    assert r_chained["ok"] is True
    assert r_chained["result"] == "-3 < x < 3", f"Expected '-3 < x < 3', got {r_chained['result']}"

    r_num_chain = solve_latex("1 < 2 < 3")
    assert r_num_chain["ok"] is True
    assert r_num_chain["result"] == "True"

    r_num1 = solve_latex("2 < 3")
    assert r_num1["ok"] is True
    assert r_num1["result"] == "True"

    r_num2 = solve_latex(r"5 \leq 4")
    assert r_num2["ok"] is True
    assert r_num2["result"] == "False"
    rev_num2 = review(r"5 \leq 4", "Arithmetic")
    assert any("false" in w.lower() for w in rev_num2["warnings"])

    print("PASS: Phase 2 fixes follow-up checks")


def test_latex_answers():
    # 1. 2 x + 6 = 90
    r1 = solve_latex("2 x + 6 = 90")
    assert r1["ok"] is True
    assert r1["latex"] == "x = 42"
    assert r1["result"] == "x = 42"

    # 2. \frac { 3 } { 4 } + \frac { 1 } { 2 }
    r2 = solve_latex(r"\frac { 3 } { 4 } + \frac { 1 } { 2 }")
    assert r2["ok"] is True
    assert r2["latex"] == r"\frac{5}{4} = 1.25"
    assert r2["result"] == "5/4"

    # 3. x ^ 2 = 9
    r3 = solve_latex("x ^ 2 = 9")
    assert r3["ok"] is True
    assert r3["latex"] == r"x = -3,\; 3"
    assert r3["result"] == "x = -3, 3"

    # 4. x ^ 2 + 1 = 0
    r4 = solve_latex("x ^ 2 + 1 = 0")
    assert r4["ok"] is True
    assert r4["latex"] == r"\text{no real solutions}"
    assert "no real solutions" in r4["result"]

    # 5. x ^ 3 - 2 x ^ 2 + 5 x = 8 (cubic: must produce short decimal answer, not closed form)
    r5 = solve_latex("x ^ 3 - 2 x ^ 2 + 5 x = 8")
    assert r5["ok"] is True
    assert r5["latex"] == r"x \approx 1.75217"
    assert len(r5["latex"]) < 30, f"Expected short decimal answer, got {r5['latex']}"
    assert r5.get("exact_latex") is not None, "Expected exact_latex in expander"
    assert "sqrt" in r5["exact_latex"].lower() or "frac" in r5["exact_latex"].lower()
    assert r5.get("note") is not None and r"\approx" in r5["note"], f"Expected decimal complex roots note: {r5.get('note')}"

    # 6. x + 3 < 5
    r6 = solve_latex("x + 3 < 5")
    assert r6["ok"] is True
    assert r6["latex"] == "x < 2"
    assert r6["result"] == "x < 2"

    # 7. x ^ 2 > 9
    r7 = solve_latex("x ^ 2 > 9")
    assert r7["ok"] is True
    assert r7["latex"] == r"x < -3 \text{ or } x > 3"
    assert r7["result"] == "x < -3 or x > 3"

    # 8. 2 x = 2 x
    r8 = solve_latex("2 x = 2 x")
    assert r8["ok"] is True
    assert r8["latex"] == r"\text{true for all real } x"
    assert r8["result"] == "true for all real x"

    # 9. x = x + 1
    r9 = solve_latex("x = x + 1")
    assert r9["ok"] is True
    assert r9["latex"] == r"\text{no solution}"
    assert r9["result"] == "x = no solution"

    print("PASS: UI Round (latex answers on all 9 required expressions)")


if __name__ == "__main__":
    test_phase1_identity()
    test_phase1_frac_aliases()
    test_phase1_trailing_equals()
    test_phase1_complex_roots()
    test_phase1_false_statement()
    test_phase1_nth_root()
    test_followup_negative_base_roots()
    test_phase2_inequalities()
    test_phase2_fixes_followup()
    test_latex_answers()
    print("\nALL PHASE 1, PHASE 2 & UI ROUND TESTS PASSED!")

