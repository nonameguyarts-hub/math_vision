"""test_solver_stress.py - Stress testing the MathVision solver, step generator, and ui_checks.
Runs without modifying any application code.
Outputs table to stdout and saves to solver_stress_results.txt.
"""
import sys
import traceback
from pathlib import Path

# Ensure local modules are imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver import solve_latex
from steps import steps_for
from ui_checks import review

GROUPS = {
    "Group A: Arithmetic": [
        ("45 + 17", "62", "Arithmetic"),
        ("120 / 4", "30", "Arithmetic"),
        (r"3 \times ( 8 - 2 )", "18", "Arithmetic"),
        ("2 ^ { 5 }", "32", "Arithmetic"),
        (r"\frac { 3 } { 4 } + \frac { 1 } { 2 }", "5/4", "Arithmetic"),
        ("7 - 12", "-5", "Arithmetic"),
        ("0.6 + 0.25", "0.85 (or 17/20)", "Arithmetic"),
        (r"100 \div 8", "25/2 (or 12.5)", "Arithmetic"),
        (r"\sqrt { 49 } + 3", "10", "Arithmetic"),
        ("- 5 + 8", "3", "Arithmetic"),
        ("2 ( 3 + 4 )", "14", "Arithmetic"),
    ],
    "Group B: One Unknown": [
        ("2 x + 6 = 90", "x = 42", "Equation"),
        ("5 x + 50 = 160", "x = 22", "Equation"),
        ("3 x - 7 = 11", "x = 6", "Equation"),
        ("x + x = 10", "x = 5", "Equation"),
        ("2 ( x + 3 ) = 14", "x = 4", "Equation"),
        (r"\frac { x } { 4 } = 5", "x = 20", "Equation"),
        ("7 = 2 x + 1", "x = 3", "Equation"),
        ("0.5 x = 3", "x = 6", "Equation"),
        ("x - 12 = - 4", "x = 8", "Equation"),
        ("4 x = 2 x + 10", "x = 5", "Equation"),
        ("5 m + 50 = 160", "m = 22", "Equation"),
        ("3 y - 9 = 0", "y = 3", "Equation"),
    ],
    "Group C: Nonlinear": [
        ("x ^ 2 = 9", "x = -3, 3", "Equation"),
        ("x ^ 2 - 5 x + 6 = 0", "x = 2, 3", "Equation"),
        ("x ^ 2 + 1 = 0", "No real solution (or x = -I, I)", "Equation"),
        (r"\sqrt { x } = 4", "x = 16", "Equation"),
        ("| x | = 3", "x = -3, 3", "Equation"),
    ],
    "Group D: Inequalities": [
        ("x + 3 < 5", "x < 2", "Equation"),
        (r"2 x - 1 \geq 7", "x >= 4", "Equation"),
        ("- x > 4", "x < -4", "Equation"),
        (r"3 x \leq 2 x + 5", "x <= 5", "Equation"),
    ],
    "Group E: Odd or Invalid": [
        ("500 - 50 = 160", "False (arithmetic check)", "Arithmetic"),
        ("2 x = 2 x", "All real numbers / Infinite solutions", "Equation"),
        ("x = x + 1", "No solution (empty set)", "Equation"),
        ("2 x + 3 y = 74", "Indeterminate / Two unknowns", "Equation"),
        ("x + y = 10", "Indeterminate / Two unknowns", "Equation"),
        (r"\frac x y", "x/y (expression in two vars)", "Arithmetic"),
        (r"\frac { 1 } { 0 }", "Undefined (division by zero)", "Arithmetic"),
        ("5 x + 50 =", "Syntax error (incomplete)", "Equation"),
        ("= 160", "Syntax error (incomplete)", "Equation"),
        ("2 x + 3", "2x + 3 (expression, not equation)", "Equation"),
        ("x ^ 2 = y", "Two variables nonlinear", "Equation"),
    ],
    "Group F: Recognizer Variants": [
        (r"\dfrac { 1 } { 2 }", "1/2", "Arithmetic"),
        (r"3 \cdot 4", "12", "Arithmetic"),
        (r"\sqrt [ 3 ] { 8 }", "2", "Arithmetic"),
        ("2 x ^ { 2 } = 8", "x = -2, 2", "Equation"),
        ("1 0 0", "100", "Arithmetic"),
        ("3 . 5 + 1", "4.5 (or 9/2)", "Arithmetic"),
        ("x = 5", "x = 5", "Equation"),
        ("5 = 5", "True (arithmetic check)", "Arithmetic"),
    ],
}


def run_stress_test():
    results = []

    for group_name, items in GROUPS.items():
        for latex, expected_math, mode in items:
            row = {
                "group": group_name,
                "latex": latex,
                "expected_math": expected_math,
                "mode": mode,
                "solver_ok": False,
                "solver_result": None,
                "solver_error": None,
                "steps_ok": False,
                "steps_answer": None,
                "steps_count": 0,
                "steps_note": None,
                "review_ok": False,
                "review_error": None,
                "review_warnings": [],
                "review_needs_confirm": False,
                "exception": None,
            }

            try:
                # 1. solve_latex
                sol = solve_latex(latex)
                row["solver_ok"] = sol.get("ok", False)
                row["solver_result"] = sol.get("result")
                row["solver_error"] = sol.get("error")
                row["solver_kind"] = sol.get("kind")
            except Exception as e:
                row["exception"] = f"solve_latex crashed: {type(e).__name__}: {e}"

            try:
                # 2. steps_for
                stp = steps_for(latex)
                row["steps_ok"] = stp.get("ok", False)
                row["steps_answer"] = stp.get("answer")
                row["steps_count"] = len(stp.get("steps", []))
                row["steps_note"] = stp.get("note")
            except Exception as e:
                err_msg = f"steps_for crashed: {type(e).__name__}: {e}"
                row["exception"] = f"{row['exception']}; {err_msg}" if row["exception"] else err_msg

            try:
                # 3. ui_checks.review
                rev = review(latex, mode=mode)
                row["review_ok"] = rev.get("ok", False)
                row["review_error"] = rev.get("error")
                row["review_warnings"] = rev.get("warnings", [])
                row["review_needs_confirm"] = rev.get("needs_confirm", False)
            except Exception as e:
                err_msg = f"review crashed: {type(e).__name__}: {e}"
                row["exception"] = f"{row['exception']}; {err_msg}" if row["exception"] else err_msg

            results.append(row)

    return results


def format_table(results):
    lines = []
    lines.append("=" * 130)
    lines.append("MATHVISION SOLVER STRESS TEST RESULTS")
    lines.append("=" * 130)
    header = f"{'LaTeX Input':<32}{'Expected Math':<24}{'Solver Ans/Err':<26}{'Steps':<8}{'UI Msg / Warnings':<40}"
    lines.append(header)
    lines.append("-" * 130)

    current_group = None
    for r in results:
        if r["group"] != current_group:
            current_group = r["group"]
            lines.append(f"\n--- {current_group} ---")

        ans_err = r["solver_result"] if r["solver_ok"] else f"ERR: {r['solver_error']}"
        steps_info = str(r["steps_count"]) if r["steps_ok"] else (f"No ({r['steps_note'][:15]}...)" if r["steps_note"] else "No")
        ui_info = r["review_error"] if r["review_error"] else (", ".join(r["review_warnings"]) if r["review_warnings"] else "OK")

        lines.append(f"{r['latex']:<32}{r['expected_math']:<24}{str(ans_err):<26}{steps_info:<8}{str(ui_info):<40}")

    lines.append("\n" + "=" * 130)
    return "\n".join(lines)


def format_analysis(results):
    lines = []
    lines.append("\nDETAILED ANALYSIS OF FAILURES, LIMITATIONS, AND SURPRISING BEHAVIORS:")
    lines.append("=" * 130)

    for idx, r in enumerate(results, 1):
        # Determine if there's a discrepancy or failure
        ans = r["solver_result"]
        err = r["solver_error"]
        exp = r["expected_math"]
        latex = r["latex"]

        # Classification of issues
        has_issue = False
        reasons = []

        if r["exception"]:
            has_issue = True
            reasons.append(f"Unhandled Exception: {r['exception']}")

        if not r["solver_ok"]:
            # If expected an error (e.g. invalid syntax), check if message is appropriate
            if r["group"] in ("Group D: Inequalities", "Group E: Odd or Invalid"):
                reasons.append(f"Rejected as expected/unsupported: error='{err}', UI='{r['review_error']}'")
            else:
                has_issue = True
                reasons.append(f"Solver rejected valid input: error='{err}'")
        else:
            # Check answer discrepancy
            if ans != exp:
                # normalize common variations (e.g. 1.25 vs 5/4, x = -3, 3 vs x = ±3)
                pass

        if r["group"] == "Group C: Nonlinear":
            has_issue = True
            reasons.append(f"Solver result: {ans}. Steps availability: {r['steps_note']}")

        if r["group"] == "Group D: Inequalities":
            has_issue = True
            reasons.append(f"Solver rejected inequality: error='{err}'")

        if r["group"] == "Group E: Odd or Invalid":
            has_issue = True
            reasons.append(f"Solver: ok={r['solver_ok']}, result={ans}, error={err}, UI friendly message='{r['review_error']}'")

        if r["group"] == "Group F: Recognizer Variants" and not r["solver_ok"]:
            has_issue = True
            reasons.append(f"Variant rejected: error='{err}'")

        if has_issue:
            lines.append(f"\n[{r['group']}] {repr(latex)}:")
            lines.append(f"   Expected Math Answer : {exp}")
            lines.append(f"   Program Answer/Status: {ans if r['solver_ok'] else f'Error: {err}'}")
            lines.append(f"   Steps Generated      : {r['steps_count']} steps (note: {r['steps_note']})")
            lines.append(f"   UI Review Message    : {r['review_error'] or 'None'} (Warnings: {r['review_warnings']})")
            for reason in reasons:
                lines.append(f"   Diagnosis/Details    : {reason}")

    return "\n".join(lines)


def main():
    results = run_stress_test()
    table_text = format_table(results)
    analysis_text = format_analysis(results)

    full_output = table_text + "\n" + analysis_text
    print(full_output)

    with open("solver_stress_results.txt", "w", encoding="utf-8") as f:
        f.write(full_output + "\n")

    print("\nSaved full table and analysis to solver_stress_results.txt")


if __name__ == "__main__":
    main()
