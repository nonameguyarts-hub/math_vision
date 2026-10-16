"""eval_recognizer.py - Resolution sweep and evaluation harness for MathVision recognizer.
Sweeps max_side across [512, 640, 768, 1024].
Scores exact-match and mathematical equivalence via SymPy parsing.
Outputs a summary table and list of wrong predictions to stdout and eval_results.txt.
"""
import csv
import difflib
import os
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp
from recognizer import recognize
from solver import normalize_latex, split_eq, to_sympy, ParseError

RESOLUTIONS = [512, 640, 768, 1024]
PHOTOS_DIR = Path("test_photos")
GROUND_TRUTH_FILE = Path("ground_truth.csv")
OUTPUT_FILE = Path("eval_results.txt")


def clean_str(s: str) -> str:
    """Strip all whitespace for exact match."""
    return "".join(str(s).split())


def character_diff(ground_truth: str, prediction: str):
    """Analyze character-by-character differences and confusion patterns."""
    gt_clean = clean_str(ground_truth)
    pred_clean = clean_str(prediction)
    matcher = difflib.SequenceMatcher(None, gt_clean, pred_clean)
    diff_parts = []
    diff_patterns = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            diff_parts.append(gt_clean[i1:i2])
        elif tag == "replace":
            diff_parts.append(f"[{gt_clean[i1:i2]} -> {pred_clean[j1:j2]}]")
            diff_patterns.append(f"'{gt_clean[i1:i2]}' read as '{pred_clean[j1:j2]}'")
        elif tag == "delete":
            diff_parts.append(f"[-{gt_clean[i1:i2]}]")
            diff_patterns.append(f"'{gt_clean[i1:i2]}' omitted")
        elif tag == "insert":
            diff_parts.append(f"[+{pred_clean[j1:j2]}]")
            diff_patterns.append(f"'{pred_clean[j1:j2]}' inserted")
    aligned = "".join(diff_parts)
    patterns = ", ".join(diff_patterns) if diff_patterns else "none"
    return aligned, patterns


def parse_to_expr_or_diff(latex_str: str):
    """Parse latex into (is_equation, sympy_expr). For equations, returns (True, simplify(lhs - rhs))."""
    toks = normalize_latex(latex_str).split()
    parts = split_eq(toks)
    if len(parts) == 1:
        return False, sp.simplify(to_sympy(parts[0]))
    elif len(parts) == 2:
        lhs = to_sympy(parts[0])
        rhs = to_sympy(parts[1])
        return True, sp.simplify(lhs - rhs)
    else:
        raise ParseError("Multiple equals signs not supported for equivalence check")


def are_equivalent(pred: str, gt: str) -> bool:
    """Check mathematical equivalence using project solver; fallback to exact match."""
    # First check exact match
    if clean_str(pred) == clean_str(gt):
        return True
    try:
        is_eq_p, expr_p = parse_to_expr_or_diff(pred)
        is_eq_g, expr_g = parse_to_expr_or_diff(gt)
        if is_eq_p != is_eq_g:
            return False
        if is_eq_p:
            # For equations, (lhs - rhs) matches up to sign after simplification
            diff_sub = sp.simplify(expr_p - expr_g)
            diff_add = sp.simplify(expr_p + expr_g)
            return (diff_sub == 0) or (diff_add == 0)
        else:
            # For expressions, difference simplifies to 0
            return sp.simplify(expr_p - expr_g) == 0
    except Exception:
        return clean_str(pred) == clean_str(gt)


def load_ground_truth(csv_path: Path):
    items = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["filename"].strip(), row["latex"].strip()))
    return items


def main():
    if not GROUND_TRUTH_FILE.exists():
        print(f"Error: {GROUND_TRUTH_FILE} does not exist.")
        sys.exit(1)

    dataset = load_ground_truth(GROUND_TRUTH_FILE)
    if not dataset:
        print("No items in ground_truth.csv")
        sys.exit(1)

    print(f"Loaded {len(dataset)} items from {GROUND_TRUTH_FILE}")

    # Warmup recognizer so model loading is not included in inference timings
    dummy_img = Image.open(PHOTOS_DIR / dataset[0][0])
    print("Warming up model on CPU...")
    _ = recognize(dummy_img, max_side=512)
    print("Warmup complete. Beginning resolution sweep...\n")

    summary_rows = []
    all_wrong_predictions = []

    for max_side in RESOLUTIONS:
        print(f"--- Running sweep for max_side = {max_side} ---")
        times = []
        exact_matches = 0
        equiv_matches = 0

        for idx, (filename, gt_latex) in enumerate(dataset):
            img_path = PHOTOS_DIR / filename
            if not img_path.exists():
                print(f"Warning: {img_path} not found. Skipping.")
                continue

            img = Image.open(img_path)
            t_start = time.time()
            pred_latex = recognize(img, max_side=max_side)
            elapsed = time.time() - t_start

            times.append(elapsed)

            is_exact = clean_str(pred_latex) == clean_str(gt_latex)
            is_equiv = are_equivalent(pred_latex, gt_latex)

            if is_exact:
                exact_matches += 1
            if is_equiv:
                equiv_matches += 1
            else:
                all_wrong_predictions.append({
                    "max_side": max_side,
                    "filename": filename,
                    "ground_truth": gt_latex,
                    "prediction": pred_latex
                })

            print(f"  [{filename}] {elapsed:.2f}s | Pred: {pred_latex} | Exact: {is_exact} | Equiv: {is_equiv}")

        mean_time = sum(times) / len(times) if times else 0.0
        exact_pct = (exact_matches / len(dataset)) * 100
        equiv_pct = (equiv_matches / len(dataset)) * 100

        summary_rows.append({
            "max_side": max_side,
            "exact_pct": exact_pct,
            "equiv_pct": equiv_pct,
            "mean_seconds": mean_time
        })

    # Prepare formatted results
    lines = []
    lines.append("=" * 70)
    lines.append("RESOLUTION SWEEP EVALUATION RESULTS")
    lines.append("=" * 70)
    lines.append(f"{'max_side':<12}{'Exact Match %':<16}{'Equivalent %':<16}{'Mean Sec/Photo':<16}")
    lines.append("-" * 70)
    for r in summary_rows:
        lines.append(f"{r['max_side']:<12}{r['exact_pct']:<16.1f}{r['equiv_pct']:<16.1f}{r['mean_seconds']:<16.2f}")
    lines.append("-" * 70)

    lines.append("\nWRONG PREDICTIONS (with character-level difference patterns):")
    if not all_wrong_predictions:
        lines.append("None (100% accuracy on all tested resolutions!)")
    else:
        for w in all_wrong_predictions:
            aligned, patterns = character_diff(w['ground_truth'], w['prediction'])
            lines.append(f"  [max_side={w['max_side']}] {w['filename']}:")
            lines.append(f"     Ground Truth : {w['ground_truth']}")
            lines.append(f"     Prediction   : {w['prediction']}")
            lines.append(f"     Char Diff    : {aligned}")
            lines.append(f"     Patterns     : {patterns}\n")

    output_text = "\n".join(lines)
    print("\n" + output_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")
    print(f"\nSaved evaluation summary and errors to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
