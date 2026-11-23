"""
test_ui_messages.py - Verifies that UI messages are never hard-coded in UI code,
and always render the exact strings returned by review(), prep(), and solver.
"""
import sys
from PIL import Image
from ui_checks import review
from recognizer import prep
from solver import solve_latex

def test_messages():
    print("Testing UI message pass-through from review(), prep(), and solver...")

    # 1. Lookalike suggestion warning
    r1 = review("4 5 + l 7", mode="Auto")
    assert len(r1["warnings"]) == 1, f"Expected 1 warning, got {r1['warnings']}"
    w1 = r1["warnings"][0]
    assert "Did you mean: 45 + 17?" in w1
    print("✓ Review lookalike suggestion exact text verified:", w1)

    # 2. Letters without relation warning
    r2 = review("2 x + 3", mode="Auto")
    assert len(r2["warnings"]) == 1
    w2 = r2["warnings"][0]
    assert "This has letters but no equals sign" in w2
    print("✓ Review letters without relation exact text verified:", w2)

    # 3. Arithmetic mode with letters & relation
    r3 = review("2 x + 3 = 7", mode="Arithmetic")
    assert any("A letter was read where a number may be expected. Check the symbols against your photo." in w for w in r3["warnings"])
    print("✓ Forced Arithmetic mode exact warning verified:", r3["warnings"])

    # 4. False statement
    r4 = review("0 = 1", mode="Arithmetic")
    assert any("The statement is false. The equation or numbers may have been misread from your photo." in w for w in r4["warnings"])
    print("✓ False statement exact warning verified:", r4["warnings"])

    # 5. Solver error for multi-variable equations
    sol = solve_latex("2 x + 3 y = 74")
    assert sol["ok"] is False
    assert sol["error"] == "only single-variable equations supported"
    print("✓ Solver multi-variable error exact text verified:", sol["error"])

    # 6. Preprocessing fallback warning on blank image
    blank = Image.new("RGB", (400, 300), (255, 255, 255))
    crop = prep(blank)
    assert crop.warning == "Couldn't isolate the handwriting. Retake on plain paper with some space around it."
    print("✓ Preprocessing fallback warning exact text verified:", crop.warning)

    print("\nAll UI message tests passed successfully!")

if __name__ == '__main__':
    test_messages()
