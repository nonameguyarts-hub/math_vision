import os
import sys
from streamlit.testing.v1 import AppTest

def test_apptest_stubs():
    print("Testing Headless Streamlit AppTest with stubbed inputs...")
    os.environ["MATHVISION_DEMO"] = "1"
    
    stubs = [
        ("5 x + 50 = 160", "x = 22", True, []),
        ("2 x + 3 y = 74", None, False, ["only single-variable equations supported"]),
        ("500 - 50 = 160", "False", True, ["The statement is false. The equation or numbers may have been misread from your photo."]),
        ("4 5 + l 7", None, True, ["This has letters but no equals sign. If it should be plain arithmetic, a digit may have been misread as a letter. Check against your photo. Did you mean: 45 + 17?"])
    ]
    
    for latex_stub, expected_ans, expected_ok, expected_warnings in stubs:
        os.environ["MATHVISION_STUB_LATEX"] = latex_stub
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        
        # In the app, input latex into text_input
        # Find the latex text_input
        latex_input = None
        for ti in at.text_input:
            if "Recognized LaTeX" in ti.label:
                latex_input = ti
                break
        
        assert latex_input is not None, "Could not find LaTeX text_input widget"
        latex_input.input(latex_stub).run()
        
        # Check warnings
        if expected_warnings:
            rendered_markdown = " ".join(m.value for m in at.markdown)
            for w in expected_warnings:
                assert w in rendered_markdown, f"Expected warning '{w}' not found in rendered output: {rendered_markdown}"
                print(f"  ✓ Found expected warning: '{w[:40]}...'")
        
        # If confirm checkbox is present, check it
        confirm_box = None
        for cb in at.checkbox:
            if "The symbols match my photo" in cb.label:
                confirm_box = cb
                break
        if confirm_box is not None:
            confirm_box.check().run()
            
        # Click Solve button
        solve_btn = None
        for b in at.button:
            if b.label == "Solve":
                solve_btn = b
                break
        assert solve_btn is not None, "Could not find Solve button"
        solve_btn.click().run()
        
        rendered_all = " ".join(m.value for m in at.markdown)
        if expected_ans:
            # Check answer presence (either in markdown or latex)
            latex_vals = " ".join(l.value for l in at.latex)
            assert expected_ans in rendered_all or expected_ans in latex_vals, f"Expected answer '{expected_ans}' not found"
            print(f"  ✓ Solved '{latex_stub}' -> expected answer '{expected_ans}' verified")
        else:
            print(f"  ✓ Handled '{latex_stub}' with expected warning/error verified")

    print("\nHeadless Streamlit AppTest passed all 4 cases!")

if __name__ == '__main__':
    test_apptest_stubs()
