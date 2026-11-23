import os
import sys
from streamlit.testing.v1 import AppTest

def test_apptest_stubs():
    print("Testing Headless Streamlit AppTest with stubbed inputs...")
    os.environ["MATHVISION_DEMO"] = "1"
    test_photo = r"test_photos\WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg"

    # 1. Full Flow: Upload photo, click Read equation, click Solve with stub "5 x + 50 = 160"
    print("\n1. Testing photo upload, Read equation, and Solve for '5 x + 50 = 160':")
    os.environ["MATHVISION_STUB_LATEX"] = "5 x + 50 = 160"
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    assert len(at.file_uploader) > 0, "File uploader not found"
    at.file_uploader[0].upload(filename="photo.jpg", content=open(test_photo, "rb").read()).run()

    read_btns = [b for b in at.button if "Read equation" in b.label]
    assert len(read_btns) > 0, "Read equation button not found after upload"
    read_btns[0].click().run()

    # Check that recognized LaTeX was populated into text_input
    latex_inputs = [ti for ti in at.text_input if "Recognized LaTeX" in ti.label]
    assert len(latex_inputs) > 0 and latex_inputs[0].value == "5 x + 50 = 160", "Recognized LaTeX not populated"

    # Click Solve
    solve_btns = [b for b in at.button if b.label == "Solve"]
    assert len(solve_btns) > 0, "Solve button not found"
    solve_btns[0].click().run()

    # Verify answer and steps
    all_markdown = " ".join(m.value for m in at.markdown)
    all_latex = " ".join(l.value for l in at.latex)
    assert "x = 22" in all_markdown or "x = 22" in all_latex, "Expected answer 'x = 22' not found"
    assert "Step-by-step working" in all_markdown and "Step 1" in all_markdown, "Expected step-by-step working not found"
    print("  ✓ Solved '5 x + 50 = 160' -> answer 'x = 22' and step-by-step working verified")

    # 2. Check stub "2 x + 3 y = 74" (Two unknowns error)
    print("\n2. Testing stub '2 x + 3 y = 74':")
    at2 = AppTest.from_file("app.py", default_timeout=30)
    at2.run()
    ti2 = [ti for ti in at2.text_input if "Recognized LaTeX" in ti.label][0]
    ti2.input("2 x + 3 y = 74").run()
    sb2 = [b for b in at2.button if b.label == "Solve"][0]
    sb2.click().run()
    all_md2 = " ".join(m.value for m in at2.markdown)
    expected_err2 = "Only equations with one unknown (such as x) are supported."
    assert expected_err2 in all_md2, f"Expected error '{expected_err2}' not found: {all_md2}"
    print(f"  ✓ Handled '2 x + 3 y = 74' with verified error: '{expected_err2}'")

    # 3. Check stub "500 - 50 = 160" (False statement warning & solve check)
    print("\n3. Testing stub '500 - 50 = 160':")
    at3 = AppTest.from_file("app.py", default_timeout=30)
    at3.run()
    ti3 = [ti for ti in at3.text_input if "Recognized LaTeX" in ti.label][0]
    ti3.input("500 - 50 = 160").run()
    all_md3_pre = " ".join(m.value for m in at3.markdown)
    expected_warn3 = "The statement is false. The equation or numbers may have been misread from your photo."
    assert expected_warn3 in all_md3_pre, f"Expected warning '{expected_warn3}' not found"
    # Confirm checkbox
    cbs3 = [cb for cb in at3.checkbox if "The symbols match my photo" in cb.label]
    assert len(cbs3) > 0, "Confirmation checkbox not found"
    cbs3[0].check().run()
    sb3 = [b for b in at3.button if b.label == "Solve"][0]
    sb3.click().run()
    all_md3_post = " ".join(m.value for m in at3.markdown)
    all_latex3 = " ".join(l.value for l in at3.latex)
    assert "False" in all_md3_post or "\\text{False}" in all_latex3, "Expected result 'False' not found"
    print(f"  ✓ Found expected warning: '{expected_warn3[:45]}...' and verified answer 'False'")

    # 4. Check stub "4 5 + l 7" (Letter in arithmetic warning)
    print("\n4. Testing stub '4 5 + l 7':")
    at4 = AppTest.from_file("app.py", default_timeout=30)
    at4.run()
    ti4 = [ti for ti in at4.text_input if "Recognized LaTeX" in ti.label][0]
    ti4.input("4 5 + l 7").run()
    all_md4 = " ".join(m.value for m in at4.markdown)
    expected_warn4 = "This has letters but no equals sign. If it should be plain arithmetic, a digit may have been misread as a letter. Check against your photo. Did you mean: 45 + 17?"
    assert expected_warn4 in all_md4, f"Expected warning '{expected_warn4}' not found: {all_md4}"
    print(f"  ✓ Found expected warning: '{expected_warn4[:45]}...'")

    print("\nHeadless Streamlit AppTest passed all 4 cases!")

if __name__ == '__main__':
    test_apptest_stubs()
