"""
MathVision UI - University Project Redesign
Photograph handwritten mathematics. Check it. Solve it.
"""
import os
import time
import html as html_lib
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(
    page_title="MathVision",
    layout="centered",
    initial_sidebar_state="collapsed"
)

from preprocess import looks_sideways, preprocess
from recognizer import DEFAULT_MAX_SIDE, prep, recognize
from steps import steps_for
from ui_checks import MODES, detect_mode, has_nested_radicals, review
from ui_components import (
    html,
    load_css,
    render_header,
    render_stepper,
    render_empty_photo_tips,
    render_skeleton,
    render_callout,
    render_footer
)

# Load global CSS
load_css()

# Environment settings
BACKEND = os.environ.get("MATHVISION_BACKEND", "uni-mumer")
DEMO_MODE = os.environ.get("MATHVISION_DEMO", "0") == "1"
if BACKEND == "legacy":
    import recognizer_old

# Render Header
render_header()

# Determine stepper step based on session state
has_photo = ("uploaded_image" in st.session_state and st.session_state["uploaded_image"] is not None)
has_latex = bool(st.session_state.get("latex", "").strip())
has_solved = ("solved_for" in st.session_state and st.session_state["solved_for"] is not None)

if has_solved:
    current_step = 3
elif has_photo or has_latex:
    current_step = 2
else:
    current_step = 1

render_stepper(current_step)

# --------------------------------------------------------------------------
# Step 1: Upload Photo Card
# --------------------------------------------------------------------------
with st.container(border=True, key="card_step1"):
    html('<div class="mv-card-header"><h3 class="mv-card-title">1. Upload handwritten equation</h3></div>')

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        file = st.file_uploader("Your photo", type=["png", "jpg", "jpeg"], help="Upload an image containing handwritten math")
        turn = st.radio(
            "Rotate photo",
            ["None", "Left 90°", "Right 90°", "180°"],
            horizontal=True,
            help="Use this if the writing is sideways or upside down."
        )

    active_image = None
    if file is not None:
        try:
            raw_img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
            rot_angle = {"None": 0, "Left 90°": 90, "Right 90°": 270, "180°": 180}[turn]
            active_image = raw_img.rotate(rot_angle, expand=True)
            st.session_state["uploaded_image"] = active_image
        except Exception:
            render_callout("Couldn't open this image file. Please upload a valid PNG, JPG, or JPEG photo.", kind="error")
            active_image = None

    with col2:
        if active_image is not None:
            if looks_sideways(active_image):
                render_callout("The writing looks sideways. Try Left 90° or Right 90° above.", kind="info")
            html('<div class="mv-photo-frame">')
            st.image(active_image, caption="Uploaded photo", use_container_width=True)
            html('</div>')
        else:
            render_empty_photo_tips()

# --------------------------------------------------------------------------
# Step 2: Check & Edit Reading Card
# --------------------------------------------------------------------------
with st.container(border=True, key="card_step2"):
    html('<div class="mv-card-header"><h3 class="mv-card-title">2. Check &amp; edit reading</h3></div>')

    if active_image is not None:
        if st.button("Read equation", type="primary", use_container_width=True):
            placeholder = st.empty()
            with placeholder.container():
                render_skeleton()
            with st.spinner("Recognising equation in image..."):
                if DEMO_MODE:
                    stub_sleep = float(os.environ.get("MATHVISION_STUB_SLEEP", "0"))
                    if stub_sleep > 0:
                        time.sleep(stub_sleep)
                    raw_latex = os.environ.get("MATHVISION_STUB_LATEX", "5 x + 50 = 160")
                    crop_img = active_image
                else:
                    rot_angle = {"None": 0, "Left 90°": 90, "Right 90°": 270, "180°": 180}[turn]
                    if BACKEND == "legacy":
                        raw_latex = recognizer_old.recognize(active_image, rotate=rot_angle)
                        crop_img = active_image
                    else:
                        crop_img = prep(active_image, rotate=rot_angle, max_side=DEFAULT_MAX_SIDE)
                        raw_latex = recognize(active_image, rotate=rot_angle, crop=crop_img)

            placeholder.empty()

            st.session_state["latex"] = raw_latex
            st.session_state["crop_img"] = crop_img
            st.session_state.pop("solved_for", None)

            prep_info = getattr(crop_img, "prep_info", None)
            if prep_info:
                if prep_info.get("warning"):
                    st.session_state["prep_warning"] = prep_info["warning"]
                if prep_info.get("num_lines", 1) > 1:
                    st.session_state["multi_line_warning"] = (
                        "Multiple lines of writing detected. Recognition works best with "
                        "one line at a time; if the reading looks wrong, crop or retake."
                    )
            elif getattr(crop_img, "warning", None):
                st.session_state["prep_warning"] = crop_img.warning

    if st.session_state.get("prep_warning"):
        render_callout(st.session_state["prep_warning"], kind="warning")
    if st.session_state.get("multi_line_warning"):
        render_callout(st.session_state["multi_line_warning"], kind="warning")

    if st.session_state.get("crop_img") is not None:
        with st.expander("What the model sees"):
            st.image(st.session_state["crop_img"], caption="Preprocessed crop", use_container_width=True)

    # LaTeX Editor & Quick Symbol Palette
    current_latex = st.session_state.get("latex", "")
    latex = st.text_input(
        "Recognized LaTeX (editable)",
        value=current_latex,
        help="Edit the equation if the model misread any handwritten symbol."
    )
    if latex != current_latex:
        st.session_state["latex"] = latex
        st.session_state.pop("solved_for", None)

    # Symbol insertion palette
    st.caption("Quick insert:")
    sym_cols = st.columns(6)
    symbols = [
        (r"\frac{a}{b}", "a/b"),
        (r"\sqrt{x}", "√x"),
        (r"^2", "x²"),
        (r"^{n}", "xⁿ"),
        (r"\times", "×"),
        (r"\pi", "π")
    ]
    for idx, (sym_code, sym_label) in enumerate(symbols):
        with sym_cols[idx]:
            if st.button(sym_label, key=f"quick_sym_{idx}", help=f"Insert {sym_code}", use_container_width=True):
                st.session_state["latex"] = latex + (" " if latex and not latex.endswith(" ") else "") + sym_code
                st.rerun()

    if latex.strip():
        try:
            st.latex(latex)
        except Exception:
            render_callout("Couldn't preview LaTeX. Check for unmatched brackets or syntax errors.", kind="error")

    # Advanced Mode Override
    with st.expander("Advanced options"):
        mode_override = st.radio(
            "Interpretation Mode",
            ["Auto", "Equation", "Arithmetic"],
            index=0,
            horizontal=True,
            help="Auto detects Equation mode if letters are present, otherwise Arithmetic."
        )

    rev = review(latex, mode_override)
    for w in rev["warnings"]:
        render_callout(w, kind="warning")

    confirmed = True
    if rev["needs_confirm"]:
        confirmed = st.checkbox("The symbols match my photo")

    solve_disabled = not bool(latex.strip())
    if st.button("Solve", type="primary", disabled=solve_disabled, use_container_width=True):
        st.session_state["solved_for"] = (latex, mode_override)

# --------------------------------------------------------------------------
# Step 3: Answer & Steps Card (Shown only for exact text that was solved)
# --------------------------------------------------------------------------
if st.session_state.get("solved_for") == (latex, mode_override):
    with st.container(border=True, key="card_step3"):
        html('<div class="mv-card-header"><h3 class="mv-card-title">3. Solution</h3></div>')

        if not rev["ok"]:
            render_callout(f"{rev['error']} Edit the LaTeX above and try again.", kind="error")
        elif not confirmed:
            render_callout("Tick the box above once you've checked the symbols.", kind="info")
        else:
            res = rev["result"]

            # Store in session recent solutions
            if "recent_solutions" not in st.session_state:
                st.session_state["recent_solutions"] = []
            sol_display = res.get("latex") or res.get("result") or ""
            recent_entry = (latex, sol_display)
            if recent_entry not in st.session_state["recent_solutions"]:
                st.session_state["recent_solutions"].append(recent_entry)
                if len(st.session_state["recent_solutions"]) > 5:
                    st.session_state["recent_solutions"].pop(0)

            # Verification Chips
            chips_html = []
            if res.get("kind") == "check":
                chips_html.append('<span class="mv-chip">Checked numerically</span>')
            elif res.get("kind") == "solve":
                chips_html.append('<span class="mv-chip">Equation Solved</span>')
            elif res.get("kind") == "value":
                chips_html.append('<span class="mv-chip">Arithmetic Evaluated</span>')
            elif res.get("kind") == "identity":
                chips_html.append('<span class="mv-chip">Identity</span>')

            chips_str = f'<div class="mv-chips-row">{"".join(chips_html)}</div>' if chips_html else ""

            # Answer Card Container
            with st.container(border=True, key="card_answer"):
                html(f"""
<div class="mv-answer-header">
<span class="mv-answer-label">Answer</span>
{chips_str}
</div>
""")
                rendered_answer = False
                if res.get("latex"):
                    try:
                        st.latex(res["latex"])
                        rendered_answer = True
                    except Exception:
                        rendered_answer = False
                if not rendered_answer:
                    st.write(res["result"])

                exact = res.get("exact_latex")
                if exact and len(exact) <= 120 and not has_nested_radicals(exact):
                    with st.expander("Exact form"):
                        st.latex(exact)
                elif res.get("approx") is not None and not res.get("latex"):
                    st.caption(f"About {res['approx']}")

                if res.get("note"):
                    st.caption(res["note"])

            # Step-by-Step Working Timeline
            out = steps_for(latex)
            if out["ok"]:
                html('<h4 style="color: #1F3A5F; margin-top: 1.5rem; margin-bottom: 0.5rem;">Step-by-step working</h4>')
                for i, row in enumerate(out["steps"], 1):
                    with st.container(border=True, key=f"step_item_{i}"):
                        st.markdown(f"**Step {i}**")
                        st.latex(row["latex"])
                        if row.get("why"):
                            html(f'<div class="mv-timeline-why">{html_lib.escape(row["why"])}</div>')
            elif out.get("note"):
                st.info(out["note"])

            # Session Recent Solutions
            if st.session_state.get("recent_solutions"):
                with st.expander("Recent solutions (this session)"):
                    for past_latex, past_sol in reversed(st.session_state["recent_solutions"]):
                        st.markdown(f"- `${past_latex}$ $\\to$ **`${past_sol}`**")

            # Solution Export Options
            with st.expander("Export solution"):
                md_report = f"# MathVision Solution Report\n\n"
                md_report += f"**Equation / Expression:** `{latex}`\n\n"
                md_report += f"**Answer:** {res.get('result', '')}\n\n"
                if out["ok"]:
                    md_report += "### Steps:\n"
                    for i, row in enumerate(out["steps"], 1):
                        md_report += f"{i}. `{row['latex']}` — {row['why']}\n"
                st.download_button(
                    label="Download solution (.md)",
                    data=md_report,
                    file_name="mathvision_solution.md",
                    mime="text/markdown",
                    use_container_width=True
                )

# Render Footer
render_footer()
