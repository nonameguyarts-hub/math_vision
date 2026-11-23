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
    PROJECT_TITLE,
    PROJECT_TAGLINE,
    INSTITUTION,
    TEAM_MEMBERS,
    GUIDE,
    YEAR,
    html,
    load_css,
    render_header,
    render_stepper,
    render_empty_photo_tips,
    render_skeleton,
    render_callout,
    render_footer,
    render_svg_pipeline
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

# Main Application Tabs
tab_solve, tab_how, tab_examples, tab_about = st.tabs([
    "Solve", "How it works", "Examples", "About"
])

# ==============================================================================
# TAB 1: SOLVE TAB
# ==============================================================================
with tab_solve:
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

        # Check if an example was selected from Examples tab
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
        elif "example_image_path" in st.session_state and st.session_state["example_image_path"]:
            ex_path = st.session_state["example_image_path"]
            if os.path.exists(ex_path):
                try:
                    raw_img = ImageOps.exif_transpose(Image.open(ex_path)).convert("RGB")
                    rot_angle = {"None": 0, "Left 90°": 90, "Right 90°": 270, "180°": 180}[turn]
                    active_image = raw_img.rotate(rot_angle, expand=True)
                    st.session_state["uploaded_image"] = active_image
                except Exception:
                    render_callout("Couldn't open example image file.", kind="error")
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
        html('<div class="mv-card-header"><h3 class="mv-card-title">2. Check the reading</h3></div>')

        crop = None
        if active_image is not None and BACKEND != "legacy":
            try:
                crop = prep(active_image, max_side=DEFAULT_MAX_SIDE)
                if getattr(crop, "warning", None):
                    render_callout(crop.warning, kind="warning")
                if getattr(crop, "num_lines", 1) > 1:
                    st.caption("Multiple lines of writing detected.")
            except Exception as e:
                render_callout(f"No handwriting found: {e}", kind="warning")

        if active_image is not None:
            with st.expander("What the model sees"):
                try:
                    if BACKEND == "legacy":
                        canvas, _ = preprocess(active_image)
                        st.image((canvas * 255).astype("uint8"), caption="Cropped handwriting", use_container_width=True)
                    elif crop is not None:
                        st.image(crop, caption="Cropped handwriting", use_container_width=True)
                except Exception as e:
                    render_callout(f"No handwriting found: {e}", kind="warning")

            if st.button("Read equation", type="secondary"):
                st.session_state.pop("solved_for", None)
                skeleton_placeholder = st.empty()
                with skeleton_placeholder:
                    render_skeleton()
                try:
                    with st.spinner("Recognising equation in image..."):
                        if DEMO_MODE and os.environ.get("MATHVISION_STUB_SLEEP"):
                            time.sleep(float(os.environ.get("MATHVISION_STUB_SLEEP", "0")))
                        if DEMO_MODE and os.environ.get("MATHVISION_STUB_LATEX"):
                            st.session_state["latex"] = os.environ.get("MATHVISION_STUB_LATEX")
                        elif BACKEND == "legacy":
                            st.session_state["latex"] = recognizer_old.recognize(active_image)
                        else:
                            st.session_state["latex"] = recognize(active_image, max_side=DEFAULT_MAX_SIDE, crop=crop)
                except NotImplementedError as e:
                    render_callout(f"{e} Type or paste the LaTeX below.", kind="info")
                except Exception as e:
                    render_callout(f"Couldn't read this image: {e}. Try a closer photo, or type the LaTeX below.", kind="error")
                finally:
                    skeleton_placeholder.empty()

        # LaTeX Editing Aid (Insert Buttons)
        html('<div style="font-size: 0.88rem; font-weight: 600; color: #1F3A5F; margin-top: 10px; margin-bottom: 4px;">Insert math symbol:</div>')
        symbols = [
            (r"\frac{a}{b}", "a/b", "frac"),
            (r"\sqrt{x}", "√x", "sqrt"),
            (r"x^2", "x²", "pow"),
            (r"x_i", "xᵢ", "sub"),
            (r"\pi", "π", "pi"),
            (r"\pm", "±", "pm"),
            (r"\le", "≤", "le"),
            (r"\ge", "≥", "ge"),
            (r"\infty", "∞", "inf"),
            (r"\int", "∫", "int"),
            (r"\sum", "∑", "sum"),
            (r"\lim", "lim", "lim")
        ]
        cols_r1 = st.columns(6)
        for idx, (sym_code, sym_display, label) in enumerate(symbols[:6]):
            if cols_r1[idx].button(sym_display, key=f"btn_sym_{label}", help=f"Insert {sym_code}"):
                current_val = st.session_state.get("latex", "")
                separator = " " if current_val and not current_val.endswith(" ") else ""
                st.session_state["latex"] = current_val + separator + sym_code
                st.rerun()

        cols_r2 = st.columns(6)
        for idx, (sym_code, sym_display, label) in enumerate(symbols[6:]):
            if cols_r2[idx].button(sym_display, key=f"btn_sym_{label}", help=f"Insert {sym_code}"):
                current_val = st.session_state.get("latex", "")
                separator = " " if current_val and not current_val.endswith(" ") else ""
                st.session_state["latex"] = current_val + separator + sym_code
                st.rerun()

        # Text Input for LaTeX (Exact label preserved)
        latex = st.text_input(
            "Recognized LaTeX, edit if anything is wrong",
            key="latex",
            placeholder="2x + 3 = 7"
        )

        if latex.strip():
            st.latex(latex)

        with st.expander("Advanced"):
            mode_override = st.radio(
                "Mode override",
                ["Auto", "Arithmetic", "Equation"],
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
                col_exp1, col_exp2 = st.columns([1, 1], gap="small")
                with col_exp1:
                    sol_text = res.get("latex") or res.get("result") or ""
                    st.text_area("Copy LaTeX solution", value=sol_text, height=68, help="Click inside and Ctrl+A / Cmd+A to copy")
                with col_exp2:
                    st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
                    md_report = f"# MathVision Solution\n\n**Input LaTeX:** `{latex}`\n\n**Mode:** `{mode_override}`\n\n**Result:** `{sol_text}`\n"
                    if out["ok"]:
                        md_report += "\n## Working Steps:\n"
                        for i, row in enumerate(out["steps"], 1):
                            md_report += f"{i}. `{row['latex']}` — {row['why']}\n"
                    st.download_button(
                        label="Download solution (.md)",
                        data=md_report,
                        file_name="mathvision_solution.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

# ==============================================================================
# TAB 2: HOW IT WORKS TAB
# ==============================================================================
with tab_how:
    with st.container(border=True, key="card_how_pipeline"):
        html('<h3 class="mv-section-header">Architecture &amp; Pipeline</h3>')
        render_svg_pipeline()
        st.markdown("""
        **Processing Stages Overview:**
        1. **Photo Input:** Mathematical handwriting is captured via image upload or camera.
        2. **Crop & Clean:** Contrast-aware component isolation strips ghost text and isolates active handwriting strokes.
        3. **Recognition:** The Uni-MuMER vision-language foundation model transcribes visual strokes into LaTeX.
        4. **Editable LaTeX:** Guardrails validate syntax, surface friendly warnings, and provide quick math symbols for edits.
        5. **SymPy Solver:** The validated AST is solved symbolically, verifies inequalities, or identifies algebraic identities.
        6. **Answer & Steps:** Final answers are displayed with exact forms and decimal approximations where appropriate. For arithmetic expressions and linear single-variable equations, pedagogical step-by-step working is rendered as a visual timeline.
        """)

    # Supported Maths Battery Report
    with st.container(border=True, key="card_how_battery"):
        html('<h3 class="mv-section-header">Capabilities &amp; Supported Maths</h3>')
        st.caption("Empirically determined by running the solver battery across 40 representative problem types.")

        c_sup1, c_sup2 = st.columns([1, 1], gap="medium")
        with c_sup1:
            html("""
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 16px;">
                <div style="font-weight: 700; color: #166534; margin-bottom: 8px;">✓ Supported Problem Classes</div>
                <ul style="font-size: 0.9rem; color: #166534; margin: 0; padding-left: 20px;">
                    <li><strong>Arithmetic:</strong> Operations with integers, decimals, powers, fractions (+, -, ×, ÷, ^)</li>
                    <li><strong>Linear Equations:</strong> Single variable with integer, decimal, or fractional coefficients (with steps)</li>
                    <li><strong>Quadratics:</strong> Pure, monic factorable, and non-monic equations (real and complex roots)</li>
                    <li><strong>Higher Polynomials:</strong> Cubics and quartics solved to exact or decimal approximations</li>
                    <li><strong>Inequalities:</strong> Linear inequalities (&lt;, &gt;, ≤, ≥) with direction-aware solution steps</li>
                    <li><strong>Radicals:</strong> Square roots, surd simplification, and integer n-th roots (<sup>n</sup>√x)</li>
                    <li><strong>Identities & Checks:</strong> True identity verification and false statement detection</li>
                </ul>
            </div>
            """)

        with c_sup2:
            html("""
            <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 16px;">
                <div style="font-weight: 700; color: #991B1B; margin-bottom: 8px;">✗ Not Supported Yet</div>
                <ul style="font-size: 0.9rem; color: #991B1B; margin: 0; padding-left: 20px;">
                    <li><strong>Two Unknowns:</strong> Systems of linear equations (e.g. 2x + 3y = 74)</li>
                    <li><strong>Trigonometry:</strong> Trigonometric functions (sin, cos, tan, sec, csc, cot)</li>
                    <li><strong>Logarithms:</strong> Logarithmic and exponential functions (ln, log)</li>
                    <li><strong>Calculus:</strong> Derivatives (d/dx), indefinite/definite integrals (∫), and limits (lim)</li>
                    <li><strong>Matrices & Vectors:</strong> Matrix notation and vector algebra</li>
                    <li><strong>Piecewise & Absolute:</strong> Absolute value equations (|x|)</li>
                </ul>
            </div>
            """)

# ==============================================================================
# TAB 3: EXAMPLES TAB
# ==============================================================================
with tab_examples:
    with st.container(border=True, key="card_examples"):
        html('<h3 class="mv-section-header">Sample Test Photos</h3>')
        st.caption("Click 'Try this example' on any sample card to test it on the Solve tab.")

        photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_photos")
        if not os.path.exists(photos_dir):
            render_callout("No example photos found in test_photos/ directory.", kind="info")
        else:
            # Load ground truth if exists
            gt_map = {}
            gt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.csv")
            if os.path.exists(gt_file):
                import csv
                with open(gt_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            gt_map[row[0].strip()] = row[1].strip()

            # Known stored predictions for real photos
            stored_examples = {
                "WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg": "2 x + 3 y = 74",
                "WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg": "5 n + 50 = 160",
                "WhatsApp Image 2026-11-13 at 9.28.44 AM.jpeg": "205 = x ^ { 4 } + 9 x ^ { 3 } + 13 x ^ { 2 } - x"
            }

            photo_files = sorted([f for f in os.listdir(photos_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            for idx, p_file in enumerate(photo_files):
                p_path = os.path.join(photos_dir, p_file)
                with st.container(border=True, key=f"card_ex_{idx}"):
                    ex_c1, ex_c2 = st.columns([1, 2], gap="medium")
                    with ex_c1:
                        st.image(p_path, caption=p_file, use_container_width=True)
                    with ex_c2:
                        st.markdown(f"**Filename:** `{p_file}`")
                        gt_val = gt_map.get(p_file)
                        if gt_val:
                            st.markdown(f"**Ground truth formula:** `${gt_val}$`")
                            from solver import solve_latex
                            gt_sol = solve_latex(gt_val)
                            if gt_sol.get("ok"):
                                st.markdown(f"**Expected result:** `{gt_sol.get('latex') or gt_sol.get('result')}`")
                            else:
                                st.markdown(f"**Expected result:** *{gt_sol.get('error')}*")
                        else:
                            st.markdown("*No ground truth entry in ground_truth.csv*")

                        stored_pred = stored_examples.get(p_file)
                        if stored_pred:
                            st.markdown(f"**Stored example reading:** `${stored_pred}$` *(stored example)*")

                        if st.button(f"Try this example", key=f"btn_ex_{p_file}"):
                            st.session_state["example_image_path"] = p_path
                            st.session_state.pop("latex", None)
                            st.session_state.pop("solved_for", None)
                            st.info(f"Loaded {p_file}. Switch to the 'Solve' tab to run recognition!")

# ==============================================================================
# TAB 4: ABOUT TAB
# ==============================================================================
with tab_about:
    with st.container(border=True, key="card_about"):
        html(f'<h3 class="mv-section-header">About {html_lib.escape(PROJECT_TITLE)}</h3>')
        st.markdown("""
        MathVision is an end-to-end handwriting recognition and mathematical computing application.
        It integrates modern vision-language models with symbolic computer algebra systems to transcribe,
        validate, and compute exact step-by-step solutions from photographs of mathematical notes.
        """)

        # Optional metadata (hidden if empty)
        if INSTITUTION or TEAM_MEMBERS or GUIDE or YEAR:
            st.markdown("#### Project Details")
            if INSTITUTION:
                st.markdown(f"- **Institution / Department:** {INSTITUTION}")
            if TEAM_MEMBERS:
                st.markdown(f"- **Team Members:** {TEAM_MEMBERS}")
            if GUIDE:
                st.markdown(f"- **Project Guide:** {GUIDE}")
            if YEAR:
                st.markdown(f"- **Year:** {YEAR}")

        # Built With component with verifiable source-read licenses
        st.markdown("#### Built With")
        st.markdown("""
        | Library / Technology | Component Role | License (Read from Source Metadata) |
        | :--- | :--- | :--- |
        | **Uni-MuMER** | Handwritten vision-language mathematical transcription | see the project page |
        | **Streamlit** (`1.66.0`) | Interactive web application frontend & layout engine | Apache-2.0 |
        | **SymPy** (`1.14.0`) | Computer algebra system, AST translation & step solving | BSD |
        | **OpenCV** (`5.0.0`) | Morphological image preprocessing & ink isolation | Apache 2.0 |
        | **PyTorch** (`2.14.1+cpu`) | Tensor operations & transformer neural execution | Apache-2.0 / BSD / MIT |
        | **Transformers** (`5.20.0`) | Vision-language pipeline loader & tokenizer execution | Apache 2.0 |
        """)

# Render Footer
render_footer()
