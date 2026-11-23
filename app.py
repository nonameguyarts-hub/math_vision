"""
MathVision UI - University Project Redesign
Photograph handwritten mathematics. Check it. Solve it.
"""
import os
import time
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
    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
    st.markdown('<div class="mv-card-header"><h3 class="mv-card-title">1. Upload handwritten equation</h3></div>', unsafe_allow_html=True)

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
            st.markdown('<div class="mv-photo-frame">', unsafe_allow_html=True)
            st.image(active_image, caption="Uploaded photo", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            render_empty_photo_tips()

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Step 2: Check & Edit Reading Card
    # --------------------------------------------------------------------------
    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
    st.markdown('<div class="mv-card-header"><h3 class="mv-card-title">2. Check the reading</h3></div>', unsafe_allow_html=True)

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
            skeleton_placeholder.markdown('<div class="mv-skeleton" aria-hidden="true"></div>', unsafe_allow_html=True)
            try:
                with st.spinner("Recognising equation in image..."):
                    # Check for simulated demo state only if MATHVISION_DEMO=1 (Rule 8)
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
    st.markdown('<div style="font-size: 0.88rem; font-weight: 600; color: #1F3A5F; margin-top: 10px; margin-bottom: 4px;">Insert math symbol:</div>', unsafe_allow_html=True)
    symbols = [
        (r"\frac{a}{b}", "frac"),
        (r"\sqrt{x}", "sqrt"),
        (r"x^2", "pow"),
        (r"x_i", "sub"),
        (r"\pi", "pi"),
        (r"\pm", "pm"),
        (r"\le", "le"),
        (r"\ge", "ge"),
        (r"\infty", "inf"),
        (r"\int", "int"),
        (r"\sum", "sum"),
        (r"\lim", "lim")
    ]
    cols_r1 = st.columns(6)
    for idx, (sym_code, label) in enumerate(symbols[:6]):
        if cols_r1[idx].button(sym_code, key=f"btn_sym_{label}", help=f"Insert {sym_code}"):
            current_val = st.session_state.get("latex", "")
            separator = " " if current_val and not current_val.endswith(" ") else ""
            st.session_state["latex"] = current_val + separator + sym_code
            st.rerun()

    cols_r2 = st.columns(6)
    for idx, (sym_code, label) in enumerate(symbols[6:]):
        if cols_r2[idx].button(sym_code, key=f"btn_sym_{label}", help=f"Insert {sym_code}"):
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
        st.markdown('<div style="margin: 8px 0;">', unsafe_allow_html=True)
        st.latex(latex)
        st.markdown('</div>', unsafe_allow_html=True)

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

    if st.button("Solve", type="primary", use_container_width=True):
        st.session_state["solved_for"] = (latex, mode_override)

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Step 3: Answer & Steps Card (Shown only for exact text that was solved)
    # --------------------------------------------------------------------------
    if st.session_state.get("solved_for") == (latex, mode_override):
        st.markdown('<div class="mv-card">', unsafe_allow_html=True)
        st.markdown('<div class="mv-card-header"><h3 class="mv-card-title">3. Solution</h3></div>', unsafe_allow_html=True)

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
            st.markdown(f"""
            <div class="mv-answer-box">
                <div class="mv-answer-header">
                    <span class="mv-answer-label">Answer</span>
                    {chips_str}
                </div>
            """, unsafe_allow_html=True)

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

            st.markdown('</div>', unsafe_allow_html=True)

            # Step-by-Step Working Timeline
            out = steps_for(latex)
            if out["ok"]:
                st.markdown('<h4 style="color: #1F3A5F; margin-top: 1.5rem; margin-bottom: 0.5rem;">Step-by-step working</h4>', unsafe_allow_html=True)
                st.markdown('<div class="mv-timeline">', unsafe_allow_html=True)
                for i, row in enumerate(out["steps"], 1):
                    st.markdown(f"""
                    <div class="mv-timeline-item">
                        <div class="mv-timeline-badge">{i}</div>
                        <div class="mv-timeline-content">
                    """, unsafe_allow_html=True)
                    st.latex(row["latex"])
                    st.markdown(f"""
                            <div class="mv-timeline-why">{row["why"]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                if out.get("note") and out.get("note") != res.get("note") and "not available" not in out["note"].lower():
                    st.caption(out["note"])

            # Session Extras: Download, Copyable LaTeX, Recent solutions
            st.markdown('<div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #E2E6EF;">', unsafe_allow_html=True)
            ex_col1, ex_col2 = st.columns([1, 1])
            with ex_col1:
                # Markdown export format
                export_text = f"# MathVision Solution Export\n\n**Expression:**\n$${latex}$$\n\n**Answer:**\n$${res.get('latex') or res.get('result')}$$\n"
                if out["ok"]:
                    export_text += "\n## Working Steps\n"
                    for idx_s, st_row in enumerate(out["steps"], 1):
                        export_text += f"{idx_s}. $${st_row['latex']}$$ — *{st_row['why']}*\n"
                st.download_button(
                    label="📥 Download Solution (.md)",
                    data=export_text,
                    file_name="mathvision_solution.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with ex_col2:
                with st.expander("Copyable LaTeX code"):
                    st.code(latex, language="latex")

            if st.session_state.get("recent_solutions"):
                with st.expander("Recent solutions (this session)"):
                    for r_latex, r_ans in reversed(st.session_state["recent_solutions"]):
                        st.markdown(f"- **Expression:** `{r_latex}` $\\to$ **Answer:** `{r_ans}`")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Persistent Footer Strip
    render_footer()

# ==============================================================================
# TAB 2: HOW IT WORKS
# ==============================================================================
with tab_how:
    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="mv-section-header">Architecture & Pipeline</h3>', unsafe_allow_html=True)
    render_svg_pipeline()

    st.markdown("""
    #### Processing Stages
    1. **Photo Input:** Captures high-resolution images of handwritten mathematical equations from paper. Exif metadata is normalized automatically so rotated or mirrored smartphone photos align consistently.
    2. **Crop & Clean:** Low-frequency illumination shadows and faint ghost text are filtered using contrast-aware thresholding. Connected component analysis clusters handwriting without clipping thin fraction bars or radical lines.
    3. **Recognition Model:** The Uni-MuMER vision-language model parses the preprocessed handwriting canvas directly into LaTeX syntax. Greedy autoregressive decoding produces fully deterministic token sequences across repeated runs.
    4. **Editable LaTeX:** Recognized formulas are presented in an editable text box with live KaTeX visual rendering. Intelligent guardrails check for common character confusions and prompt verification when relations are ambiguous.
    5. **SymPy Solver:** LaTeX syntax is translated into an abstract syntax tree and verified using computer algebra. The engine evaluates numeric values, determines roots of polynomial equations, and verifies algebraic identities.
    6. **Answer & Steps:** Final answers are displayed with exact forms and decimal approximations where appropriate. For arithmetic expressions and linear single-variable equations, pedagogical step-by-step working is rendered as a visual timeline.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Supported Maths Battery Report
    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="mv-section-header">Capabilities & Supported Maths</h3>', unsafe_allow_html=True)
    st.caption("Empirically determined by running the solver battery across 40 representative problem types.")

    c_sup1, c_sup2 = st.columns([1, 1], gap="medium")
    with c_sup1:
        st.markdown("""
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
        """, unsafe_allow_html=True)

    with c_sup2:
        st.markdown("""
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
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 3: EXAMPLES TAB
# ==============================================================================
with tab_examples:
    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="mv-section-header">Sample Test Photos</h3>', unsafe_allow_html=True)
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
        for p_file in photo_files:
            p_path = os.path.join(photos_dir, p_file)
            st.markdown('<div style="border: 1px solid #E2E6EF; border-radius: 10px; padding: 16px; margin-bottom: 16px; background: #FFFFFF;">', unsafe_allow_html=True)
            ex_c1, ex_c2 = st.columns([1, 2], gap="medium")
            with ex_c1:
                st.image(p_path, caption=p_file, use_container_width=True)
            with ex_c2:
                st.markdown(f"**Filename:** `{p_file}`")
                gt_val = gt_map.get(p_file)
                if gt_val:
                    st.markdown(f"**Ground truth formula:** `${gt_val}$`")
                    # Compute expected result live
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
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 4: ABOUT TAB
# ==============================================================================
with tab_about:
    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 class="mv-section-header">About {PROJECT_TITLE}</h3>', unsafe_allow_html=True)
    st.markdown("""
    MathVision is an end-to-end handwriting recognition and mathematical computing application.
    It integrates modern vision-language models with symbolic computer algebra systems to transcribe,
    validate, and compute exact step-by-step solutions from photographs of mathematical notes.
    """, unsafe_allow_html=True)

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
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
