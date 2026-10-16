"""MathVision UI.   Run:  streamlit run app.py"""
import streamlit as st
from PIL import Image, ImageOps

from preprocess import looks_sideways, preprocess
import recognizer_old
from recognizer import DEFAULT_MAX_SIDE, prep, recognize
from steps import steps_for
from ui_checks import MODES, review

st.set_page_config(page_title="MathVision", page_icon="∑")
st.title("MathVision")
st.caption("Photograph a handwritten equation, check what was read, then solve it.")

backend = st.sidebar.radio(
    "Recognizer Backend",
    ["Uni-MuMER (Qwen3-VL-2B)", "Legacy (MathVision CNN+GRU)"],
    help="Uni-MuMER is the fine-tuned vision-language model. Legacy is the custom CNN+GRU model."
)

mode = st.radio("Mode", MODES, horizontal=True,
                help="Arithmetic expects numbers only. Equation expects one unknown, such as 2x + 3 = 7.")

# 1. photo
file = st.file_uploader("Your photo", type=["png", "jpg", "jpeg"])
if file is not None:
    image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    turn = st.radio("Rotate photo", ["None", "Left 90°", "Right 90°", "180°"], horizontal=True,
                    help="Use this if the writing is sideways or upside down.")
    image = image.rotate({"None": 0, "Left 90°": 90, "Right 90°": 270, "180°": 180}[turn], expand=True)
    if looks_sideways(image):
        st.info("The writing looks sideways. Try Left 90° or Right 90° above.")
    st.image(image, width="stretch")
    with st.expander("What the model sees"):
        try:
            if backend.startswith("Uni-MuMER"):
                crop = prep(image, max_side=DEFAULT_MAX_SIDE)
                st.image(crop, caption=f"Cropped handwriting (size {crop.size}). Dark ink on light background.", width="stretch")
            else:
                canvas, _ = preprocess(image)
                st.image((canvas * 255).astype("uint8"), caption="128px normalized strip (Legacy CNN+GRU).", width="stretch")
        except Exception as e:
            st.warning(f"No handwriting found: {e}")
    if st.button("Read equation"):
        try:
            if backend.startswith("Uni-MuMER"):
                with st.spinner("Recognizing equation with Uni-MuMER (CPU)..."):
                    st.session_state["latex"] = recognize(image, max_side=DEFAULT_MAX_SIDE)
            else:
                with st.spinner("Recognizing equation with Legacy model..."):
                    st.session_state["latex"] = recognizer_old.recognize(image)
        except NotImplementedError as e:
            st.info(f"{e} Type or paste the LaTeX below.")
        except Exception as e:
            st.error(f"Couldn't read this image: {e}. Try a closer photo, or type the LaTeX below.")

# 2. what was read (editable)
st.subheader("What was read")
latex = st.text_input("Recognized LaTeX, edit if anything is wrong", key="latex",
                      placeholder="2x + 3 = 7")
if latex.strip():
    st.latex(latex)

rev = review(latex, mode)
for w in rev["warnings"]:
    st.warning(w)
confirmed = True
if rev["needs_confirm"]:
    confirmed = st.checkbox("The symbols match my photo")

if st.button("Solve"):
    st.session_state["solved_for"] = (latex, mode)

# 3. answer and steps (shown only for the exact text that was solved)
if st.session_state.get("solved_for") == (latex, mode):
    st.subheader("Answer")
    if not rev["ok"]:
        st.error(f"{rev['error']} Edit the LaTeX above and try again.")
    elif not confirmed:
        st.info("Tick the box above once you've checked the symbols.")
    else:
        out = steps_for(latex)
        if out["ok"]:
            st.latex(out["answer"])
            st.subheader("Steps")
            for i, row in enumerate(out["steps"], 1):
                c1, c2, c3 = st.columns([1, 5, 4])
                c1.write(i)
                c2.latex(row["latex"])
                c3.write(row["why"])
        else:
            res = rev["result"]
            st.write(res["result"])
            if res["approx"] is not None:
                st.caption(f"About {res['approx']}")
            st.caption(out["note"])

st.divider()
st.caption("Always compare the recognized equation with your photo before trusting an answer.")