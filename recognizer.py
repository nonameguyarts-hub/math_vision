import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".hf_cache"))

import re
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "phxember/Uni-MuMER-Qwen3-VL-2B"
PROMPT = ("I have an image of a handwritten mathematical expression. "
          "Please write out the expression of the formula in the image using LaTeX format.")
DEFAULT_MAX_SIDE = 512


try:
    import streamlit as st
    _cache_decorator = st.cache_resource
except Exception:
    _cache_decorator = lambda f: lru_cache(maxsize=1)(f)


@_cache_decorator
def _load():
    """Load the model once per process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    return processor, model, device


def prep(img, rotate=0, margin=0.25, max_side=DEFAULT_MAX_SIDE):
    """Remove ruled lines, crop to the main cluster of ink, keep dark ink on light."""
    img = img.convert("RGB")
    if rotate:
        img = img.rotate(rotate, expand=True)
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 15)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, img.width // 6), 1))
    bw = cv2.subtract(bw, cv2.morphologyEx(bw, cv2.MORPH_OPEN, k))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    merged = cv2.dilate(bw, np.ones((15, 40), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(merged)
    if n <= 1:
        return img
    best = max(range(1, n), key=lambda i: int((bw[lab == i] > 0).sum()))
    x, y, w, h = stats[best][:4]
    mx, my = w * margin, max(h * margin, 20)
    out = img.crop((int(max(x - mx, 0)), int(max(y - my, 0)),
                    int(min(x + w + mx, img.width)), int(min(y + h + my, img.height))))
    s = max_side / max(out.size)
    return out.resize((int(out.width * s), int(out.height * s))) if s < 1 else out


def normalize(s):
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "").strip()
    s = re.sub(r"(?<=\d) (?=\d)", "", s)                  # "7 4" -> "74"
    s = re.sub(r"(?<=\d) (?=\.)|(?<=\.) (?=\d)", "", s)   # "3 . 5" -> "3.5"
    s = re.sub(r"\\frac\s+([a-zA-Z0-9])\s+([a-zA-Z0-9])", r"\\frac{\1}{\2}", s)  # "\frac x y" -> "\frac{x}{y}"
    return s


def recognize(image: Image.Image, rotate: int = 0, max_new_tokens: int = 512, max_side: int = DEFAULT_MAX_SIDE) -> str:
    """Photo in, LaTeX string out."""
    processor, model, device = _load()
    crop = prep(image, rotate=rotate, max_side=max_side)
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[crop], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return normalize(processor.batch_decode(gen, skip_special_tokens=True)[0])