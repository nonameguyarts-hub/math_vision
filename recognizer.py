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


def prep(img: Image.Image, rotate: int = 0, margin: float = 0.12, max_side: int = DEFAULT_MAX_SIDE) -> Image.Image:
    """Robust ink detection and crop: never clip handwriting columns or rows."""
    img = img.convert("RGB")
    if rotate:
        img = img.rotate(rotate, expand=True)
    w, h = img.size
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # 1. Polarity detection (handle light ink on dark background)
    bw_margin = max(5, int(w * 0.05))
    bh_margin = max(5, int(h * 0.05))
    border = np.concatenate([g[:bh_margin, :].ravel(), g[-bh_margin:, :].ravel(),
                             g[:, :bw_margin].ravel(), g[:, -bw_margin:].ravel()])
    if np.median(border) < 110:
        g = cv2.bitwise_not(g)

    # 2. Background estimation (large closing / blur) and difference
    k_size = max(31, int(min(h, w) * 0.08) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)))
    diff = cv2.subtract(bg, g)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    t, _ = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Sensitivity threshold to catch faint pen strokes (~25% contrast)
    ink = (diff_norm >= max(10, int(t * 0.45))).astype(np.uint8) * 255

    # 3. Ignore ruled lines, page edges and table borders
    line_w = max(40, int(w * 0.28))
    line_h = max(40, int(h * 0.28))
    lines_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1)))
    lines_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_h)))
    ink = cv2.subtract(ink, cv2.bitwise_or(lines_h, lines_v))

    # Also drop hairline strokes (length/thickness >= 8 and long)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n > 1:
        cw = stats[1:, cv2.CC_STAT_WIDTH]
        ch = stats[1:, cv2.CC_STAT_HEIGHT]
        ca = stats[1:, cv2.CC_STAT_AREA]
        long_dim = np.maximum(cw, ch)
        short_dim = np.maximum(np.minimum(cw, ch), 1)
        hairline = (long_dim / short_dim >= 8) & (ca / long_dim <= 2.5) & (long_dim >= 30)
        rule_mask = np.zeros(n, bool)
        rule_mask[1:] = hairline
        ink[rule_mask[lab]] = 0

    # 4. Remove tiny specks (area below fraction of median stroke size)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n > 1:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 5]
        med_a = np.median(areas) if len(areas) > 0 else 20
        keep = np.zeros(n, bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= max(3, 0.04 * med_a):
                keep[i] = True
        ink = (keep[lab] * 255).astype(np.uint8)

    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        out = img.copy()
        out.warning = None
        out.clipped_by_crop = False
        out.num_lines = 1
        out.crop_box = (0, 0, w, h)
        out.ink_box = (0, 0, w, h)
        out.prep_info = {"warning": None, "clipped_by_crop": False, "num_lines": 1,
                         "crop_box": (0, 0, w, h), "ink_box": (0, 0, w, h)}
        return out

    # Union of all remaining ink components
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    ink_h = max_y - min_y + 1
    ink_w = max_x - min_x + 1

    # Pad every side by max(10 px, 12% of ink height). Clamp to image, never crop inside ink box
    pad = max(10, int(round(margin * ink_h)))
    crop_x0 = max(0, min_x - pad)
    crop_y0 = max(0, min_y - pad)
    crop_x1 = min(w, max_x + 1 + pad)
    crop_y1 = min(h, max_y + 1 + pad)

    # Multi-line detection
    num_lines = 1
    proj_y = ink[min_y:max_y + 1, min_x:max_x + 1].sum(axis=1)
    if len(proj_y) > 30:
        smoothed = cv2.GaussianBlur(proj_y.astype(float), (21, 1), 0).ravel()
        troughs = (smoothed < 0.1 * smoothed.max())
        if troughs[int(0.2 * len(proj_y)):int(0.8 * len(proj_y))].any():
            num_lines = 2

    # Edge guardrail: check whether ink touches crop's outer 1% while original had background beyond,
    # or whether ink touches original photo border
    xm = max(1, int(round(0.01 * (crop_x1 - crop_x0))))
    ym = max(1, int(round(0.01 * (crop_y1 - crop_y0))))
    clipped_by_crop = False
    if crop_x0 > 0 and (min_x - crop_x0) < xm:
        clipped_by_crop = True
    if crop_x1 < w and (crop_x1 - max_x) < xm:
        clipped_by_crop = True
    if crop_y0 > 0 and (min_y - crop_y0) < ym:
        clipped_by_crop = True
    if crop_y1 < h and (crop_y1 - max_y) < ym:
        clipped_by_crop = True

    border_touch = (min_x <= 1 or max_x >= w - 2 or min_y <= 1 or max_y >= h - 2)
    warning = None
    if clipped_by_crop or border_touch:
        warning = "Writing may be cut off at the edge. Retake the photo with more space around the equation."

    crop = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    # Proportional scaling: never distort aspect ratio or cut columns/rows
    s = max_side / max(crop.size)
    if s < 1.0:
        cw, ch = crop.width, crop.height
        best_w, best_h = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
        best_diff = abs(best_w / cw - best_h / ch) / max(best_w / cw, best_h / ch)

        if cw >= ch:
            raw_h = max(1.0, ch * s)
            for dh in range(-2, 3):
                cand_h = max(1, int(round(raw_h)) + dh)
                cand_w = max(1, int(round(cand_h * cw / ch)))
                if cand_w <= max_side and cand_h <= max_side:
                    diff = abs(cand_w / cw - cand_h / ch) / max(cand_w / cw, cand_h / ch)
                    if diff < best_diff:
                        best_diff = diff
                        best_w, best_h = cand_w, cand_h
        else:
            raw_w = max(1.0, cw * s)
            for dw in range(-2, 3):
                cand_w = max(1, int(round(raw_w)) + dw)
                cand_h = max(1, int(round(cand_w * ch / cw)))
                if cand_w <= max_side and cand_h <= max_side:
                    diff = abs(cand_w / cw - cand_h / ch) / max(cand_w / cw, cand_h / ch)
                    if diff < best_diff:
                        best_diff = diff
                        best_w, best_h = cand_w, cand_h

        crop = crop.resize((best_w, best_h), Image.Resampling.LANCZOS)

    # Attach metadata to crop
    crop.warning = warning
    crop.clipped_by_crop = clipped_by_crop
    crop.num_lines = num_lines
    crop.crop_box = (crop_x0, crop_y0, crop_x1, crop_y1)
    crop.ink_box = (min_x, min_y, max_x, max_y)
    crop.prep_info = {
        "warning": warning,
        "clipped_by_crop": clipped_by_crop,
        "num_lines": num_lines,
        "crop_box": (crop_x0, crop_y0, crop_x1, crop_y1),
        "ink_box": (min_x, min_y, max_x, max_y),
    }
    return crop


def normalize(s):
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "").strip()
    s = re.sub(r"(?<=\d) (?=\d)", "", s)                  # "7 4" -> "74"
    s = re.sub(r"(?<=\d) (?=\.)|(?<=\.) (?=\d)", "", s)   # "3 . 5" -> "3.5"
    s = re.sub(r"\\frac\s+([a-zA-Z0-9])\s+([a-zA-Z0-9])", r"\\frac{\1}{\2}", s)  # "\frac x y" -> "\frac{x}{y}"
    return s


def recognize(image: Image.Image, rotate: int = 0, max_new_tokens: int = 512, max_side: int = DEFAULT_MAX_SIDE, crop: Image.Image = None) -> str:
    """Photo in, LaTeX string out. Reuses single crop PIL object if provided."""
    processor, model, device = _load()
    if crop is None:
        crop = prep(image, rotate=rotate, max_side=max_side)
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[crop], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return normalize(processor.batch_decode(gen, skip_special_tokens=True)[0])