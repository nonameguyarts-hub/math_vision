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

    # 2. Low-frequency shadow removal: large blur subtraction before thresholding
    k_blur = max(31, int(min(w, h) * 0.22) | 1)
    blur = cv2.GaussianBlur(g, (k_blur, k_blur), 0)
    diff = cv2.subtract(blur, g)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    t, _ = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Sensitivity threshold to catch faint pen strokes (~25% contrast)
    ink = (diff_norm >= max(12, int(t * 0.45))).astype(np.uint8) * 255

    # 3. Ignore ruled lines, page edges and table borders
    line_w = max(40, int(w * 0.25))
    line_h = max(40, int(h * 0.25))
    lines_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1)))
    lines_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_h)))
    ink = cv2.subtract(ink, cv2.bitwise_or(lines_h, lines_v))

    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        out = img.copy()
        out.warning = "Couldn't isolate the handwriting. Retake on plain paper with some space around it."
        out.num_lines = 1
        out.crop_box = (0, 0, w, h)
        out.ink_box = (0, 0, w, h)
        out.prep_info = {"warning": out.warning, "num_lines": 1, "crop_box": out.crop_box, "ink_box": out.ink_box}
        return out

    # 4. Discard components touching border that are large or long (page edge, desk, shadow)
    # and filter out ruled line segments and tiny specks
    candidates = []
    touches = {}
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        tb = (x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1)
        long_s = max(cw, ch)
        short_s = max(min(cw, ch), 1)
        aspect = long_s / short_s

        # Discard border components that are large or long (page edge, desk, shadow)
        if tb:
            is_huge = (area > 400 or area > 0.005 * (w * h))
            is_long_border = (aspect > 3.0 and long_s > max(35, int(0.12 * max(w, h))))
            is_spanning = (cw > 0.35 * w or ch > 0.35 * h)
            if is_huge or is_long_border or is_spanning:
                continue
        # Discard ruled / margin / grid lines
        if short_s == 1 and aspect > 4.0:
            continue
        if aspect > 5.5 and long_s > 35:
            continue
        if area < 5:
            continue
        candidates.append(i)
        touches[i] = tb

    if not candidates:
        out = img.copy()
        out.warning = "Couldn't isolate the handwriting. Retake on plain paper with some space around it."
        out.num_lines = 1
        out.crop_box = (0, 0, w, h)
        out.ink_box = (0, 0, w, h)
        out.prep_info = {"warning": out.warning, "num_lines": 1, "crop_box": out.crop_box, "ink_box": out.ink_box}
        return out

    # Plausible height relative to median stroke height
    h_list = [stats[i, cv2.CC_STAT_HEIGHT] for i in candidates if stats[i, cv2.CC_STAT_AREA] >= 15 and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / max(min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]), 1) < 4.0]
    med_h = np.median(h_list) if h_list else 25

    # Filter out far-away stroke outliers or massive noise
    plausible_cands = []
    for i in candidates:
        long_s = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        area = stats[i, cv2.CC_STAT_AREA]
        if long_s < 0.15 * med_h and area < 10:
            continue
        if long_s > 4.5 * med_h:
            continue
        plausible_cands.append(i)

    if not plausible_cands:
        plausible_cands = candidates

    # Group components into lines / text cluster using directional proximity
    thresh_line = max(75, int(med_h * 4.5))
    thresh_cross = max(45, int(med_h * 2.5))
    boxes = [stats[i, :4] for i in plausible_cands]
    nc = len(plausible_cands)
    adj = [[] for _ in range(nc)]
    for i in range(nc):
        x1, y1, w1, h1 = boxes[i]
        for j in range(i + 1, nc):
            x2, y2, w2, h2 = boxes[j]
            dx = max(0, max(x1 - (x2 + w2), x2 - (x1 + w1)))
            dy = max(0, max(y1 - (y2 + h2), y2 - (y1 + h1)))
            # Link if within line gap (horizontal writing line or vertical writing line)
            linked_h = (dx <= thresh_line and dy <= thresh_cross)
            linked_v = (dy <= thresh_line and dx <= thresh_cross)
            if linked_h or linked_v:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * nc
    clusters = []
    for i in range(nc):
        if not visited[i]:
            comp = []
            q = [i]
            visited[i] = True
            for u in q:
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            clusters.append(comp)

    # Rank clusters: prefer clusters with multiple components, then by total ink area
    clusters.sort(key=lambda c: (len(c) > 1, sum(stats[plausible_cands[u], cv2.CC_STAT_AREA] for u in c)), reverse=True)
    best_cluster_cands = [plausible_cands[u] for u in clusters[0]]

    # Union bounding box across dominant cluster
    min_x = min(stats[i, cv2.CC_STAT_LEFT] for i in best_cluster_cands)
    max_x = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in best_cluster_cands)
    min_y = min(stats[i, cv2.CC_STAT_TOP] for i in best_cluster_cands)
    max_y = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in best_cluster_cands)
    ink_h = max_y - min_y + 1
    ink_w = max_x - min_x + 1

    # Check if text-sized ink itself reaches the photo border (after cleanup)
    writing_touches_border = any(touches.get(i, False) for i in best_cluster_cands)

    # Multi-line detection
    num_lines = 1
    proj_y = ink[min_y:max_y + 1, min_x:max_x + 1].sum(axis=1)
    if len(proj_y) > 30:
        smoothed = cv2.GaussianBlur(proj_y.astype(float), (21, 1), 0).ravel()
        troughs = (smoothed < 0.1 * smoothed.max())
        if troughs[int(0.2 * len(proj_y)):int(0.8 * len(proj_y))].any():
            num_lines = 2

    # Pad every side by max(10 px, 12% of ink height)
    pad = max(10, int(round(margin * ink_h)))
    crop_x0 = max(0, min_x - pad)
    crop_y0 = max(0, min_y - pad)
    crop_x1 = min(w, max_x + pad)
    cy1 = min(h, max_y + pad)
    crop_y1 = cy1

    # Clipping guard: measure ink in a thin ring just OUTSIDE the crop box in the original
    # If that ring contains ink above a small threshold, expand until clean or frame edge reached
    ring_thick = max(2, int(0.03 * ink_h))
    step = max(4, int(0.05 * ink_h))
    for _ in range(15):
        expanded = False
        if crop_x0 > 0:
            rx0 = max(0, crop_x0 - ring_thick)
            if (ink[crop_y0:crop_y1, rx0:crop_x0] > 0).sum() >= 2:
                crop_x0 = max(0, crop_x0 - step)
                expanded = True
        if crop_x1 < w:
            rx1 = min(w, crop_x1 + ring_thick)
            if (ink[crop_y0:crop_y1, crop_x1:rx1] > 0).sum() >= 2:
                crop_x1 = min(w, crop_x1 + step)
                expanded = True
        if crop_y0 > 0:
            ry0 = max(0, crop_y0 - ring_thick)
            if (ink[ry0:crop_y0, crop_x0:crop_x1] > 0).sum() >= 2:
                crop_y0 = max(0, crop_y0 - step)
                expanded = True
        if crop_y1 < h:
            ry1 = min(h, crop_y1 + ring_thick)
            if (ink[crop_y1:ry1, crop_x0:crop_x1] > 0).sum() >= 2:
                crop_y1 = min(h, crop_y1 + step)
                expanded = True
        if not expanded:
            break

    # If final box still covers > 85% of frame width AND height, treat detection as failed
    box_w = crop_x1 - crop_x0
    box_h = crop_y1 - crop_y0
    failed_isolation = (box_w > 0.85 * w and box_h > 0.85 * h)

    warning = None
    if failed_isolation:
        warning = "Couldn't isolate the handwriting. Retake on plain paper with some space around it."
        crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, w, h
    elif writing_touches_border:
        warning = "Writing touches the edge of the photo."

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
    crop.num_lines = num_lines
    crop.crop_box = (crop_x0, crop_y0, crop_x1, crop_y1)
    crop.ink_box = (min_x, min_y, max_x, max_y)
    crop.prep_info = {
        "warning": warning,
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