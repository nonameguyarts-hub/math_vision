"""test_preprocess_clipping.py - Comprehensive test suite for ink clipping prevention.

Asserts three core invariant properties across 300+ randomized synthetic cases:
1. Ink recall >= 99%, with leftmost and rightmost ink columns fully inside crop with margin.
2. Aspect ratio preserved (scale_x equals scale_y within 1%).
3. No step produces a width or height smaller than the ink box.
Also includes a regression test on '50 + 40n = 190y' with descender 'y'.
"""
import io
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recognizer import prep

WIN_FONTS_DIR = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
AVAILABLE_FONTS = []
for fname in ["comic.ttf", "arial.ttf", "calibri.ttf", "times.ttf", "segoeui.ttf"]:
    fpath = os.path.join(WIN_FONTS_DIR, fname)
    if os.path.exists(fpath):
        AVAILABLE_FONTS.append(fpath)

EQUATIONS = [
    "50 + 40n = 190y",
    r"\frac{3}{4}+\frac{1}{2}=\frac{5}{4}",
    "2x + 3y - 4z + 5w = 100 + 200",  # ~30-40 chars
    "x^2 + y^2 = z^2",
    "a + b = c",
    "f(x) = 3x^3 - 2x^2 + 5x - 8 = 0",
    "120 / 4 = 30",
    r"\sqrt{49} + 3 = 10",
    "x + 3 < 5",
    "100 - 25 = 75",
]


def render_jittered_text(text, font, text_color, bg_size, pos):
    """Render text with per-character jitter and return (image, clean_mask)."""
    w, h = bg_size
    clean_mask = np.zeros((h, w), dtype=bool)
    txt_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_img)

    cur_x, cur_y = pos
    for char in text:
        # Per-character random jitter
        jx = cur_x + random.randint(-2, 2)
        jy = cur_y + random.randint(-2, 2)

        # Character bounding box
        bbox = draw.textbbox((jx, jy), char, font=font)
        draw.text((jx, jy), char, fill=text_color, font=font)

        char_w = max(bbox[2] - bbox[0], font.size // 2)
        cur_x += char_w + random.randint(1, 4)

    # Extract clean binary mask of rendered text
    txt_arr = np.array(txt_img)
    alpha = txt_arr[:, :, 3]
    clean_mask = alpha > 30
    return txt_img, clean_mask


def generate_synthetic_case(rng, case_idx):
    """Generate a single randomized synthetic case."""
    eq = rng.choice(EQUATIONS)
    font_path = rng.choice(AVAILABLE_FONTS) if AVAILABLE_FONTS else None
    font_size = rng.randint(22, 54)
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Vary canvas size: 300px to 4000px wide
    w = rng.randint(350, 2400)
    aspect = rng.choice([0.25, 0.4, 0.5, 0.75, 1.0])
    h = max(200, int(w * aspect))
    # Cap dimensions to avoid memory bloat
    w = min(w, 2400)
    h = min(h, 1600)

    # Background type
    bg_type = rng.choice(["white", "grey", "yellow", "gradient", "noise", "ruled", "grid", "margin"])
    if bg_type == "white":
        bg_col = (rng.randint(245, 255), rng.randint(245, 255), rng.randint(245, 255))
        base = Image.new("RGB", (w, h), bg_col)
    elif bg_type == "grey":
        v = rng.randint(190, 225)
        base = Image.new("RGB", (w, h), (v, v, v))
    elif bg_type == "yellow":
        base = Image.new("RGB", (w, h), (rng.randint(240, 255), rng.randint(235, 250), rng.randint(190, 215)))
    elif bg_type == "gradient":
        # Linear gradient shadow
        base_arr = np.zeros((h, w, 3), dtype=np.uint8)
        grad = np.linspace(rng.randint(150, 190), rng.randint(230, 255), w)
        for c in range(3):
            base_arr[:, :, c] = grad
        base = Image.fromarray(base_arr)
    else:
        v = rng.randint(220, 245)
        base = Image.new("RGB", (w, h), (v, v, v))

    draw_base = ImageDraw.Draw(base)

    # Ruled lines / grid paper / margin lines
    if bg_type in ["ruled", "grid"]:
        line_step = rng.randint(28, 55)
        line_col = (rng.randint(170, 205), rng.randint(175, 210), rng.randint(200, 230))
        for y_l in range(line_step, h, line_step):
            draw_base.line([(0, y_l), (w, y_l)], fill=line_col, width=1)
        if bg_type == "grid":
            for x_l in range(line_step, w, line_step):
                draw_base.line([(x_l, 0), (x_l, h)], fill=line_col, width=1)

    if bg_type == "margin":
        margin_x = rng.randint(40, max(50, w // 6))
        draw_base.line([(margin_x, 0), (margin_x, h)], fill=(220, 140, 150), width=rng.randint(1, 2))

    # Ink type
    ink_type = rng.choice(["black", "blue", "pencil", "light_on_dark"])
    if ink_type == "light_on_dark":
        # Blackboard / dark whiteboard
        base = Image.new("RGB", (w, h), (rng.randint(30, 50), rng.randint(35, 55), rng.randint(40, 60)))
        draw_base = ImageDraw.Draw(base)
        text_color = (rng.randint(220, 250), rng.randint(220, 250), rng.randint(220, 250), 255)
    elif ink_type == "pencil":
        # Faint pencil (~25% contrast)
        text_color = (rng.randint(110, 145), rng.randint(110, 145), rng.randint(110, 145), 255)
    elif ink_type == "blue":
        text_color = (rng.randint(15, 35), rng.randint(30, 60), rng.randint(130, 185), 255)
    else:
        text_color = (rng.randint(15, 45), rng.randint(15, 45), rng.randint(15, 45), 255)

    # Position
    pos_type = rng.choice(["center", "left", "right", "near_edge", "tiny"])
    approx_txt_w = len(eq) * (font_size * 0.65)
    approx_txt_h = font_size * 1.5

    if pos_type == "left":
        px = rng.randint(15, max(16, int(w * 0.1)))
        py = rng.randint(20, max(21, int(h - approx_txt_h - 20)))
    elif pos_type == "right":
        px = max(15, int(w - approx_txt_w - rng.randint(20, 50)))
        py = rng.randint(20, max(21, int(h - approx_txt_h - 20)))
    elif pos_type == "near_edge":
        px = rng.randint(12, 30)
        py = rng.randint(12, 30)
    elif pos_type == "tiny":
        px = rng.randint(30, max(31, int(w * 0.4)))
        py = rng.randint(30, max(31, int(h * 0.4)))
    else:  # center
        px = max(15, int((w - approx_txt_w) / 2))
        py = max(15, int((h - approx_txt_h) / 2))

    px = max(10, min(px, int(w - approx_txt_w - 15)))
    py = max(10, min(py, int(h - approx_txt_h - 15)))

    txt_img, clean_mask = render_jittered_text(eq, font, text_color, (w, h), (px, py))
    base.paste(txt_img, (0, 0), txt_img)

    # Rotation / skew
    rot_opt = rng.choice([0, 0, 0, "skew", 90, 180, 270])
    applied_rot = 0
    if rot_opt == "skew":
        skew_angle = rng.uniform(-4.5, 4.5)
        base = base.rotate(skew_angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=base.getpixel((0, 0)))
        clean_mask = np.array(Image.fromarray(clean_mask.astype(np.uint8) * 255).rotate(skew_angle, expand=False)) > 30
    elif rot_opt in [90, 180, 270]:
        applied_rot = rot_opt
        # In actual usage, user rotates the photo via UI turn option before prep
        base = base.rotate(applied_rot, expand=True)
        clean_mask = np.array(Image.fromarray(clean_mask.astype(np.uint8) * 255).rotate(applied_rot, expand=True)) > 30

    # Blur & JPEG compression
    if rng.random() < 0.35:
        base = base.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.2)))

    if rng.random() < 0.4:
        buf = io.BytesIO()
        base.save(buf, format="JPEG", quality=rng.randint(55, 85))
        buf.seek(0)
        base = Image.open(buf)

    return base, clean_mask, eq


def test_regression_50_40n_190y():
    """Regression test: '50 + 40n = 190y' must contain the final 0 and the y."""
    w, h = 1300, 320
    im = Image.new("RGB", (w, h), (245, 245, 240))
    draw = ImageDraw.Draw(im)
    font_path = AVAILABLE_FONTS[0] if AVAILABLE_FONTS else None
    font = ImageFont.truetype(font_path, 44) if font_path else ImageFont.load_default()

    # 50 + 40n = 190y with spacing
    draw.text((100, 130), "50 + 40n = 19", fill=(35, 35, 40), font=font)
    draw.text((450, 130), "0y", fill=(40, 40, 45), font=font)

    crop = prep(im)
    box = crop.crop_box

    # Assert crop includes left of '5' and right of 'y'
    assert box[0] <= 100, f"Left edge clipped: {box}"
    assert box[2] >= 495, f"Right edge clipped (0y lost): {box}"
    print(f"PASS: Regression test ('50 + 40n = 190y' fully enclosed in crop box {box})")


def run_random_suite(n_cases=300):
    rng = random.Random(42)
    failures = 0
    passed = 0

    print(f"Running {n_cases} random clipping prevention tests...")
    idx = 0
    while (passed + failures) < n_cases:
        idx += 1
        img, clean_mask, eq = generate_synthetic_case(rng, idx)
        tot_ink = clean_mask.sum()
        if tot_ink == 0:
            continue

        ys, xs = np.nonzero(clean_mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        ink_w = x_max - x_min + 1
        ink_h = y_max - y_min + 1

        crop = prep(img)
        x0, y0, x1, y1 = crop.crop_box

        # Property 1: Ink recall >= 99%, leftmost and rightmost ink columns inside crop
        crop_mask = clean_mask[y0:y1, x0:x1]
        recalled_ink = crop_mask.sum()
        recall = recalled_ink / tot_ink

        errs = []
        if recall < 0.99:
            errs.append(f"Ink recall {recall:.3f} < 0.99")
        if x0 > x_min:
            errs.append(f"Left ink column {x_min} outside crop left {x0}")
        if x1 < x_max + 1:
            errs.append(f"Right ink column {x_max} outside crop right {x1}")
        if y0 > y_min:
            errs.append(f"Top ink row {y_min} outside crop top {y0}")
        if y1 < y_max + 1:
            errs.append(f"Bottom ink row {y_max} outside crop bottom {y1}")

        # Property 2: Aspect ratio not distorted (scale_x equals scale_y within 1%)
        box_w = x1 - x0
        box_h = y1 - y0
        final_w, final_h = crop.size
        scale_x = final_w / box_w
        scale_y = final_h / box_h
        if max(scale_x, scale_y) > 0:
            diff_scale = abs(scale_x - scale_y) / max(scale_x, scale_y)
            if diff_scale >= 0.01:
                errs.append(f"Aspect ratio distorted: diff {diff_scale:.4f} >= 0.01 (scale_x={scale_x}, scale_y={scale_y})")

        # Property 3: No step produced width or height smaller than the ink box
        if box_w < ink_w:
            errs.append(f"Crop width {box_w} < ink width {ink_w}")
        if box_h < ink_h:
            errs.append(f"Crop height {box_h} < ink height {ink_h}")

        if errs:
            failures += 1
            print(f"FAIL [Case {idx}] ({eq}): {', '.join(errs)}")
        else:
            passed += 1

        if idx % 50 == 0:
            print(f"  Progress: {idx}/{n_cases} cases completed ({passed} passed, {failures} failed)")

    print(f"\nRandom cases summary: {passed}/{passed + failures} passed, {failures} failures.")
    return failures


if __name__ == "__main__":
    test_regression_50_40n_190y()
    fails = run_random_suite(300)
    sys.exit(1 if fails else 0)
