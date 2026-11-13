import os
import io
import random
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from recognizer import prep, DEFAULT_MAX_SIDE

AVAILABLE_FONTS = []
font_candidates = [
    "arial.ttf", "calibri.ttf", "times.ttf", "comic.ttf", "segoepb.ttf",
    "cour.ttf", "georgia.ttf", "impact.ttf", "tahoma.ttf", "trebuc.ttf"
]
for fc in font_candidates:
    try:
        f = ImageFont.truetype(fc, 30)
        AVAILABLE_FONTS.append(fc)
    except Exception:
        pass


def render_jittered_text(text: str, font, fill_color, img_size, start_pos, rng: random.Random = None):
    """Render text with per-character position and baseline jitter."""
    r = rng if rng is not None else random
    txt_layer = Image.new("RGBA", img_size, (0, 0, 0, 0))
    clean_mask = np.zeros((img_size[1], img_size[0]), dtype=bool)

    cur_x, cur_y = start_pos
    draw = ImageDraw.Draw(txt_layer)

    for ch in text:
        jx = r.randint(-2, 3)
        jy = r.randint(-3, 3)
        pos = (cur_x + jx, cur_y + jy)
        draw.text(pos, ch, font=font, fill=fill_color)

        bbox = font.getbbox(ch)
        cw = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]
        x0, y0 = max(0, pos[0]), max(0, pos[1])
        x1, y1 = min(img_size[0], pos[0] + max(cw, 8)), min(img_size[1], pos[1] + max(ch_h, 8))
        clean_mask[y0:y1, x0:x1] = True
        cur_x += max(cw, 8) + r.randint(2, 7)

    alpha = np.array(txt_layer)[:, :, 3]
    ink_mask = alpha > 40
    clean_mask = clean_mask & ink_mask
    return txt_layer, clean_mask


def generate_synthetic_case(rng: random.Random, idx: int):
    equations = [
        "x + 3 = 10",
        "2x + 5 = 15",
        "50 + 40n = 190y",
        "\\frac{3}{4} + \\frac{1}{2} = \\frac{5}{4}",
        "2(x + 3) - 5 = 4(2x - 1) + 8",
        "\\sqrt{x^2 + 16} = 2x - 3",
        "y = mx + b",
        "3x^2 - 12x + 9 = 0",
        "a + b + c + d + e + f = 100",
        "x_1 + x_2 + x_3 + x_4 = 40"
    ]
    eq = rng.choice(equations)

    font_path = rng.choice(AVAILABLE_FONTS) if AVAILABLE_FONTS else None
    font_size = rng.randint(26, 56)
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Vary canvas size: 300px to 4000px wide
    # Also include phone-photo aspect ratio 9:16
    aspect_choice = rng.choice([0.25, 0.4, 0.5, 0.75, 1.0, "9:16"])
    approx_txt_w = len(eq) * (font_size * 0.65)
    approx_txt_h = font_size * 1.5

    if aspect_choice == "9:16":
        w = rng.randint(450, 720)
        h = int(w * 16 / 9)
    else:
        w = rng.randint(400, 2400)
        h = max(220, int(w * aspect_choice))

    w = max(int(approx_txt_w + 80), min(w, 2400))
    h = max(int(approx_txt_h + 80), min(h, 1600))

    # Background type: includes real-photo-style artifacts
    bg_type = rng.choice([
        "white", "grey", "yellow", "gradient", "ruled", "grid", "margin",
        "paper_on_desk", "strong_shadow", "finger_curl"
    ])

    if bg_type == "white":
        bg_col = (rng.randint(245, 255), rng.randint(245, 255), rng.randint(245, 255))
        base = Image.new("RGB", (w, h), bg_col)
    elif bg_type == "grey":
        v = rng.randint(190, 225)
        base = Image.new("RGB", (w, h), (v, v, v))
    elif bg_type == "yellow":
        base = Image.new("RGB", (w, h), (rng.randint(240, 255), rng.randint(235, 250), rng.randint(190, 215)))
    elif bg_type in ["gradient", "strong_shadow"]:
        base_arr = np.zeros((h, w, 3), dtype=np.uint8)
        min_v = rng.randint(90, 140) if bg_type == "strong_shadow" else rng.randint(150, 190)
        grad = np.linspace(min_v, rng.randint(235, 255), w)
        for c in range(3):
            base_arr[:, :, c] = grad
        base = Image.fromarray(base_arr)
    else:
        v = rng.randint(220, 245)
        base = Image.new("RGB", (w, h), (v, v, v))

    draw_base = ImageDraw.Draw(base)

    # Real-photo-style desk surface along bottom or side
    if bg_type == "paper_on_desk":
        desk_col = (rng.randint(45, 80), rng.randint(35, 70), rng.randint(30, 60))
        desk_side = rng.choice(["bottom", "right", "top"])
        if desk_side == "bottom":
            desk_h = rng.randint(30, max(35, int(h * 0.18)))
            draw_base.rectangle([0, h - desk_h, w, h], fill=desk_col)
        elif desk_side == "right":
            desk_w = rng.randint(30, max(35, int(w * 0.15)))
            draw_base.rectangle([w - desk_w, 0, w, h], fill=desk_col)
        elif desk_side == "top":
            desk_h = rng.randint(25, max(30, int(h * 0.12)))
            draw_base.rectangle([0, 0, w, desk_h], fill=desk_col)

    # Real-photo-style finger / curl at border
    if bg_type == "finger_curl":
        finger_col = (rng.randint(180, 210), rng.randint(130, 160), rng.randint(110, 140))
        fc_w = rng.randint(35, max(40, int(w * 0.12)))
        fc_h = rng.randint(40, max(45, int(h * 0.15)))
        draw_base.ellipse([w - fc_w, h - fc_h, w + fc_w, h + fc_h], fill=finger_col)

    # Ruled lines / grid paper / margin lines
    if bg_type in ["ruled", "grid"]:
        line_step = rng.randint(28, 55)
        line_col = (rng.randint(170, 205), rng.randint(175, 210), rng.randint(200, 230))
        for y_l in range(line_step, h, line_step):
            draw_base.line([(0, y_l), (w, y_l)], fill=line_col, width=1)
        if bg_type == "grid":
            for x_l in range(line_step, w, line_step):
                draw_base.line([(x_l, 0), (x_l, h)], fill=line_col, width=1)

    if bg_type in ["margin", "strong_shadow"]:
        margin_x = rng.randint(40, max(50, w // 7))
        draw_base.line([(margin_x, 0), (margin_x, h)], fill=(210, 130, 140), width=rng.randint(1, 2))

    # Ink type
    ink_type = rng.choice(["black", "blue", "pencil", "light_on_dark"])
    if ink_type == "light_on_dark":
        base = Image.new("RGB", (w, h), (rng.randint(30, 50), rng.randint(35, 55), rng.randint(40, 60)))
        draw_base = ImageDraw.Draw(base)
        text_color = (rng.randint(220, 250), rng.randint(220, 250), rng.randint(220, 250), 255)
    elif ink_type == "pencil":
        if bg_type in ["gradient", "strong_shadow"]:
            text_color = (rng.randint(25, 60), rng.randint(25, 60), rng.randint(25, 60), 255)
        else:
            text_color = (rng.randint(90, 130), rng.randint(90, 130), rng.randint(90, 130), 255)
    elif ink_type == "blue":
        text_color = (rng.randint(15, 35), rng.randint(30, 60), rng.randint(130, 185), 255)
    else:
        text_color = (rng.randint(15, 45), rng.randint(15, 45), rng.randint(15, 45), 255)

    # Position: center, left, right, small writing in middle of 9:16
    approx_txt_w = len(eq) * (font_size * 0.65)
    approx_txt_h = font_size * 1.5

    if aspect_choice == "9:16":
        px = max(25, int((w - approx_txt_w) / 2))
        py = max(25, int((h - approx_txt_h) / 2) + rng.randint(-40, 40))
    else:
        pos_type = rng.choice(["center", "left", "right", "tiny"])
        if pos_type == "left":
            px = rng.randint(20, max(25, int(w * 0.15)))
            py = rng.randint(30, max(35, int(h - approx_txt_h - 40)))
        elif pos_type == "right":
            px = max(20, int(w - approx_txt_w - rng.randint(30, 60)))
            py = rng.randint(30, max(35, int(h - approx_txt_h - 40)))
        elif pos_type == "tiny":
            px = rng.randint(40, max(45, int(w * 0.35)))
            py = rng.randint(40, max(45, int(h * 0.35)))
        else:
            px = max(20, int((w - approx_txt_w) / 2))
            py = max(25, int((h - approx_txt_h) / 2))

    px = max(15, min(px, int(w - approx_txt_w - 20)))
    py = max(15, min(py, int(h - approx_txt_h - 20)))

    txt_img, clean_mask = render_jittered_text(eq, font, text_color, (w, h), (px, py), rng=rng)
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
    # 1. Synthetic case with far-right descender y
    w, h = 1300, 320
    im = Image.new("RGB", (w, h), (245, 245, 240))
    draw = ImageDraw.Draw(im)
    font_path = AVAILABLE_FONTS[0] if AVAILABLE_FONTS else None
    font = ImageFont.truetype(font_path, 44) if font_path else ImageFont.load_default()

    draw.text((100, 130), "50 + 40n = 19", fill=(35, 35, 40), font=font)
    draw.text((450, 130), "0y", fill=(40, 40, 45), font=font)

    crop = prep(im)
    box = crop.crop_box

    assert box[0] <= 100, f"Left edge clipped: {box}"
    assert box[2] >= 495, f"Right edge clipped (0y lost): {box}"
    print(f"PASS: Synthetic regression test ('50 + 40n = 190y' crop box {box})")

    # 2. Real photo regression test
    real_photo_path = "test_photos/WhatsApp Image 2026-10-16 at 12.42.32 PM.jpeg"
    if os.path.exists(real_photo_path):
        real_im = Image.open(real_photo_path)
        # Upright orientation is rotate=270
        real_crop = prep(real_im, rotate=270)
        rbox = real_crop.crop_box
        # Equation spans from 213 to 608 in x
        assert rbox[0] <= 213, f"Real photo left edge clipped: {rbox}"
        assert rbox[2] >= 608, f"Real photo right edge clipped (0y lost): {rbox}"
        print(f"PASS: Real photo regression test ({real_photo_path} at rot 270 crop box {rbox})")


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

        w, h = img.size
        crop = prep(img)
        x0, y0, x1, y1 = crop.crop_box
        box_w = x1 - x0
        box_h = y1 - y0

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

        # Property 2: Non-trivial crop metric
        # When writing occupies < 30% of photo area, crop area must be < 60% of photo area
        writing_area_frac = (ink_w * ink_h) / (w * h)
        crop_area_frac = (box_w * box_h) / (w * h)
        if writing_area_frac < 0.30 and crop_area_frac >= 0.60:
            errs.append(f"Trivial crop: crop area {crop_area_frac:.2f} >= 0.60 for writing area {writing_area_frac:.2f}")

        # Property 3: Aspect ratio not distorted (scale_x equals scale_y within 1%)
        final_w, final_h = crop.size
        scale_x = final_w / box_w
        scale_y = final_h / box_h
        if max(scale_x, scale_y) > 0:
            diff_scale = abs(scale_x - scale_y) / max(scale_x, scale_y)
            if diff_scale >= 0.01:
                errs.append(f"Aspect ratio distorted: diff {diff_scale:.4f} >= 0.01 (scale_x={scale_x}, scale_y={scale_y})")

        # Property 4: No step produced width or height smaller than the ink box
        if box_w < ink_w:
            errs.append(f"Crop width {box_w} < ink width {ink_w}")
        if box_h < ink_h:
            errs.append(f"Crop height {box_h} < ink height {ink_h}")

        if errs:
            failures += 1
            print(f"FAIL [Case {idx}] ({eq}): {', '.join(errs)}")
        else:
            passed += 1

        total_done = passed + failures
        if total_done % 50 == 0:
            print(f"  Progress: {total_done}/{n_cases} cases completed ({passed} passed, {failures} failed)")

    print(f"\nRandom cases summary: {passed}/{passed + failures} passed, {failures} failures.")
    return failures


if __name__ == "__main__":
    test_regression_50_40n_190y()
    fails = run_random_suite(300)
    sys.exit(1 if fails else 0)
