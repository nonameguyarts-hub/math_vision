import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def prep_v2(img: Image.Image, rotate: int = 0, margin: float = 0.12, max_side: int = 512):
    img = img.convert("RGB")
    if rotate:
        img = img.rotate(rotate, expand=True)
    w, h = img.size
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # 1. Polarity detection (dark background like blackboards)
    bw_margin = max(5, int(w * 0.05))
    bh_margin = max(5, int(h * 0.05))
    border = np.concatenate([g[:bh_margin, :].ravel(), g[-bh_margin:, :].ravel(),
                             g[:, :bw_margin].ravel(), g[:, -bw_margin:].ravel()])
    if np.median(border) < 110:
        g = cv2.bitwise_not(g)

    # 2. Low-frequency shadow removal: large blur subtraction
    k_blur = max(31, int(min(w, h) * 0.22) | 1)
    blur = cv2.GaussianBlur(g, (k_blur, k_blur), 0)
    diff = cv2.subtract(blur, g)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Otsu threshold on difference
    t, _ = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (diff_norm >= max(12, int(t * 0.45))).astype(np.uint8) * 255

    # Remove thin ruled lines before connected components
    line_w = max(40, int(w * 0.25))
    line_h = max(40, int(h * 0.25))
    lines_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1)))
    lines_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_h)))
    ink = cv2.subtract(ink, cv2.bitwise_or(lines_h, lines_v))

    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        out = img.copy()
        out.warning = "Couldn't isolate the handwriting. Retake on plain paper with some space around it."
        out.crop_box = (0, 0, w, h)
        out.ink_box = (0, 0, w, h)
        return out

    # 3. Discard components touching border that are large or long (page edge, desk, shadow)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[0] = False
    
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches_border = (x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1)
        long_side = max(cw, ch)
        short_side = max(min(cw, ch), 1)
        aspect = long_side / short_side
        
        # Ruled line segment / hairline check
        if aspect > 5.5 and long_side > 60:
            keep_mask[i] = False
            continue
            
        # Tiny specks
        if area < 5:
            keep_mask[i] = False
            continue

        # Border components
        if touches_border:
            # Clearly text-sized: small dimension and area
            is_text_sized = (cw <= 0.10 * w and ch <= 0.10 * h and area <= 0.003 * (w * h) and aspect < 4.0)
            if not is_text_sized:
                keep_mask[i] = False

    candidate_indices = [i for i in range(1, n) if keep_mask[i]]
    if not candidate_indices:
        out = img.copy()
        out.warning = "Couldn't isolate the handwriting. Retake on plain paper with some space around it."
        out.crop_box = (0, 0, w, h)
        out.ink_box = (0, 0, w, h)
        return out

    # Median height of text-like components
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in candidate_indices if stats[i, cv2.CC_STAT_AREA] >= 15]
    if not heights:
        heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in candidate_indices]
    med_h = max(10, np.median(heights))

    # Keep components within plausible height range
    text_indices = [i for i in candidate_indices if 0.20 * med_h <= stats[i, cv2.CC_STAT_HEIGHT] <= 4.0 * med_h]
    if not text_indices:
        text_indices = candidate_indices

    # Text orientation & dominant cluster
    y_cents = [cents[i][1] for i in text_indices]
    x_cents = [cents[i][0] for i in text_indices]
    
    is_horizontal = (np.std(x_cents) >= np.std(y_cents))
    
    if is_horizontal:
        # Cluster horizontally into 1 or 2 lines
        bin_h = max(12, int(med_h * 1.5))
        hist, bin_edges = np.histogram(y_cents, bins=max(5, int(h / bin_h)))
        peak_idx = np.argmax(hist)
        line_y1 = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0
        
        # Check second line
        hist_copy = hist.copy()
        hist_copy[max(0, peak_idx - 2):min(len(hist), peak_idx + 3)] = 0
        peak2_idx = np.argmax(hist_copy)
        if hist_copy[peak2_idx] >= max(2, int(0.25 * hist[peak_idx])):
            line_y2 = (bin_edges[peak2_idx] + bin_edges[peak2_idx + 1]) / 2.0
            def in_line(cy):
                return abs(cy - line_y1) <= 2.2 * med_h or abs(cy - line_y2) <= 2.2 * med_h
            num_lines = 2
        else:
            def in_line(cy):
                return abs(cy - line_y1) <= 2.2 * med_h
            num_lines = 1
            
        cluster_indices = [i for i in candidate_indices if in_line(cents[i][1])]
    else:
        # Vertical text line
        bin_w = max(12, int(med_h * 1.5))
        hist, bin_edges = np.histogram(x_cents, bins=max(5, int(w / bin_w)))
        peak_idx = np.argmax(hist)
        line_x = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0
        num_lines = 1
        cluster_indices = [i for i in candidate_indices if abs(cents[i][0] - line_x) <= 2.2 * med_h]

    if not cluster_indices:
        cluster_indices = text_indices

    # Check if text-sized ink itself reaches the photo border
    writing_touches_border = False
    for i in cluster_indices:
        x, y, cw, ch, _ = stats[i]
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            writing_touches_border = True
            break

    # Union bounding box across cluster
    min_x = min(stats[i, cv2.CC_STAT_LEFT] for i in cluster_indices)
    max_x = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in cluster_indices)
    min_y = min(stats[i, cv2.CC_STAT_TOP] for i in cluster_indices)
    max_y = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in cluster_indices)
    
    ink_w = max_x - min_x
    ink_h = max_y - min_y
    
    # Pad every side by max(10 px, 12% of ink height)
    pad = max(10, int(round(margin * ink_h)))
    crop_x0 = max(0, min_x - pad)
    crop_y0 = max(0, min_y - pad)
    crop_x1 = min(w, max_x + pad)
    crop_y1 = min(h, max_y + pad)

    # 4. Clipping Guard: Thin ring just OUTSIDE the crop box
    ring_thick = max(2, int(0.02 * ink_h))
    step = max(5, int(0.05 * ink_h))
    
    for _ in range(8):
        expanded = False
        # Left
        if crop_x0 > 0:
            rx0 = max(0, crop_x0 - ring_thick)
            if (ink[crop_y0:crop_y1, rx0:crop_x0] > 0).sum() > 10:
                crop_x0 = max(0, crop_x0 - step)
                expanded = True
        # Right
        if crop_x1 < w:
            rx1 = min(w, crop_x1 + ring_thick)
            if (ink[crop_y0:crop_y1, crop_x1:rx1] > 0).sum() > 10:
                crop_x1 = min(w, crop_x1 + step)
                expanded = True
        # Top
        if crop_y0 > 0:
            ry0 = max(0, crop_y0 - ring_thick)
            if (ink[ry0:crop_y0, crop_x0:crop_x1] > 0).sum() > 10:
                crop_y0 = max(0, crop_y0 - step)
                expanded = True
        # Bottom
        if crop_y1 < h:
            ry1 = min(h, crop_y1 + ring_thick)
            if (ink[crop_y1:ry1, crop_x0:crop_x1] > 0).sum() > 10:
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
    
    # Proportional scaling
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

    crop.warning = warning
    crop.crop_box = (crop_x0, crop_y0, crop_x1, crop_y1)
    crop.ink_box = (min_x, min_y, max_x, max_y)
    crop.num_lines = num_lines
    return crop

for p in [
    "test_photos/WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg",
    "test_photos/WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg",
    "test_photos/WhatsApp Image 2026-10-16 at 12.42.32 PM.jpeg"
]:
    im = Image.open(p)
    w, h = im.size
    crop = prep_v2(im)
    box = crop.crop_box
    area_pct = ((box[2] - box[0]) * (box[3] - box[1])) / (w * h) * 100
    print(f"{p} ({w}x{h}): box={box}, final_size={crop.size}, area={area_pct:.1f}%, warn={crop.warning}")
