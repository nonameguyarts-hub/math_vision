import random
from test_preprocess_clipping import generate_synthetic_case
import cv2, numpy as np

def run():
    rng = random.Random(42)
    passed = 0
    failed = 0
    failures = []

    for idx in range(1, 301):
        img, clean_mask, eq = generate_synthetic_case(rng, idx)
        w, h = img.size
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        bw_margin = max(5, int(w * 0.05))
        bh_margin = max(5, int(h * 0.05))
        border = np.concatenate([g[:bh_margin, :].ravel(), g[-bh_margin:, :].ravel(),
                                 g[:, :bw_margin].ravel(), g[:, -bw_margin:].ravel()])
        if np.median(border) < 110:
            g = cv2.bitwise_not(g)

        k_blur = max(31, int(min(w, h) * 0.22) | 1)
        blur = cv2.GaussianBlur(g, (k_blur, k_blur), 0)
        diff = cv2.subtract(blur, g)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        t, _ = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (diff_norm >= max(10, int(t * 0.40))).astype(np.uint8) * 255

        line_w = max(40, int(w * 0.25))
        line_h = max(40, int(h * 0.25))
        lines_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1)))
        lines_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_h)))
        ink = cv2.subtract(ink, cv2.bitwise_or(lines_h, lines_v))

        n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
        cands = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            tb = (x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1)
            long_s = max(cw, ch)
            short_s = max(min(cw, ch), 1)
            aspect = long_s / short_s

            if area < 4:
                continue
            if aspect > 6.0 and long_s > 80:
                continue
            if tb and (cw > 0.15 * w or ch > 0.15 * h or area > 0.005 * (w * h) or (aspect > 6.0 and long_s > 80)):
                continue
            cands.append(i)

        if not cands:
            failed += 1
            failures.append((idx, eq, "no_cands"))
            continue

        # Dominant ink box: min and max of candidates
        min_x = min(stats[i, 0] for i in cands)
        max_x = max(stats[i, 0] + stats[i, 2] for i in cands)
        min_y = min(stats[i, 1] for i in cands)
        max_y = max(stats[i, 1] + stats[i, 3] for i in cands)
        ink_h = max_y - min_y
        ink_w = max_x - min_x

        pad = max(10, int(0.12 * max(ink_h, ink_w * 0.1)))
        x0 = max(0, min_x - pad)
        y0 = max(0, min_y - pad)
        x1 = min(w, max_x + pad)
        y1 = min(h, max_y + pad)

        tot = clean_mask.sum()
        if tot == 0: continue
        crop_mask = clean_mask[y0:y1, x0:x1]
        recall = crop_mask.sum() / tot
        
        # Check non-trivial metric: when writing occupies < 30%, crop area must be < 60%
        writing_frac = (clean_mask.any(axis=0).sum() * clean_mask.any(axis=1).sum()) / (w * h)
        crop_frac = ((x1 - x0) * (y1 - y0)) / (w * h)

        if recall < 0.99:
            failed += 1
            failures.append((idx, eq, f"recall={recall:.3f}"))
        elif writing_frac < 0.30 and crop_frac >= 0.60:
            failed += 1
            failures.append((idx, eq, f"trivial_crop={crop_frac:.2f}"))
        else:
            passed += 1

    print(f"Results: {passed} passed, {failed} failed")
    if failures:
        print("First 10 failures:", failures[:10])

if __name__ == "__main__":
    run()
