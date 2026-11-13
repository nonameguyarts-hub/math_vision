import random
from test_preprocess_clipping import generate_synthetic_case
import cv2, numpy as np

def run_test():
    rng = random.Random(42)
    passed = 0
    failed = 0
    fail_cases = []

    for idx in range(1, 301):
        img, clean_mask, eq = generate_synthetic_case(rng, idx)
        w, h = img.size
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
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
            if (long_s / short_s > 6.0 and long_s > 60) or area < 4:
                continue
            if tb and not (cw <= 0.10 * w and ch <= 0.10 * h and area <= 0.003 * (w * h) and (long_s / short_s) < 4.0):
                continue
            cands.append(i)

        if not cands:
            failed += 1
            continue

        heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in cands if stats[i, cv2.CC_STAT_AREA] >= 15]
        med_h = max(10, np.median(heights)) if heights else 15

        xs = [cents[i][0] for i in cands]
        ys = [cents[i][1] for i in cands]
        is_vertical = (np.std(ys) > 1.2 * np.std(xs))

        if not is_vertical:
            bin_h = max(15, int(med_h * 2.0))
            hist, bin_edges = np.histogram(ys, bins=max(4, int(h / bin_h)))
            peak_idx = np.argmax(hist)
            line_y1 = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0
            hist_copy = hist.copy()
            hist_copy[max(0, peak_idx - 2):min(len(hist), peak_idx + 3)] = 0
            peak2_idx = np.argmax(hist_copy)
            if hist_copy[peak2_idx] >= max(2, int(0.20 * hist[peak_idx])):
                line_y2 = (bin_edges[peak2_idx] + bin_edges[peak2_idx + 1]) / 2.0
                def in_band(cy, ch):
                    return (abs(cy - line_y1) <= max(3.5 * med_h, ch)) or (abs(cy - line_y2) <= max(3.5 * med_h, ch))
            else:
                def in_band(cy, ch):
                    return abs(cy - line_y1) <= max(3.5 * med_h, ch)
            cluster = [i for i in cands if in_band(cents[i][1], stats[i, cv2.CC_STAT_HEIGHT])]
        else:
            bin_w = max(15, int(med_h * 2.0))
            hist, bin_edges = np.histogram(xs, bins=max(4, int(w / bin_w)))
            peak_idx = np.argmax(hist)
            line_x1 = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0
            def in_band(cx, cw):
                return abs(cx - line_x1) <= max(3.5 * med_h, cw)
            cluster = [i for i in cands if in_band(cents[i][0], stats[i, cv2.CC_STAT_WIDTH])]

        if not cluster:
            cluster = cands

        min_x = min(stats[i, 0] for i in cluster)
        max_x = max(stats[i, 0] + stats[i, 2] for i in cluster)
        min_y = min(stats[i, 1] for i in cluster)
        max_y = max(stats[i, 1] + stats[i, 3] for i in cluster)

        pad = max(10, int(0.12 * (max_y - min_y)))
        x0, y0 = max(0, min_x - pad), max(0, min_y - pad)
        x1, y1 = min(w, max_x + pad), min(h, max_y + pad)

        tot = clean_mask.sum()
        if tot == 0: continue
        crop_mask = clean_mask[y0:y1, x0:x1]
        recall = crop_mask.sum() / tot
        if recall < 0.99:
            failed += 1
            fail_cases.append((idx, eq, recall))
        else:
            passed += 1

    print(f"Results: {passed} passed, {failed} failed")
    if fail_cases:
        print("First 5 failed:", fail_cases[:5])

if __name__ == "__main__":
    run_test()
