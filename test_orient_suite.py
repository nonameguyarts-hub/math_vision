import random
from test_preprocess_clipping import generate_synthetic_case
import cv2, numpy as np

def run_suite():
    rng = random.Random(42)
    passed = 0
    failed = 0
    fails = []

    for idx in range(1, 301):
        img, clean_mask, eq = generate_synthetic_case(rng, idx)
        w, h = img.size
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        # Polarity
        bw_margin = max(5, int(w * 0.05))
        bh_margin = max(5, int(h * 0.05))
        border = np.concatenate([g[:bh_margin, :].ravel(), g[-bh_margin:, :].ravel(),
                                 g[:, :bw_margin].ravel(), g[:, -bw_margin:].ravel()])
        if np.median(border) < 110:
            g = cv2.bitwise_not(g)

        # Shadow removal
        k_blur = max(31, int(min(w, h) * 0.22) | 1)
        blur = cv2.GaussianBlur(g, (k_blur, k_blur), 0)
        diff = cv2.subtract(blur, g)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        t, _ = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (diff_norm >= max(10, int(t * 0.40))).astype(np.uint8) * 255

        # Lines removal
        line_len = max(38, min(60, int(0.06 * max(w, h))))
        lines_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1)))
        lines_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len)))
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
            fails.append((idx, eq, "no_cands"))
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
                    return (abs(cy - line_y1) <= max(2.5 * med_h, ch * 0.7)) or (abs(cy - line_y2) <= max(2.5 * med_h, ch * 0.7))
            else:
                def in_band(cy, ch):
                    return abs(cy - line_y1) <= max(2.5 * med_h, ch * 0.7)
            line_indices = [i for i in cands if in_band(cents[i][1], stats[i, cv2.CC_STAT_HEIGHT])]
            
            # Group along X
            if len(line_indices) > 1:
                line_indices.sort(key=lambda i: stats[i, 0])
                groups = [[line_indices[0]]]
                max_gap = max(90, int(4.0 * med_h))
                for c in line_indices[1:]:
                    prev_r = max(stats[m, 0] + stats[m, 2] for m in groups[-1])
                    curr_l = stats[c, 0]
                    if curr_l - prev_r <= max_gap:
                        groups[-1].append(c)
                    else:
                        groups.append([c])
                groups.sort(key=lambda grp: sum(stats[i, 4] for i in grp), reverse=True)
                cluster = groups[0]
            else:
                cluster = line_indices
        else:
            bin_w = max(15, int(med_h * 2.0))
            hist, bin_edges = np.histogram(xs, bins=max(4, int(w / bin_w)))
            peak_idx = np.argmax(hist)
            line_x1 = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0
            def in_band(cx, cw):
                return abs(cx - line_x1) <= max(2.5 * med_h, cw * 0.7)
            line_indices = [i for i in cands if in_band(cents[i][0], stats[i, cv2.CC_STAT_WIDTH])]
            if len(line_indices) > 1:
                line_indices.sort(key=lambda i: stats[i, 1])
                groups = [[line_indices[0]]]
                max_gap = max(90, int(4.0 * med_h))
                for c in line_indices[1:]:
                    prev_b = max(stats[m, 1] + stats[m, 3] for m in groups[-1])
                    curr_t = stats[c, 1]
                    if curr_t - prev_b <= max_gap:
                        groups[-1].append(c)
                    else:
                        groups.append([c])
                groups.sort(key=lambda grp: sum(stats[i, 4] for i in grp), reverse=True)
                cluster = groups[0]
            else:
                cluster = line_indices

        if not cluster:
            cluster = cands

        min_x = min(stats[i, 0] for i in cluster)
        max_x = max(stats[i, 0] + stats[i, 2] for i in cluster)
        min_y = min(stats[i, 1] for i in cluster)
        max_y = max(stats[i, 1] + stats[i, 3] for i in cluster)

        pad = max(10, int(0.12 * max(max_y - min_y, (max_x - min_x) * 0.1)))
        x0 = max(0, min_x - pad)
        y0 = max(0, min_y - pad)
        x1 = min(w, max_x + pad)
        y1 = min(h, max_y + pad)

        tot = clean_mask.sum()
        if tot == 0: continue
        crop_mask = clean_mask[y0:y1, x0:x1]
        recall = crop_mask.sum() / tot
        
        # Metric check: when writing occupies < 30%, crop area must be < 60%
        writing_frac = (clean_mask.any(axis=0).sum() * clean_mask.any(axis=1).sum()) / (w * h)
        crop_frac = ((x1 - x0) * (y1 - y0)) / (w * h)

        if recall < 0.99:
            failed += 1
            fails.append((idx, eq, f"recall={recall:.3f}"))
        elif writing_frac < 0.30 and crop_frac >= 0.60:
            failed += 1
            fails.append((idx, eq, f"trivial_crop={crop_frac:.2f}"))
        else:
            passed += 1

    print(f"Results: {passed} passed, {failed} failed")
    if fails:
        print("First 10 fails:", fails[:10])

if __name__ == "__main__":
    run_suite()
