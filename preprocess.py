"""preprocess.py - photo -> model input. No torch needed.

Mirrors the training pipeline: binarize (the Step 17D method), 128 px tall, white strokes on a
black canvas, width capped at 1280. Added on top: an optional auto-crop to the handwriting, because
the training images were tight crops of single equations.

Assumes dark ink on light paper (pen or pencil on a page). White chalk on a dark board is not handled.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_H, MAX_W = 128, 1280
THR_PHOTO = 1.0      # threshold = Otsu x this. 1.0 keeps pen strokes but drops faint ruled lines
THR_FAINT = 0.8      # fallback for very light writing (the value used when binarizing HME100K)
MIN_AREA = 10        # smallest kept blob in pixels at 128 px height


def to_gray(image) -> np.ndarray:
    """PIL image, numpy array or file path -> uint8 grayscale."""
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image).convert("L")     # phones store rotation in EXIF
        return np.array(image, dtype=np.uint8)
    g = np.asarray(image)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    return g.astype(np.uint8)


def _drop_rules(ink: np.ndarray, max_thick: float) -> np.ndarray:
    """Remove ruled lines and page edges: thin strokes that cross (almost) the whole image, and
    hairline fragments (about 1-2 px thick, long and straight) left where a line fades.
    A lone minus sign or fraction bar is kept, because it is then the only thing in the image."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return ink
    h, w = ink.shape
    cw, ch, area = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
    rule = np.zeros(n, bool)
    long_, short_ = np.maximum(cw, ch), np.maximum(np.minimum(cw, ch), 1)
    spans = ((cw >= 0.9 * w) & (ch <= 0.08 * h)) | ((ch >= 0.9 * h) & (cw <= 0.08 * w))
    hairline = (area / long_ <= max_thick) & (long_ / short_ >= 8)
    rule[1:] = spans | hairline
    if rule[1:].all():
        return ink
    out = ink.copy()
    out[rule[lab]] = 0
    return out


def _keep_text(ink: np.ndarray) -> np.ndarray:
    """Keep the handwriting and drop stray specks, shadow blobs and fragments away from it.
    The core is the cluster of strokes that line up in one row; anything within a few
    stroke-heights of the core stays (operators, dots, exponents), the rest goes."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 2:
        return ink
    x, y, w, h, a = (st[1:, i].astype(float) for i in range(5))
    cy = y + h / 2
    cand = np.where(a >= 0.1 * a.max())[0]
    best, best_key = None, (-1, -1.0)
    for i in cand:                                           # strokes sharing this stroke's row
        row = cand[np.abs(cy[cand] - cy[i]) <= 0.75 * max(h[i], 4)]
        row = row[np.argsort(x[row])]
        gap_limit = 4 * np.median(h[row])
        clusters, cur = [], [row[0]]
        for j in row[1:]:                                    # split the row at big horizontal gaps
            if x[j] - (x[cur[-1]] + w[cur[-1]]) > gap_limit:
                clusters.append(cur); cur = [j]
            else:
                cur.append(j)
        clusters.append(cur)
        for c in clusters:
            key = (len(c), float(a[c].sum()))
            if key > best_key:
                best, best_key = c, key
    core = np.array(best)
    cx0, cx1 = x[core].min(), (x + w)[core].max()
    cy0, cy1 = y[core].min(), (y + h)[core].max()
    hc = max(cy1 - cy0, 1)
    ex, ey = 3 * hc, 2.5 * hc       # stacked fractions span about three stroke-heights
    mx, my = x + w / 2, y + h / 2
    near = (mx >= cx0 - ex) & (mx <= cx1 + ex) & (my >= cy0 - ey) & (my <= cy1 + ey)
    keep = np.zeros(n, bool)
    keep[1:] = near
    out = ink.copy()
    out[~keep[lab]] = 0
    return out


def _ink(g: np.ndarray, target_h: int, min_area: int, thr: float = THR_PHOTO) -> np.ndarray:
    """Gray photo -> binary ink mask (ink = 255) at height target_h."""
    h, w = g.shape
    nw = max(1, int(round(w * target_h / h)))
    g = cv2.resize(g, (nw, target_h), interpolation=cv2.INTER_CUBIC if target_h > h else cv2.INTER_AREA)
    k = int(np.clip(target_h // 3, 11, 41)) | 1
    bg = cv2.medianBlur(cv2.dilate(g, np.ones((3, 3), np.uint8)), k)       # estimated paper
    norm = cv2.normalize(cv2.subtract(bg, g), None, 0, 255, cv2.NORM_MINMAX)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, ink = cv2.threshold(norm, t * thr, 255, cv2.THRESH_BINARY)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    ink = _drop_rules((keep[lab] * 255).astype(np.uint8), 2.0 if target_h > 200 else 1.3)
    return _keep_text(ink)


def _ink_bbox(g: np.ndarray, thr: float = THR_PHOTO):
    """Bounding box (y0, y1, x0, x1) of the handwriting in full-resolution coordinates, and the scale used."""
    h, w = g.shape
    scale = min(1.0, 800 / max(h, w))
    small = cv2.resize(g, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else g
    sh, sw = small.shape
    ink = _ink(small, sh, max(15, int(sh * sw * 2e-5)), thr)
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        raise ValueError("No handwriting found in the image.")
    return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1), scale, (sh, sw)


def looks_sideways(image) -> bool:
    """True when the handwriting is clearly taller than wide, i.e. the photo needs rotating."""
    try:
        (y0, y1, x0, x1), scale, (sh, sw) = _ink_bbox(to_gray(image))
        ink_h = y1 - y0
        ink_w = x1 - x0
    except ValueError:
        return False

    if ink_h <= 1.35 * ink_w:
        return False

    # Fraction guard: if a component with width >= 60% of the crop width and aspect > 3
    # exists with ink both above and below it, treat the crop as an upright fraction and do not rotate.
    try:
        g = to_gray(image)
        scale_val = min(1.0, 800 / max(g.shape))
        small = cv2.resize(g, (max(1, round(g.shape[1] * scale_val)), max(1, round(g.shape[0] * scale_val))), interpolation=cv2.INTER_AREA) if scale_val < 1 else g
        ink = _ink(small, small.shape[0], max(15, int(small.shape[0] * small.shape[1] * 2e-5)), THR_PHOTO)
        crop_ink = ink[y0:y1, x0:x1]
        crop_w = max(1, ink_w)
        n, lab, st, _ = cv2.connectedComponentsWithStats(crop_ink, connectivity=8)
        for i in range(1, n):
            cw = st[i, cv2.CC_STAT_WIDTH]
            ch = st[i, cv2.CC_STAT_HEIGHT]
            cy = st[i, cv2.CC_STAT_TOP]
            aspect = cw / max(ch, 1)
            if cw >= 0.60 * crop_w and aspect > 3.0:
                ink_above = np.count_nonzero(crop_ink[:cy, :])
                ink_below = np.count_nonzero(crop_ink[cy + ch:, :])
                if ink_above >= 10 and ink_below >= 10:
                    return False
    except Exception:
        pass

    return True


def crop_to_ink(g: np.ndarray, margin: float = 0.12, thr: float = THR_PHOTO) -> np.ndarray:
    """Crop the gray photo to the handwriting, with a margin."""
    (y0, y1, x0, x1), scale, (sh, sw) = _ink_bbox(g, thr)
    pad = max(8, int(margin * (y1 - y0)))
    y0, y1 = max(0, y0 - pad), min(sh, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(sw, x1 + pad)
    return g[int(y0 / scale):int(np.ceil(y1 / scale)), int(x0 / scale):int(np.ceil(x1 / scale))]


def preprocess(image, autocrop: bool = True, rotate: int = 0):
    """Returns (canvas float32 [IMG_H, w] with ink = 1.0, width in pixels).
    rotate = degrees counter-clockwise (0, 90, 180, 270) for sideways photos."""
    g0 = to_gray(image)
    if rotate % 360:
        g0 = np.ascontiguousarray(np.rot90(g0, k=(rotate // 90) % 4))
    ink = None
    for thr in (THR_PHOTO, THR_FAINT):               # normal first, then the faint-writing fallback
        try:
            g = crop_to_ink(g0, thr=thr) if autocrop else g0
            cand = _ink(g, IMG_H, MIN_AREA, thr)
        except ValueError:
            continue
        if cand.any() and ink is None:
            ink = cand
        if cand.mean() / 255 >= 0.01:                # enough ink: accept this setting
            ink = cand
            break
    if ink is None or not ink.any():
        raise ValueError("No handwriting found in the image.")
    h, w = ink.shape
    if w > MAX_W:                                   # same width cap as training (new_size)
        s = MAX_W / w
        h, w = max(1, round(h * s)), MAX_W
        ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_H, w), np.float32)
    o = (IMG_H - h) // 2
    canvas[o:o + h] = ink.astype(np.float32) / 255.0
    return canvas, w