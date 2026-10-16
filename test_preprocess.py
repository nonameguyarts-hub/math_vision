"""python test_preprocess.py - checks the photo preprocessing on synthetic images (no model needed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
from preprocess import preprocess, looks_sideways, IMG_H, MAX_W

def synth(text, big=False, tilt_noise=True):
    h, w = (1600, 1200) if big else (180, 520)
    img = np.full((h, w), 225, np.uint8)
    if tilt_noise:                                     # uneven lighting + sensor noise
        img = (img * np.linspace(0.75, 1.0, w)[None, :]).astype(np.uint8)
        img = np.clip(img.astype(int) + np.random.RandomState(0).randint(-8, 8, img.shape), 0, 255).astype(np.uint8)
    org = (w // 3, h // 2) if big else (30, 120)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 3.0 if big else 3.2, 30, 10 if big else 8, cv2.LINE_AA)
    return img

fails = 0
def check(name, cond):
    global fails
    if not cond: fails += 1; print("FAIL |", name)

for name, img in [("small tight", synth("12+3")), ("big page, small text", synth("12+3", big=True))]:
    canvas, w = preprocess(img)
    check(f"{name}: height {canvas.shape[0]}", canvas.shape[0] == IMG_H)
    check(f"{name}: width matches {w}", canvas.shape[1] == w and 40 < w <= MAX_W)
    check(f"{name}: dtype/range", canvas.dtype == np.float32 and 0 <= canvas.min() and canvas.max() <= 1)
    frac = float((canvas > 0.5).mean())
    check(f"{name}: ink fraction {frac:.3f} plausible", 0.03 < frac < 0.5)      # white strokes on black, not inverted
    check(f"{name}: corners are background", canvas[:4, :4].mean() < 0.2)
    print(f"{name}: shape {canvas.shape}, ink fraction {frac:.3f}")

blank = np.full((200, 400), 230, np.uint8)
try:
    preprocess(blank); check("blank image should raise", False)
except ValueError:
    pass

wide = np.full((120, 4000), 225, np.uint8)
for k in range(20): cv2.putText(wide, "8", (20 + k * 190, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.8, 30, 7, cv2.LINE_AA)
c, w = preprocess(wide, autocrop=False)
check(f"very wide capped at {MAX_W}: got {w}", w == MAX_W and c.shape[0] == IMG_H)

# ruled paper photographed sideways (like a phone photo of a notebook page turned 90 degrees)
def ruled_page(text, sideways):
    h, w = 700, 1300
    img = np.full((h, w), 225, np.float32) * np.linspace(0.7, 1.0, w)[None, :]
    for y in range(40, h, 90):
        img[y:y + 3, :] = 165                                   # faint ruled lines
    img = np.clip(img + np.random.RandomState(1).randint(-6, 6, img.shape), 0, 255).astype(np.uint8)
    cv2.putText(img, text, (300, 330), cv2.FONT_HERSHEY_SIMPLEX, 2.6, 40, 8, cv2.LINE_AA)
    return np.rot90(img, k=-1).copy() if sideways else img          # k=-1: clockwise

upright, sideways = ruled_page("5x+50=180", False), ruled_page("5x+50=180", True)
check("upright page not flagged sideways", not looks_sideways(upright))
check("sideways page flagged", looks_sideways(sideways))
cu, wu = preprocess(upright)
cs, ws = preprocess(sideways, rotate=90)                       # 90 = counter-clockwise turn
check(f"ruled lines removed (ink fraction {float((cu > 0.5).mean()):.3f})", (cu > 0.5).mean() < 0.4)
check(f"rotated sideways page matches upright (width {ws} vs {wu})", abs(ws - wu) <= 0.1 * wu)
check("rotated ink fraction matches", abs(float((cs > 0.5).mean()) - float((cu > 0.5).mean())) < 0.03)
cv2.imwrite("/tmp/sideways_fixed.png", (cs * 255).astype(np.uint8))
print("ruled page: upright width", wu, "| sideways+rotate width", ws)

# dust, a shadow blob and a line fragment far from the writing must not widen the crop
clean = ruled_page("5x+50=180", False)
dirty = clean.copy()
cv2.circle(dirty, (1250, 640), 30, 90, -1)                    # shadow blob, bottom right
dirty[60:64, 1100:1180] = 70                                  # dark hairline fragment
for p_ in [(80, 600), (1200, 60), (40, 40)]:
    cv2.circle(dirty, p_, 4, 60, -1)                          # dust
c1, w1 = preprocess(clean); c2, w2 = preprocess(dirty)
check(f"stray marks do not change the crop (width {w1} vs {w2})", abs(w1 - w2) <= 0.08 * w1)

# a stacked fraction: numerator, bar and denominator must all survive the cleanup
fr = np.full((420, 600), 225, np.uint8)
cv2.putText(fr, "123", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 40, 8, cv2.LINE_AA)
cv2.line(fr, (180, 200), (470, 200), 40, 8)
cv2.putText(fr, "456", (200, 330), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 40, 8, cv2.LINE_AA)
cf, wf = preprocess(fr)
third = IMG_H // 3
rows = [float((cf[i * third:(i + 1) * third] > 0.5).mean()) for i in range(3)]
check(f"fraction keeps numerator, bar and denominator (row ink {[round(r, 2) for r in rows]})", all(r > 0.02 for r in rows))
print("Preprocess tests:", "all passed" if not fails else f"{fails} failed")
sys.exit(1 if fails else 0)