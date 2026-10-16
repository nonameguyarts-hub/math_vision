"""Smoke test:  python check_recognizer.py photo1.jpg photo2.png ...   (or a folder)
Prints the recognized LaTeX, the solver answer and the time per image. Optional: --ckpt path/to/best.pt"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--no-crop", action="store_true", help="skip auto-crop (use for images already cropped to the equation)")
a = ap.parse_args()

files = []
for p in a.paths:
    files += sorted(glob.glob(os.path.join(p, "*"))) if os.path.isdir(p) else [p]
files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]

from recognizer import Recognizer
from solver import solve_latex
t0 = time.time()
rec = Recognizer(a.ckpt) if a.ckpt else Recognizer()
print(f"Loaded model (epoch {rec.epoch}, {len(rec.vocab)} symbols) in {time.time() - t0:.1f}s\n")
for f in files:
    t0 = time.time()
    try:
        latex = rec.recognize(f, autocrop=not a.no_crop)
    except Exception as e:
        print(f"{os.path.basename(f)}: ERROR {type(e).__name__}: {e}\n"); continue
    dt = time.time() - t0
    r = solve_latex(latex)
    ans = r["result"] if r["ok"] else f"(solver: {r['error']})"
    print(f"{os.path.basename(f)}  [{dt:.2f}s]\n   latex : {latex}\n   answer: {ans}\n")
