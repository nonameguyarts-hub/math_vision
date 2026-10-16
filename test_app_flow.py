import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recognizer import DEFAULT_MAX_SIDE, prep, recognize
from steps import steps_for
from ui_checks import review

photos = [
    ("WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg", "Photo 1", "Equation"),
    ("WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg", "Photo 2", "Equation"),
]

print(f"Running app verification with DEFAULT_MAX_SIDE = {DEFAULT_MAX_SIDE}...\n")

for path, label, mode in photos:
    print("=" * 60)
    print(f"Testing {label} ({path}) in Mode: {mode}")
    print("=" * 60)
    
    img = Image.open(path)
    
    # 1. Preview ("What the model sees")
    crop = prep(img, max_side=DEFAULT_MAX_SIDE)
    print(f"1. What the model sees: Crop size = {crop.size}")
    
    # 2. Read equation
    latex = recognize(img, max_side=DEFAULT_MAX_SIDE)
    print(f"2. Recognized LaTeX: {repr(latex)}")
    
    # 3. UI Review / Guardrails
    rev = review(latex, mode)
    print(f"3. Guardrail review: ok={rev['ok']}, error={repr(rev['error'])}, warnings={rev['warnings']}")
    
    # 4. Solve and Steps
    if not rev["ok"]:
        print(f"4. App UI display: Error message -> '{rev['error']}'")
    else:
        out = steps_for(latex)
        if out["ok"]:
            print(f"4. Solved successfully: Answer = {out['answer']}")
            print(f"   Steps count: {len(out['steps'])}")
            for idx, s in enumerate(out["steps"], 1):
                print(f"      Step {idx}: {s['latex']:<25} | Why: {s['why']}")
        else:
            print(f"4. Solved without steps: {rev['result']['result']}")
    print()
