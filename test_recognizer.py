"""test_recognizer.py - standalone verification of the new Uni-MuMER recognizer.
Tests cold-load, first/second inference times, raw vs normalized outputs,
saves prep() crops to debug_crops/, and checks speed scaling with max_side.
"""
import os
import sys
import time
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recognizer
from recognizer import _load, prep, normalize, PROMPT

PHOTOS = [
    {
        "name": "photo1 (2.14.59 PM)",
        "path": "test_photos/WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg",
        "expected": "2x+3y=74"
    },
    {
        "name": "photo2 (2.16.35 PM)",
        "path": "test_photos/WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg",
        "expected": "5x+50=170 (or 160)"
    }
]

def run_inference(processor, model, device, crop, max_new_tokens=512):
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[crop], return_tensors="pt").to(device)
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return raw, normalize(raw)

def main():
    crop_dir = Path("debug_crops")
    crop_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 3: TESTING UNI-MUMER RECOGNIZER")
    print("=" * 60)

    # 1. Cold load
    print("\n--- 1. Cold Load Model ---")
    t0 = time.time()
    processor, model, device = _load()
    cold_load_time = time.time() - t0
    print(f"Device: {device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print(f"Cold-load time: {cold_load_time:.2f}s")

    # 2. Test each photo
    slowest_time = 0.0
    for idx, p in enumerate(PHOTOS, 1):
        print(f"\n--- Photo {idx}: {p['name']} ---")
        img_path = Path(p["path"])
        if not img_path.exists():
            print(f"Error: {img_path} not found!")
            continue

        raw_img = Image.open(img_path)
        print(f"Original image size: {raw_img.size}")

        # Save prep() crop
        crop = prep(raw_img, rotate=0, max_side=1024)
        crop_path = crop_dir / f"crop_{idx}_{img_path.stem[:20]}.png"
        crop.save(crop_path)
        print(f"Prep crop saved to: {crop_path.resolve()} (size: {crop.size})")

        # First inference (warmup for this image)
        t_start1 = time.time()
        raw1, norm1 = run_inference(processor, model, device, crop)
        t_infer1 = time.time() - t_start1

        # Second inference (steady state)
        t_start2 = time.time()
        raw2, norm2 = run_inference(processor, model, device, crop)
        t_infer2 = time.time() - t_start2

        slowest_time = max(slowest_time, t_infer1, t_infer2)

        print(f"Expected:           {p['expected']}")
        print(f"Raw model output:   {repr(raw1)}")
        print(f"Normalized output:  {repr(norm1)}")
        print(f"First inference:    {t_infer1:.2f}s")
        print(f"Second inference:   {t_infer2:.2f}s")

    # 3. CPU Speed Check: If > 20s, test lower max_side (640, 512)
    print("\n" + "=" * 60)
    print(f"Inference speed check (max observed: {slowest_time:.2f}s)")
    if slowest_time > 20.0:
        print("Inference took > 20s. Benchmarking max_side = 640 and 512...")
        for ms in [640, 512]:
            print(f"\n--- Benchmarking max_side = {ms} ---")
            for idx, p in enumerate(PHOTOS, 1):
                img_path = Path(p["path"])
                raw_img = Image.open(img_path)
                crop_res = prep(raw_img, rotate=0, max_side=ms)
                t_bench = time.time()
                r, n = run_inference(processor, model, device, crop_res)
                dt = time.time() - t_bench
                print(f"Photo {idx} (max_side={ms}, crop size {crop_res.size}): {dt:.2f}s -> {repr(n)}")
    else:
        print("Inference is under 20s threshold.")
    print("=" * 60)

if __name__ == "__main__":
    main()
