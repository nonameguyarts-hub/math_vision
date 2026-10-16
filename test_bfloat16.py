import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".hf_cache"))

import sys
import time
import psutil
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import recognizer
from recognizer import prep, normalize, PROMPT, MODEL_ID

PHOTOS = [
    ("WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg", "Photo 1 (2x+3y=74)"),
    ("WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg", "Photo 2 (5x+50=160)")
]

def test_dtype(target_dtype, dtype_name):
    print(f"\n==================== Testing {dtype_name} on CPU (max_side=512) ====================")
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024 * 1024)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=target_dtype).to("cpu").eval()
    load_time = time.time() - t0

    mem_after_load = proc.memory_info().rss / (1024 * 1024)
    print(f"Loaded in {load_time:.2f}s | Process RAM after load: {mem_after_load:.1f} MB (Delta: {mem_after_load - mem_before:.1f} MB)")

    # Warmup with small dummy or photo 1
    p1_img = Image.open(PHOTOS[0][0])
    crop1 = prep(p1_img, rotate=0, max_side=512)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs1 = processor(text=[text], images=[crop1], return_tensors="pt").to("cpu")
    with torch.no_grad():
        _ = model.generate(**inputs1, max_new_tokens=64, do_sample=False)

    peak_ram = mem_after_load
    for path, label in PHOTOS:
        img = Image.open(path)
        crop = prep(img, rotate=0, max_side=512)
        inputs = processor(text=[text], images=[crop], return_tensors="pt").to("cpu")
        
        t_start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen_time = time.time() - t_start

        current_ram = proc.memory_info().rss / (1024 * 1024)
        peak_ram = max(peak_ram, current_ram)

        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(gen, skip_special_tokens=True)[0]
        norm = normalize(raw)
        print(f"[{label}] Time: {gen_time:.2f}s | Output: {repr(norm)}")

    print(f"Peak Working Set (RAM): {peak_ram:.1f} MB ({peak_ram/1024:.2f} GB)")
    return peak_ram, load_time

if __name__ == "__main__":
    dtype_arg = sys.argv[1] if len(sys.argv) > 1 else "bfloat16"
    if dtype_arg == "bfloat16":
        test_dtype(torch.bfloat16, "torch.bfloat16")
    else:
        test_dtype(torch.float32, "torch.float32")
