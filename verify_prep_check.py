import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import recognizer
from recognizer import prep, recognize

def make_side_by_side(orig: Image.Image, crop: Image.Image, box, out_path: str):
    # Draw green rectangle on a copy of original
    marked = orig.copy().convert("RGB")
    draw = ImageDraw.Draw(marked)
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=max(2, int(0.005 * max(orig.size))))

    # Create side-by-side composite
    target_h = 500
    w_marked = int(marked.width * (target_h / marked.height))
    marked_resized = marked.resize((w_marked, target_h), Image.Resampling.LANCZOS)

    w_crop = int(crop.width * (target_h / crop.height))
    crop_resized = crop.resize((w_crop, target_h), Image.Resampling.LANCZOS)

    sep_w = 20
    comp_w = w_marked + sep_w + w_crop
    comp = Image.new("RGB", (comp_w, target_h), (255, 255, 255))
    comp.paste(marked_resized, (0, 0))
    comp.paste(crop_resized, (w_marked + sep_w, 0))
    comp.save(out_path)

def compute_ink_recall(orig: Image.Image, box):
    # Robust ink mask from background subtraction
    gray = cv2.cvtColor(np.array(orig.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    k = max(21, (min(h, w) // 15) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    diff = cv2.subtract(bg, gray)
    _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    tot_ink = bw.sum() / 255.0
    if tot_ink == 0:
        return 1.0
    x0, y0, x1, y1 = box
    crop_ink = bw[y0:y1, x0:x1].sum() / 255.0
    return crop_ink / tot_ink

def main():
    os.makedirs("prep_check", exist_ok=True)

    items = [
        ("WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg", "WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg"),
        ("WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg", "WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg"),
    ]

    # Create synthetic failing photo of "50 + 40n = 190y"
    failing_path = "failing_50_40n_190y.png"
    w, h = 1400, 350
    im = Image.new("RGB", (w, h), (242, 240, 235))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    draw.text((120, 140), "50 + 40n = 19", fill=(30, 30, 35), font=font)
    draw.text((510, 140), "0y", fill=(32, 32, 38), font=font)
    im.save(failing_path)
    items.append(("failing_50_40n_190y.png", failing_path))

    for label, path in items:
        img = Image.open(path)
        crop = prep(img)
        x0, y0, x1, y1 = crop.crop_box
        recall = compute_ink_recall(img, crop.crop_box)
        side_path = os.path.join("prep_check", f"{os.path.splitext(os.path.basename(path))[0]}_side_by_side.png")
        make_side_by_side(img, crop, crop.crop_box, side_path)

        print("=" * 60)
        print(f"PHOTO: {label}")
        print(f"  Original size:     {img.size}")
        print(f"  Ink bbox:          {crop.ink_box}")
        print(f"  Crop bbox:         {crop.crop_box}")
        print(f"  Final crop size:   {crop.size}")
        print(f"  Ink recall:        {recall * 100:.2f}%")
        print(f"  Warning:           {crop.warning}")
        print(f"  Side-by-side saved:{side_path}")

        # Run recognition
        pred = recognize(img, crop=crop)
        print(f"  Recognized LaTeX:  {pred}")

if __name__ == "__main__":
    main()
