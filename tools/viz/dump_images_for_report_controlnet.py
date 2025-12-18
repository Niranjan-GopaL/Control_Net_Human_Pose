import os
from PIL import Image

BASE_DIR = "coco_pose"
SAMPLES_DIR = os.path.join(BASE_DIR, "samples_controlnet")
OUT_DIR = "output"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# Helper
# -------------------------------
def load(img_name):
    return Image.open(img_name).convert("RGB")

def vstack(images):
    w = max(img.width for img in images)
    h = sum(img.height for img in images)
    out = Image.new("RGB", (w, h))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height
    return out

# ===============================
# 1. Hint + Final Image
# ===============================
hint = load(os.path.join(BASE_DIR, "hint.png"))
final_img = load(os.path.join(SAMPLES_DIR, "x0_0.png"))

img1 = vstack([hint, final_img])
img1.save(os.path.join(OUT_DIR, "hint_and_final.png"))

print("Saved:", os.path.join(OUT_DIR, "hint_and_final.png"))

# ===============================
# 2. Denoising Progress
# ===============================
steps = [999, 700, 400, 100]
imgs = [
    load(os.path.join(SAMPLES_DIR, f"x0_{i}.png"))
    for i in steps
]

img2 = vstack(imgs)
img2.save(os.path.join(OUT_DIR, "denoising_progress.png"))

print("Saved:", os.path.join(OUT_DIR, "denoising_progress.png"))
