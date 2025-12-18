import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
IMAGE_DIR = "data/coco_pose/images"
POSE_DIR = "data/coco_pose/poses"
VAE_SAMPLE_DIR = "coco_pose/vae_autoencoder_samples"
OUTPUT_DIR = "output"

NUM_SAMPLES = 4
FIGSIZE = (8, 12)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================
# 1️⃣ ORIGINAL IMAGE vs CONTROL SIGNAL (POSE)
# =================================================

image_files = sorted(os.listdir(IMAGE_DIR))
sample_files = random.sample(image_files, NUM_SAMPLES)

fig, axes = plt.subplots(NUM_SAMPLES, 2, figsize=FIGSIZE)

for i, fname in enumerate(sample_files):
    img_path = os.path.join(IMAGE_DIR, fname)
    pose_path = os.path.join(POSE_DIR, fname.replace(".jpg", ".png"))

    img = Image.open(img_path).convert("RGB")
    pose = Image.open(pose_path).convert("RGB")

    axes[i, 0].imshow(img)
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(pose)
    axes[i, 1].set_title("Control Signal (Pose)")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "original_vs_control_4x2.png"), dpi=200)
plt.close()

print("Saved: output/original_vs_control_4x2.png")

# =================================================
# 2️⃣ VAE AUTOENCODER SAMPLES (4x2 GRID)
# =================================================

vae_files = sorted(os.listdir(VAE_SAMPLE_DIR))
vae_samples = random.sample(vae_files, NUM_SAMPLES)

fig, axes = plt.subplots(NUM_SAMPLES, 2, figsize=FIGSIZE)

for i, fname in enumerate(vae_samples):
    img = Image.open(os.path.join(VAE_SAMPLE_DIR, fname)).convert("RGB")

    w, h = img.size
    top = img.crop((0, 0, w, h // 2))       # original
    bottom = img.crop((0, h // 2, w, h))    # reconstruction

    axes[i, 0].imshow(top)
    axes[i, 0].set_title("Input")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(bottom)
    axes[i, 1].set_title("Reconstruction")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_autoencoder_4x2.png"), dpi=200)
plt.close()

print("Saved: output/vae_autoencoder_4x2.png")
