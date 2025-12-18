import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# Paths
# -----------------------------
img1_path = "uncond_ldm_out.png"
img2_path = "x0_0.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load images
# -----------------------------
to_tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img1 = to_tensor(Image.open(img1_path).convert("RGB"))
img2 = to_tensor(Image.open(img2_path).convert("RGB"))

# -----------------------------
# SSIM (expects numpy, [0,1])
# -----------------------------
img1_np = img1.permute(1, 2, 0).numpy()
img2_np = img2.permute(1, 2, 0).numpy()

ssim_val = ssim(
    img1_np,
    img2_np,
    channel_axis=2,
    data_range=1.0
)

# -----------------------------
# LPIPS (expects [-1,1])
# -----------------------------
lpips_model = lpips.LPIPS(net="alex").to(device)

img1_lp = (img1 * 2 - 1).unsqueeze(0).to(device)
img2_lp = (img2 * 2 - 1).unsqueeze(0).to(device)

lpips_val = lpips_model(img1_lp, img2_lp).item()

# -----------------------------
# Print results
# -----------------------------
print(f"SSIM  : {ssim_val:.4f}  (higher is better)")
print(f"LPIPS : {lpips_val:.4f}  (lower is better)")
