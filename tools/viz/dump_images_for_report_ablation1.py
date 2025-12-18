from PIL import Image
import os

# Paths
image_paths = [
    "output/hint.png",
    "output/scale_0p5.png",
    "output/scale_1p0.png",
    "output/scale_1p5.png",
    "output/scale_2p0.png",
]

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "controlnet_scale_ablation.png")

# Load images
images = [Image.open(p).convert("RGB") for p in image_paths]

# Ensure same width
width = images[0].width
images = [img.resize((width, img.height)) for img in images]

# Compute total height
total_height = sum(img.height for img in images)

# Create canvas
stacked = Image.new("RGB", (width, total_height))

# Paste images
y_offset = 0
for img in images:
    stacked.paste(img, (0, y_offset))
    y_offset += img.height

# Save
stacked.save(output_path)

print(f"Saved stacked image → {output_path}")
