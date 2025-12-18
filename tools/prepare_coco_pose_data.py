# tools/prepare_coco_pose_data.py
import pickle, os, cv2

with open("coco_pose_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

os.makedirs("data/coco_pose/images", exist_ok=True)
os.makedirs("data/coco_pose/poses", exist_ok=True)

for i, sample in enumerate(dataset):
    img_id = str(sample["image_id"]).zfill(6)

    img = cv2.cvtColor(sample["img"], cv2.COLOR_RGB2BGR)
    pose = cv2.cvtColor(sample["skeleton"], cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"data/coco_pose/images/{img_id}.jpg", img)
    cv2.imwrite(f"data/coco_pose/poses/{img_id}.png", pose)
