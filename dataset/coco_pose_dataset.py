# datasets/coco_pose_dataset.py
import os, glob, cv2
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset
from utils.diffusion_utils import load_latents

class CocoPoseDataset(Dataset):
    def __init__(
        self,
        im_path,
        pose_path,
        im_size=256,
        use_latents=False,
        latent_path=None,
        return_hint=False,
    ):
        self.images = sorted(glob.glob(os.path.join(im_path, "*.jpg")))
        self.poses = sorted(glob.glob(os.path.join(pose_path, "*.png")))
        assert len(self.images) == len(self.poses)

        self.im_size = im_size
        self.use_latents = use_latents
        self.return_hints = return_hint

        self.latent_maps = None
        if use_latents:
            self.latent_maps = load_latents(latent_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        pose_path = self.poses[idx]

        if self.use_latents:
            key = img_path.replace("\\", "/")
            latent = self.latent_maps[key]

            if self.return_hints:
                pose = Image.open(pose_path)
                pose = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(self.im_size),
                    torchvision.transforms.CenterCrop(self.im_size),
                    torchvision.transforms.ToTensor()
                ])(pose)
                pose = (2 * pose) - 1
                return latent, pose

            return latent

        im = Image.open(img_path)
        im = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor()
        ])(im)
        im = (2 * im) - 1

        if self.return_hints:
            pose = Image.open(pose_path)
            pose = torchvision.transforms.ToTensor()(pose)
            pose = (2 * pose) - 1
            return im, pose

        return im

