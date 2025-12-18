import torch
import torchvision
import argparse
import yaml
import os
import random
from torchvision.utils import make_grid
from tqdm import tqdm
from models.controlnet_ldm import ControlNet
from dataset.celeb_dataset import CelebDataset
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

from dataset.coco_pose_dataset import CocoPoseDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, scheduler, train_config, diffusion_model_config,
        autoencoder_model_config, diffusion_config, dataset_config, vae, dataset):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((train_config['num_samples'],
                    autoencoder_model_config['z_channels'],
                    im_size,
                    im_size)).to(device)

    # Get random hints for the desired number of samples
    hint_cache_path = os.path.join(train_config['task_name'], "fixed_hints.pt")
    if os.path.exists(hint_cache_path):
        print("Loading fixed hints")
        hints = torch.load(hint_cache_path, map_location=device)
    else:
        print("Sampling and saving fixed hints")
        hints = []
        rng = random.Random(42)  # deterministic
        for _ in range(train_config['num_samples']):
            hint_idx = rng.randint(0, len(dataset) - 1)
            hints.append(dataset[hint_idx][1].unsqueeze(0))
        hints = torch.cat(hints, dim=0)
        torch.save(hints, hint_cache_path)
    hints = hints.to(device)


    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), hints, scale = 100.0)

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.to(device).decode(xt)
        else:
            ims = xt

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        if not os.path.exists(os.path.join(train_config['task_name'], 'samples_controlnet')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples_controlnet'))
        img.save(os.path.join(train_config['task_name'], 'samples_controlnet', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
        ldm_scheduler=True
    )

    coco_pose = CocoPoseDataset(
        im_path=dataset_config['im_path'],
        pose_path=dataset_config['pose_path'],
        im_size=dataset_config['im_size'],
        use_latents=True,
        latent_path=os.path.join(
            train_config['task_name'],
            train_config['vae_latent_dir_name']
        ),
        return_hint=True
    )


    latent_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    downscale_factor = dataset_config['im_size'] // latent_size


    model = ControlNet(im_channels=autoencoder_model_config['z_channels'],
                    model_config=diffusion_model_config,
                    model_locked=True,
                    model_ckpt=os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']),
                    device=device,
                    down_sample_factor=downscale_factor).to(device)
    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                    train_config['controlnet_ckpt_name'])), "Train ControlNet first"
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                train_config['controlnet_ckpt_name']),
                                    map_location=device))
    print('Loaded controlnet checkpoint')

    vae = VAE(im_channels=dataset_config['im_channels'],
            model_config=autoencoder_model_config)
    vae.eval()

    # Load vae if found
    assert os.path.exists(os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])), \
        "VAE checkpoint not present. Train VAE first."
    vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                train_config['vae_autoencoder_ckpt_name']),
                                map_location=device), strict=True)
    print('Loaded vae checkpoint')

    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
            autoencoder_model_config, diffusion_config, dataset_config, vae, coco_pose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ldm controlnet generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq.yaml', type=str)
    args = parser.parse_args()
    infer(args)
