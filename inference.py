import argparse

from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    # Image paths
    parser.add_argument('--output_path', type=str, default='result.png')
    parser.add_argument('--lq_path', type=str, default='assets/demo/lq.png')
    parser.add_argument('--ref_paths', nargs='+', default=['assets/demo/ref0.png', 'assets/demo/ref1.png', 'assets/demo/ref2.png', 'assets/demo/ref3.png'])
    # Model paths
    parser.add_argument('--model_config_path', type=str, default='configs/refldm.yaml')
    parser.add_argument('--model_ckpt_path', type=str, default='ckpts/refldm.ckpt')
    parser.add_argument('--vae_ckpt_path', type=str, default='ckpts/vqgan.ckpt')
    # Inference settings
    parser.add_argument('--ddim_step', type=int, default=50)
    parser.add_argument('--ddim_schedule', type=str, default='uniform_trailing')
    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    return args


def read_and_normalize_image(image_path, resize=(512, 512)):
    image_path = str(image_path)
    image = Image.open(image_path)
    image = image.convert("RGB")
    if image.size != resize:
        image = image.resize(resize)
    image = pil_to_tensor(image)
    image = image / 127.5 - 1.0  # [0, 255] to [-1, 1]
    return image


def main():
    args = parse_args()

    # Load model
    model_config = OmegaConf.load(args.model_config_path)
    model_config.model.params.first_stage_config.params.ckpt_path = args.vae_ckpt_path
    for k in ['ckpt_path', 'perceptual_loss_config']:
        model_config.model.params.pop(k, None)
    model = instantiate_from_config(model_config.model)
    model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    sampler = DDIMSampler(model, print_tqdm=True, schedule=args.ddim_schedule)
    
    # Sample initial latent xT
    latent_shape = [model.model.diffusion_model.out_channels, model.image_size, model.image_size]
    torch.manual_seed(args.seed)
    xT = torch.randn([1, *latent_shape], device=device)

    # Inference
    torch.set_grad_enabled(False)
    # prepare condition
    lq_image = read_and_normalize_image(args.lq_path)
    ref_image = torch.concat([read_and_normalize_image(p) for p in args.ref_paths], axis=-1)
    c = {
        'lq_image': lq_image.unsqueeze(0).to(device),
        'ref_image': ref_image.unsqueeze(0).to(device),
    }
    # encode condition from image to latent
    c = model.get_learned_conditioning(c)
    # CFG null condition = no reference
    uc = None
    if args.cfg_scale != 1.0:
        uc = {k: c[k].detach().clone() for k in c.keys()}
        uc['ref_image'] *= 0
    # diffusion denoising
    with model.ema_scope():
        output_latent, _ = sampler.sample(
            S=args.ddim_step,
            unconditional_guidance_scale=args.cfg_scale,
            conditioning=c,
            unconditional_conditioning=uc,
            shape=latent_shape,
            x_T=xT,
            batch_size=1,
            verbose=False,
        )
    # decode output latent to image
    output_image = model.decode_first_stage(output_latent)
    # save result    
    output_image = ((output_image + 1.0) / 2.0).clamp(0.0, 1.0)  # [-1, 1] to [0, 1]
    output_image = output_image.squeeze(0).cpu()
    to_pil_image(output_image).save(args.output_path)


if __name__ == '__main__':
    main()