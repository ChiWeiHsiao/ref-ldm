import itertools
from pathlib import Path
from shutil import copy
from glob import glob

import argparse
from omegaconf import OmegaConf
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import torch
from typing import Tuple

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.ir import ImageRestorationDataset, unnormalize_image


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Select testing dataset
data_tag = 'celebatest'  # 'ffhqtest_severe', 'ffhqtest_moderate', 'celebatest'

# Set model path
model_tag = 'refldm'
model_ckpt = 'ckpts/refldm-bfr/model.ckpt'
model_config = 'ckpts/refldm-bfr/config.yaml'
config = OmegaConf.load(model_config)

# Set output directory
out_dir = Path(f'results/{data_tag}/{model_tag}')
out_dir.mkdir(parents=True, exist_ok=True)

# Set inference params
save_data = False
max_num_refs = 5
ddim_schedule = 'uniform' # 'uniform', 'uniform_trailing'
ddim_step_lst = [100]
cfg_key = 'ref_image'
cfg_scale_lst = [1.5]
seed = 2024

# Instantiate dataset
data_tag_to_data_path_dict = {
    'ffhqtest_severe': {
        'file_list': 'data/ffhq/file_list/test_references_arcface01-04.csv',
        'lq_dir': 'data/ffhq/ffhq_test_LQ_images/severe_b8-16_r8-32_n0-20_j30-100',
        'gt_dir': 'data/ffhq/images512x512',
        'ref_dir': 'data/ffhq/images512x512',
    },
    'ffhqtest_moderate': {
        'file_list': 'data/ffhq/file_list/test_references_arcface01-04.csv',
        'lq_dir': 'data/ffhq/ffhq_test_LQ_images/moderate_b0-8_r1-8_n0-15_j60-100',
        'gt_dir': 'data/ffhq/images512x512',
        'ref_dir': 'data/ffhq/images512x512',
    },
    'celebatest': {
        'file_list': 'data/celebahq/file_list/test_references_arcface01-04.csv',
        'lq_dir': 'data/celebahq/img_test_LQ',
        'gt_dir': 'data/celebahq/img_test_HQ',
        'ref_dir': 'data/celebahq/img_all',
    },
}
data_path_dict = data_tag_to_data_path_dict[data_tag]
use_given_ref = config.data.params.train.params.get('use_given_ref', False)
dataset = ImageRestorationDataset(
    dup_to_max_num_refs=False,
    max_num_refs=max_num_refs,
    use_given_lq=True,
    use_given_ref=use_given_ref,
    file_list=data_path_dict['file_list'],
    lq_dir=data_path_dict['lq_dir'],
    gt_dir=data_path_dict['gt_dir'],
    ref_dir=data_path_dict['ref_dir'],
    image_size=(512, 512),
)

# Load model
# TODO ddpm.py del lpips, if config use
for k in ['ckpt_path', 'perceptual_loss_config']:
    config.model.params.pop(k, None)
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load(model_ckpt)['state_dict'], strict=False)
# del model.lpips_loss  # only needed in training  # TODO del
model.to(device)
model.eval()

sampler = DDIMSampler(model, print_tqdm=True, schedule=ddim_schedule)

# Sample shared xT
torch.manual_seed(seed)
latent_shape = [model.model.diffusion_model.out_channels, model.image_size, model.image_size]
xT = torch.randn([1, *latent_shape], device=device)

# Inference
with torch.no_grad():
    for i in range(len(dataset)):
        paths = dataset.df.iloc[i]
        gt_path = paths['gt_image']
        lq_path = paths['lq_image']
        ref_paths = paths['ref_image'][:max_num_refs] if use_given_ref else []
        name = Path(lq_path).stem

        # Copy input data to output dir
        if save_data:
            copy(gt_path, out_dir / f'{name}_data_gt.png')
            copy(lq_path, out_dir / f'{name}_data_lq.png')
            if len(ref_paths) > 0:
                for ref_path in ref_paths:
                    refid = Path(ref_path).stem
                    copy(ref_path, out_dir / f'{name}_data_ref={refid}.png')

        # Make torch batch data
        data_dict = dataset.__getitem__(i)
        batch = {}
        for k in data_dict.keys():
            batch[k] = torch.from_numpy(data_dict[k][None]).to(device)

        # Inference model
        # get encoded condition
        _, c = model.get_input(batch, model.first_stage_key)
        c = model.get_learned_conditioning(c)

        for ddim_step, cfg_scale in itertools.product(ddim_step_lst, cfg_scale_lst):
            str_trailing = '_trailing' if ddim_schedule == 'uniform_trailing' else ''
            sub_out_dir = out_dir / f'step={ddim_step}{str_trailing}_cfg={cfg_scale}'
            sub_out_dir.mkdir(parents=True, exist_ok=True)

            # create uc from c for classifier-free guidance
            if cfg_scale == 1.0:
                uc = None
            else:
                uc = {k: c[k].detach().clone() for k in c.keys()}
                uc[cfg_key] *= 0

            out_path = sub_out_dir / f'{name}.png'
            if out_path.is_file():
                print(f'Skip existing: {out_path}')
                continue
            else:
                print(f'Generating: {out_path}')

            # run with ema model
            with model.ema_scope():
                samples_ddim, _ = sampler.sample(S=ddim_step,
                                                 conditioning=c,
                                                 unconditional_guidance_scale=cfg_scale,
                                                 unconditional_conditioning=uc,
                                                 batch_size=1,
                                                 shape=latent_shape,
                                                 x_T=xT,
                                                 eta=0,
                                                 verbose=False)
                out_image = model.decode_first_stage(samples_ddim)
                out_image = ((out_image + 1.0) / 2.0).clamp(0.0, 1.0)
                out_image = out_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                out_image = Image.fromarray(out_image.astype(np.uint8))
                out_image.save(str(out_path))