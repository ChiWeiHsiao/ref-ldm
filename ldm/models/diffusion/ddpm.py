"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
from collections import defaultdict
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
# import pytorch_ssim
# import lpips
# from pytorch_memlab import profile, profile_every

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like, make_ddim_timesteps
from ldm.models.diffusion.ddim import DDIMSampler
from ldm import cache_kv


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False, ckpt_extra_w_scale=0.):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]

        # if input_channels is different from that of pretrained model
        for in_weight_key in [
            'model.diffusion_model.input_blocks.0.0.weight',
            'model_ema.diffusion_modelinput_blocks00weight',
        ]:
            ckpt_in_weight = sd[in_weight_key]
            ckpt_in_chs = ckpt_in_weight.shape[1]
            new_in_chs = self.model.diffusion_model.in_channels
            if new_in_chs > ckpt_in_chs:
                print('Input channels: current config > pretrained ckpt')
                print(f'-> init weights of extra channels = kaiming_normal * {ckpt_extra_w_scale}')
                pad_in_chs = new_in_chs - ckpt_in_chs
                (b, c, *d) = ckpt_in_weight.shape
                pad_in_weight = torch.zeros([b, pad_in_chs, *d])
                torch.nn.init.kaiming_normal_(pad_in_weight, nonlinearity='relu')
                pad_in_weight *= ckpt_extra_w_scale
                sd[in_weight_key] = torch.cat([ckpt_in_weight, pad_in_weight], axis=1)

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 use_cache_kv=False,
                 perceptual_loss_scale=0.0,
                 perceptual_loss_config=None,
                 perceptual_loss_weight_by_t=True,
                 val_loss_avg_num_timesteps=0,
                 val_loss_run_ddim_steps=200,
                 val_loss_run_ddim_cfg_scale=1.,
                 val_loss_run_ddim_cfg_key=None,
                 ckpt_extra_w_scale=0.,
                 sample_t_by_weight_exp=None,
                 diff_lr_keys=None,
                 diff_lr_scale=10,
                 *args, **kwargs):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(*args, **kwargs)

        self.first_stage_model = self.instantiate_first_stage_model(first_stage_config)
        if isinstance(cond_stage_config, list):
            self.cond_stage_key = []  # ex. ['LR_image', ]
            self.cond_stage_model = {}  # ex. {'LR_image': model_obj, }
            self.conditioning_key = {}  # ex. {'LR_image': 'concat', }
            self.uncond_prob = {}
            for cfg in cond_stage_config:
                key = cfg['cond_stage_key']
                cond_model_config = cfg['cond_model_config']
                self.cond_stage_key.append(key)
                self.cond_stage_model[key] = self.instantiate_cond_stage_model(cond_model_config)
                self.conditioning_key[key] = cfg.get('conditioning_key', 'concat')
                self.uncond_prob[key] = cfg.get('uncond_prob', 0.0)
            self.cond_stage_model = nn.ModuleDict(self.cond_stage_model)
        else:
            # for backwards compatibility after implementation of DiffusionWrapper
            if conditioning_key is None:
                conditioning_key = 'concat' if concat_mode else 'crossattn'
            if cond_stage_config == '__is_unconditional__':
                conditioning_key = None
            if isinstance(cond_stage_config, dict):
                cond_stage_config['freeze'] = not cond_stage_trainable
            self.cond_stage_key = cond_stage_key
            self.cond_stage_model = self.instantiate_cond_stage_model(cond_stage_config)
            self.conditioning_key = conditioning_key
            self.uncond_prob = cond_stage_config.get('uncond_prob', 0.0) if cond_stage_config != '__is_unconditional__' else 0.0

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.scale_by_std = scale_by_std
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, ckpt_extra_w_scale=ckpt_extra_w_scale)
            self.restarted_from_ckpt = True

        self.use_cache_kv = use_cache_kv
        if perceptual_loss_config is not None:
            self.perceptual_loss_scale = perceptual_loss_scale
            self.perceptual_loss_weight_by_t = perceptual_loss_weight_by_t
            self.perceptual_loss = instantiate_from_config(perceptual_loss_config)
            self.freeze_model(self.perceptual_loss)

        # self.lpips_loss = lpips.LPIPS(net='vgg', model_path='pretrained/lpips_vgg.pth')
        # self.freeze_model(self.lpips_loss)

        if val_loss_avg_num_timesteps > 0:
            self.val_loss_avg_timesteps = make_ddim_timesteps(
                ddim_discr_method='uniform',
                num_ddim_timesteps=val_loss_avg_num_timesteps,
                num_ddpm_timesteps=self.num_timesteps,
            )
        else:
            self.val_loss_avg_timesteps = []
        self.val_loss_run_ddim_steps = val_loss_run_ddim_steps
        self.val_loss_run_ddim_cfg_scale = val_loss_run_ddim_cfg_scale
        self.val_loss_run_ddim_cfg_key = val_loss_run_ddim_cfg_key
        self.sample_t_by_weight_exp = sample_t_by_weight_exp
        self.diff_lr_keys = diff_lr_keys
        self.diff_lr_scale = diff_lr_scale

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def freeze_model(self, model):
        model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False

    def instantiate_first_stage_model(self, config):
        model = instantiate_from_config(config)
        self.freeze_model(model)
        return model

    def instantiate_cond_stage_model(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            model = self.first_stage_model
        elif config == "__is_unconditional__":
            print(f"Training {self.__class__.__name__} as an unconditional model.")
            model = None
        else:
            model = instantiate_from_config(config)
            freeze = config.get('freeze', False)
            if freeze:
                self.freeze_model(model)
        return model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def apply_condition_model(self, cond: torch.Tensor, model: nn.Module):
        if self.cond_stage_forward is None:
            if hasattr(model, 'encode') and callable(model.encode):
                cond = model.encode(cond)
                if isinstance(cond, DiagonalGaussianDistribution):
                    cond = cond.mode()
            else:
                cond = model(cond)
        else:
            assert hasattr(model, self.cond_stage_forward)
            cond = getattr(model, self.cond_stage_forward)(cond)
        return cond

    def get_learned_conditioning(self, conds: Union[Dict, torch.Tensor]):
        if isinstance(conds, dict):
            encoded_conds = {}
            for k, c in conds.items():
                model = self.cond_stage_model[k]
                encoded_conds[k] = self.apply_condition_model(c, model)
            return encoded_conds
        else:
            c = conds
            model = self.cond_stage_model
            return self.apply_condition_model(c, model)

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, cond_key=None, bs=None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        # z will be HR // 8 since it's 8x SR

        if self.cond_stage_key is not None:
            cond_key = self.cond_stage_key if cond_key is None else cond_key
            if cond_key == self.first_stage_key:
                c = x
            elif cond_key in ['caption', 'coordinates_bbox']:
                c = batch[cond_key]
            elif cond_key == 'class_label':
                c = batch
            elif isinstance(cond_key, list):
                c = {}
                for ck in cond_key:
                    c[ck] = super().get_input(batch, ck).to(self.device)
            else:
                c = super().get_input(batch, cond_key).to(self.device)

            if bs is not None:
                if isinstance(c, dict):
                    c = {k: v[:bs] for k, v in c.items()}
                else:
                    c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])

        return out

    def _decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        return self._decode_first_stage(z, predict_cids, force_not_quantize)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c, img, img_rec = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True)
        loss_weight_map = batch.get('loss_weight_map', None)
        loss = self(x, c, img_rec, loss_weight_map)
        return loss

    def set_uncondition(self, conds: Union[Dict, torch.Tensor]):
        if isinstance(conds, dict):
            for k in conds.keys():
                s = torch.rand(conds[k].shape[0]) > self.uncond_prob[k]
                s = s.reshape([-1] + [1] * (conds[k].ndim - 1))
                conds[k] *= s.to(self.device)
        else:
            s = torch.rand(conds.shape[0]) > self.uncond_prob
            s = s.reshape([-1] + [1] * (conds.ndim - 1))
            conds *= s.to(self.device)
        return conds

    def forward(self, x, c, img, loss_weight_map, *args, **kwargs):
        '''
        Args:
            c: raw condition(s) before condition model, can be a single tensor or
                a dict of tensors, ex. {'LR_image': tensor, ...}
        '''
        if c is not None:
            c = self.get_learned_conditioning(c)
        if self.training:
            c = self.set_uncondition(c)
            if self.sample_t_by_weight_exp is None:
                t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
            else:
                # for faster finetune from pretrained, sample more larger t
                t_cdf = torch.arange(self.num_timesteps)
                #  t_cdf = t_cdf ** self.sample_t_by_weight_exp
                t_cdf = t_cdf ** (self.sample_t_by_weight_exp / (self.global_step + 1))
                t_cdf = t_cdf / t_cdf.sum()
                t_cdf = t_cdf.cumsum(dim=0)
                t = torch.rand((x.shape[0],))
                t = torch.searchsorted(t_cdf, t).to(self.device)
            loss, loss_dict = self.p_losses(x, c, t, img, loss_weight_map, *args, **kwargs)
        else:
            # validation loss
            loss_dict = {}
            loss = 0
            # average loss over multiple timesteps
            if len(self.val_loss_avg_timesteps) > 0:
                loss_lst, loss_dict_lst = [], []
                for t in self.val_loss_avg_timesteps:
                    t = torch.full((x.shape[0],), t, device=self.device).long()
                    t_loss, t_loss_dict = self.p_losses(x, c, t, img, loss_weight_map, *args, **kwargs)
                    loss_lst.append(t_loss)
                    loss_dict_lst.append(t_loss_dict)
                loss += sum(loss_lst) / len(loss_lst)
                for k in loss_dict_lst[0].keys():
                    loss_dict[k] = sum(d[k] for d in loss_dict_lst) / len(loss_dict_lst)
            # run DDIM inference and compute reconstruction loss
            if self.val_loss_run_ddim_steps is not None:
                x_ddim, _ = self.sample_log(
                    cond=c, batch_size=x.shape[0], ddim=True, ddim_steps=self.val_loss_run_ddim_steps,
                )
                img_ddim = self.decode_first_stage(x_ddim)
                loss_dict['val/ddim_latent_l1_loss'] = (x - x_ddim).abs().mean()
                # loss_dict['val/ddim_image_lpips'] = self.lpips_loss(img, img_ddim).mean()
                #  loss_dict['val/ddim_image_l1_loss'] = (img - img_ddim).abs().mean()
                #  loss_dict['val/ddim_image_ssim'] = pytorch_ssim.ssim(img, img_ddim, size_average=True)
                if hasattr(self, 'perceptual_loss'):
                    loss_dict['val/ddim_image_perceptual_loss'] = self.perceptual_loss(img_ddim, img).mean()
                if loss_weight_map is not None:
                    weighted_loss_map = (x - x_ddim).abs() * F.adaptive_max_pool2d(loss_weight_map.unsqueeze(1), x.shape[-2:])
                    loss_dict['val/ddim_latent_weighted_l1_loss'] = weighted_loss_map.mean()
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        '''
        Args:
            cond: condition(s) after condition model, can be a single tensor or a dict of tensors,
                    ex. {'LR_image': tensor, }
        '''
        # ex. {'c_concat': [cond_tensor1, cond_tensor2], ]}
        if cond is None:
            arg_to_conds = {}
        else:
            arg_to_conds = defaultdict(list)
            if isinstance(cond, dict):
                conditioning_keys = [self.conditioning_key[k] for k in self.cond_stage_key]
                cond_tensors = [cond[k] for k in self.cond_stage_key]
            else:
                conditioning_keys = [self.conditioning_key]
                cond_tensors = [cond]
            for conditioning_key, cond_tensor in zip(conditioning_keys, cond_tensors):
                arg = {
                    'concat': 'c_concat',
                    'crossattn': 'c_crossattn',
                    'spatial_concat': 'c_spatial_concat',
                    'cache_kv': 'c_cache_kv',
                }[conditioning_key]
                arg_to_conds[arg].append(cond_tensor)

        # x_recone shape is the same as x_noisy (i.e., LR size)
        # later should be passed through AE's decoder to up-sample
        # but decoder is unnecessary for training right? loss only depend on DM
        x_recon = self.model(x_noisy, t, **arg_to_conds)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, img, loss_weight_map=None, noise=None):
        # In SR3, x_start is HR image from training dataset
        # here x_start is the latent HR image, size is the same as LR image
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # from clean to noisy
        if self.use_cache_kv:
            cache_kv.mode = 'save'
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False)
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        if loss_weight_map is not None:
            loss_simple_weighted = loss_simple * F.adaptive_max_pool2d(loss_weight_map.unsqueeze(1), loss_simple.shape[-2:])
            loss_dict.update({f'{prefix}/loss_simple_weighted': loss_simple_weighted.mean()})
            loss_simple += loss_simple_weighted
        loss_simple = loss_simple.mean([1, 2, 3])

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        if hasattr(self, "perceptual_loss") and self.perceptual_loss_scale > 0:
            model_x_start = self.predict_start_from_noise(x_noisy, t, model_output)
            model_img_start = self._decode_first_stage(model_x_start)
            loss_perceptual = self.perceptual_loss(model_img_start, img)
            if self.perceptual_loss_weight_by_t is True:
                loss_perceptual *= self.sqrt_alphas_cumprod[t]
            # ablation timestep scaling factors for identity loss
            elif self.perceptual_loss_weight_by_t == 't<500':
                loss_perceptual *= (t < 500)
            elif self.perceptual_loss_weight_by_t == 't<100':
                loss_perceptual *= (t < 100)
            loss_perceptual = loss_perceptual.mean()
            loss += self.perceptual_loss_scale * loss_perceptual
            loss_dict.update({f'{prefix}/loss_perceptual': loss_perceptual})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps, cfg_scale=1., cfg_key=None, **kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            # classifier-free guidance
            cfg_scale = self.val_loss_run_ddim_cfg_scale
            cfg_key = self.val_loss_run_ddim_cfg_key
            if cfg_scale != 1.:
                ucond = {k: cond[k].detach().clone() for k in cond.keys()}
                ucond[cfg_key] *= 0
            else:
                ucond = None
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,
                                                        unconditional_guidance_scale=cfg_scale,
                                                        unconditional_conditioning=ucond,
                                                        **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    @torch.no_grad()
    def log_images(self, batch, N=2, n_row=2, sample=True, ddim_steps=200, ddim_eta=0., return_keys=None,
                   quantize_denoised=False, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, xc, x, xrec = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           bs=N)
        if xc is None:
            c = None
        else:
            c = self.get_learned_conditioning(xc)

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec

        #  if self.conditioning_key is not None:
            #  if hasattr(self.cond_stage_model, "decode"):
                #  xc = self.cond_stage_model.decode(c)
                #  log["conditioning"] = xc
            #  elif self.cond_stage_key in ["caption"]:
                #  xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                #  log["conditioning"] = xc
            #  elif self.cond_stage_key == 'class_label':
                #  xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                #  log['conditioning'] = xc
            #  elif isimage(xc):
                #  log["conditioning"] = xc
            #  if ismap(xc):
                #  log["original_conditioning"] = self.to_rgb(xc)

        if self.cond_stage_key is not None:
            if isinstance(self.cond_stage_key, list):
                for cond_key in self.cond_stage_key:
                    log[f"raw_condition_{cond_key}"] = xc[cond_key][:, :3]
            else:
                cond_key = self.cond_stage_key
                log[f"raw_condition_{cond_key}"] = xc[:, :3]

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        name_params = list(self.model.named_parameters())
        if self.cond_stage_model is not None:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if isinstance(self.cond_stage_model, nn.ModuleDict):
                cond_models = self.cond_stage_model.values()
            else:
                cond_models = [self.cond_stage_model]
            for model in cond_models:
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        name_params.append((n, p))
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            name_params.append(('logvar', self.logvar))

        if self.diff_lr_keys is not None:
            #  use_diff_lr = lambda n: any([n for k in self.diff_lr_keys if n.startswith(k)])
            import re
            diff_lr_pattern = re.compile('|'.join(self.diff_lr_keys))
            use_diff_lr = lambda n: any([n for k in self.diff_lr_keys if diff_lr_pattern.fullmatch(n) is not None])
            diff_lr_names = [n for n, p in name_params if use_diff_lr(n)]
            print(f'{len(diff_lr_names)} parmas use diff_lr:\n{diff_lr_names}')

            diff_lr_params = [p for n, p in name_params if use_diff_lr(n)]
            orig_lr_params = [p for n, p in name_params if not use_diff_lr(n)]
            grouped_params = [
                {'params': diff_lr_params, 'lr': lr * self.diff_lr_scale},
                {'params': orig_lr_params, 'lr': lr},
            ]
            opt = torch.optim.AdamW(grouped_params, lr=lr)
        else:
            params = [p for n, p in name_params]
            opt = torch.optim.AdamW(params, lr=lr)

        if hasattr(self, 'lr_lambda'):
            sch = {'scheduler': LambdaLR(opt, self.lr_lambda), 'interval': 'step'}
            return [opt], [sch]
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    '''Wrap U-net for self.model in LatentDiffusion

    Attributes:
        diffusion: U-net model object, instantiated from unet_config
    '''
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    #  @profile_every(1)  # TODO for gpu memory profiling
    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_spatial_concat: list = None, c_cache_kv: list = None):
        if c_concat is not None:
            xc = torch.cat([x] + c_concat, dim=1)
        else:
            xc = x
        if c_crossattn is not None:
            cc = torch.cat(c_crossattn, 1)
        else:
            cc = None
        if c_spatial_concat is not None:
            for i in range(len(c_spatial_concat)):
                pad_channels = xc.shape[1] - c_spatial_concat[i].shape[1]
                c_spatial_concat[i] = F.pad(c_spatial_concat[i], (0, 0, 0, 0, 0, pad_channels))
            xc = torch.cat([xc] + c_spatial_concat, dim=-1)  # concat along W
        if c_cache_kv is not None:
            assert len(c_cache_kv) == 1
            xrs = c_cache_kv[0]
            if cache_kv.mode == 'save':
                cache_kv.clear_cache()
                xrs = xrs.split(x.shape[-1], dim=-1)
                for xr in xrs:
                    xrc = F.pad(xr, (0, 0, 0, 0, 0, xc.shape[1] - xr.shape[1]))
                    _ = self.diffusion_model(xrc, torch.zeros_like(t), context=cc, is_ref=True)
                cache_kv.mode = 'use'
        out = self.diffusion_model(xc, t, context=cc)
        b, c, h, w = x.shape
        out = out[:, :, :h, :w]
        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs
