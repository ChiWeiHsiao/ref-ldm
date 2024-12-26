import os
import math
import random
from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.ndimage import binary_dilation
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T

from ldm.data import degradations


def normalize_image(image):
    '''[0, 255] to [-1, 1]'''
    return image / 127.5 - 1.0


def unnormalize_image(image):
    '''[-1, 1] to [0, 255]'''
    return (image.clip(-1.0, 1.0) + 1.0) * 127.5


def read_image(image_path, resize=None):
    '''Read image and resize
    return [0, 255] pil image
    '''
    image_path = str(image_path)
    image = Image.open(image_path)
    if resize is not None and image.size != resize:
        image = image.resize(resize)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def sample_degraded_image(
    gt_image,
    blur_kernel_list, blur_kernel_prob, blur_kernel_size, blur_sigma,
    downsample_range, noise_range, jpeg_range,
):
    '''Generate LQ-image by applying random sampled degradation to input-image
    Args:
        gt_image: [0, 255] np image in rgb order

    Return:
        lq_image: [0, 255] np image in rgb order

    Reference:
        https://github.com/TencentARC/GFPGAN/blob/master/gfpgan/data/ffhq_degradation_dataset.py#L145
    '''
    lq_image = gt_image
    h, w, _ = lq_image.shape
    # degradations functions require input in [0, 1] and BGR-format
    lq_image = cv2.cvtColor(lq_image, cv2.COLOR_RGB2BGR)
    lq_image = lq_image / 255
    # apply blur
    kernel = degradations.random_mixed_kernels(
        blur_kernel_list,
        blur_kernel_prob,
        blur_kernel_size,
        blur_sigma,
        blur_sigma,
    )
    lq_image = cv2.filter2D(lq_image, -1, kernel)
    # apply downsample
    scale = np.random.uniform(downsample_range[0], downsample_range[1])
    #  interp = np.random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA])  # CelebA-Test use LINEAR
    lq_image = cv2.resize(lq_image, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
    # apply noise
    if noise_range is not None:
        lq_image = degradations.random_add_gaussian_noise(lq_image, noise_range)
    # apply jpeg compression
    if jpeg_range is not None:
        lq_image = degradations.random_add_jpg_compression(lq_image, jpeg_range)
    # upsample to original size
    lq_image = cv2.resize(lq_image, (w, h), interpolation=cv2.INTER_LINEAR)
    lq_image = cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB)
    lq_image = np.clip((lq_image * 255).round(), 0, 255)
    return lq_image


class ImageRestorationDataset(Dataset):
    '''Dataset for image restoration
    Args:
        file_list:
            csv file path (gt_image, lq_image, ref_image)
        use_given_lq:
            If true, LQ = given lq_image. Otherwise, LQ = gt_image + random degradation.
    '''
    def __init__(
        self, file_list, gt_dir='', lq_dir='', ref_dir='', semantic_dir='',
        use_given_lq=False,
        use_given_ref=False, max_num_refs=None, dup_to_max_num_refs=True, cat_refs_axis='width',
        loss_weight_by_semantic=None,
        use_sample_weight=False,
        ref_rand_aug=False, shuffle_refs_prob=0.,
        image_size=(512, 512), degrad_opt=None, lr_flip_aug=False,
    ):
        df = pd.read_csv(file_list)
        df['gt_image'] = df['gt_image'].apply(lambda x: os.path.join(gt_dir, x))
        if use_given_lq:
            df['lq_image'] = df['lq_image'].apply(lambda x: os.path.join(lq_dir, x))
        self.use_given_lq = use_given_lq
        if use_given_ref:
            df['ref_image'] = df['ref_image'].apply(literal_eval)
            df['ref_image'] = df['ref_image'].apply(lambda xs: [os.path.join(ref_dir, x) for x in xs])
            self.max_num_refs = max_num_refs
            self.dup_to_max_num_refs = dup_to_max_num_refs
            assert cat_refs_axis in ['width', 'channel']
            self.cat_refs_axis = cat_refs_axis
            self.shuffle_refs_prob = shuffle_refs_prob
            if ref_rand_aug:
                self.ref_rand_aug = T.Compose([
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                    T.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05), interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomPerspective(distortion_scale=0.2, p=1.0, interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomHorizontalFlip(p=0.5),
                ])
            else:
                self.ref_rand_aug = None
        if use_sample_weight and 'weight' in df.columns:
            self.sampler = self.create_weighted_sampler(df)
        self.semantic_dir = semantic_dir
        self.loss_weight_by_semantic = loss_weight_by_semantic
        self.use_given_ref = use_given_ref
        self.df = df
        self.image_size = image_size
        self.lr_flip_aug = lr_flip_aug
        self.degrad_opt = degrad_opt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        d = {}
        r = self.df.iloc[i]
        lr_flip = np.random.choice([True, False]) if self.lr_flip_aug else False
        # GT image
        gt_image = read_image(r['gt_image'], self.image_size)
        gt_image = np.array(gt_image, dtype=np.uint8)
        d['gt_image'] = normalize_image(gt_image)
        # loss weight map from semantic map
        if self.loss_weight_by_semantic is not None:
            gt_image_name = Path(r['gt_image']).stem
            gt_semantic_map = np.load(os.path.join(self.semantic_dir, f'{gt_image_name}.npy'))
            d['loss_weight_map'] = self.create_loss_weight_map_from_semantic_map(gt_semantic_map)
        # LQ image
        if self.use_given_lq:
            lq_image = read_image(r['lq_image'], self.image_size)
            lq_image = np.array(lq_image, dtype=np.uint8)
        else:
            lq_image = sample_degraded_image(gt_image, **self.degrad_opt)
        d['lq_image'] = normalize_image(lq_image)
        # Reference images
        if self.use_given_ref:
            # preprocess the list of images
            paths = r['ref_image'].copy()
            if self.shuffle_refs_prob > random.random():
                random.shuffle(paths)
            if self.max_num_refs is not None:
                # duplicate ref images if not enough
                if self.dup_to_max_num_refs:
                    paths = paths * math.ceil(self.max_num_refs / len(paths))
                paths = paths[:self.max_num_refs]
            # read images
            ref_images = []
            for path in paths:
                ref_image = read_image(path, self.image_size)
                if self.ref_rand_aug is not None:
                    ref_image = self.ref_rand_aug(ref_image)
                ref_image = np.array(ref_image, dtype=np.uint8)
                ref_image = normalize_image(ref_image)
                ref_images.append(ref_image)
            if self.cat_refs_axis == 'width':
                ref_images = np.concatenate(ref_images, axis=1)  # [H, nRef*W, 3]
            elif self.cat_refs_axis == 'channel':
                ref_images = np.concatenate(ref_images, axis=2)  # [H, W, nRef*3]
            d['ref_image'] = ref_images
        if self.lr_flip_aug and random.random() > 0.5:
            for k in d.keys():
                if k in d:
                    d[k] = np.fliplr(d[k]).copy()
        return d

    def create_loss_weight_map_from_semantic_map(self, semantic_map, dilation=5):
        '''
        semantic classes of CelebAMask-HQ: {
            0: 'background', 1: 'neck', 2: 'face', 3: 'cloth', 4: 'rr', 5: 'lr',
            6: 'rb', 7: 'lb', 8: 're', 9: 'le', 10: 'nose', 11: 'imouth', 12: 'llip',
            13: 'ulip', 14: 'hair', 15: 'eyeg', 16: 'hat', 17: 'earr', 18: 'neck_l',
        }
        loss_weight_by_semantic: {1: [14], 2:[2, 4, 5], 3: [10, 11, 12, 13], 9: [6, 7, 8, 9, 15]}
        loss_weight_by_semantic: {1: [2, 4, 5, 14], 2:[10, 11, 12, 13], 9: [6, 7, 8, 9, 15]}
        '''
        loss_weight_map = np.zeros_like(semantic_map)
        for weight, class_ids in sorted(self.loss_weight_by_semantic.items()):
            for class_id in class_ids:
                binary_map = (semantic_map == class_id)
                binary_map = binary_dilation(binary_map, iterations=dilation)
                loss_weight_map[binary_map] = weight
        loss_weight_map = loss_weight_map.astype(float)
        return loss_weight_map

    def create_weighted_sampler(self, df):
        num_samples = len(df)
        weights = df['weight'].to_numpy()
        return WeightedRandomSampler(weights, num_samples)


if __name__ == '__main__':
    # test dataset class
    degrad_opt = {
        'blur_kernel_list': ['iso', 'aniso'],
        'blur_kernel_prob': [0.5, 0.5],
        'blur_kernel_size': 41,
        'blur_sigma': [0, 16], 'downsample_range': [1, 32], 'noise_range': [0, 20], 'jpeg_range': [30, 100],
    }

    dataset = ImageRestorationDataset(
        file_list='data/ffhq/file_list/train_references_arcface0-04.csv',
        gt_dir='data/ffhq/images512x512',
        ref_dir='data/ffhq/images512x512',
        semantic_dir='data/ffhq/face_parsing/semantic_map',
        lq_dir='',
        use_given_lq=False,
        use_given_ref=True,
        loss_weight_by_semantic={1: [2, 4, 5, 14], 2:[10, 11, 12, 13], 5: [6, 7, 8, 9, 15]},
        max_num_refs=3,
        image_size=(512, 512),
        degrad_opt=degrad_opt,
        lr_flip_aug=True,
        ref_rand_aug=True, shuffle_refs_prob=1.,
    )

    ''' Visualize  '''
    #  from tqdm import tqdm
    #  save_dir = Path('./vis_aug_ref')
    #  save_dir.mkdir(parents=True, exist_ok=True)
    #  for i in tqdm(list(range(0, 40))):
        #  data = dataset.__getitem__(i)
        #  Image.fromarray(np.uint8(unnormalize_image(data['gt_image']))).save(save_dir / f'{i}_gt.png')
        #  Image.fromarray(np.uint8(unnormalize_image(data['lq_image']))).save(save_dir / f'{i}_lq.png')
        #  Image.fromarray(np.uint8(unnormalize_image(data['ref_image'][..., :3]))).save(save_dir / f'{i}_ref.png')
        #  #  Image.fromarray(np.uint8(data['ref_image'][..., 3] * 10)).save(save_dir / f'{i}_ref_semantic.png')
        #  loss_weight_map = data['loss_weight_map']
        #  loss_weight_map = torch.nn.functional.adaptive_max_pool2d(torch.Tensor(loss_weight_map).unsqueeze(0), (64, 64))[0].numpy()
        #  loss_weight_map = loss_weight_map.repeat(8, axis=0).repeat(8, axis=1)
        #  Image.fromarray(np.uint8(loss_weight_map * 50)).save(save_dir / f'{i}_gt_loss_weight_map.png')


    '''Sample input LQ from HQ'''
    #  from tqdm import tqdm
    #  random.seed(2024); np.random.seed(2024)
    #  gt_dir = Path('data/ffhq/images512x512')
    #  #  lq_dir = Path('data/ffhq/ffhq_test_LQ_images/b8-15_r16-32_n0-15_j30-100')
    #  lq_dir = Path('data/ffhq/ffhq_test_LQ_images/b10_r8_n15_j60')
    #  lq_dir = Path('data/ffhq/ffhq_test_LQ_images/b2_r4_n8_j80')
    #  gt_names = pd.read_csv('data/ffhq/file_list/test_references_arcface01-04.csv')['gt_image'].tolist()
    #  lq_dir.mkdir(parents=True)
    #  print(lq_dir)
    #  for gt_name in tqdm(gt_names):
        #  gt_image = read_image(str(gt_dir / gt_name), (512, 512))
        #  gt_image = np.array(gt_image).astype(np.float32)
        #  lq_image = sample_degraded_image(gt_image, **degrad_opt)
        #  lq_image = lq_image.astype(np.uint8)
        #  Image.fromarray(lq_image).save(str(lq_dir / gt_name))
