from pathlib import Path
from glob import glob
from collections import defaultdict
from functools import partial

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pytorch_ssim
import lpips

from ldm.modules.losses.identity_loss import IdentityLoss
import pyiqa  # TODO: /proj/gpu_mtk26829/.cache/torch/hub/checkpoints/PUT_WEIGHT_HERE


# Initialize metric functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
identity_loss = IdentityLoss(model_path='pretrained/insightface_webface_r50.onnx').to(device)  # input -1~1
lpips = pyiqa.create_metric('lpips-vgg', device=device)
psnr = pyiqa.create_metric('psnr', device=device)
ssim = pyiqa.create_metric('ssim', device=device)
musiq = pyiqa.create_metric('musiq', device=device)
niqe = pyiqa.create_metric('niqe', device=device)
fid = pyiqa.create_metric('fid', device=device)
fid_ffhq = partial(fid, dataset_name='FFHQ', dataset_res=512, dataset_split='trainval70k')


# Specify what to run
summary_csv_path = Path('results/eval.csv')
save_per_image_csv_dir = Path('results/eval_per_image')
save_per_image_csv = False

gt_pred_dirs = [
    ('data/ffhq/images512x512', 'results/ffhqtest_moderate/refldm/step=100_cfg=1.5'),
    ('data/ffhq/images512x512', 'results/ffhqtest_severe/refldm/step=100_cfg=1.5'),
    ('data/celebahq/img_test_HQ', 'results/celebatest/refldm/step=100_cfg=1.5'),
]


# # Semantic map -> binary face mask
# dataset_to_semantic_dir = {
#     'celebahq': 'data/celebahq/face_parsing/semantic_map',
#     'ffhq': 'data/ffhq/face_parsing/semantic_map',
# }
# def create_facemask_from_semanticmap(semantic_map_path, definition='face'):
#     '''
#     semantic classes of CelebAMask-HQ: {
#         0: 'background', 1: 'neck', 2: 'face', 3: 'cloth', 4: 'rr', 5: 'lr',
#         6: 'rb', 7: 'lb', 8: 're', 9: 'le', 10: 'nose', 11: 'imouth', 12: 'llip',
#         13: 'ulip', 14: 'hair', 15: 'eyeg', 16: 'hat', 17: 'earr', 18: 'neck_l',
#     }
#     '''
#     face_class_idxs = {
#         'face': [2, 4, 5, 10, 11, 12, 13, 6, 7, 8, 9, 15],
#         'face_hair': [2, 4, 5, 10, 11, 12, 13, 6, 7, 8, 9, 15, 14],
#         'face_component': [10, 11, 12, 13, 6, 7, 8, 9, 15],
#     }[definition]
#     if not Path(semantic_map_path).is_file():
#         print(f'Warning: semantic map not found, {semantic_map_path}')
#         return None
#     semantic_map = np.load(semantic_map_path)
#     binary_mask = np.isin(semantic_map, face_class_idxs)
#     return binary_mask

def normalize(x):
    '''Normalize from [0, 1] to [-1, 1]
    '''
    return x * 2. - 1.

def read_image(path, resize=(512, 512), mask=None):
    '''Read image and convert as torch model input
    Returns:
        [1, 3, H, W] torch_tensor in 0 ~ 1
    '''
    img = Image.open(str(path))
    if resize is not None and img.size != resize:
        img = img.resize(resize)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)
    if mask is not None:
        img = img * mask[..., None]
    # [0, 255] -> [0, 1]
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)[None]
    img = torch.from_numpy(img)
    img = img / 255.
    return img


for gt_dir, pred_dir in gt_pred_dirs:
    print(f'pred_dir: {pred_dir}')
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    # semantic_dir = Path(dataset_to_semantic_dir['celebahq' if ('celebahq' in str(gt_dir)) else 'ffhq'])
    img_names = sorted([path.name for path in pred_dir.glob('*.png')])
    pred_name = '-'.join(str(pred_dir).split('/')[1:])
    for img_name in img_names: assert (gt_dir / img_name).is_file()

    data = defaultdict(list)
    for img_name in tqdm(img_names):
        data['img_name'].append(img_name)
        gt = read_image(gt_dir / img_name).to(device)
        pred = read_image(pred_dir / img_name).to(device)
        # mask = create_facemask_from_semanticmap(semantic_dir / f'{Path(img_name).stem}.npy')
        # gt_masked = read_image(gt_dir / img_name, mask=mask).to(device)
        # pred_masked = read_image(pred_dir / img_name, mask=mask).to(device)
        with torch.no_grad():
            data['ids'].append(1.0 - float(identity_loss(normalize(pred), normalize(gt))))
            data['lpips'].append(float(lpips(pred, gt)))
            # data['Flpips'].append(float(lpips(pred_masked, gt_masked)))
            data['psnr'].append(float(psnr(pred, gt)))
            data['ssim'].append(float(ssim(pred, gt)))
            data['niqe'].append(float(niqe(pred)))
            data['musiq'].append(float(musiq(pred)))

    # Save per-image metrics (row = img)
    if save_per_image_csv:
        save_per_image_csv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_csv(save_per_image_csv_dir / f'{pred_name}.csv', index=False)  # for picking qualitative

    # Register to summary table (row = method)
    metric_names = [m for m in data.keys() if m != 'img_name']
    metric_to_mean = {m: np.mean(data[m]) for m in metric_names}
    metric_to_mean['fid_ffhq'] = float(fid_ffhq(pred_dir))
    metric_to_mean['n_img'] = len(img_names)

    if summary_csv_path.is_file():
        df_sum = pd.read_csv(summary_csv_path, index_col=0)
    else:
        df_sum = pd.DataFrame(columns=metric_to_mean.keys())
    df_sum.loc[pred_name] = metric_to_mean
    df_sum.sort_index(inplace=True)
    pd.set_option('display.max_colwidth', None)
    print(df_sum)
    df_sum.to_csv(summary_csv_path)