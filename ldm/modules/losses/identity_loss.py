from pathlib import Path

import torch
import torch.nn as nn
import onnx2torch

#  from ldm.util import freeze_model


MODEL_PATH = 'pretrained/insightface_webface_r50.onnx'


class IdentityLoss(nn.Module):
    """Compute cosine distance on face recognition model embedding space

    Forward:
        pred: image tensor, shape [B, 3, H, W], values in [-1, 1]
        target: image tensor, shape [B, 3, H, W], values in [-1, 1]
    """
    def __init__(self, model_path=MODEL_PATH, center_crop=True, resize_hw=(112, 112)):
        super().__init__()
        model_path = Path(model_path)
        if model_path.suffix == ".onnx":
            model = onnx2torch.convert(model_path)
        elif model_path.suffix == ".pt":
            model = torch.load(model_path)
        else:
            raise NotImplementedError
        #  freeze_model(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.center_crop = center_crop
        self.resize_hw = resize_hw

    def preprocess(self, img):
        if self.center_crop:
            h, w = img.shape[-2:]
            img = img[:, :, int(h * 0.0625): int(h * 0.9375), int(w * 0.0625): int(w * 0.9375)]
        if self.resize_hw is not None:
            img = nn.functional.interpolate(img, self.resize_hw, mode='area')
        return img

    def forward(self, pred, target):
        pred = self.preprocess(pred)
        target = self.preprocess(target)

        pred = self.model(pred)
        target = self.model(target)

        loss = 1 - nn.functional.cosine_similarity(pred, target, dim=1)
        return loss

class IdentityEncoder(nn.Module):
    """Compute cosine distance on face recognition model embedding space

    Forward:
        pred: image tensor, shape [B, 3, H, W], values in [-1, 1]
        target: image tensor, shape [B, 3, H, W], values in [-1, 1]
    """
    def __init__(self, model_path=MODEL_PATH, center_crop=True, resize_hw=(112, 112)):
        super().__init__()
        model_path = Path(model_path)
        if model_path.suffix == ".onnx":
            model = onnx2torch.convert(model_path)
        elif model_path.suffix == ".pt":
            model = torch.load(model_path)
        else:
            raise NotImplementedError
        #  freeze_model(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.center_crop = center_crop
        self.resize_hw = resize_hw

    def preprocess(self, img):
        if self.center_crop:
            h, w = img.shape[-2:]
            img = img[:, :, int(h * 0.0625): int(h * 0.9375), int(w * 0.0625): int(w * 0.9375)]
        if self.resize_hw is not None:
            img = nn.functional.interpolate(img, self.resize_hw, mode='area')
        return img

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x).unsqueeze(1) # B, Tokens (1), Channels (512)
        return x
