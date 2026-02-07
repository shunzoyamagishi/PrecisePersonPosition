import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import swin_v2_s, Swin_V2_S_Weights, swin_v2_t, Swin_V2_T_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
import os
import numpy as np

from .depth import DepthFeatureExtractor
from .swintransformerhead import SwinTransformerHead

from src.utils import deep_merge_dict


class PositionEstimator(nn.Module):
    def __init__(self, cfg:dict|None=None, device:str='cuda'):
        super().__init__()

        default_cfg = {
            "DepthAnythingV2_encoder": "vits",
            "in_size": 224,
            "pretrain_model": None,
            "perspective_heatmap" : True
        }
        if cfg is None:
            self.cfg = default_cfg
        else:
            self.cfg = deep_merge_dict(default_cfg, cfg)

        self.device = device

        depthmodel_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }        
        self.in_size = self.cfg["in_size"]
        self.encoder = self.cfg["DepthAnythingV2_encoder"]

        if self.cfg.get("pretrain_model") and os.path.isfile(self.cfg["pretrain_model"]):
            ckpt = torch.load(self.cfg["pretrain_model"], map_location='cpu')
            self.in_size = ckpt.get("in_size", self.in_size)
            self.encoder = ckpt.get("DepthAnythingV2_encoder", self.encoder)
            # self.head = SwinTransformerHead(in_size=self.in_size, in_channel=depthmodel_configs[self.encoder]['features'], out_channel=2, device=self.device)
            self.head = SwinTransformerHead(in_size=self.in_size, in_channel=depthmodel_configs[self.encoder]['features'], out_channel=2, device=self.device)
            self.head.load_state_dict(ckpt["model"])
            print(f"Head checkpoint {self.cfg['pretrain_model']} is loaded.")
        else:
            # self.head = SwinTransformerHead(in_size=self.in_size, in_channel=depthmodel_configs[self.encoder]['features'], out_channel=2, device=self.device)
            self.head = SwinTransformerHead(in_size=self.in_size, in_channel=depthmodel_configs[self.encoder]['features'], out_channel=2, device=self.device)
        self.depth_model = DepthFeatureExtractor(**depthmodel_configs[self.encoder], device=self.device)
        if os.path.isfile(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{self.encoder}.pth'):
            self.depth_model.load_state_dict(torch.load(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cpu'))
            print(f"Depth Anything V2 checkpoint depth_anything_v2_{self.encoder}.pth is loaded.")
        else:
            print(f"Depth Anything V2 Checkpoint depth_anything_v2_{self.encoder}.pth is not found.")

        self.depth_model = self.depth_model.to(self.device).eval()
        self.head = self.head.to(self.device).eval()

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        feature, depth = self.depth_model.forward(x)
        pos = self.head.forward(feature)

        return depth, pos
    
    @torch.inference_mode()
    def inference(self, raw_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, (h, w) = self.depth_model.image2tensor(raw_image, self.in_size)
        
        depth, pos = self.forward(image)

        if self.cfg.get("perspective_heatmap", True):
            pos = pos[:,0:1] + torch.prod(pos, dim=1, keepdim=True)
        else:
            pos = pos[:,0:1]
        pos = self.head.normalize_zscore(pos)
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        pos = F.interpolate(pos, (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy(), pos.cpu().numpy()