import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import swin_v2_s, Swin_V2_S_Weights, swin_v2_t, Swin_V2_T_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
import os
import numpy as np
import warnings

from .depth import DepthFeatureExtractor
from .swintransformerhead import SwinTransformerHead, SwinTransformerHeadWithBBox

from src.utils import deep_merge_dict


class PositionEstimator(nn.Module):
    def __init__(self, cfg:dict|None=None, device:str='cuda'):
        super().__init__()

        default_cfg = {
            "DepthAnythingV2_encoder": "vits",
            "head": "swintransformer",
            "head_depth": 2,
            "in_size": 224,
            "pretrain_model": None,
            "perspective_heatmap" : True,
            "targetstd": 5.0
        }
        if cfg is None:
            self.cfg = default_cfg
        else:
            self.cfg = deep_merge_dict(default_cfg, cfg)
            if self.cfg["head_depth"] < 1 or 4 < self.cfg["head_depth"]:
                self.cfg["head_depth"] = min(max(1, self.cfg["head_depth"]), 4)
                warnings.warn(
                    f"The depth of head must be ranga of [1, 4]. It has been automatically changed to {self.cfg["head_depth"]}.",
                    UserWarning
                )
            if self.cfg["in_size"] % (8*self.cfg["head_depth"]) != 0:
                self.cfg["in_size"] = 224
                warnings.warn(
                    f"The input size is recommended to be a multiple of 7 and 8×head depth(={8*self.cfg["head_depth"]}). It has been automatically changed to {self.cfg["in_size"]}.",
                    UserWarning
                )

        self.device = device

        depthmodel_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if self.cfg.get("pretrain_model") and os.path.isfile(self.cfg["pretrain_model"]):
            ckpt = torch.load(self.cfg["pretrain_model"], map_location='cpu')
            self.cfg["in_size"] = ckpt.get("in_size", self.cfg["in_size"])
            self.cfg["DepthAnythingV2_encoder"] = ckpt.get("DepthAnythingV2_encoder", self.cfg["DepthAnythingV2_encoder"])
            self.cfg["head"] = ckpt.get("model_name", self.cfg["head"])
            self.cfg["head_depth"] = ckpt.get("model_depth", self.cfg["head_depth"])
            self.cfg["targetstd"] = ckpt.get("targetstd", self.cfg["targetstd"])
            
            self.head = self.build_head(
                model_name=self.cfg["head"],
                in_channel=depthmodel_configs[self.cfg["DepthAnythingV2_encoder"]]['features'],
                bottleneck_depth=self.cfg["head_depth"],
                out_channel=2,
                device=self.device
            )
            self.head.load_state_dict(ckpt["model"])
            print(f"Head checkpoint {self.cfg['pretrain_model']} is loaded.")
        else:
            if self.cfg.get("pretrain_model"):
                print(f"Head checkpoint {self.cfg['pretrain_model']} is not found.")
            self.head = self.build_head(
                model_name=self.cfg["head"],
                in_channel=depthmodel_configs[self.cfg["DepthAnythingV2_encoder"]]['features'],
                bottleneck_depth=self.cfg["head_depth"],
                out_channel=2,
                device=self.device
            )
        self.depth_model = DepthFeatureExtractor(**depthmodel_configs[self.cfg["DepthAnythingV2_encoder"]], device=self.device)
        if os.path.isfile(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{self.cfg["DepthAnythingV2_encoder"]}.pth'):
            self.depth_model.load_state_dict(torch.load(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{self.cfg["DepthAnythingV2_encoder"]}.pth', map_location='cpu'))
            print(f"Depth Anything V2 checkpoint depth_anything_v2_{self.cfg["DepthAnythingV2_encoder"]}.pth is loaded.")
        else:
            print(f"Depth Anything V2 Checkpoint depth_anything_v2_{self.cfg["DepthAnythingV2_encoder"]}.pth is not found.")

        self.depth_model = self.depth_model.to(self.device).eval()
        self.head = self.head.to(self.device).eval()

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        feature, depth = self.depth_model.forward(x)
        pos = self.head.forward(feature)

        return depth, pos
    
    def build_head(self, *, model_name: str, in_channel: int, bottleneck_depth: int, out_channel: int, device: str):
        MODEL_REGISTRY = {
            "swintransformer": SwinTransformerHead,
            # You can add your favorite models as needed.
        }
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_name={model_name}. choices={list(MODEL_REGISTRY)}")
        
        cls = MODEL_REGISTRY[model_name]
        return cls(
            in_channel=in_channel,
            bottleneck_depth=bottleneck_depth,
            out_channel=out_channel,
            device=device,
        )

    @torch.inference_mode()
    def inference(self, raw_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, (h, w) = self.depth_model.image2tensor(raw_image, self.cfg["in_size"])
        
        depth, pos = self.forward(image)

        if self.cfg.get("perspective_heatmap", True):
            pos = pos[:,0:1] + torch.prod(pos, dim=1, keepdim=True)
        else:
            pos = pos[:,0:1]
        pos = self.head.normalize_zscore(pos)
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        pos = F.interpolate(pos, (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy(), pos.cpu().numpy()
    

class PositionEstimatorWithBBox(PositionEstimator):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def forward(self, x: torch.Tensor, bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        feature, depth = self.depth_model.forward(x)
        pos = self.head.forward(feature, bbox)

        return depth, pos
    
    def build_head(self, *, model_name: str, in_channel: int, bottleneck_depth: int, out_channel: int, device: str):
        MODEL_REGISTRY = {
            "swintransformer": SwinTransformerHeadWithBBox,
            # You can add your favorite models as needed.
        }
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_name={model_name}. choices={list(MODEL_REGISTRY)}")
        
        cls = MODEL_REGISTRY[model_name]
        return cls(
            in_channel=in_channel+8,
            bottleneck_depth=bottleneck_depth,
            out_channel=out_channel,
            device=device,
        )