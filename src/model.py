import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import swin_v2_s, Swin_V2_S_Weights, swin_v2_t, Swin_V2_T_Weights
from src.depth import DepthFeatureExtractor
import os
import numpy as np

class SwinTransformerHead(nn.Module):
    def __init__(self, in_size:int=224, in_channel:int=3, out_channel:int=1, device:str='cuda'):
        """
        Swin Transformer Encoder-Decoder ネットワーク
        """
        super().__init__()
        self.in_size = in_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        # Swin Transformer backbone
        swin = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        if in_channel != 3:
            swin.features[0] = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=swin.features[0][0].out_channels, kernel_size=(1,1), stride=(1,1)),
                torchvision.ops.misc.Permute(dims=[0,2,3,1]),
                nn.LayerNorm(normalized_shape=96)
            )

        self.encoder = swin.features

        self.layersize = self._calc_layersize()

        self.head = nn.Sequential(
            torchvision.ops.misc.Permute(dims=[0,3,1,2]),
            nn.ConvTranspose2d(in_channels=96*8, out_channels=out_channel, kernel_size=(8,8), stride=(8,8)),
            nn.BatchNorm2d(num_features=out_channel)
        )

        self.device = device
        self.to(device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.head(x)

        return x

    def _calc_layersize(self) -> dict:
        """
        中間層の出力サイズを推定
        """
        l0 = (self.in_channel, self.in_size, self.in_size)
        # l1_h = ((l0[1] - 4) // 4 + 1)
        # l1_w = ((l0[2] - 4) // 4 + 1)
        l1_h = l0[1]
        l1_w = l0[2]
        l1 = (l1_h, l1_w, 96)

        def down(s): return -(-s // 2)
        l2 = (down(l1[0]), down(l1[1]), l1[2] * 2)
        l3 = (down(l2[0]), down(l2[1]), l2[2] * 2)
        l4 = (down(l3[0]), down(l3[1]), l3[2] * 2)

        return {
            "input": l0,
            "layer1": l1,
            "layer2": l2,
            "layer3": l3,
            "layer4": l4
        }
    
    def normalize_zscore(self, depth_map: torch.Tensor) -> torch.Tensor:
        mean = depth_map.mean(dim=[-2, -1], keepdim=True)
        std  = depth_map.std(dim=[-2, -1], keepdim=True)
        return (depth_map - mean) / (std + 1e-8)


class PositionEstimator(nn.Module):
    def __init__(self, cfg:dict, DEVICE:str='cuda'):
        super().__init__()
        
        pretrain_path = cfg.get("pretrain_model") if (cfg.get("pretrain_model") and os.path.isfile(cfg["pretrain_model"])) else None
        
        self.in_size = int(cfg["in_size"])
        self.DEVICE = DEVICE 

        depthmodel_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = cfg["DepthAnythingV2_encoder"]
        self.depth_model = DepthFeatureExtractor(**depthmodel_configs[encoder], device=self.DEVICE)
        self.head = SwinTransformerHead(in_size=self.in_size, in_channel=depthmodel_configs['vits']['features'], out_channel=2, device=self.DEVICE)

        if os.path.isfile(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth'):
            self.depth_model.load_state_dict(torch.load(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        else:
            print(f"Depth Anything V2 Checkpoint depth_anything_v2_{encoder}.pth is not found.")
        if pretrain_path:
            self.head.load_state_dict(torch.load(pretrain_path, map_location='cpu')["model"]) 
        else:
            print(f"Checkpoint {cfg.get("pretrain_model")} is not found.")
        
        self.depth_model = self.depth_model.to(DEVICE).eval()
        self.head = self.head.to(DEVICE).eval()

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.DEVICE)
        feature, depth = self.depth_model.forward(x)
        pos = self.head.forward(feature)

        return depth, pos
    
    @torch.inference_mode()
    def inference(self, raw_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, (h, w) = self.depth_model.image2tensor(raw_image, self.in_size)
        
        depth, pos = self.forward(image)

        pos = pos[:,0:1] + torch.prod(pos, dim=1, keepdim=True)
        pos = self.head.normalize_zscore(pos)
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        pos = F.interpolate(pos, (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy(), pos.cpu().numpy()
    

class PositionEstimator_single(nn.Module):
    def __init__(self, cfg:dict, DEVICE:str='cuda'):
        super().__init__()
        
        pretrain_path = cfg.get("pretrain_model") if (cfg.get("pretrain_model") and os.path.isfile(cfg["pretrain_model"])) else None
        
        self.in_size = int(cfg["in_size"])
        self.DEVICE = DEVICE 

        depthmodel_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = cfg["DepthAnythingV2_encoder"]
        self.depth_model = DepthFeatureExtractor(**depthmodel_configs[encoder], device=self.DEVICE)
        self.head = SwinTransformerHead(in_size=self.in_size, in_channel=depthmodel_configs['vits']['features'], out_channel=1, device=self.DEVICE)

        if os.path.isfile(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth'):
            self.depth_model.load_state_dict(torch.load(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        else:
            print(f"Depth Anything V2 Checkpoint depth_anything_v2_{encoder}.pth is not found.")
        if pretrain_path:
            self.head.load_state_dict(torch.load(pretrain_path, map_location='cpu')["model"]) 
        else:
            print(f"Checkpoint {cfg.get("pretrain_model")} is not found.")
        
        self.depth_model = self.depth_model.to(DEVICE).eval()
        self.head = self.head.to(DEVICE).eval()

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.DEVICE)
        feature, depth = self.depth_model.forward(x)
        pos = self.head.forward(feature)
        return depth, pos
    
    @torch.inference_mode()
    def inference(self, raw_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, (h, w) = self.depth_model.image2tensor(raw_image, self.in_size)
        
        depth, pos = self.forward(image)
        pos = self.head.normalize_zscore(pos)
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        pos = F.interpolate(pos, (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy(), pos.cpu().numpy()