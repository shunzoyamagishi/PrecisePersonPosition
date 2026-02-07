import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import swin_v2_s, Swin_V2_S_Weights, swin_v2_t, Swin_V2_T_Weights


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

        # self.head = nn.Sequential(
        #     torchvision.ops.misc.Permute(dims=[0,3,1,2]),
        #     nn.ConvTranspose2d(in_channels=96*8, out_channels=out_channel, kernel_size=(8,8), stride=(8,8)),
        #     nn.BatchNorm2d(num_features=out_channel)
        # )
        self.head = nn.Sequential(
            torchvision.ops.misc.Permute(dims=[0,3,1,2]),  # (B,C,H,W)

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96*8, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),

            nn.Conv2d(64, out_channel, 1)
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