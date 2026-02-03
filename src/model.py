import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import swin_v2_s, Swin_V2_S_Weights, swin_v2_t, Swin_V2_T_Weights
import os
import numpy as np

from src.depth import DepthFeatureExtractor
from src.utils import deep_merge_dict


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


class DecoderBlock(nn.Module):
    """
    アップサンプリングとスキップ接続の結合を行うブロック
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # チャンネルラスト(B, H, W, C)からチャンネルファースト(B, C, H, W)への変換は呼び出し元で行う前提
        
        # 1. アップサンプリング (2倍)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 2. 結合後の畳み込み
        # 入力ch + スキップ接続ch -> 出力ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # x: 深い層からの入力 (B, C_in, H, W)
        # skip: 浅い層からの入力 (B, C_skip, H*2, W*2)
        x = self.up(x)
        
        # サイズ調整（padding等のズレ防止）
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = torch.nn.functional.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)
            
        x = torch.cat([x, skip], dim=1) # チャンネル方向(dim=1)で結合
        x = self.conv(x)
        return x


class SwinTransformerUNetHead(nn.Module):
    def __init__(self, in_size:int=224, in_channel:int=3, out_channel:int=1, device:str='cuda'):
        """
        Swin Transformer Encoder + U-Net style Decoder
        """
        super().__init__()
        self.in_size = in_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.device = device

        # --- Encoder (Backbone) ---
        # Weights.DEFAULTを使用
        swin = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)

        # 入力チャンネル数の変更対応
        if in_channel != 3:
            # 元のパッチサイズ(4x4)を維持しつつ、入力chを変更
            original_layer = swin.features[0][0]
            swin.features[0][0] = nn.Conv2d(
                in_channels=in_channel, # ここを修正 (64ではなくin_channel)
                out_channels=original_layer.out_channels, # 通常は96
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride
            )

        # Swin Transformerの各ステージを分割保持
        # torchvisionのSwin実装は features[0]~[7] で構成される
        # 0,1 -> Stage 1 (H/4, C=96)
        # 2,3 -> Stage 2 (H/8, C=192)
        # 4,5 -> Stage 3 (H/16, C=384)
        # 6,7 -> Stage 4 (H/32, C=768)
        
        self.encoder_stage1 = nn.Sequential(swin.features[0], swin.features[1])
        self.encoder_stage2 = nn.Sequential(swin.features[2], swin.features[3])
        self.encoder_stage3 = nn.Sequential(swin.features[4], swin.features[5])
        self.encoder_stage4 = nn.Sequential(swin.features[6], swin.features[7])
        
        self.norm = swin.norm # 最終層のLayerNorm

        # 各ステージのチャンネル数 (Swin-Tの場合)
        c1, c2, c3, c4 = 96, 192, 384, 768

        # --- Decoder (U-Net style) ---
        # Stage 4 -> Stage 3
        self.decoder3 = DecoderBlock(in_channels=c4, skip_channels=c3, out_channels=c3)
        # Stage 3 -> Stage 2
        self.decoder2 = DecoderBlock(in_channels=c3, skip_channels=c2, out_channels=c2)
        # Stage 2 -> Stage 1
        self.decoder1 = DecoderBlock(in_channels=c2, skip_channels=c1, out_channels=c1)

        # Final Upsampling (Stage 1 -> Input Size)
        # H/4 -> Hへ戻す (4倍アップサンプリング)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, out_channel, kernel_size=1) # 最終出力
        )
        
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder Path ---
        # Swinは (B, H, W, C) で出力するため、Conv2dに入れる前に (B, C, H, W) へpermuteが必要
        
        # Stage 1
        x1 = self.encoder_stage1(x) # (B, H/4, W/4, 96)
        x1_p = x1.permute(0, 3, 1, 2) # (B, 96, H/4, W/4)
        
        # Stage 2
        x2 = self.encoder_stage2(x1) # (B, H/8, W/8, 192)
        x2_p = x2.permute(0, 3, 1, 2)
        
        # Stage 3
        x3 = self.encoder_stage3(x2) # (B, H/16, W/16, 384)
        x3_p = x3.permute(0, 3, 1, 2)
        
        # Stage 4 (Bottleneck)
        x4 = self.encoder_stage4(x3) # (B, H/32, W/32, 768)
        x4 = self.norm(x4) # Swinの最後のNorm
        x4_p = x4.permute(0, 3, 1, 2)

        # --- Decoder Path ---
        # x4とx3を結合
        d3 = self.decoder3(x4_p, x3_p) # -> (B, 384, H/16, W/16)
        
        # d3とx2を結合
        d2 = self.decoder2(d3, x2_p)   # -> (B, 192, H/8, W/8)
        
        # d2とx1を結合
        d1 = self.decoder1(d2, x1_p)   # -> (B, 96, H/4, W/4)
        
        # --- Final Head ---
        out = self.final_up(d1)        # -> (B, out_channel, H, W)

        return out

    def normalize_zscore(self, depth_map: torch.Tensor) -> torch.Tensor:
        mean = depth_map.mean(dim=[-2, -1], keepdim=True)
        std  = depth_map.std(dim=[-2, -1], keepdim=True)
        return (depth_map - mean) / (std + 1e-8)
    

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
            self.head = SwinTransformerHead(in_size=self.in_size, in_channel=depthmodel_configs[self.encoder]['features'], out_channel=2, device=self.device)
            self.head.load_state_dict(ckpt["model"])
            print(f"Head checkpoint {self.cfg['pretrain_model']} is loaded.")
        else:
            self.head = SwinTransformerUNetHead(in_size=self.in_size, in_channel=depthmodel_configs[self.encoder]['features'], out_channel=2, device=self.device)

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