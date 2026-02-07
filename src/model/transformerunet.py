import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import swin_v2_s, Swin_V2_S_Weights, swin_v2_t, Swin_V2_T_Weights, convnext_tiny, ConvNeXt_Tiny_Weights


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