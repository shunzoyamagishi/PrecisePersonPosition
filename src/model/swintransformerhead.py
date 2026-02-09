import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

class UpBlock(nn.Module):
    """
    2倍のアップサンプリングを行うブロック
    ConvTranspose2d ではなく、Upsample + Conv を使用することで
    チェッカーボード歪みを防ぎ、より滑らかな出力を得やすくします。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # 1. バイリニア補間で2倍に拡大 (滑らかさ重視)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 2. 畳み込みで特徴を整える
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SwinTransformerHead(nn.Module):
    def __init__(self, in_size:int=224, in_channel:int=3, out_channel:int=1, device:str='cuda'):
        super().__init__()
        self.in_size = in_size
        self.device = device

        # --- Encoder (Backbone) ---
        swin = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        features_ch = [96, 192, 384, 768]  # Swin-Tの各ステージのチャンネル数
        
        # 入力チャンネル調整
        if in_channel != 3:
            original_layer = swin.features[0][0]
            swin.features[0][0] = nn.Conv2d(
                in_channels=in_channel,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride
            )
        
        # Swinは最後のGlobal Average Pooling手前までを使用
        # 出力サイズ: (B, H/32, W/32, 768)
        # self.encoder = swin.features
        self.encoder = swin.features[0:4]
        # self.norm = swin.norm 
        self.norm = nn.LayerNorm(192)

        # --- Decoder (Gradual Upsampling) ---
        # Swin-Tの最終出力は 768ch, サイズは 1/32 (224pxなら7x7)
        # これを 1/1 (224x224) まで5段階で戻す
        
        self.decoder = nn.Sequential(
            # 1/32 -> 1/16
            # UpBlock(768, 384),
            # 1/16 -> 1/8
            # UpBlock(384, 192),
            # 1/8  -> 1/4
            UpBlock(192, 96),
            # 1/4  -> 1/2
            UpBlock(96, 48),
            # 1/2  -> 1/1 (Original Size)
            UpBlock(48, 24)
        )

        # 最終出力層 (1x1 Conv)
        self.final_conv = nn.Conv2d(24, out_channel, kernel_size=1)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.encoder(x)      # (B, H/32, W/32, 768)
        x = self.norm(x)         # LayerNorm
        
        # Permute: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2) 

        # Decoder (段階的に大きくする)
        x = self.decoder(x)
        
        # Final prediction
        x = self.final_conv(x)

        return x

    def normalize_zscore(self, depth_map: torch.Tensor) -> torch.Tensor:
        mean = depth_map.mean(dim=[-2, -1], keepdim=True)
        std  = depth_map.std(dim=[-2, -1], keepdim=True)
        return (depth_map - mean) / (std + 1e-8)