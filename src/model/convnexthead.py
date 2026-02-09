import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtHead(nn.Module):
    def __init__(self, in_size:int=224, in_channel:int=3, out_channel:int=1, embedding_dim:int=128, device:str='cuda'):
        """
        ConvNeXt Tiny Backbone + SegFormer (All-MLP) Head
        軽量・高速・高精度なヒートマップ回帰モデル
        """
        super().__init__()
        self.device = device
        self.in_size = in_size
        
        # --- Backbone: ConvNeXt Tiny ---
        # weights=DEFAULT (ImageNet pretrained) を使用
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        base_model = convnext_tiny(weights=weights)
        
        # 入力チャンネル数の変更対応 (Stem部分の最初のConv2dを置換)
        # ConvNeXtのfeatures[0]はConv2dNormActivationブロック
        if in_channel != 3:
            original_conv = base_model.features[0][0]
            base_model.features[0][0] = nn.Conv2d(
                in_channels=in_channel,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding
            )

        # 特徴抽出用レイヤーの分離
        # torchvisionのConvNeXt features構造:
        # [0]: Stem (H/4)
        # [1]: Stage 1 (H/4)
        # [2]: Downsample (H/8)
        # [3]: Stage 2 (H/8)
        # [4]: Downsample (H/16)
        # [5]: Stage 3 (H/16)
        # [6]: Downsample (H/32)
        # [7]: Stage 4 (H/32)
        
        self.features = base_model.features
        
        # 各ステージのチャンネル数 (ConvNeXt Tiny)
        dims = [96, 192, 384, 768]
        
        # --- Decoder: SegFormer Style (All-MLP) ---
        # embedding_dimを小さくするほど軽量になります (例: 256 -> 128)
        
        # 1. 各ステージの出力を同じチャンネル数(embedding_dim)に変換
        self.linear_c1 = nn.Conv2d(dims[0], embedding_dim, 1)
        self.linear_c2 = nn.Conv2d(dims[1], embedding_dim, 1)
        self.linear_c3 = nn.Conv2d(dims[2], embedding_dim, 1)
        self.linear_c4 = nn.Conv2d(dims[3], embedding_dim, 1)

        # 2. 統合レイヤー (4つのステージを結合した後)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 最終出力ヘッド
        self.dropout = nn.Dropout(0.1) # 過学習抑制
        self.head = nn.Conv2d(embedding_dim, out_channel, kernel_size=1)
        
        self.to(device)

    def forward(self, x):
        # --- Encoder Path ---
        # ConvNeXtの各ステージを通しながら出力を保持
        
        # Stem + Stage 1 -> H/4
        x = self.features[0](x)
        c1 = self.features[1](x)
        
        # Downsample + Stage 2 -> H/8
        x = self.features[2](c1)
        c2 = self.features[3](x)
        
        # Downsample + Stage 3 -> H/16
        x = self.features[4](c2)
        c3 = self.features[5](x)
        
        # Downsample + Stage 4 -> H/32
        x = self.features[6](c3)
        c4 = self.features[7](x)

        # --- Decoder Path (All-MLP) ---
        
        # 1. 次元圧縮 (Channel Projection)
        c1_emb = self.linear_c1(c1)
        c2_emb = self.linear_c2(c2)
        c3_emb = self.linear_c3(c3)
        c4_emb = self.linear_c4(c4)
        
        # 2. アップサンプリングしてサイズを合わせる (全て H/4 に統一)
        # c1は既に H/4 なのでそのまま
        target_size = c1_emb.shape[2:]
        
        c2_up = F.interpolate(c2_emb, size=target_size, mode='bilinear', align_corners=True)
        c3_up = F.interpolate(c3_emb, size=target_size, mode='bilinear', align_corners=True)
        c4_up = F.interpolate(c4_emb, size=target_size, mode='bilinear', align_corners=True)
        
        # 3. 特徴結合 (Concatenate) -> (B, 4*emb, H/4, W/4)
        x_cat = torch.cat([c4_up, c3_up, c2_up, c1_emb], dim=1)
        
        # 4. 特徴融合 (Fusion)
        x_fuse = self.linear_fuse(x_cat)
        
        # 5. 最終予測 (Prediction)
        x_out = self.dropout(x_fuse)
        x_out = self.head(x_out)
        
        # 6. 入力サイズに戻す (H/4 -> H)
        x_out = F.interpolate(x_out, size=(self.in_size, self.in_size), mode='bilinear', align_corners=True)
        
        return x_out

    def normalize_zscore(self, depth_map: torch.Tensor) -> torch.Tensor:
        mean = depth_map.mean(dim=[-2, -1], keepdim=True)
        std  = depth_map.std(dim=[-2, -1], keepdim=True)
        return (depth_map - mean) / (std + 1e-8)
    
