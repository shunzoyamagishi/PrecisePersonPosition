import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from .bboxencoding import BboxEncoding

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SwinTransformerHead(nn.Module):
    def __init__(self, in_channel:int=3, bottleneck_depth:int=2, out_channel:int=1, device:str='cuda'):
        super().__init__()
        self.bottleneck_depth = max(1, min(bottleneck_depth, 4))
        self.device = device

        # --- Encoder (Backbone) ---
        swin = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        features_ch = [96, 192, 384, 768]  # Number of Channels in Each Stage of Swin-T
        
        if in_channel != 3:
            original_layer = swin.features[0][0]
            swin.features[0][0] = nn.Conv2d(
                in_channels=in_channel,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride
            )
        
        ch = features_ch[self.bottleneck_depth-1]
        self.encoder = swin.features[0:2*self.bottleneck_depth]
        self.norm = nn.LayerNorm(ch)
        
        decoder_layers = []
        while ch > 24:
            decoder_layers.append(UpBlock(ch, ch//2))
            ch = ch//2
        self.decoder = nn.Sequential(*decoder_layers)

        self.final_conv = nn.Conv2d(24, out_channel, kernel_size=1)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.encoder(x)      # (B, H/32, W/32, 768)
        x = self.norm(x)         # LayerNorm
        
        # Permute: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2) 

        # Decoder
        x = self.decoder(x)
        
        # Final prediction
        x = self.final_conv(x)

        return x

    def normalize_zscore(self, depth_map: torch.Tensor) -> torch.Tensor:
        mean = depth_map.mean(dim=[-2, -1], keepdim=True)
        std  = depth_map.std(dim=[-2, -1], keepdim=True)
        return (depth_map - mean) / (std + 1e-8)


class SwinTransformerHeadWithBBox(SwinTransformerHead):
    def __init__(self, in_channel:int=3, bottleneck_depth:int=2, out_channel:int=1, device:str='cuda'):
        super().__init__(in_channel, bottleneck_depth, out_channel, device)
        self.bboxencoder = BboxEncoding(out_channel=8, device=device)
        self.to(device)
    
    def forward(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.bboxencoder(x, bbox)
        x = self.encoder(x)      
        x = self.norm(x)         
        
        # Permute: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2) 

        # Decoder
        x = self.decoder(x)
        
        # Final prediction
        x = self.final_conv(x)

        return x