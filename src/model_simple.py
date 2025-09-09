import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_s, Swin_V2_S_Weights
from einops import rearrange
import copy

class SwinTransformerEncoder(nn.Module):
    def __init__(self, imgsize: tuple[int, int], num_classes: int):
        """
        Swin Transformer Encoder-Decoder ネットワーク
        """
        super().__init__()
        self.imgsize = imgsize
        self.height, self.width = imgsize
        self.num_classes = num_classes

        # Swin Transformer backbone
        swin = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
        layers = list(swin.features.children())

        self.encoders = nn.ModuleList([
            nn.Sequential(*layers[0:2]),  # Layer 1
            nn.Sequential(*layers[2:4]),  # Layer 2
            nn.Sequential(*layers[4:6]),  # Layer 3
            nn.Sequential(*layers[6:8])   # Layer 4
        ])

        self.layersize = self._calc_layersize()

        self.decoders = nn.ModuleList([
            PatchExpanding(self.layersize["layer4"][2]),
            PatchExpanding(self.layersize["layer3"][2]),
            PatchExpanding(self.layersize["layer2"][2]),
            Unpatch(self.layersize["layer1"][2], imgsize=(self.height, self.width))
        ])

        # 分類器 (仮に1x1 convで定義)
        self.classifier = nn.Conv2d(
            in_channels=self.layersize["layer1"][2],
            out_channels=num_classes,
            kernel_size=1
        )
        # self.classifier = nn.Linear(in_features=self.layersize["layer1"][2], out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)

        x = self.decoders[0](x, features[2])
        x = self.decoders[1](x, features[1])
        x = self.decoders[2](x, features[0])
        x = self.decoders[3](x)

        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.classifier(x)
        return x

    def _calc_layersize(self) -> dict:
        """
        中間層の出力サイズを推定
        """
        l0 = (3, self.height, self.width)
        l1_h = ((l0[1] - 4) // 4 + 1)
        l1_w = ((l0[2] - 4) // 4 + 1)
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

class PatchExpanding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm((dim // 2,))
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = rearrange(
            x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=2, p2=2, c=self.dim // 2
        )
        x = x.permute(0, 3, 1, 2)
        skip = skip.permute(0, 3, 1, 2)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class Unpatch(nn.Module):
    def __init__(self, dim: int, imgsize: tuple[int, int]):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.imgsize = imgsize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = rearrange(
            x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=4, p2=4, c=self.dim
        )
        x = x.permute(0, 3, 1, 2)

        if x.shape[2:] != self.imgsize:
            x = F.interpolate(x, size=self.imgsize, mode='bilinear', align_corners=False)

        x = x.permute(0, 2, 3, 1)
        return x

class TemporalCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(2, 0, 1)  # (N, B, C)

        out, _ = self.attn(x_flat, x_flat, x_flat)
        out = out.permute(1, 2, 0).view(B, C, H, W)
        return out


class SwinTransformerUnet_S(nn.Module):
    def __init__(self, imgsize: tuple[int, int], num_classes: int):
        super().__init__()
        self.imgsize = imgsize
        self.height, self.width = imgsize
        self.num_classes = num_classes

        swin = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
        layers = list(swin.features.children())

        self.encoders = nn.ModuleList([
            nn.Sequential(*layers[0:2]),
            nn.Sequential(*layers[2:4]),
            nn.Sequential(*layers[4:6]),
            nn.Sequential(*layers[6:8])
        ])

        self.layersize = self.calc_layersize()

        self.decoders = nn.ModuleList([
            #nn.Sequential(*self._create_decoder_block(layers[7])),
            PatchExpanding(self.layersize["layer4"][2]),
            nn.Sequential(*self._create_decoder_block(layers[5])),
            PatchExpanding(self.layersize["layer3"][2]),
            nn.Sequential(*self._create_decoder_block(layers[3])),
            PatchExpanding(self.layersize["layer2"][2]),
            nn.Sequential(*self._create_decoder_block(layers[1])),
            Unpatch(self.layersize["layer1"][2], imgsize=(self.height, self.width))
        ])

        self.classifier = nn.Conv2d(
            in_channels=self.layersize["layer1"][2],
            out_channels=num_classes,
            kernel_size=1
        )
        # self.classifier = nn.Linear(in_features=self.layersize["layer1"][2], out_features=num_classes)

    def forward(self, x: torch.Tensor):
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)

        # x = self.decoders[0](x)
        x = self.decoders[0](x, features[2])
        x = self.decoders[1](x)
        x = self.decoders[2](x, features[1])
        x = self.decoders[3](x)
        x = self.decoders[4](x, features[0])
        x = self.decoders[5](x)
        x = self.decoders[6](x)

        x = x.permute(0, 3, 1, 2)
        x = self.classifier(x)
        return x

    def calc_layersize(self) -> dict:
        l0 = (3, self.height, self.width)
        l1 = ((l0[1] - 4) // 4 + 1, (l0[2] - 4) // 4 + 1, 96)

        def down(s): return -(-s // 2)
        l2 = (down(l1[0]), down(l1[1]), l1[2] * 2)
        l3 = (down(l2[0]), down(l2[1]), l2[2] * 2)
        l4 = (down(l3[0]), down(l3[1]), l3[2] * 2)

        return {"input": l0, "layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4}

    def _create_decoder_block(self, layer):
        return [copy.deepcopy(m) for m in layer.children()]