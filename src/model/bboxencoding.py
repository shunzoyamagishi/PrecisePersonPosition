import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BboxEncoding(nn.Module):
    def __init__(self, out_channel:int=8, device:str='cuda'):
        super().__init__()
        self.out_channel = out_channel
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.out_channel//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel//2, self.out_channel, 1),
            nn.GroupNorm(num_groups=8, num_channels=self.out_channel)
        )
        
    
    def forward(self, feature: torch.Tensor, bboxes: torch.Tensor, ) -> torch.Tensor:
        # Encoder
        x = self.encoding(bboxes, feature)
        x = self.encoder(x)
        x = torch.concatenate([x, feature], dim=1)
        return x
    

    def encoding(self, bboxes:torch.Tensor, feature:torch.Tensor)->torch.Tensor:
        """
        bboxes: Tensor [B,4] = (x1,y1,x2,y2)
        return: Tensor [B,1,H,W] where H=W=self.out_size
        """
        assert bboxes.ndim == 2 and bboxes.size(-1) == 4, f"bboxes must be [B,4], got {tuple(bboxes.shape)}"

        device = self.device
        dtype = feature.dtype
        B = bboxes.size(0)
        H = feature.size(2)
        W = feature.size(3)

        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        # center and half-size (broadcast-friendly: [B,1,1])
        xc = (x1 + x2) * 0.5
        yc = (y1 + y2) * 0.5
        xs = (x2 - x1) * 0.5
        ys = (y2 - y1) * 0.5

        xc = xc[:, None, None]
        yc = yc[:, None, None]
        xs = xs[:, None, None]
        ys = ys[:, None, None]

        # grid [1,H,W] -> broadcast to [B,H,W]
        x = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        y = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")  # [H,W]
        X = X[None, ...]  # [1,H,W]
        Y = Y[None, ...]  # [1,H,W]

        # rectangle SDF ingredients
        qx = torch.abs(X - xc) - xs
        qy = torch.abs(Y - yc) - ys

        # outside distance
        dx = torch.clamp(qx, min=0.0)
        dy = torch.clamp(qy, min=0.0)
        dist_out = torch.sqrt(dx * dx + dy * dy)

        # inside distance (distance to closest edge)
        dist_in = torch.minimum(xs - torch.abs(X - xc), ys - torch.abs(Y - yc))
        dist_in = torch.clamp(dist_in, min=0.0)

        inside = (qx <= 0) & (qy <= 0)
        S = torch.where(inside, dist_in, -dist_out)  # [B,H,W]

        # per-sample normalize so each map has max positive = 1
        max_pos = S.amax(dim=(1, 2), keepdim=True)  # [B,1,1]
        S = S / (max_pos + 1e-8)

        return torch.unsqueeze(S, dim=1)