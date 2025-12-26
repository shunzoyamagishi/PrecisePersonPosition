import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class SoftArgmaxLoss(nn.Module):
    def __init__(self, beta: float = 100.0):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        pred: [B,1,H,W]
        targets: [B,2] (x,y)
        """
        _, _, H, W = pred.shape
        pred_coords = self.soft_argmax_2d(pred, beta=self.beta)

        # 正規化 (0〜1)
        pred_coords_norm = torch.zeros_like(pred_coords)
        pred_coords_norm[:,0] = pred_coords[:,0] / W
        pred_coords_norm[:,1] = pred_coords[:,1] / H

        targets_norm = torch.zeros_like(targets)
        targets_norm[:,0] = targets[:,0] / W
        targets_norm[:,1] = targets[:,1] / H

        loss = self.mse(pred_coords, targets)
        return loss

    def soft_argmax_2d(self, heatmap: torch.Tensor, beta: float = 100.0):
        B, C, H, W = heatmap.shape

        # Flatten and softmax
        heatmap_flat = heatmap.view(B, -1)
        softmax = F.softmax(heatmap_flat * beta, dim=1)
        softmax = softmax.view(B, H, W)

        # coordinate grids
        xs = torch.linspace(0, W-1, W, device=heatmap.device)
        ys = torch.linspace(0, H-1, H, device=heatmap.device)

        exp_x = torch.sum(softmax * xs.view(1, 1, -1), dim=(1,2))
        exp_y = torch.sum(softmax * ys.view(1, -1, 1), dim=(1,2))

        coords = torch.stack([exp_x, exp_y], dim=1)
        return coords


class KLDivLoss_old(nn.Module):
    def __init__(self, std:float=5.0):
        super().__init__()
        self.gausianstd = std

    def forward(self, pred:torch.Tensor, targets:torch.Tensor,eps:float=1e-6) -> torch.Tensor:
        B = pred.size(0)
        target_map = self.generate_gaussian_heatmap(pred, targets)

        # reshape
        pred = pred.view(B, 1, -1)
        target_map = target_map.view(B, 1, -1)

        # log_softmax
        pred_logsoftmax = F.log_softmax(pred, dim=2)

        # 正規化して確率分布に
        target_probs = F.softmax(target_map, dim=2) + eps
        target_probs = target_probs / target_probs.sum(dim=2, keepdim=True)


        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(pred_logsoftmax, target_probs)
        return loss
    
    
    def generate_gaussian_heatmap(self, pred: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, 1, H, W]
        centers: [B, 2] (x, y)
        Returns:
            heatmap: [B, 1, H, W]
        """
        B, _, H, W = pred.shape
        sigma = self.gausianstd

        # (H, W) grid
        yy, xx = torch.meshgrid(
            torch.arange(H, device=pred.device).float(),
            torch.arange(W, device=pred.device).float(),
            indexing="ij"
        )
        # xx: [H, W], yy: [H, W]

        # output heatmap tensor
        heatmap = torch.zeros((B, 1, H, W), dtype=torch.float32, device=pred.device)

        for i in range(B):
            x0, y0 = centers[i].float()
            # Gaussian formula over the grid
            distance_sq = (xx - x0)**2 + (yy - y0)**2
            gaussian = torch.exp(-distance_sq / (2 * sigma**2))
            heatmap[i, 0] = gaussian

        return heatmap


class CrossEntropyLoss(nn.Module):
    def __init__(self, std: float = 5.0, alpha: float = 10.0):
        super().__init__()
        self.gausianstd = std
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        pred: (B, 1, H, W) logits
        targets: (B, 2) - center coordinates (x, y)
        """
        B = pred.size(0)
        target_map = self.generate_gaussian_heatmap(pred, targets)

        # reshape
        pred_flat = pred.view(B, -1)           # (B, H*W)
        target_flat = target_map.view(B, -1)   # (B, H*W)

        # Softmaxで確率
        pred_probs = F.softmax(pred_flat, dim=1)
        log_pred_probs = torch.log(pred_probs + eps)

        # targetを正規化
        target_probs = target_flat / (target_flat.sum(dim=1, keepdim=True) + eps)

        # 重み付きクロスエントロピー
        weights = 1.0 + self.alpha * target_probs
        loss = - (weights * target_probs * log_pred_probs).sum(dim=1)

        return loss.mean()   
       
    def generate_gaussian_heatmap(self, pred: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, 1, H, W]
        centers: [B, 2] (x, y)
        Returns:
            heatmap: [B, 1, H, W]
        """
        B, _, H, W = pred.shape
        sigma = self.gausianstd

        # Create coordinate grid
        yy, xx = torch.meshgrid(
            torch.arange(H, device=pred.device).float(),
            torch.arange(W, device=pred.device).float(),
            indexing="ij"
        )

        heatmap = torch.zeros((B, 1, H, W), dtype=torch.float32, device=pred.device)

        for i in range(B):
            x0, y0 = centers[i].float()
            dist_sq = (xx - x0) ** 2 + (yy - y0) ** 2
            gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
            heatmap[i, 0] = gaussian

        return heatmap


class MSELoss(nn.Module):
    def __init__(self, std: float = 5.0, alpha: float = 10.0):
        super().__init__()
        self.gausianstd = std
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        pred: (B, 1, H, W) logits
        targets: (B, 2) - center coordinates (x, y)
        """
        B = pred.size(0)
        target_map = self.generate_gaussian_heatmap(pred, targets)

        # reshape
        pred_flat = pred.view(B, -1)           # (B, H*W)
        target_flat = target_map.view(B, -1)   # (B, H*W)

        # Softmaxで確率
        pred_probs = F.softmax(pred_flat, dim=1)

        # targetを正規化
        target_probs = target_flat / (target_flat.sum(dim=1, keepdim=True) + eps)

        loss = F.mse_loss(pred_probs, target_probs)

        return loss  
       
    def generate_gaussian_heatmap(self, pred: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, 1, H, W]
        centers: [B, 2] (x, y)
        Returns:
            heatmap: [B, 1, H, W]
        """
        B, _, H, W = pred.shape
        sigma = self.gausianstd

        # Create coordinate grid
        yy, xx = torch.meshgrid(
            torch.arange(H, device=pred.device).float(),
            torch.arange(W, device=pred.device).float(),
            indexing="ij"
        )

        heatmap = torch.zeros((B, 1, H, W), dtype=torch.float32, device=pred.device)

        for i in range(B):
            x0, y0 = centers[i].float()
            dist_sq = (xx - x0) ** 2 + (yy - y0) ** 2
            gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
            heatmap[i, 0] = gaussian

        return heatmap
    

class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred:torch.Tensor, target_map:torch.Tensor,eps:float=1e-6) -> torch.Tensor:
        B = pred.size(0)
        
        # reshape
        pred = pred.reshape(B, 1, -1)
        target_map = target_map.reshape(B, 1, -1)

        # log_softmax
        pred_logsoftmax = F.log_softmax(pred, dim=2)

        # 正規化して確率分布に
        target_probs = F.softmax(target_map, dim=2) + eps
        target_probs = target_probs / target_probs.sum(dim=2, keepdim=True)


        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(pred_logsoftmax, target_probs)
        return loss