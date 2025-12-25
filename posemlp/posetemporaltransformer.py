import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple
import os
import json
import numpy as np

# --- Sinusoidal Positional Encoding（動的にTへ合わせて生成） ---
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        B, T, D = x.shape
        device = x.device
        pos = torch.arange(T, device=device).float().unsqueeze(1)              # [T,1]
        i = torch.arange(D, device=device).float().unsqueeze(0)                # [1,D]
        div = torch.exp(-(i//2)*torch.log(torch.tensor(10000.0, device=device))/D)
        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(pos * div[:, 0::2])
        pe[:, 1::2] = torch.cos(pos * div[:, 1::2])
        return x + pe.unsqueeze(0)  # [1,T,D] -> broadcast

# --- 1層ぶんのEncoder（注意重みを保持して返す） ---
class TemporalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                       # [B,T,D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B,T] True=PAD
        attn_mask: Optional[torch.Tensor] = None,         # [T,T] (未来抑制などに使うなら)
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Multi-head self-attention（時間方向）
        attn_out, attn_w = self.mha(
            x, x, x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=False  # -> [B, nhead, T, T]
        )
        x = self.ln1(x + self.dropout1(attn_out))
        x2 = self.ff(x)
        x = self.ln2(x + self.dropout2(x2))
        return x, attn_w if need_weights else None

# --- 全体モデル ---
class PoseToPointTransformer(nn.Module):
    """
    Input : [B, T, 17, 3]  (最後の3は(x,y,score)想定。use_conf=Falseなら(x,y)のみ)
    Output: [B, T, 1, 2]
    """
    def __init__(
        self,
        use_conf: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_conf = use_conf
        in_feat = 17 * (3 if use_conf else 2)

        self.input_norm = nn.LayerNorm(in_feat)
        self.embed = nn.Linear(in_feat, d_model)
        self.pe = SinusoidalPE(d_model)

        self.layers = nn.ModuleList([
            TemporalEncoderLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])

        # 時刻ごとの2D点を出すヘッド
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2)
        )

    def _make_key_padding_mask(self, lengths: torch.Tensor, T: int) -> torch.Tensor:
        # True が PAD（MultiheadAttentionの仕様）
        device = lengths.device
        ar = torch.arange(T, device=device).unsqueeze(0)  # [1,T]
        return ar >= lengths.unsqueeze(1)                 # [B,T]

    def forward(
        self,
        x: torch.Tensor,                       # [B,T,17,3] or [B,T,17,2]
        lengths: Optional[torch.Tensor] = None,# [B] 実長（無ければ全長）
        return_attn: bool = False
    ):
        B, T, J, C = x.shape
        assert J == 17 and C >= 2
        if not self.use_conf and C == 3:
            x = x[..., :2]

        x = x.reshape(B, T, -1)                # [B,T,in_feat]
        x = self.input_norm(x)
        x = self.embed(x)                      # [B,T,D]
        x = self.pe(x)                         # 位置エンコーディング加算

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self._make_key_padding_mask(lengths, T)  # [B,T]

        attn_list: List[torch.Tensor] = []
        for layer in self.layers:
            x, attn_w = layer(x, key_padding_mask=key_padding_mask, need_weights=return_attn)
            if return_attn:
                # attn_w: [B, nhead, T, T]
                attn_list.append(attn_w)

        y = self.head(x)           # [B,T,2]
        y = y.unsqueeze(2)         # [B,T,1,2]
        return (y, attn_list) if return_attn else y



class PoseDataset_Temporal(Dataset):
    def __init__(self, root_dir:str):
        """
        root_dir: 動画ごとのフォルダが並んでいるルートディレクトリ
        transform: 画像に適用する変換
        """
        self.root_dir = root_dir
        self.samples = []


        # フォルダごとに探索
        for motion_folder in sorted(os.listdir(root_dir)):
            motion_path = os.path.join(root_dir, motion_folder)
            for video_folder in sorted(os.listdir(motion_path)):
                video_path = os.path.join(motion_path, video_folder)
                with open(os.path.join(video_path, "data.json"), 'r') as json_data:
                    data = json.load(json_data)

                data['imgs'] = [
                    img for img in data['imgs']
                    if img.get("ViTPose") is not None
                ]
                pose = []
                pos = []
                for img in data['imgs']:
                    pose.append(img['ViTPose'])
                    pos.append(img['pos'])
                
                self.samples.append({'kpt':pose, 'pos':pos})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
       
        kp, (root, scale) = self.normalize_pose_batch(np.array(item['kpt']))
        pos_gt = np.array(item['pos'])
        pos_gt = (pos_gt-root)/scale
        pos_gt = np.expand_dims(pos_gt, axis=1)
        
        return kp.astype(np.float32), pos_gt.astype(np.float32), (root.astype(np.float32), scale.astype(np.float32))

    def normalize_pose_batch(self, kp: np.ndarray):
        """
        kp: (B, J=17, C=3)  ... [x,y,conf] 形式
        return:
            kp_norm: (B, J, 3)
            root:    (B, 2)
            scale:   (B, 1)
        """
        # 腰（左右 11,12 番目）の中心を root に
        root = kp[:, 11:13, 0:2].mean(axis=1)    # (B,2)

        # scale を左腰(11)と root の距離で定義
        scale = np.linalg.norm(kp[:, 11, 0:2] - root, axis=1, keepdims=True)  # (B,1)

        # 正規化
        kp_norm = kp.copy()
        kp_norm[:, :, 0:2] = (kp[:, :, 0:2] - root[:, None, :]) / scale[:, None, :]

        return kp_norm, (root, scale)


def collate_fn(batch):
    # batch: list of (kp:[Ti,17,3], pos:[Ti,1,2], (root:[Ti,2], scale:[Ti,1]))
    kps, poss, metas = zip(*batch)
    roots, scales = zip(*metas)

    lengths = [kp.shape[0] for kp in kps]
    max_len = max(lengths)

    pad_kps, pad_pos, pad_root, pad_scale = [], [], [], []
    for kp, pos, root, scale in zip(kps, poss, roots, scales):
        Ti = kp.shape[0]
        pad_t = max_len - Ti

        if pad_t > 0:
            kp = np.concatenate([kp,    np.zeros((pad_t, kp.shape[1],    kp.shape[2]),    dtype=kp.dtype)],   axis=0)
            pos = np.concatenate([pos,  np.zeros((pad_t, pos.shape[1],   pos.shape[2]),   dtype=pos.dtype)],  axis=0)
            root = np.concatenate([root,np.zeros((pad_t, root.shape[1]), dtype=root.dtype)],                 axis=0)
            scale= np.concatenate([scale,np.zeros((pad_t, scale.shape[1]),dtype=scale.dtype)],               axis=0)

        pad_kps.append(kp)
        pad_pos.append(pos)
        pad_root.append(root)
        pad_scale.append(scale)

    pad_kps = torch.from_numpy(np.stack(pad_kps, axis=0))    # [B,T,17,3]
    pad_pos = torch.from_numpy(np.stack(pad_pos, axis=0))    # [B,T,1,2]
    pad_root= torch.from_numpy(np.stack(pad_root, axis=0))   # [B,T,2]
    pad_scale=torch.from_numpy(np.stack(pad_scale, axis=0))  # [B,T,1]
    lengths = torch.tensor(lengths, dtype=torch.long)        # [B]

    # True=valid のマスク（損失/評価で使用）
    B, T = pad_kps.shape[:2]
    valid = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1))  # [B,T] bool
    mask = valid.unsqueeze(-1).unsqueeze(-1)                        # [B,T,1,1]

    return pad_kps, pad_pos, pad_root, pad_scale, lengths, mask



@torch.no_grad()
def abnormalize_points(pred_norm, lengths, metas):
    """
    pred_norm: [B,T,1,2]  正規化座標の出力
    lengths:  [B]
    metas:    list of (root:[2], scale:float)
    return:   list of [Ti,1,2] の torch.Tensor をバッチごとに返す（可変長のためリスト）
    """
    B, T = pred_norm.shape[:2]
    outs = []
    for b in range(B):
        Ti = int(lengths[b].item())
        root, scale = metas[b]
        root_t = torch.as_tensor(root, device=pred_norm.device, dtype=pred_norm.dtype)  # [2]
        scale_t = torch.as_tensor(scale, device=pred_norm.device, dtype=pred_norm.dtype)  # []

        denorm = pred_norm[b, :Ti] * scale_t + root_t  # [Ti,1,2] （broadcast）
        outs.append(denorm)
    return outs
