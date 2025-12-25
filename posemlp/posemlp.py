import torch, torch.nn as nn
import numpy as np
import os
import sys
import cv2
import numpy as np
from typing import Any
from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from src.dataloader import PersonPositionDataset_D
from vitpose.vitpose import vitpose_inference

class Pose2PosMLP(nn.Module):
    def __init__(self, J=17, C=3, out_dim=2, hidden=256, use_mask=True, uncertainty=False):
        super().__init__()
        in_dim = J*C + (J if use_mask else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim if not uncertainty else out_dim*2)
        )
        self.uncertainty = uncertainty
        self.use_mask = use_mask

    def forward(self, kp, mask=None):
        # kp: (B, J, C)  例: root差分＋スケール正規化済み
        B, J, C = kp.shape
        x = kp.reshape(B, J*C)
        if self.use_mask:
            assert mask is not None
            x = torch.cat([x, mask.float()], dim=-1)  # (B, J*C + J)
        out = self.mlp(x)
        if self.uncertainty:
            mu, log_var = out.chunk(2, dim=-1)
            return mu, log_var
        return out
    

class PoseDataset(PersonPositionDataset_D):
    def __init__(self, root_dir:str, input_size:int=224, augmentation:Any=None, random_crop:bool=False, DEVICE:str=None):
        super(PoseDataset, self).__init__(root_dir, input_size, augmentation, random_crop)
        if DEVICE is None:
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.DEVICE = DEVICE
            
        # Detect humans in the image
        self.person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=self.DEVICE)

        # Detect keypoints for each person found
        self.vitpose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.vitpose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=self.DEVICE)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = cv2.imread(item['img_name'])
        pos_gt = np.array(item['pos'])
        bbox_pref = np.array(item['bbox'])        
        P = np.array(item['PerspectiveProjectionMatrix'])
        if self.augmentation:
            augmented = self.augmentation(image=image, keypoints=np.expand_dims(pos_gt,0), bboxes=np.expand_dims(bbox_pref,0))
            image = augmented["image"]
            pos_gt = augmented['keypoints'][0]
            bbox_pref = augmented['bboxes'][0].astype(np.int32)
            flip_applied = any(
                t.get("__class_fullname__", "").endswith("HorizontalFlip") and t.get("applied", False)
                for t in augmented["replay"]["transforms"]
            )
            if flip_applied:
                Hx = np.array([[-1, 0, image.shape[1] - 1],
                            [ 0, 1, 0],
                            [ 0, 0, 1]], dtype=float)
                P = Hx @ P

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if self.random_crop:
            image, bbox, side = self.random_square_crop(image, pos_gt, ref_bbox=bbox_pref)
        else:
            image, bbox, side = self.square_crop(image, pos_gt, bbox_pref)
        image = cv2.resize(image,(self.input_size, self.input_size))
        pos = np.array([self.input_size/side, self.input_size/side])*(pos_gt - bbox[0:2])

        P = np.array([[self.input_size/side,0,0],[0,self.input_size/side,0],[0,0,1]])@np.array([[1,0,-bbox[0]],[0,1,-bbox[1]],[0,0,1]])@P

        kp, _, _ = vitpose_inference(image, self.person_image_processor, self.person_model, self.vitpose_image_processor, self.vitpose_model)
        ind = np.argmax(kp.confidence.sum(axis=1))
        kp_vector = np.hstack([kp.xy[ind], kp.confidence[ind:ind+1].T])
        # kp_vector = np.hstack([kp.xy[0], kp.confidence[0:1].T])
        kp_vector, (root, scale) = self.normalize_pose(kp_vector)
        pos = (pos-root)/scale
                
        return kp_vector.astype(np.float32), pos.astype(np.float32), (root.astype(np.float32), scale.astype(np.float32))


    def normalize_pose(self, kp:np.ndarray)->tuple[np.ndarray,tuple[np.ndarray,np.ndarray]]:
        # kp: (J=17, C=3 [x,y,conf])
        root = kp[11:13,0:2].mean(axis=0) # center of l&r waist

        s = np.linalg.norm(kp[11,0:2] - root)
        kp_norm = kp.copy()
        kp_norm[:, 0:2] = (kp[:,0:2]-root)/s

        return kp_norm, (root, np.expand_dims(s, 0))


class PoseAnnotatedDataset(Dataset):
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

                for img in data['imgs']:
                    img['img_name'] = os.path.join(video_path, img['img_name'])

                self.samples += data['imgs']


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        kp, (root, scale) = self.normalize_pose(np.array(item['ViTPose']))
        pos_gt = np.array(item['pos'])
        pos_gt = (pos_gt-root)/scale
        return kp.astype(np.float32), pos_gt.astype(np.float32), (root, scale)

    def normalize_pose(self, kp):
        # kp: (J=17, C=3 [x,y,conf])
        root = kp[11:13,0:2].mean(axis=0) # center of l&r waist

        s = np.linalg.norm(kp[11,0:2] - root)
        kp_norm = kp.copy()
        kp_norm[:, 0:2] = (kp[:,0:2]-root)/s

        return kp_norm, (root, np.expand_dims(s, 0))


def abnormalize_pose(kp_norm:np.ndarray, root:np.ndarray, scale:np.ndarray)->np.ndarray:
    kp_xy = scale*kp_norm[:,0:2] + root
    kp_conf = kp_norm[:,2:]
    return np.concatenate([kp_xy, kp_conf], axis=-1)

def abnormalize_pose_batch(kp_norm: np.ndarray, root: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    kp_norm: (B, 17, 3)
    root:    (B, 2)
    scale:   (B, 1)
    returns: (B, 17, 2)
    """
    # XYの復元
    kp_xy = kp_norm[..., :2] * scale[:,  None] + root[:, None, :]  # (B,17,2)
    # Zはそのまま残す
    kp_conf = kp_norm[..., 2:]  # (B,17,1)
    # 結合して (B,17,3) に戻す
    return np.concatenate([kp_xy, kp_conf], axis=-1)