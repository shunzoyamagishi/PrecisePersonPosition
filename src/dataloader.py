import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import cv2
import numpy as np
import random
from typing import Any
from Depth_Anything_V2.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

class FrameDataset(Dataset):
    def __init__(self, root_dir:str, input_size:tuple=(224,224), augmentation=None):
        """
        root_dir: 動画ごとのフォルダが並んでいるルートディレクトリ
        transform: 画像に適用する変換
        """
        self.root_dir = root_dir
        self.input_w, self.input_h = input_size
        self.augmentation = augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.samples = [] 

        # フォルダごとに探索
        for motion_folder in sorted(os.listdir(root_dir)):
            motion_path = os.path.join(root_dir, motion_folder)
            for video_folder in sorted(os.listdir(motion_path)):
                video_path = os.path.join(motion_path, video_folder)
                with open(os.path.join(video_path, "data.json"), 'r') as json_data:
                    data = json.load(json_data)

                for img in data['imgs']:
                    img['img_name'] = os.path.join(video_path, img['img_name'])

                self.samples += data['imgs']


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        item = self.samples[idx]
        # image = Image.open(item['img_name']).convert('RGB')
        image = cv2.imread(item['img_name'])
        pos = np.array(item['pos'])
        bbox = np.array(item['bbox'])
        if self.augmentation:
            augmented = self.augmentation(image=image, keypoints=np.expand_dims(pos,0), bboxes=np.expand_dims(bbox,0))
            # image = Image.fromarray(cv2.cvtColor(augmented["image"],cv2.COLOR_BGR2RGB))
            image = augmented["image"]
            pos = augmented['keypoints'][0]
            bbox = augmented['bboxes'][0].astype(np.int32)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        bbox, (bbox_w, bbox_h) = self.pad_image(bbox)
        image = self.crop_with_padding(image, bbox)
        image = cv2.resize(image,(self.input_w, self.input_h))
        pos = np.array([self.input_w/bbox_w, self.input_h/bbox_h])*(pos - bbox[0:2])
                
        return self.transform(image), torch.from_numpy(pos.astype(np.float32)).clone()


    def pad_image(self, bbox:list)->list:
        width = bbox[2]-bbox[0]
        height = bbox[3]-bbox[1]
        
        if width > height:
            bbox[3] += width-height
        elif height > width:
            bbox[2] += height-width

        return bbox, (bbox[2]-bbox[0], bbox[3]-bbox[1])
    

    def crop_with_padding(self, image:np.ndarray, crop_rect:np.ndarray)->np.ndarray:
        
        xmin, ymin, xmax, ymax = crop_rect
        crop_width = xmax - xmin
        crop_height = ymax - ymin

        if len(image.shape) == 2:
            channels = 1
        else:
            channels = image.shape[2]
        cropped = np.zeros((crop_height, crop_width, channels), dtype=image.dtype) if channels > 1 else np.zeros((crop_height, crop_width), dtype=image.dtype)

        img_h, img_w = image.shape[:2]
        x1_src = max(xmin, 0)
        y1_src = max(ymin, 0)
        x2_src = min(xmax, img_w)
        y2_src = min(ymax, img_h)

        x1_dst = x1_src - xmin
        y1_dst = y1_src - ymin
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)

        cropped[y1_dst:y2_dst, x1_dst:x2_dst] = image[y1_src:y2_src, x1_src:x2_src]

        return cropped
    

class DepthDataset(Dataset):
    def __init__(self, root_dir:str, input_size:int=224, augmentation:Any=None, random_crop:bool=True):
        """
        root_dir: 動画ごとのフォルダが並んでいるルートディレクトリ
        transform: 画像に適用する変換
        """
        self.root_dir = root_dir
        self.input_size = input_size
        self.augmentation = augmentation
        self.random_crop = random_crop
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        self.samples = [] 
        

        # フォルダごとに探索
        for motion_folder in sorted(os.listdir(root_dir)):
            motion_path = os.path.join(root_dir, motion_folder)
            for video_folder in sorted(os.listdir(motion_path)):
                video_path = os.path.join(motion_path, video_folder)
                with open(os.path.join(video_path, "data.json"), 'r') as json_data:
                    data = json.load(json_data)

                for img in data['imgs']:
                    img['img_name'] = os.path.join(video_path, img['img_name'])

                self.samples += data['imgs']


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = cv2.imread(item['img_name'])
        pos_gt = np.array(item['pos'])
        bbox_pref = np.array(item['bbox'])
        if self.augmentation:
            augmented = self.augmentation(image=image, keypoints=np.expand_dims(pos_gt,0), bboxes=np.expand_dims(bbox_pref,0))
            image = augmented["image"]
            pos_gt = augmented['keypoints'][0]
            bbox_pref = augmented['bboxes'][0].astype(np.int32)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if self.random_crop:
            image, bbox, side = self.random_square_crop(image, pos_gt, ref_bbox=bbox_pref)
        else:
            image, bbox, side = self.square_crop(image, pos_gt, bbox_pref)
        image = cv2.resize(image,(self.input_size, self.input_size))
        pos = np.array([self.input_size/side, self.input_size/side])*(pos_gt - bbox[0:2])
        
        image = image / 255.0
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image)
                
        return image, torch.from_numpy(pos.astype(np.float32)).clone()


    def square_crop(self, image:np.ndarray, point:np.ndarray, bbox:list)->tuple[np.ndarray, np.ndarray, int]:
        h, w = image.shape[:2]
        bw = bbox[2]-bbox[0]
        bh = bbox[3]-bbox[1]
        
        if bw > bh:
            bbox[1] -= (bw-bh)//2
            bbox[3] += (bw-bh)//2 + (bw-bh)%2
        elif bh > bw:
            bbox[0] -= (bh-bw)//2
            bbox[2] += (bh-bw)//2 + (bh-bw)%2

        margin = min(point[0]-bbox[0], point[1]-bbox[1], bbox[2]-point[0], bbox[3]-point[1])
        pad = random.randint(-2*margin//3, 2*margin//3)
        bbox[0] -= pad
        bbox[1] -= pad
        bbox[2] += pad
        bbox[3] += pad

        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(w, bbox[2])
        bbox[3] = min(h, bbox[3])

        new_w = bbox[2] - bbox[0]
        new_h = bbox[3] - bbox[1]
        side = min(new_w, new_h)

        bbox[2] = bbox[0] + side
        bbox[3] = bbox[1] + side

        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]], bbox, side
    

    def random_square_crop(
        self,
        image:np.ndarray,
        point:np.ndarray,                  # (x, y)
        ref_bbox:np.ndarray=None,          # あれば side の目安に使用
        rng:Any=random,
        margin_ratio:float=0.2,       # side に対する点から各辺までの最小距離の比率
        margin_px_min:int=20,          # 最低でもこの画素は離す（上限が足りなければ縮む）
        bottom_bias_gamma:float=3.0,  # 下寄りバイアス強度: 1=一様, 大きいほど下寄り
        max_tries:int=32
    )->tuple[np.ndarray, np.ndarray, int]:

        """
        点を必ず含み、点ができるだけ下方に寄りやすい正方形クロップを返す。
        返り値: np.array([x0, y0, x1, y1]) 右下は非包含 (x1=x0+side)
        """
        if rng is None:
            rng = random
        h, w = image.shape[:2]
        x, y = map(int, map(round, point))
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))

        # --- side 候補レンジ ---
        if ref_bbox is not None:
            ref_bbox = np.asarray(ref_bbox)
            ref_h = int(ref_bbox[3] - ref_bbox[1])
            pref_lo = max(1, int(np.floor(ref_h * 1)))
            pref_hi = max(1, int(np.ceil (ref_h * 4/3)))
        else:
            pref_lo, pref_hi = 16, min(w, h)

        global_max = min(w, h)
        lo = max(1, min(pref_lo, global_max))
        hi = max(lo, min(pref_hi, global_max))

        # --- 試行ループ ---
        for _ in range(max_tries):
            side = rng.randint(lo, hi)

            # 希望マージン
            m_wish = max(int(round(side * margin_ratio)), int(margin_px_min))
            m_side_max = (side - 1) // 2
            m_border_x = min(x, (w - 1) - x)
            m_border_y = min(y, (h - 1) - y)
            m = min(m_wish, m_side_max, m_border_x, m_border_y)
            if m < 0:
                continue  # この side では無理

            # x0（横）は一様にサンプル（マージン満たす範囲）
            x0_min = max(0, x - side + 1 + m)
            x0_max = min(w - side, x - m)
            if x0_min > x0_max:
                continue
            x0 = rng.randint(x0_min, x0_max)

            # y0（縦）は "下寄り" にサンプル
            # r = (y - y0)/(side-1) ∈ [r_min, r_max] で、rが大きいほど点は下寄り
            r_min = m / side
            r_max = 1.0 - (m / side)
            if r_min > r_max:
                continue

            if bottom_bias_gamma <= 1.0:
                u = rng.random()  # 一様
            else:
                u = rng.random() ** (1.0 / float(bottom_bias_gamma))  # 1 に寄る
            r = r_min + (r_max - r_min) * u

            # 連続値から y0 を計算して整数化
            y0_cont = y - r * (side - 1)
            y0_min = max(0, y - side + 1 + m)
            y0_max = min(h - side, y - m)
            if y0_min > y0_max:
                continue

            y0 = int(round(y0_cont))
            # 可行範囲にクリップ
            y0 = max(y0_min, min(y0, y0_max))

            bbox = np.array([x0, y0, x0 + side, y0 + side], dtype=np.int32)

            # 最終チェック（点が内側 & マージン満たす）
            lx = x - x0
            rx = (x0 + side - 1) - x
            uy = y - y0
            dy = (y0 + side - 1) - y
            if (bbox[0] <= x <= bbox[2]-1) and (bbox[1] <= y <= bbox[3]-1) and \
            min(lx, rx, uy, dy) >= m:
                return image[bbox[1]:bbox[3], bbox[0]:bbox[2]], bbox, side

        # フォールバック（マージン拘らず点だけは含む）
        side = min(max(lo, 1), global_max)
        x0_min = max(0, x - side + 1)
        x0_max = min(w - side, x)
        y0_min = max(0, y - side + 1)
        y0_max = min(h - side, y)
        x0 = (x0_min + x0_max) // 2
        y0 = min(y0_max, max(y0_min, y - int(0.8 * (side - 1))))  # なるべく下寄りで救済

        bbox = np.array([x0, y0, x0 + side, y0 + side], dtype=np.int32)
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]], bbox, side


class PersonPositionDataset_Full(DepthDataset):
    def __init__(self, root_dir:str, input_size:int=224, augmentation:Any=None, random_crop:bool=True):
        self.root_dir = root_dir
        self.input_size = input_size
        self.augmentation = augmentation
        self.random_crop = random_crop
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        self.samples = [] 
        
        for base_folder in ["train", "val", "test"]:
            base_path = os.path.join(root_dir, base_folder)
            for motion_folder in sorted(os.listdir(base_path)):
                motion_path = os.path.join(base_path, motion_folder)
                for video_folder in sorted(os.listdir(motion_path)):
                    video_path = os.path.join(motion_path, video_folder)
                    with open(os.path.join(video_path, "data.json"), 'r') as json_data:
                        data = json.load(json_data)

                    for img in data['imgs']:
                        img['img_name'] = os.path.join(video_path, img['img_name'])

                    self.samples += data['imgs']

    
    def __getitem__(self, idx):
        item = self.samples[idx]
        # image = Image.open(item['img_name']).convert('RGB')
        image = cv2.imread(item['img_name'])
        pos_gt = np.array(item['pos'])
        bbox_pref = np.array(item['bbox'])
        if self.augmentation:
            augmented = self.augmentation(image=image, keypoints=np.expand_dims(pos_gt,0), bboxes=np.expand_dims(bbox_pref,0))
            image = augmented["image"]
            pos_gt = augmented['keypoints'][0]
            bbox_pref = augmented['bboxes'][0].astype(np.int32)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if self.random_crop:
            image, bbox, side = self.random_square_crop(image, pos_gt, ref_bbox=bbox_pref)
        else:
            image, bbox, side = self.square_crop(image, pos_gt, bbox_pref)
        image = cv2.resize(image,(self.input_size, self.input_size))
        pos = np.array([self.input_size/side, self.input_size/side])*(pos_gt - bbox[0:2])

        image = image / 255.0
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image)
                
        return image, torch.from_numpy(pos.astype(np.float32)).clone()
    

    