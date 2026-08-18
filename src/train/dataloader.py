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
   
    
class PersonPositionDataset(Dataset):
    def __init__(self, root_dir:str, input_size:int=224, target_std:float=5.0, augmentation:Any=None, random_crop:bool=True):
        self.root_dir = root_dir
        self.input_size = input_size
        self.target_std = target_std
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
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            filenames.sort()
            if "data.json" not in filenames:
                continue

            video_path = dirpath
            json_path = os.path.join(video_path, "data.json")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "imgs" not in data:
                raise KeyError(f"`imgs` not found in {json_path}")
            cam_param = data.get("Camera Parameter", {})
            if cam_param.get("Perspective Projection Matrix", None) is None:
                raise KeyError(f"`Camera Parameter/Perspective Projection Matrix` not found in {json_path}")

            for img in data['imgs']:
                img['img_name'] = os.path.join(video_path, img['img_name'])
                img['PerspectiveProjectionMatrix'] = data["Camera Parameter"]["Perspective Projection Matrix"]

            self.samples += data['imgs']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = cv2.imread(item['img_name'])
        pos_gt = np.array(item['pos'])
        square = np.array(item['square'])
        bbox_pref = np.array(item['bbox'])         
        P = np.array(item['PerspectiveProjectionMatrix'])
        if self.augmentation:
            augmented = self.augmentation(image=image, keypoints=np.expand_dims(pos_gt,0), bboxes=np.expand_dims(square,0))
            # augmented = self.augmentation(image=image, keypoints=np.expand_dims(pos_gt,0), bboxes=np.stack([square, bbox_pref], axis=0))
            if len(augmented['keypoints']) > 0 and len(augmented['bboxes']) > 0:
                image = augmented["image"]
                pos_gt = augmented['keypoints'][0]
                square = augmented['bboxes'][0].astype(np.int32)
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
            image, square, side = self.random_square_crop(image, pos_gt, ref_bbox=square)
        else:
            image, square, side = self.square_crop(image, pos_gt, square)
        image = cv2.resize(image,(self.input_size, self.input_size))
        pos = np.array([self.input_size/side, self.input_size/side])*(pos_gt - square[0:2])

        P = np.array([[self.input_size/side,0,0],[0,self.input_size/side,0],[0,0,1]])@np.array([[1,0,-square[0]],[0,1,-square[1]],[0,0,1]])@P

        # heatmap in image coordinate
        hmap1 = self.generate_gaussian_heatmap(shape=(self.input_size,self.input_size), centers=pos, sigma=self.target_std)
        
        # heatmap in perspective
        gaussianmap_size = 400
        H_field2image = P[:,[0,1,3]]
        H_gaussian2field = np.array([[2/gaussianmap_size, 0, -1], [0, 2/gaussianmap_size, -1], [0,0,1]], dtype=np.float32) 
        r = self.local_radius_for_target_area(H_field2image@H_gaussian2field, center=(0,0), A_tar=self.target_std*self.target_std*np.pi)

        hmap2 = self.generate_gaussian_heatmap(shape=(gaussianmap_size,gaussianmap_size), sigma=r)
        hmap2 = cv2.warpPerspective(
            hmap2, H_field2image@H_gaussian2field, (self.input_size,self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0)
        )
        hmap2 /= (hmap2.max() + 1e-8)
        heatmap = np.stack([hmap1,hmap2]) 
        
        image = image / 255.0
        image = self.transform({'image': image})['image']
                
        return torch.from_numpy(image).clone(), torch.from_numpy(pos.astype(np.float32)).clone(), torch.from_numpy(heatmap.astype(np.float32)).clone(), torch.from_numpy(P.astype(np.float32)).clone()
    

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

        margin = int(min(point[0]-bbox[0], point[1]-bbox[1], bbox[2]-point[0], bbox[3]-point[1]))
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
        margin_ratio:float=0.2,       # The ratio of the minimum distances from a point on the side to each edge
        margin_px_min:int=20,         # Leave at least this many pixels between them (if there isn't enough space, they will shrink).
        bottom_bias_gamma:float=3.0,  # Downside Bias Strength: 1 = Uniform; the higher the value, the stronger the downside bias
        max_tries:int=32
    )->tuple[np.ndarray, np.ndarray, int]:

        """
        Returns a square crop that definitely includes the point and is positioned as low as possible.
        Return value: np.array([x0, y0, x1, y1]) The bottom-right corner is excluded (x1 = x0 + side)
        """
        if rng is None:
            rng = random
        h, w = image.shape[:2]
        x, y = map(int, map(round, point))
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))

        # --- side candidate ---
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

        for _ in range(max_tries):
            side = rng.randint(lo, hi)

            m_wish = max(int(round(side * margin_ratio)), int(margin_px_min))
            m_side_max = (side - 1) // 2
            m_border_x = min(x, (w - 1) - x)
            m_border_y = min(y, (h - 1) - y)
            m = min(m_wish, m_side_max, m_border_x, m_border_y)
            if m < 0:
                continue

            x0_min = max(0, x - side + 1 + m)
            x0_max = min(w - side, x - m)
            if x0_min > x0_max:
                continue
            x0 = rng.randint(x0_min, x0_max)

            r_min = m / side
            r_max = 1.0 - (m / side)
            if r_min > r_max:
                continue

            if bottom_bias_gamma <= 1.0:
                u = rng.random()
            else:
                u = rng.random() ** (1.0 / float(bottom_bias_gamma))
            r = r_min + (r_max - r_min) * u

            y0_cont = y - r * (side - 1)
            y0_min = max(0, y - side + 1 + m)
            y0_max = min(h - side, y - m)
            if y0_min > y0_max:
                continue

            y0 = int(round(y0_cont))
            y0 = max(y0_min, min(y0, y0_max))

            bbox = np.array([x0, y0, x0 + side, y0 + side], dtype=np.int32)

            lx = x - x0
            rx = (x0 + side - 1) - x
            uy = y - y0
            dy = (y0 + side - 1) - y
            if (bbox[0] <= x <= bbox[2]-1) and (bbox[1] <= y <= bbox[3]-1) and \
            min(lx, rx, uy, dy) >= m:
                return image[bbox[1]:bbox[3], bbox[0]:bbox[2]], bbox, side

        side = min(max(lo, 1), global_max)
        x0_min = max(0, x - side + 1)
        x0_max = min(w - side, x)
        y0_min = max(0, y - side + 1)
        y0_max = min(h - side, y)
        x0 = (x0_min + x0_max) // 2
        y0 = min(y0_max, max(y0_min, y - int(0.8 * (side - 1))))

        bbox = np.array([x0, y0, x0 + side, y0 + side], dtype=np.int32)
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]], bbox, side


    def generate_gaussian_heatmap(self, shape:tuple=(400,400), centers:np.ndarray=None, sigma:float=40.0) -> np.ndarray:
        """
        pred: [B, 1, H, W]
        centers: [B, 2] (x, y)
        sigma: float
        Returns:
            heatmap: [B, 1, H, W]
        """
        H, W = shape
        
        if centers is None:
            centers = np.array([W/2, H/2])

        # (H, W) grid
        yy, xx = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij"
        )
        
        x0, y0 = centers.astype(np.float32)
        distance_sq = (xx - x0) ** 2 + (yy - y0) ** 2
        gaussian = np.exp(-distance_sq / (2 * sigma ** 2))

        return gaussian
    
    def local_radius_for_target_area(self, H: np.ndarray, center:tuple[float,float]=(200,200), A_tar:float=100.0) -> float:
        """
        Local approximation: Returns the radius r of a circle centered at (x0, y0) on the original image.
        Determine r such that the area after projection by homography H is A_tar.

        H: 3x3 homography (any scale is acceptable)
        (x0, y0): Center of the circle (in the original image coordinate system)
        A_tar: Target area after projection (>0)

        Return value: Radius r
        """
        x0, y0 = center
        H = np.asarray(H, dtype=float)
        assert H.shape == (3, 3), "H must be 3x3"
        if A_tar <= 0:
            raise ValueError("A_tar must be positive.")

        # Jacobianの面積スケール s(x0,y0) = |det(H)| / |h31*x0 + h32*y0 + h33|^3
        detH = np.linalg.det(H)
        w = H[2,0]*x0 + H[2,1]*y0 + H[2,2]

        if np.isclose(w, 0.0):
            raise ZeroDivisionError("The local approximation formula diverges when w ≈ 0 (near the horizon). Please shift the center position.")

        s = abs(detH) / (abs(w)**3)

        if s <= 0:
            raise ValueError("An invalid scale s was calculated (H may be close to a singularity).")

        # r = sqrt( A_tar / (pi * s) )
        r = np.sqrt(A_tar / (np.pi * s))
        return float(r)
    

class TestDataset(PersonPositionDataset):
    def __init__(self, root_dir:str, input_size:int=224):
        self.root_dir = root_dir
        self.input_size = input_size
        self.samples = []
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            filenames.sort()
            if "data.json" not in filenames:
                continue

            video_path = dirpath
            json_path = os.path.join(video_path, "data.json")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "imgs" not in data:
                raise KeyError(f"`imgs` not found in {json_path}")
            cam_param = data.get("Camera Parameter", {})
            if cam_param.get("Perspective Projection Matrix", None) is None:
                raise KeyError(f"`Camera Parameter/Perspective Projection Matrix` not found in {json_path}")

            for img in data['imgs']:
                    img['img_name'] = os.path.join(video_path, img['img_name'])
                    img['PerspectiveProjectionMatrix'] = data["Camera Parameter"]["Perspective Projection Matrix"]
                    img['distancefromperson'] = data["Camera Parameter"]["Distance from Player"]
                    img['tiltangle'] = data["Camera Parameter"]["Tilt wrt Field"]
            self.samples += data['imgs']


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx)->tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, int, int]]:
        item = self.samples[idx]
        image = cv2.imread(item['img_name'])
        pos_gt = np.array(item['pos'])
        P = np.array(item['PerspectiveProjectionMatrix'])
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
        return image, pos_gt, P, (item['img_name'], item['distancefromperson'], item['tiltangle'])