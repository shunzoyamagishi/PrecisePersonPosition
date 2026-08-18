import cv2
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import supervision as sv
from typing import Tuple, Optional

from src.model import PositionEstimator
from src.utils import deep_merge_dict
from persondetection import PersonDetectionPipeline


class PositionEstimatorPipeline:
    def __init__(self, cfg: dict | None = None, device: str = "cuda"):
        default_cfg = {
            "inference":{
                "persondetection":{
                    "detector": "RTDETR",
                    "detector_threshold": 0.6,
                },
                "input_padding_bbox": 2.0,
            },
            "positionestimation":{
                "DepthAnythingV2_encoder": "vits",
                "in_size": 224,
                "pretrain_model": "checkpoints/depthhead.pt"
            },  
        }
        if cfg is None:
            self.cfg = default_cfg
        else:
            self.cfg = deep_merge_dict(default_cfg, cfg)

        self.detect_pipeline = PersonDetectionPipeline(cfg=self.cfg["inference"]["persondetection"], device=device)
        self.positionestimator = PositionEstimator(cfg=self.cfg["positionestimation"], device=device)
        self.cfg["inference"]["persondetection"] = deep_merge_dict(self.cfg["inference"]["persondetection"], self.detect_pipeline.cfg)
        self.cfg["positionestimation"] = deep_merge_dict(self.cfg["positionestimation"], self.positionestimator.cfg)

    def __call__(self, image:np.ndarray|Image.Image)->Tuple[Optional[sv.Detections], Optional[dict], Optional[np.ndarray], np.ndarray]:
        return self.potisionestimation_inference(image)
        

    def potisionestimation_inference(self, image:np.ndarray|Image.Image)->Tuple[Optional[sv.Detections], Optional[dict], Optional[np.ndarray], np.ndarray, np.ndarray]:
        if type(image) == np.ndarray:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        imagesize = image.size
        detect, kpt, annotatedimage = self.detect_pipeline(image)
        annotatedimage = cv2.cvtColor(np.array(annotatedimage), cv2.COLOR_RGB2BGR)
        heatmap = np.zeros((imagesize[1], imagesize[0]), dtype=np.float32)
        if detect is None:
            return None, None, None, annotatedimage, heatmap
        pred_pts = []
        cropped = self.crop_input(detect.xyxy, image.size, pad=self.cfg["inference"]["input_padding_bbox"])
        for bbox, square in zip(detect.xyxy, cropped):
            x1, y1, x2, y2 = map(int, square)
            input = image.crop((x1, y1, x2, y2))
            input = input.resize((self.positionestimator.cfg["in_size"], self.positionestimator.cfg["in_size"]))
            scale = self.positionestimator.cfg["in_size"]/(x2-x1)

            depth_result, pred_result = self.positionestimator.inference(np.array(input))
            mask_region = (scale*(bbox- np.tile([x1,y1],2))).astype(np.uint32)
            pred_result = self.mask_prediction(pred_result, mask_region)
            # pred_point = calc_peak_centroid(pred_result, r=min(img.shape[0:2])//10, rel_thresh=0.5)
            pred_pt = self.calc_max(pred_result)
            pred_pt = np.array([1/scale, 1/scale])*pred_pt + np.array([x1,y1])

            heatmap_region = cv2.resize(pred_result, dsize=(x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC)
            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], heatmap_region)

            annotatedimage = cv2.drawMarker(
                                img=annotatedimage,
                                position=pred_pt.astype(np.uint32),
                                color=sv.Color.from_hex("#fc0303").as_bgr(),
                                markerType=cv2.MARKER_CROSS,
                                markerSize=int(max(10, min(imagesize)//40)),
                                thickness=int(max(1, min(imagesize)//5e2)),
                                line_type=cv2.LINE_4
                            )
            pred_pts.append(pred_pt)
        
        pred_pts = np.array(pred_pts)

        cmap = cm.get_cmap("viridis")
        heatmap = cmap(heatmap/(np.max(heatmap)+1e-12))[:, :, :3]  # RGB, drop alpha

        return detect, kpt, pred_pts, annotatedimage, heatmap
    

    def crop_input(self, xyxy:np.ndarray, imgsize:Tuple[int, int], pad:float=1.0, allow_shrink:bool=True)->np.ndarray:
        img_w, img_h = imgsize

        in_region = []
        for box in xyxy:
            box = box.astype(np.int16)

            x1, y1, x2, y2 = box
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            if pad>1.0:
                side = int(pad*max(bw, bh))
            else:
                side = max(bw, bh)
            max_side = min(img_w, img_h)
            if side > max_side and allow_shrink:
                side = max_side

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            half = side / 2.0
            left   = cx - half
            right  = cx + half
            top    = cy - half
            bottom = cy + half

            if left < 0:
                shift = -left
                left += shift; right += shift
            if right > img_w:
                shift = img_w - right
                left += shift; right += shift
            if top < 0:
                shift = -top
                top += shift; bottom += shift
            if bottom > img_h:
                shift = img_h - bottom
                top += shift; bottom += shift

            left_i  = int(round(left))
            top_i   = int(round(top))
            side_i  = int(round(side))

            right_i = left_i + side_i
            bottom_i= top_i + side_i

            if right_i > img_w:
                right_i = img_w
                left_i  = right_i - side_i
            if bottom_i > img_h:
                bottom_i = img_h
                top_i    = bottom_i - side_i
            if left_i < 0:
                left_i = 0
                right_i = side_i
            if top_i < 0:
                top_i = 0
                bottom_i = side_i

            in_region.append([left_i, top_i, right_i, bottom_i])

        return np.array(in_region)


    def _validate_box(self, box: np.ndarray, W: int, H: int) -> Tuple[int, int, int, int]:
        """
        Clips box = [x1, y1, x2, y2] to the image size, rounds the values to integers, and returns the result.
        If x1 < x2 and y1 < y2 cannot be satisfied, a ValueError is raised.
        """
        x1, y1, x2, y2 = map(int, box)
        x1 = int(np.clip(x1, 0, W))
        x2 = int(np.clip(x2, 0, W))
        y1 = int(np.clip(y1, 0, H))
        y2 = int(np.clip(y2, 0, H))
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"Invalid box after clipping: {(x1, y1, x2, y2)}")
        return x1, y1, x2, y2


    def _gaussian_1d(self, L: int, sigma_ratio: float = 0.2, center: Optional[int] = None) -> np.ndarray:
        x = np.arange(L, dtype=np.float32)
        c = (L - 1) / 2.0 if center is None else float(center)
        sigma = max(L * float(sigma_ratio), 1e-6)
        w = np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2))
        return w / (w.max() + 1e-12)


    def _sigmoid_top_to_bottom(self,L: int, sharpness: float = 0.1, center: Optional[int] = None, top_scale: float = 0.4) -> np.ndarray:
        """
        A 1D sigmoid function that monotonically increases from α to 1 from the top to the bottom.
        top_scale=α (0 < α ≤ 1) is the minimum magnification at the top.
        """
        y = np.arange(L, dtype=np.float32)
        cy = (L - 1) / 2.0 if center is None else float(center)
        k = float(sharpness)
        s = 1.0 / (1.0 + np.exp(-k * (y - cy)))      # 0→1
        return float(top_scale) + (1.0 - float(top_scale)) * s  # α→1


    def make_weight(
        self,
        H: int,
        W: int,
        *,
        sigma_ratio_x: float = 0.3,
        sigma_ratio_y: float = 1.0,
        sharpness_sigmoid: float = 3e-2,
        center_x: Optional[int] = None,
        center_y: Optional[int] = None,
        top_scale: float = 0.0,
        gamma_x: float = 1.0,
        ar_thresh: float = 4/5,
    ) -> np.ndarray:
        """
        Generate a 2D weight map (H, W).
        - Rectangular (W/H < ar_thresh):   Gaussian for x, sigmoid (weaker at the top, stronger at the bottom) for y
        - Square or horizontal (W/H >= ...): Gaussian for both x and y
        """
        wx = self._gaussian_1d(W, sigma_ratio_x, center_x) ** float(gamma_x)
        if (W / H) < float(ar_thresh):
            wy = self._sigmoid_top_to_bottom(H, sharpness_sigmoid, center_y, top_scale)
        else:
            wy = self._gaussian_1d(H, sigma_ratio_y, center_y)
        return wy[:, None] * wx[None, :]


    def mask_prediction(
        self,
        heatmap: np.ndarray,
        box: np.ndarray,
        *,
        normalize: bool = True,
        eps: float = 1e-8,
        border: int = 8,
        **weight_kwargs,
    ) -> np.ndarray:
        """
        Given a 2D array heatmap as input, this function places values obtained by multiplying the weight map by the region defined by box=[x1,y1,x2,y2],
        and sets values outside this region to 0. Finally, the entire result is z-score normalized (by default).
        weight_kwargs is passed directly to make_weight.

        For the input 2D array `heatmap`, place values obtained by applying the weight map to the region `box=[x1,y1,x2,y2]`, setting values outside this region to 0. Finally, perform z-score normalization on the entire array (default).
        `weight_kwargs` is passed directly to `make_weight`.
        """
        if heatmap.ndim != 2:
            raise ValueError(f"`heatmap` must be 2D (H,W). Got shape={heatmap.shape}")

        H, W = heatmap.shape[:2]
        # Horizontal filtering is applied to the heatmap.
        x1, _, x2, _ = self._validate_box(box, W, H)
        # Vertical filtering is NOT applied to the heatmap.
        y1 = 0
        y2 = H

        mask = np.zeros_like(heatmap, dtype=np.float32)

        patch = heatmap[y1:y2, x1:x2].astype(np.float32, copy=False)
        w = self.make_weight(y2 - y1, x2 - x1, **weight_kwargs).astype(np.float32, copy=False)
        mask[y1:y2, x1:x2] = patch * w

        if border > 0:
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0

        if normalize:
            m = mask.mean(dtype=np.float64)
            s = mask.std(dtype=np.float64)
            mask = (mask - m) / (s + eps)

        return mask



    def calc_max(self, arr:np.ndarray)->np.ndarray:
        max_index_flat = np.argmax(arr) 
        max_index = np.unravel_index(max_index_flat, arr.shape)
        return np.array([max_index[1], max_index[0]]).astype(np.float32)


    def calc_peak_centroid(self, arr: np.ndarray, r: int = 20, rel_thresh: float | None = None) -> np.ndarray:
        """
        Returns the center of mass (x, y), weighted by the pixel values, within the region around the maximum pixel in `arr` (a patch with radius `r`).
        If `rel_thresh` is specified, pixels with values less than the peak value multiplied by `rel_thresh` are ignored and assigned a weight of 0.

        Args:
            arr: 2D array (H, W)
            r:   Radius from the peak center (the patch is a (2r+1) square)
            rel_thresh: Relative threshold between 0 and 1. Example: If 0.5, only values equal to or greater than 50% of the maximum value are used

        Returns:
            np.ndarray: [cx, cy] (subpixels, floating-point)
        """

        y0, x0 = np.unravel_index(np.nanargmax(arr), arr.shape)

        H, W = arr.shape[:2]
        y1 = max(0, y0 - r); y2 = min(H, y0 + r + 1)
        x1 = max(0, x0 - r); x2 = min(W, x0 + r + 1)

        patch = arr[y1:y2, x1:x2].astype(np.float64)

        if rel_thresh is not None:
            pmax = patch.max()
            if pmax > 0:
                mask = patch >= (pmax * rel_thresh)
                patch = np.where(mask, patch, 0.0)

        w = np.clip(patch, 0.0, None)
        s = w.sum()
        if s == 0:
            return np.array([float(x0), float(y0)])

        ys, xs = np.mgrid[y1:y2, x1:x2]
        cx = float((w * xs).sum() / s)
        cy = float((w * ys).sum() / s)
        return np.array([cx, cy])