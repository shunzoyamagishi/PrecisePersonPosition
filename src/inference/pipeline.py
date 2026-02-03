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
            "persondetection":{
                "detector_threshold": 0.6,
                "is_pose": True,
            },
            "positionestimation":{
                "DepthAnythingV2_encoder": "vits",
                "in_size": 224,
                "pretrain_model": "checkpoints/depthhead.pt",
            },  
        }
        if cfg is None:
            self.cfg = default_cfg
        else:
            self.cfg = deep_merge_dict(default_cfg, cfg)

        self.detect_pipeline = PersonDetectionPipeline(cfg=self.cfg["persondetection"], device=device)
        self.positionestimator = PositionEstimator(cfg=self.cfg["positionestimation"], device=device)


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
            return None, None, None, annotatedimage
        pred_pts = []
        cropped = self.crop_input(detect.xyxy, image.size, pad=1.2)
        for bbox, square in zip(detect.xyxy, cropped):
            x1, y1, x2, y2 = map(int, square)
            input = image.crop((x1, y1, x2, y2))
            input = input.resize((self.positionestimator.in_size, self.positionestimator.in_size))
            scale = self.positionestimator.in_size/(x2-x1)

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
        box = [x1, y1, x2, y2] を画像サイズにクリップし、整数化して返す。
        x1<x2, y1<y2 が満たせない場合は ValueError。
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
        """長さ L の 1D ガウス。最大値で 1 に正規化。"""
        x = np.arange(L, dtype=np.float32)
        c = (L - 1) / 2.0 if center is None else float(center)
        sigma = max(L * float(sigma_ratio), 1e-6)
        w = np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2))
        return w / (w.max() + 1e-12)


    def _sigmoid_top_to_bottom(self,L: int, sharpness: float = 0.1, center: Optional[int] = None, top_scale: float = 0.4) -> np.ndarray:
        """
        上端→中央→下端にかけて α→1 に単調増加する 1D シグモイド。
        top_scale=α (0<α<=1) が上端の最小倍率。
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
        sigma_ratio_x: float = 0.4,
        sigma_ratio_y: float = 1.0,
        sharpness_sigmoid: float = 3e-2,
        center_x: Optional[int] = None,
        center_y: Optional[int] = None,
        top_scale: float = 0.0,
        gamma_x: float = 1.0,
        ar_thresh: float = 4/5,          # アスペクト比判定の閾値 (W/H < 0.8 で縦長寄り)
    ) -> np.ndarray:
        """
        (H, W) の 2D 重みマップを生成。
        - 縦長寄り (W/H < ar_thresh):   x にガウス、y にシグモイド(上弱/下強)
        - 横長または正方 (W/H >= ...): x, y ともにガウス
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
        border: int = 7,
        **weight_kwargs,
    ) -> np.ndarray:
        """
        入力 2D 配列 heatmap に対し、box=[x1,y1,x2,y2] 領域へ重みマップを掛けた値を配置し、
        領域外は 0。最後に（デフォルト）全体を z-score 正規化。
        weight_kwargs は make_weight にそのまま渡される。

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

        # 該当領域の元値をコピーし、重みを乗算
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
        arr の最大画素付近(半径 r のパッチ)で、画素値を重みとした重心 (x, y) を返す。
        rel_thresh を指定すると、ピーク値 * rel_thresh 未満の画素は重み 0 として無視する。

        Args:
            arr: 2D 配列 (H, W)
            r:   ピーク中心からの半径 (パッチは (2r+1) 四方)
            rel_thresh: 0~1 の相対しきい値。例: 0.5 なら最大値の 50%以上のみ使用

        Returns:
            np.ndarray: [cx, cy] (サブピクセル、浮動小数)
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