import torch
import os
import cv2
import numpy as np
import time
import argparse
from datetime import datetime

from src.model import PositionEstimator
from src.utils import load_config

from ultralytics import YOLO
from ultralytics.engine.results import Results

from pathlib import Path
import mimetypes

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".ppm", ".pgm"
}
VID_EXTS = {
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".mpg", ".mpeg", ".ts", ".mxf"
}


# デバイスを指定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reference(path:str, cfg:dict):    
    model = PositionEstimator(cfg=cfg, DEVICE=DEVICE)
    yolo = YOLO("yolo11n.pt").to(DEVICE)

    print("Person Detection")
    if is_video(path):
        results = yolo.track(path, classes=[0])
    elif is_image(path):
        results = yolo.predict(path, classes=[0])

    print("Position Estimation")
    for f, r in enumerate(results):
        start = time.time()
        results[f].boxes.in_region = crop_input(r)

        pos = []
        for i, in_region in enumerate(r.boxes.in_region):
            input = cv2.cvtColor(r.orig_img[in_region[1]:in_region[3],in_region[0]:in_region[2]], cv2.COLOR_BGR2RGB)
            depth_result, pred_result = model.reference(input)
            mask_region = r.boxes.xyxy[i].to("cpu").numpy().astype(np.int16)- np.tile(r.boxes.in_region[i][0:2],2)
            pred_result = mask_prediction(pred_result, mask_region.astype(np.uint16))
            # p = calc_max(pred_result) + in_region[0:2]
            p = calc_peak_centroid(pred_result, r=min(input.shape[0:2])//10, rel_thresh=0.5) + in_region[0:2]
            pos.append(p)    
        results[f].boxes.position = np.array(pos)
        end = time.time()
        if is_video(path):
            print(f"video 1/1 (frame {f+1}/{len(results)}) {r.path}, {(end - start)*10e3:.1f} ms")
        if is_image(path):
            print(f"image 1/1 (frame {f+1}/{len(results)}) {r.path}, {(end - start)*10e3:.1f} ms")

    if is_video(path):
        cap = cv2.VideoCapture(results[0].path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width = results[0].orig_shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 出力形式 (mp4)

        output_path = os.path.join("reference", f"{os.path.splitext(os.path.basename(path))[0]}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for f, r in enumerate(results):
            frame = r.plot()
            for i in range(r.boxes.cls.shape[0]):
                pos = r.boxes.position[i].astype(np.int16)                
                frame = cv2.drawMarker(frame, position=pos, color=(90, 90, 220), markerType=cv2.MARKER_CROSS, markerSize=min(r.orig_shape)//20, thickness=min(r.orig_shape)//80, line_type=cv2.LINE_4)
            
            out.write(frame)

        cap.release()
        out.release()
        print(f"Result has been saved to: {output_path}")

    elif is_image(path):
        output_path = os.path.join("reference", f"{os.path.splitext(os.path.basename(path))[0]}.jpg")
        frame = results[0].plot()
        for i in range(results[0].boxes.cls.shape[0]):
            pos = results[0].boxes.position[i].astype(np.int16)
            frame = cv2.drawMarker(frame, position=pos, color=(90, 90, 220), markerType=cv2.MARKER_CROSS, markerSize=min(r.orig_shape)//20, thickness=min(r.orig_shape)//80, line_type=cv2.LINE_4)
        
        cv2.imwrite(output_path, frame)
        print(f"Result has been saved to: {output_path}")


def is_image(path: str) -> bool:
    ext = Path(path).suffix.lower()
    if ext in IMG_EXTS:
        return True
    mime, _ = mimetypes.guess_type(path)
    return bool(mime and mime.startswith("image/"))

def is_video(path: str) -> bool:
    ext = Path(path).suffix.lower()
    if ext in VID_EXTS:
        return True
    mime, _ = mimetypes.guess_type(path)
    return bool(mime and mime.startswith("video/"))


def crop_input(result:Results, pad:int=0, allow_shrink:bool=True)->np.ndarray:
    img_h, img_w = result.orig_shape

    in_region = []
    for box in result.boxes.xyxy:
        box = box.to("cpu").detach().numpy().astype(np.int16)

        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if pad:
            side = max(bw, bh) + pad
        else:
            side = 3*max(bw, bh)//2
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


def mask_prediction(arr:np.ndarray, box:np.ndarray)->np.ndarray:
    mask = np.zeros_like(arr)
    mask[:, box[0]:box[2]] = arr[:, box[0]:box[2]]*horizontal_weight(arr[:, box[0]:box[2]].shape, kind="gaussian", sigma_ratio=0.4)
    mask = (mask - mask.mean()) / (mask.std() + 1e-8)
    return mask


def horizontal_weight(size:tuple, kind:str="gaussian", sigma_ratio:float=0.2, gamma:float=1.0)->np.ndarray:
    H,W = size
    def _oned_weight(L:int, kind:str="gaussian", sigma_ratio:float=0.2)->np.ndarray:
        x = np.arange(L, dtype=np.float32)
        c = (L - 1) / 2.0
        if kind == "gaussian":
            sigma = max(L * sigma_ratio, 1e-6)
            w = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
        elif kind == "tri":
            w = 1.0 - np.abs(x - c) / max(c, 1e-6)
            w = np.clip(w, 0.0, 1.0)
        elif kind == "cos":
            w = 0.5 * (1.0 + np.cos(np.pi * (x - c) / max(c, 1e-6)))
        else:
            raise ValueError("kind must be 'gaussian' | 'tri' | 'cos'")
        return w / (w.max() + 1e-12)  # 最大1で正規化
    
    w_x = _oned_weight(W, kind, sigma_ratio) ** gamma         # 形 (W,)
    return np.tile(w_x[None, :], (H, 1))# 形 (H, W)


def calc_max(arr:np.ndarray)->np.ndarray:
    max_index_flat = np.argmax(arr) 
    max_index = np.unravel_index(max_index_flat, arr.shape)
    return np.array([max_index[1], max_index[0]])


def calc_peak_centroid(arr: np.ndarray, r: int = 20, rel_thresh: float | None = None) -> np.ndarray:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="referenceコード")
    parser.add_argument("input_path", help="input video/image path.")
    parser.add_argument("--config", default="config/defalut.yaml", help="設定ファイルのパス")
    args = parser.parse_args()    

    assert os.path.exists(args.path), f"input {args.path} is not exists."
    assert os.path.exists(args.config), f"config {args.config} is not exists."

    cfg = load_config(args.config)

    if not os.path.isdir("reference"):
        os.makedirs("reference")

    reference(path=args.path, cfg=cfg)