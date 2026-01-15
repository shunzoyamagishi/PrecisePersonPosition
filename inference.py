import torch
import os
import cv2
import numpy as np
import time
import argparse
from typing import Tuple, Optional
from PIL import Image

from src.model import PositionEstimator
from src.utils import load_config
from src.inference import PositionEstimatorPipeline

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


def inference(path:str, cfg:dict, save_dir:str):    
    pipeline = PositionEstimatorPipeline(cfg=cfg, device=DEVICE)

    print("Person Detection")
    if is_video(path):
        return 0
    elif is_image(path):
        img = cv2.imread(path)
        output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(path))[0]}.jpg")
        result = pipeline(img)
        cv2.imwrite(output_path, result)
        print(f"Result has been saved to: {output_path}")

    # print("Position Estimation")
    # for f, r in enumerate(results):
    #     start = time.time()
    #     results[f].boxes.in_region = crop_input(r.boxes.xyxy.to("cpu").detach().numpy(), r.orig_img.shape[0:2], pad=200)

    #     pos = []
    #     for i, in_region in enumerate(r.boxes.in_region):
    #         input = cv2.cvtColor(r.orig_img[in_region[1]:in_region[3],in_region[0]:in_region[2]], cv2.COLOR_BGR2RGB)
    #         depth_result, pred_result = model.inference(input)
    #         mask_region = r.boxes.xyxy[i].to("cpu").numpy().astype(np.int16)- np.tile(r.boxes.in_region[i][0:2],2)
    #         pred_result = mask_prediction(pred_result, mask_region.astype(np.uint16))
    #         # p = calc_max(pred_result) + in_region[0:2]
    #         p = calc_peak_centroid(pred_result, r=min(input.shape[0:2])//10, rel_thresh=0.5) + in_region[0:2]
    #         pos.append(p)    
    #     results[f].boxes.position = np.array(pos)
    #     end = time.time()
    #     if is_video(path):
    #         print(f"video 1/1 (frame {f+1}/{len(results)}) {r.path}, {(end - start)*10e3:.1f} ms")
    #     if is_image(path):
    #         print(f"image 1/1 (frame {f+1}/{len(results)}) {r.path}, {(end - start)*10e3:.1f} ms")

    # if is_video(path):
    #     cap = cv2.VideoCapture(results[0].path)
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     height, width = results[0].orig_shape
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 出力形式 (mp4)

    #     output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(path))[0]}.mp4")
    #     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    #     for f, r in enumerate(results):
    #         frame = r.plot()
    #         for i in range(r.boxes.cls.shape[0]):
    #             pos = r.boxes.position[i].astype(np.int16)                
    #             frame = cv2.drawMarker(frame, position=pos, color=(90, 90, 220), markerType=cv2.MARKER_CROSS, markerSize=min(r.orig_shape)//20, thickness=min(r.orig_shape)//80, line_type=cv2.LINE_4)
            
    #         out.write(frame)

    #     cap.release()
    #     out.release()
    #     print(f"Result has been saved to: {output_path}")

    # elif is_image(path):
    #     output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(path))[0]}.jpg")
    #     frame = results[0].plot()
    #     for i in range(results[0].boxes.cls.shape[0]):
    #         pos = results[0].boxes.position[i].astype(np.int16)
    #         frame = cv2.drawMarker(frame, position=pos, color=(90, 90, 220), markerType=cv2.MARKER_CROSS, markerSize=min(r.orig_shape)//20, thickness=min(r.orig_shape)//80, line_type=cv2.LINE_4)
        
    #     cv2.imwrite(output_path, frame)
    #     print(f"Result has been saved to: {output_path}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inferenceコード")
    parser.add_argument("input_path", help="input video/image path.")
    parser.add_argument("--config", default="config/defalut.yaml", help="設定ファイルのパス")
    args = parser.parse_args()    

    assert os.path.exists(args.input_path), f"input {args.input_path} is not exists."
    assert os.path.exists(args.config), f"config {args.config} is not exists."

    cfg = load_config(args.config)

    if not os.path.isdir("runs/inference"):
        os.makedirs("runs/inference")

    inference(path=args.input_path, cfg=cfg, save_dir="runs/inference")