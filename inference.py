import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Optional
from PIL import Image
from tqdm import tqdm

from src.model import PositionEstimator
from src.utils import load_config
from src.inference import PositionEstimatorPipeline

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

    if is_video(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open the video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(path))[0]}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_idx in tqdm(range(total_frames), desc=f"Infering {os.path.basename(path)}"):
            ret, frame = cap.read()
            if not ret:
                break
            _, _, _, _, annotatedframe = pipeline(frame)
            out.write(annotatedframe)
        cap.release()
        out.release()
        print(f"Result has been saved to: {output_path}")

    elif is_image(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to open the image: {path}")
        output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(path))[0]}.jpg")
        _, _, _, _, annotatedimg = pipeline(img)
        cv2.imwrite(output_path, annotatedimg)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inferenceコード")
    parser.add_argument("input_path", help="input video/image path.")
    parser.add_argument("--config", default="config/default.yaml", help="設定ファイルのパス")
    args = parser.parse_args()    

    assert os.path.exists(args.input_path), f"input {args.input_path} is not exists."
    assert os.path.exists(args.config), f"config {args.config} is not exists."

    cfg = load_config(args.config)

    if not os.path.isdir("runs/inference"):
        os.makedirs("runs/inference")

    inference(path=args.input_path, cfg=cfg, save_dir="runs/inference")