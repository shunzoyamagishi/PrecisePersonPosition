import torch
import os
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Tuple, Optional

from src.model import PositionEstimator
from src.utils import load_config, perspective_transform
from src.inference_utils import crop_input, mask_prediction, calc_peak_centroid, calc_max
from src.dataloader import TestDataset
from vitpose import VitPosePipeline

from ultralytics import YOLO



# デバイスを指定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(cfg:dict, save_dir:str):
    DEVICE = torch.device(f"cuda:{int(cfg["DEVICE1"])}" if torch.cuda.is_available() else "cpu")
    model = PositionEstimator(cfg=cfg, DEVICE=DEVICE)
    yolo = YOLO("yolo11n.pt").to(DEVICE)
    testdataset = TestDataset("data/test")

    df = pd.DataFrame(columns=["lower_f", "mid_f", "bc_bbox", "pred_point", "error_lower_f", "error_mid_f", "error_bc_bbox", "error_pred_point"])
    
    for i in tqdm(range(len(testdataset)), desc=f"Test : "):
        img, pos, pose_kpt, P = testdataset.__getitem__(i)
        H_image2field = np.linalg.inv(P[:,[0,1,3]])

        results = yolo(img, classes=[0], verbose=False)
        if results[0].boxes.cls.shape[0] == 0:
            continue
        
        ind = torch.argmax(results[0].boxes.conf).item()
        left_f = pose_kpt[15]
        right_f = pose_kpt[16]
        if left_f[1] > right_f[1]:
            lower_f = left_f
        else:
            lower_f = right_f
        mid_f = (left_f+right_f)/2
        bc_bbox = np.array([results[0].boxes.xywh[ind][0].to('cpu'), results[0].boxes.xyxy[ind][3].to('cpu')])

        depth_result, pred_result = model.inference(img)
        pred_point = calc_peak_centroid(pred_result, r=min(img.shape[0:2])//10, rel_thresh=0.5)
        # pred_point = calc_max(pred_result)

        lower_f = perspective_transform(H_image2field, lower_f)
        mid_f = perspective_transform(H_image2field, mid_f)
        bc_bbox = perspective_transform(H_image2field, bc_bbox)
        pred_point = perspective_transform(H_image2field, pred_point)
        
        df.loc[len(df)] = [lower_f, mid_f, bc_bbox, pred_point, np.linalg.norm(lower_f), np.linalg.norm(mid_f), np.linalg.norm(bc_bbox), np.linalg.norm(pred_point)]

  
    errors = pd.DataFrame({
        "mean": df[["error_lower_f", "error_mid_f", "error_bc_bbox", "error_pred_point"]].mean(),
        "std":  df[["error_lower_f", "error_mid_f", "error_bc_bbox", "error_pred_point"]].std(),
    })

    errors = errors.rename(index={
        "error_lower_f": "lower_f",
        "error_mid_f": "mid_f",
        "error_bc_bbox": "bc_bbox",
        "error_pred_point": "pred_point",
    })

    thresholds = [0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0]

    for thresh in thresholds:
        errors[f"acc_{thresh}m"] = None  # 先に列を作る

        for method in errors.index:
            acc = threshold_accuracy_series(
                df[f"error_{method}"],
                threshold=thresh
            )
            errors.loc[method, f"acc_{thresh}m"] = acc

    errors.to_csv(os.path.join(save_dir, 'test.csv'))


def threshold_accuracy_series(s_pred: pd.Series, threshold: float=0.5) -> float:
    
    return float((s_pred<threshold).sum()/s_pred.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inferenceコード")
    parser.add_argument("--config", default="config/default_inference.yaml", help="設定ファイルのパス")
    args = parser.parse_args()    

    assert os.path.exists(args.config), f"config {args.config} is not exists."

    cfg = load_config(args.config)

    save_dir = os.path.join("test", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    test(cfg=cfg, save_dir=save_dir)