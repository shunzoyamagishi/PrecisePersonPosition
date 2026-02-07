import cv2
import torch
import os
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Tuple, Optional

from src.utils import load_config, perspective_transform
from src.inference import PositionEstimatorPipeline
from src.dataloader import TestDataset


def test(cfg:dict, save_dir:str):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = PositionEstimatorPipeline(cfg=cfg, device=DEVICE)
    testdataset = TestDataset("data/test")

    columns = [
        "DistancefromPerson",
        "TiltAngle",
        "lower_f",
        "mid_f",
        "bc_bbox",
        "pred_point",
        "error_lower_f",
        "error_mid_f",
        "error_bc_bbox",
        "error_pred_point",
    ]

    df = pd.DataFrame(columns=columns)
    
    for i in tqdm(range(len(testdataset)), desc=f"Test : "):
        img, _, P, attribute = testdataset.__getitem__(i)
        H_image2field = np.linalg.inv(P[:,[0,1,3]]).astype(np.float32)

        detect, kpt, pred_pts, _, _ = pipeline(img)
        if detect is None:
            row = {
                "DistancefromPerson": attribute[0],
                "TiltAngle": attribute[1],
                "lower_f": np.nan,
                "mid_f": np.nan,
                "bc_bbox": np.nan,
                "pred_point": np.nan,
                "error_lower_f": np.nan,
                "error_mid_f": np.nan,
                "error_bc_bbox": np.nan,
                "error_pred_point": np.nan,
            }
            df.loc[len(df)] = row
        else:
            bbox = detect.xyxy[np.argmax(detect.confidence)]
            bc_bbox = np.array([bbox[0:3:2].mean(), bbox[3]])
            left_f = kpt.xy[np.argmax(detect.confidence)][15]
            right_f = kpt.xy[np.argmax(detect.confidence)][16]
            pred_pt = pred_pts[np.argmax(detect.confidence)]
            mid_f = (left_f + right_f)/2
            if left_f[1] > right_f[1]:
                lower_f = left_f
            else:
                lower_f = right_f

            pts_img = np.array([mid_f, lower_f, bc_bbox, pred_pt], dtype=np.float32).reshape(1,-1,2)
            pts_fld = cv2.perspectiveTransform(pts_img, H_image2field)
            mid_f, lower_f, bc_bbox, pred_pt = pts_fld[0]

            row = {
                "DistancefromPerson": attribute[0],
                "TiltAngle": attribute[1],
                "lower_f": lower_f,
                "mid_f": mid_f,
                "bc_bbox": bc_bbox,
                "pred_point": pred_pt,
                "error_lower_f": np.linalg.norm(lower_f),
                "error_mid_f": np.linalg.norm(mid_f),
                "error_bc_bbox": np.linalg.norm(bc_bbox),
                "error_pred_point": np.linalg.norm(pred_pt),
            }

            df.loc[len(df)] = row


    df.to_csv(os.path.join(save_dir, 'pred.csv'))

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

    thresholds = [0.1, 0.25, 0.5, 1.0, 2.0]
    for thresh in thresholds:
        errors[f"acc_{thresh}m"] = None 

        for method in errors.index:
            acc = threshold_accuracy_series(
                df[f"error_{method}"],
                threshold=thresh
            )
            errors.loc[method, f"acc_{thresh}m"] = acc

    for dist in [5, 20, 50]:
        for angle in [0, 30, 60]:
            df_subset = df[
                (df["DistancefromPerson"] == dist) &
                (df["TiltAngle"] == angle)
            ]
            errors[f"{method}_{dist}m_{angle}deg"] = None
            for thresh in thresholds:
                for method in errors.index:
                    acc = threshold_accuracy_series(
                        df_subset[f"error_{method}"],
                        threshold=thresh
                    )
                    errors.loc[f"{method}_{dist}m_{angle}deg", f"acc_{thresh}m"] = acc

    errors.to_csv(os.path.join(save_dir, 'test.csv'))


def threshold_accuracy_series(s_pred: pd.Series, threshold: float=0.5) -> float:
    
    return float((s_pred<threshold).sum()/s_pred.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inferenceコード")
    parser.add_argument("--config", default="config/default_test.yaml", help="設定ファイルのパス")
    args = parser.parse_args()    

    assert os.path.exists(args.config), f"config {args.config} is not exists."

    cfg = load_config(args.config)

    save_dir = os.path.join("runs/test", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    test(cfg=cfg, save_dir=save_dir)