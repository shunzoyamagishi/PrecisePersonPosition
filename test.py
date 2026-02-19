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

from src.utils import load_config, deep_merge_dict
from src.inference import PositionEstimatorPipeline
from src.test import TestDataset


def test(cfg:dict, save_dir:str):
    default_cfg = {
        "inference":{
            "persondetection":{
                    "detector": "RTDETR",
                    "detector_threshold": 0.8,
                    "pose_estimator": "ViTPose",
                },
            "input_padding_bbox": 2.0 
        },
        "positionestimation":{
            "pretrain_model": "checkpoints/transformerhead.pt",
            "perspective_heatmap": True
        }  
    }
    cfg = deep_merge_dict(default_cfg, cfg)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = PositionEstimatorPipeline(cfg=cfg, device=DEVICE)
    testdataset = TestDataset("data/test")
    pipeline.detect_pipeline.tracker = None
    pipeline.cfg["inference"]["persondetection"].pop("tracker", None)

    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(pipeline.cfg, f)

    pred_result = []
    
    for i in tqdm(range(len(testdataset)), desc=f"Test : "):
        img, _, P, attribute = testdataset.__getitem__(i)
        H_image2field = np.linalg.inv(P[:,[0,1,3]]).astype(np.float32)

        detect, kpt, pred_pts, _, _ = pipeline(img)
        if detect is None:
            row = {
                "image": attribute[0],
                "DistancefromPerson": attribute[1],
                "TiltAngle": attribute[2],
                "lower_a(image)[X]": np.nan,
                "lower_a(image)[Y]": np.nan,
                "mid_a(image)[X]": np.nan,
                "mid_a(image)[Y]": np.nan,
                "bc_bbox(image)[X]": np.nan,
                "bc_bbox(image)[Y]": np.nan,
                "pred_point(image)[X]": np.nan,
                "pred_point(image)[Y]": np.nan,
                "lower_a(field)[X]": np.nan,
                "lower_a(field)[Y]": np.nan,
                "mid_a(field)[X]": np.nan,
                "mid_a(field)[Y]": np.nan,
                "bc_bbox(field)[X]": np.nan,
                "bc_bbox(field)[Y]": np.nan,
                "pred_point(field)[X]": np.nan,
                "pred_point(field)[Y]": np.nan,
                "error_lower_a": np.nan,
                "error_mid_a": np.nan,
                "error_bc_bbox": np.nan,
                "error_pred_point": np.nan,
            }
            pred_result.append(row)
        else:
            bbox = detect.xyxy[np.argmax(detect.confidence)]
            bc_bbox = np.array([bbox[0:3:2].mean(), bbox[3]])
            left_a = kpt.xy[np.argmax(detect.confidence)][15]
            right_a = kpt.xy[np.argmax(detect.confidence)][16]
            pred_pt = pred_pts[np.argmax(detect.confidence)]
            mid_a = (left_a + right_a)/2
            if left_a[1] > right_a[1]:
                lower_a = left_a
            else:
                lower_a = right_a
                
            pts_img = np.array([lower_a, mid_a, bc_bbox, pred_pt], dtype=np.float32).reshape(1,-1,2)
            pts_fld = cv2.perspectiveTransform(pts_img, H_image2field)

            row = {
                "image": attribute[0],
                "DistancefromPerson": attribute[1],
                "TiltAngle": attribute[2],
                "lower_a(image)[X]": lower_a[0],
                "lower_a(image)[Y]": lower_a[1],
                "mid_a(image)[X]": mid_a[0],
                "mid_a(image)[Y]": mid_a[1],
                "bc_bbox(image)[X]": bc_bbox[0],
                "bc_bbox(image)[Y]": bc_bbox[1],
                "pred_point(image)[X]": pred_pt[0],
                "pred_point(image)[Y]": pred_pt[1],
                "lower_a(field)[X]": pts_fld[0][0][0],
                "lower_a(field)[Y]": pts_fld[0][0][1],
                "mid_a(field)[X]": pts_fld[0][1][0],
                "mid_a(field)[Y]": pts_fld[0][1][1],
                "bc_bbox(field)[X]": pts_fld[0][2][0],
                "bc_bbox(field)[Y]": pts_fld[0][2][1],
                "pred_point(field)[X]": pts_fld[0][3][0],
                "pred_point(field)[Y]": pts_fld[0][3][1],
                "error_lower_a": np.linalg.norm(pts_fld[0][0]),
                "error_mid_a": np.linalg.norm(pts_fld[0][1]),
                "error_bc_bbox": np.linalg.norm(pts_fld[0][2]),
                "error_pred_point": np.linalg.norm(pts_fld[0][3]),
            }

            pred_result.append(row)

    df = pd.DataFrame(pred_result)
    df.to_csv(os.path.join(save_dir, 'pred.csv'), index=False)

    evaluate_pipeline(df)
    

def evaluate_pipeline(df:pd.DataFrame):
    errors = pd.DataFrame({
        "mean(error)": df[["error_lower_a", "error_mid_a", "error_bc_bbox", "error_pred_point"]].mean(),
        "std(error)":  df[["error_lower_a", "error_mid_a", "error_bc_bbox", "error_pred_point"]].std(),
    })

    errors = errors.rename(index={
        "error_lower_a": "lower_a",
        "error_mid_a": "mid_a",
        "error_bc_bbox": "bc_bbox",
        "error_pred_point": "pred_point",
    })
    
    errors["unbiased_var"] = np.nan

    for method in errors.index:
        df_subset = df[df[f"error_{method}"]<1.0]
        coords = df_subset[[f'{method}(field)[X]', f'{method}(field)[Y]']]
        mean_coord = coords.mean()
        error_frommean = coords - mean_coord
        dist_error = np.sqrt((error_frommean**2).sum(axis=1))
        unbiased_var = dist_error.var(ddof=1)
        errors.loc[method, "unbiased_var"] = unbiased_var

    thresholds = [0.1, 0.25, 0.5, 1.0, 2.0]
    for thresh in thresholds:
        errors[f"acc@{thresh}m"] = np.nan

        for method in errors.index:
            acc = threshold_accuracy_series(
                df[f"error_{method}"],
                threshold=thresh
            )
            errors.loc[method, f"acc@{thresh}m"] = acc

    for dist in [5, 20, 50]:
        for angle in [0, 30, 60]:
            df_subset = df.query(f"DistancefromPerson=={dist} & TiltAngle=={angle}")
            for method in errors.index:
                errors.loc[method, f"mean(dist={dist}m,tilt={angle}deg)"] = df_subset[f"error_{method}"].mean()
                errors.loc[method, f"std(dist={dist}m,tilt={angle}deg)"] = df_subset[f"error_{method}"].std()
            for thresh in thresholds:
                errors[f"acc@{thresh}m(dist={dist}m,tilt={angle}deg)"] = np.nan
                for method in errors.index:
                    if len(df_subset):
                        acc = threshold_accuracy_series(
                            df_subset[f"error_{method}"],
                            threshold=thresh
                        )
                        errors.loc[method, f"acc@{thresh}m(dist={dist}m,tilt={angle}deg)"] = acc
              
    errors.to_csv(os.path.join(save_dir, 'test.csv'))
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.width', None
    ):
        print(errors.iloc[:, [0,2,3,4,5,6,7]])


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

    test(cfg=cfg, save_dir=save_dir)