import cv2
import torch
import os
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from pathlib import Path
from typing import Sequence

from src.utils import load_config, deep_merge_dict
from src.inference import PositionEstimatorPipeline
from src.test import TestDataset


METHODS = (
    "lower_a",
    "mid_a",
    "bc_bbox",
    "pred_point",
)

DEFAULT_THRESHOLDS = (0.1, 0.25, 0.5, 1.0, 2.0)
DEFAULT_DISTANCES = (5, 20, 50)
DEFAULT_TILT_ANGLES = (0, 30, 60)


def test(cfg:dict, save_dir:str):
    cfg["inference"]["persondetection"].pop("tracker", None)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = PositionEstimatorPipeline(cfg=cfg, device=DEVICE)
    testdataset = TestDataset("data/test")

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


def threshold_accuracy_series(
    errors: pd.Series,
    threshold: float = 0.5,
) -> float:
    """Return the fraction of samples whose error is below the threshold."""
    if errors.empty:
        return np.nan

    return float((errors < threshold).mean())


def compute_coordinate_statistics(
    df: pd.DataFrame,
    method: str,
) -> tuple[float, float]:
    """Compute positional spread around the sample mean coordinate.

    Returns
    -------
    median_distance_from_mean:
        Median Euclidean distance from the sample mean coordinate.
    variance_distance_from_mean:
        Unbiased variance of Euclidean distances from the sample mean coordinate.
    """
    coord_columns = [
        f"{method}(field)[X]",
        f"{method}(field)[Y]",
    ]

    coords = df[coord_columns].dropna()

    if coords.empty:
        return np.nan, np.nan

    mean_coord = coords.mean()
    centered_coords = coords - mean_coord

    distances_from_mean = np.linalg.norm(
        centered_coords.to_numpy(),
        axis=1,
    )

    median_distance_from_mean = float(
        np.median(distances_from_mean)
    )

    variance_distance_from_mean = (
        float(np.var(distances_from_mean, ddof=1))
        if len(distances_from_mean) >= 2
        else np.nan
    )

    return (
        median_distance_from_mean,
        variance_distance_from_mean,
    )


def evaluate_pipeline(
    df: pd.DataFrame,
    save_dir: Path | str | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    valid_error_threshold: float = 1.0,
) -> pd.DataFrame:
    """Evaluate position estimation methods."""

    errors = pd.DataFrame(index=METHODS)

    # ================================================================
    # Overall evaluation
    # ================================================================
    for method in METHODS:
        error_series = df[f"error_{method}"]

        errors.loc[method, "samples"] = len(error_series)
        errors.loc[method, "mean_error"] = error_series.mean()
        errors.loc[method, "std_error"] = error_series.std()

        (
            median_distance_from_mean,
            variance_distance_from_mean,
        ) = compute_coordinate_statistics(df, method)

        errors.loc[
            method,
            "median_distance_from_mean",
        ] = median_distance_from_mean

        errors.loc[
            method,
            "unbiasedvar_distance_from_mean",
        ] = variance_distance_from_mean

        # Statistics for samples with sufficiently small estimation error.
        valid_df = df[error_series < valid_error_threshold]

        (
            valid_median_distance_from_mean,
            valid_variance_distance_from_mean,
        ) = compute_coordinate_statistics(
            valid_df,
            method,
        )
        errors.loc[
            method, 
            f"valid_samples(error<{valid_error_threshold})",
        ] = len(valid_df)

        errors.loc[
            method,
            f"median_distance_from_mean(error<{valid_error_threshold})",
        ] = valid_median_distance_from_mean

        errors.loc[
            method,
            f"unbiasedvar_distance_from_mean(error<{valid_error_threshold})",
        ] = valid_variance_distance_from_mean

    # ================================================================
    # Overall threshold accuracy
    # ================================================================
    for threshold in thresholds:
        column = f"acc@{threshold}m"

        for method in METHODS:
            errors.loc[method, column] = threshold_accuracy_series(
                df[f"error_{method}"],
                threshold=threshold,
            )

    # ================================================================
    # Evaluation by camera condition
    # ================================================================
    for distance in DEFAULT_DISTANCES:
        for tilt_angle in DEFAULT_TILT_ANGLES:
            subset = df[
                (df["DistancefromPerson"] == distance)
                & (df["TiltAngle"] == tilt_angle)
            ]

            condition = f"dist={distance}m,tilt={tilt_angle}deg"

            for method in METHODS:
                error_series = subset[f"error_{method}"]

                errors.loc[
                    method,
                    f"mean_error({condition})",
                ] = error_series.mean()

                errors.loc[
                    method,
                    f"std_error({condition})",
                ] = error_series.std()

                for threshold in thresholds:
                    errors.loc[
                        method,
                        f"acc@{threshold}m({condition})",
                    ] = threshold_accuracy_series(
                        error_series,
                        threshold=threshold,
                    )

    # ================================================================
    # Save
    # ================================================================
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        errors.to_csv(save_dir / "test.csv")

    # ================================================================
    # Display representative metrics
    # ================================================================
    summary_columns = [
        "mean_error",
        "std_error",
        "median_distance_from_mean",
        "unbiasedvar_distance_from_mean",
        f"median_distance_from_mean(error<{valid_error_threshold})",
        f"unbiasedvar_distance_from_mean(error<{valid_error_threshold})",
        "acc@0.1m",
        "acc@0.25m",
        "acc@0.5m",
        "acc@1.0m",
    ]

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
    ):
        print(errors[summary_columns])

    return errors

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_threshold_accuracy_curve(
    df: pd.DataFrame,
    threshold_min: float = 0.0,
    threshold_max: float = 2.0,
    threshold_step: float = 0.1,
    methods: Sequence[str] = METHODS,
    save_dir: Path | str | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Plot threshold-accuracy curves for each position estimation method.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation dataframe containing columns named ``error_{method}``.

    threshold_min : float
        Minimum error threshold [m].

    threshold_max : float
        Maximum error threshold [m].

    threshold_step : float
        Interval between thresholds [m].

    methods : Sequence[str]
        Methods to evaluate.

    save_dir : Path | str | None
        Directory in which to save the plot.
        If None, the plot is not saved.

    show : bool
        Whether to display the plot.

    Returns
    -------
    pd.DataFrame
        DataFrame containing threshold-accuracy values.
        Rows correspond to thresholds and columns correspond to methods.
    """

    if threshold_step <= 0:
        raise ValueError("threshold_step must be greater than 0.")

    if threshold_max < threshold_min:
        raise ValueError(
            "threshold_max must be greater than or equal to threshold_min."
        )

    # np.arange can miss the endpoint because of floating-point errors.
    num_steps = int(
        np.floor((threshold_max - threshold_min) / threshold_step)
    )

    thresholds = (
        threshold_min
        + np.arange(num_steps + 1) * threshold_step
    )

    # Explicitly include threshold_max if necessary.
    if thresholds[-1] < threshold_max:
        thresholds = np.append(thresholds, threshold_max)

    # ================================================================
    # Calculate threshold accuracy
    # ================================================================
    accuracy_df = pd.DataFrame(
        index=thresholds,
        columns=methods,
        dtype=float,
    )
    accuracy_df.index.name = "threshold"

    for method in methods:
        error_series = df[f"error_{method}"]

        for threshold in thresholds:
            accuracy_df.loc[threshold, method] = threshold_accuracy_series(
                error_series,
                threshold=threshold,
            )

    # ================================================================
    # Plot
    # ================================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in methods:
        ax.plot(
            accuracy_df.index,
            accuracy_df[method],
            label=method,
            marker='o'
        )

    ax.set_xlabel("Error threshold [m]")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(threshold_min, threshold_max)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    # ================================================================
    # Save
    # ================================================================
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_dir / "threshold_accuracy_curve.png",
            dpi=300,
            bbox_inches="tight",
        )

        accuracy_df.to_csv(
            save_dir / "threshold_accuracy_curve.csv",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return accuracy_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test code")
    parser.add_argument("--config", default="config/default_test.yaml", help="path to config")
    args = parser.parse_args()    

    assert os.path.exists(args.config), f"config {args.config} is not exists."

    cfg = load_config(args.config)

    save_dir = os.path.join("runs/test", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)

    test(cfg=cfg, save_dir=save_dir)