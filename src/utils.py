import cv2
import numpy as np
import yaml
import os
import logging


def load_config(config_path:str)->dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # コンソール用ハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # ファイル用ハンドラ
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 既存のハンドラを消す（重複防止）
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # ハンドラを登録
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def perspective_transform(P:np.ndarray, points:np.ndarray)->np.ndarray:
    if points.ndim == 1:
        points = np.expand_dims(points, 0)
        if points.shape[-1] != P.shape[1]:
            points = np.concatenate([points, np.ones((points.shape[0],1))],axis=1)
        uv1 = P @ points.T
        uv1 /= uv1[-1,:]+1e-8
        return uv1.T[0][:-1]
    else:
        if points.shape[-1] != P.shape[1]:
            points = np.concatenate([points, np.ones((points.shape[0],1))],axis=1)
        uv1 = P @ points.T
        uv1 /= uv1[-1,:]+1e-8
        return uv1.T[:,:-1]
