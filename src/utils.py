import numpy as np
import yaml
import os
import logging
from typing import Any, Mapping, MutableMapping
import copy


def load_config(config_path:str)->dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def deep_merge_dict(default: Mapping[str, Any], user: Mapping[str, Any]) -> dict[str, Any]:
    """
    Recursively merge `user` into `default` and return a NEW dictionary.
    - `default` is the base
    - `user` is the overwrite
    - Merge recursively only when both are dictionaries
    """
    merged: dict[str, Any] = copy.deepcopy(dict(default))  # default を汚さない

    for k, v in user.items():
        if (
            k in merged
            and isinstance(merged[k], dict)
            and isinstance(v, Mapping)
        ):
            merged[k] = deep_merge_dict(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)

    return merged


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
