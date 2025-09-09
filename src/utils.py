import cv2
import numpy as np
import yaml


def load_config(config_path:str)->dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
