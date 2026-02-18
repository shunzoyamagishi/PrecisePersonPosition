import torch
from src.inference import PositionEstimatorPipeline

def preprocess_notebook(pipeline:PositionEstimatorPipeline):
    ckt_vit = torch.load(f'../Depth_Anything_V2/checkpoints/depth_anything_v2_{pipeline.positionestimator.cfg["DepthAnythingV2_encoder"]}.pth', map_location='cpu')
    pipeline.positionestimator.depth_model.load_state_dict(ckt_vit)