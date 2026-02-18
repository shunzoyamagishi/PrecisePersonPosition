from trackers import ByteTrackTracker, SORTTracker
from transformers import AutoProcessor, RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor, VitPoseForPoseEstimation
from .inference import persondetection_inference
from PIL import Image
from src.utils import deep_merge_dict

"""
This project uses ViTPose via Hugging Face Transformers.
Reference: Hugging Face Transformers – ViTPose Documentation
  https://huggingface.co/docs/transformers/model_doc/vitpose

"""

class PersonDetectionPipeline:
    def __init__(self, cfg:dict|None=None, device="cuda"):
        default_cfg = {
            "person_detector": "RTDETR",
            "pose_estimator": "ViTPose",
            "is_pose": True,
            "detector_threshold": 0.3,
            "tracker": None
        }
        if cfg is None:
            self.cfg = default_cfg
        else:
            self.cfg = deep_merge_dict(default_cfg, cfg)

        DETECTOR_REGISTRY = {
            "RTDETR": {
                "person_image_processor": AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365"),
                "person_model": RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
            },
            "RTDETRv2": {
                "person_image_processor": RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd"),
                "person_model": RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", device_map=device)
            },
            # You can add your favorite detector as needed.
        }

        POSE_REGISTRY = {
            "ViTPose": {
                "pose_image_processor": AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple"),
                "pose_model": VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)
            },
            # You can add your favorite pose estimator as needed.
        }
        if self.cfg["person_detector"] not in DETECTOR_REGISTRY:
            raise ValueError(f"Unknown detector_name={self.cfg["person_detector"]}. choices={list(DETECTOR_REGISTRY)}")
        if self.cfg["pose_estimator"] not in POSE_REGISTRY:
            raise ValueError(f"Unknown pose_estimator_name={self.cfg["pose_estimator"]}. choices={list(POSE_REGISTRY)}")
            
        self.person_image_processor = DETECTOR_REGISTRY[self.cfg["person_detector"]]["person_image_processor"]
        self.person_model = DETECTOR_REGISTRY[self.cfg["person_detector"]]["person_model"]
        self.vitpose_image_processor = POSE_REGISTRY[self.cfg["pose_estimator"]]["pose_image_processor"]
        self.vitpose_model = POSE_REGISTRY[self.cfg["pose_estimator"]]["pose_model"]
        
        TRACKER_REGISTRY = {
            "ByteTrack": ByteTrackTracker(),
            "SORT": SORTTracker()
        }
        if self.cfg["tracker"] is not None and self.cfg["tracker"] in TRACKER_REGISTRY:
            self.tracker = TRACKER_REGISTRY[self.cfg["tracker"]]
        else:
            self.tracker = None

    def __call__(self, image:Image)->dict:
        return persondetection_inference(
            image,
            self.person_image_processor,
            self.person_model,
            self.vitpose_image_processor,
            self.vitpose_model,
            self.cfg["is_pose"],
            float(self.cfg["detector_threshold"]),
            tracker=self.tracker
        )