from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
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
            "person_image_processor": "PekingU/rtdetr_r50vd_coco_o365",
            "person_model": "PekingU/rtdetr_r50vd_coco_o365",
            "vitpose_image_processor": "usyd-community/vitpose-base-simple",
            "vitpose_model": "usyd-community/vitpose-base-simple",
            "is_pose": True,
            "detector_threshold": 0.3,
        }
        if cfg is None:
            self.cfg = default_cfg
        else:
            self.cfg = deep_merge_dict(default_cfg, cfg)

        self.person_image_processor = AutoProcessor.from_pretrained(
            self.cfg["person_image_processor"]
        )
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            self.cfg["person_model"],
            device_map=device
        )

        self.vitpose_image_processor = AutoProcessor.from_pretrained(
            self.cfg["vitpose_image_processor"]
        )
        self.vitpose_model = VitPoseForPoseEstimation.from_pretrained(
            self.cfg["vitpose_model"],
            device_map=device
        )
        

    def __call__(self, image:Image)->dict:
        return persondetection_inference(
            image,
            self.person_image_processor,
            self.person_model,
            self.vitpose_image_processor,
            self.vitpose_model,
            self.cfg["is_pose"],
            float(self.cfg["detector_threshold"])
        )