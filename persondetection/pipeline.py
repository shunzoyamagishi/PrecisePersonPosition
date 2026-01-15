from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
from .inference import persondetection_inference
from PIL import Image

"""
This project uses ViTPose via Hugging Face Transformers.
Reference: Hugging Face Transformers – ViTPose Documentation
  https://huggingface.co/docs/transformers/model_doc/vitpose

"""

class PersonDetectionPipeline:
    def __init__(self, is_pose:bool=True, device="cuda"):
        self.person_image_processor = AutoProcessor.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365"
        )
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365",
            device_map=device
        )

        self.vitpose_image_processor = AutoProcessor.from_pretrained(
            "usyd-community/vitpose-base-simple"
        )
        self.vitpose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-base-simple",
            device_map=device
        )
        self.is_pose = is_pose

    def __call__(self, image:Image, threshold:float=0.3)->dict:
        return persondetection_inference(
            image,
            self.person_image_processor,
            self.person_model,
            self.vitpose_image_processor,
            self.vitpose_model,
            self.is_pose,
            threshold
        )