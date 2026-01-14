from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
from .inference import detr_inference
from typing import Iterable, Optional
from PIL import Image


"""
This project uses ViTPose via Hugging Face Transformers.
Reference: Hugging Face Transformers – ViTPose Documentation
  https://huggingface.co/docs/transformers/model_doc/vitpose

"""

class DeTRPipeline:
    def __init__(self, device="cuda"):
        self.person_image_processor = AutoProcessor.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365"
        )
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365",
            device_map=device
        )

    def __call__(self, image:Image, class_ids:Optional[Iterable[int]] = None, threshold:float=0.3)->dict:
        return detr_inference(
            image,
            self.person_image_processor,
            self.person_model,
            class_ids=class_ids,
            threshold=threshold
        )