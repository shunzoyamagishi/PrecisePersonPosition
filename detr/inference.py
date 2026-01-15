import torch
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection
from typing import Iterable, Optional
import cv2

"""
This project uses ViTPose via Hugging Face Transformers.
Reference: Hugging Face Transformers – ViTPose Documentation
  https://huggingface.co/docs/transformers/model_doc/vitpose

"""

def detr_inference(
        image:Image,
        person_image_processor:AutoProcessor,
        person_model:RTDetrForObjectDetection,
        class_ids: Optional[Iterable[int]] = None,
        threshold: float = 0.3,
    )->tuple[dict, Image]:
    if type(image) == np.ndarray:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Detect humans in the image
    inputs = person_image_processor(images=image, return_tensors="pt").to(person_model.device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=threshold
    )

    result = results[0]

    if class_ids is not None:
        class_ids = set(class_ids)

        keep = [
            i for i, label in enumerate(result["labels"])
            if int(label) in class_ids
        ]

        for k in ["scores", "labels", "boxes"]:
            result[k] = result[k][keep]
    
    if len(result["boxes"]) == 0:
        return None, image
    
    detections = sv.Detections.from_transformers(
        transformers_results=result
    )

    box_annotator = sv.BoxAnnotator(
        color=sv.Color.from_hex('#fcba03'),
        thickness=1
    )
    annotated_frame = box_annotator.annotate(
        scene=image.copy(), 
        detections=detections
    )

    return detections, annotated_frame