import torch
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
import cv2

"""
This project uses ViTPose via Hugging Face Transformers.
Reference: Hugging Face Transformers – ViTPose Documentation
  https://huggingface.co/docs/transformers/model_doc/vitpose

"""

def vitpose_inference(
        image:Image,
        person_image_processor:AutoProcessor,
        person_model:RTDetrForObjectDetection,
        vitpose_image_processor:AutoProcessor,
        vitpose_model:VitPoseForPoseEstimation
    )->tuple[sv.KeyPoints, np.ndarray, Image]:
    if type(image) == np.ndarray:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Detect humans in the image
    inputs = person_image_processor(images=image, return_tensors="pt").to(person_model.device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    )
    result = results[0]

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()
    person_boxes_xyxy = person_boxes.copy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    if len(person_boxes) == 0:
        return None, None, None
    
    # Detect keypoints for each person found
    inputs = vitpose_image_processor(image, boxes=[person_boxes], return_tensors="pt").to(vitpose_model.device)

    with torch.no_grad():
        outputs = vitpose_model(**inputs)

    pose_results = vitpose_image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
    image_pose_result = pose_results[0]

    xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
    scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

    key_points = sv.KeyPoints(
        xy=xy, confidence=scores
    )

    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.GREEN,
        thickness=1
    )
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.RED,
        radius=2
    )
    annotated_frame = edge_annotator.annotate(
        scene=image.copy(),
        key_points=key_points
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )

    return key_points, person_boxes_xyxy, annotated_frame