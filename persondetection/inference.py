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

def persondetection_inference(
        image:Image,
        person_image_processor:AutoProcessor,
        person_model:RTDetrForObjectDetection,
        vitpose_image_processor:AutoProcessor,
        vitpose_model:VitPoseForPoseEstimation,
        is_pose:bool=False,
        threshold:float=0.3,
    )->tuple[dict, sv.KeyPoints, Image.Image]:
    if type(image) == np.ndarray:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    imagesize = image.size
    # Detect humans in the image
    inputs = person_image_processor(images=image, return_tensors="pt").to(person_model.device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=threshold
    )
    result = results[0]

    keep = [
        i for i, label in enumerate(result["labels"])
        if int(label) == 0
    ]

    for k in ["scores", "labels", "boxes"]:
        result[k] = result[k][keep]

    detections = sv.Detections.from_transformers(
        transformers_results=result
    )

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    if len(person_boxes) == 0:
        return None, None, image

    box_annotator = sv.BoxAnnotator(
        color=sv.Color.from_hex('#fcba03'),
        thickness=int(max(1, min(imagesize)//1e3))
    )
    annotated_frame = box_annotator.annotate(
        scene=image.copy(), 
        detections=detections
    )

    if not is_pose:
        return detections, None, annotated_frame
    
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
        color=sv.Color.from_hex("#73fc03"),
        thickness=int(max(1, min(imagesize)//1e3))
    )
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#ff00ff'),
        radius=int(max(2, min(imagesize)//5e2))
    )
    
    annotated_frame = edge_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )
    
    return detections, key_points, annotated_frame