import os
import json
import cv2
import numpy as np
from ..train import PersonPositionDataset
   

class TestDataset(PersonPositionDataset):
    def __init__(self, root_dir:str, input_size:int=224):
        self.root_dir = root_dir
        self.input_size = input_size
        self.samples = []
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            filenames.sort()
            if "data.json" not in filenames:
                continue

            video_path = dirpath
            json_path = os.path.join(video_path, "data.json")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "imgs" not in data:
                raise KeyError(f"`imgs` not found in {json_path}")
            cam_param = data.get("Camera Parameter", {})
            if cam_param.get("Perspective Projection Matrix", None) is None:
                raise KeyError(f"`Camera Parameter/Perspective Projection Matrix` not found in {json_path}")

            for img in data['imgs']:
                    img['img_name'] = os.path.join(video_path, img['img_name'])
                    img['PerspectiveProjectionMatrix'] = data["Camera Parameter"]["Perspective Projection Matrix"]
                    img['distancefromperson'] = data["Camera Parameter"]["Distance from Player"]
                    img['tiltangle'] = data["Camera Parameter"]["Tilt wrt Field"]
            self.samples += data['imgs']


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx)->tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, int, int]]:
        item = self.samples[idx]
        image = cv2.imread(item['img_name'])
        pos_gt = np.array(item['pos'])
        P = np.array(item['PerspectiveProjectionMatrix'])
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
        return image, pos_gt, P, (item['img_name'], item['distancefromperson'], item['tiltangle'])