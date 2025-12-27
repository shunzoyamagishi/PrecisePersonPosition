import albumentations as A

def create_augmentation():
    aug = A.ReplayCompose([        
        A.HorizontalFlip(p=0.5)
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    bbox_params=A.BboxParams(format="pascal_voc")
    )
    return aug