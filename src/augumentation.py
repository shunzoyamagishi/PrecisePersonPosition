import albumentations as A

def create_augmentation():
    aug = A.ReplayCompose([        
        A.HorizontalFlip(p=0.5),
        A.Compose(
            [
                A.CoarseDropout(
                    num_holes_range=(1,2),
                    hole_height_range=(0.05,0.4),
                    hole_width_range=(0.05,0.4),
                    p=1.0
                ),
                A.GaussNoise(
                    std_range = (0.2, 0.4),
                    per_channel=True,
                    p=1.0
                ),
                A.ImageCompression(
                    quality_range=(90,100),
                    p=1.0
                )
            ],
            p=0.25
        )
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    bbox_params=A.BboxParams(format="pascal_voc")
    )
    return aug