import albumentations as A

def create_augmentation():
    aug = A.ReplayCompose([        
        A.HorizontalFlip(p=0.5),
        # A.Compose(
        #     [
        #         A.CoarseDropout(
        #             num_holes_range=(5,10),
        #             hole_height_range=(0.05,0.1),
        #             hole_width_range=(0.05,0.1),
        #             p=1.0
        #         ),
        #         A.GaussNoise(
        #             std_range = (0.2, 0.4),
        #             per_channel=True,
        #             p=1.0
        #         ),
        #     ],
        #     p=0.25
        # ),
        A.GaussNoise(
            std_range = (0.2, 0.4),
            per_channel=True,
            p=0.2
        ),
        A.ImageCompression(
            quality_range=(60,100),
            p=0.5
        )
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    bbox_params=A.BboxParams(
            format="pascal_voc", 
            min_area=0,        # 面積が小さくなっても削除しない
            min_visibility=0,  # 変換ではみ出しても少しでも残っていれば保
        )
    )
    return aug