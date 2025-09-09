import kornia.augmentation as augm
import torch.nn as nn

augmentation_transforms = nn.Sequential(
    augm.RandomSharpness(sharpness=1., p=0.2),  # シャープネス
    augm.RandomBrightness(brightness=(0.8, 1.2), p=0.3),  # 明るさ
    augm.RandomContrast(contrast=(0.8, 1.2), p=0.3),  # コントラスト
    augm.RandomPosterize(bits=3, p=0.2),  # ポスタリゼーション
    augm.RandomGaussianNoise(mean=0., std=0.05, p=0.2),  # ガウシアンノイズ
    augm.RandomHue(hue=0.5, p=0.5),  # 色相のシフト
    augm.RandomSaturation(saturation=(0.5, 1.5), p=0.5),  # 彩度の変化
    augm.RandomChannelShuffle(p=0.5),  # チャンネルのシャッフル
    augm.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # 複合的な色変換
)