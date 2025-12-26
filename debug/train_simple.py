import os
import copy
import yaml
import torch
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A

from src.model_simple import SwinTransformerUnet_S, SwinTransformerEncoder
from src.loss import KLDivLoss, CrossEntropyLoss, MSELoss
from src.dataloader import FrameDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    cfg:dict,
    save_dir:str,
):
    # config 
    input_size = tuple(cfg["imagesize"])
    batch_size = int(cfg["batch"])
    epochs = int(cfg["epoch"])
    # pretrained model
    pretrain_model = cfg["pretrain"] if (cfg["pretrain"] and os.path.isfile(cfg["pretrain"])) else None
    # Augmentation
    augmentation = create_augmentation(cfg["augmentation"]) if(cfg["augmentation"] != None) else None
 
    loss_functions = {
        "KLDiv":KLDivLoss(std=10.0),
        "CrossEntropy":CrossEntropyLoss(std=10.0,alpha=15.0),
        "MSE":MSELoss,
    }
    models = {
        "Unet":SwinTransformerUnet_S(imgsize=input_size, num_classes=1),
        "Encoder":SwinTransformerEncoder(imgsize=input_size, num_classes=1)
    }
    
    logger = setup_logger(save_dir)
    logger.info("Logger initialized.")

    writer = SummaryWriter(log_dir=save_dir)
    
    logger.info("Loading Dataset...")
    train_dataset = FrameDataset("data/train", input_size=input_size, augmentation=augmentation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = FrameDataset("data/val", input_size=input_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # model
    model = models[cfg["model"]]
    model = model.to(device)
    logger.info(f"Model: model({input_size[0]},{input_size[1]})_{cfg["model"]}")

    if pretrain_model is not None and os.path.isfile(pretrain_model):
        model.load_state_dict(torch.load(pretrain_model, weights_only=True))
        logger.info(f"Loaded pretrained model: {pretrain_model}")

    best_val_loss = float('inf')
    best_model_weights = None
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criteria = loss_functions[cfg["loss"]]
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch, (images, pos) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")):
            images = images.to(device)
            pos = pos.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criteria(outputs, pos)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, pos in tqdm(valid_loader, desc="Validating"):
                images = images.to(device)
                pos = pos.to(device)
                outputs = model(images)
                val_loss += criteria(outputs, pos).item()

        avg_val_loss = val_loss / len(valid_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch+1)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            logger.info(f"Best model updated (val_loss={best_val_loss:.6f})")
            torch.save(
                best_model_weights,
                os.path.join(save_dir, f"model({input_size[0]},{input_size[1]})_s_best_epoch{epoch+1}.pth")
            )

        torch.cuda.empty_cache()

    writer.close()
    # Save final best model
    # os.makedirs("models", exist_ok=True)
    # torch.save(
    #     best_model_weights,
    #     os.path.join("models", f"model({input_size[0]},{input_size[1]})_s_final_best.pth")
    # )


def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # コンソール用ハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # ファイル用ハンドラ
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 既存のハンドラを消す（重複防止）
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # ハンドラを登録
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path:str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def create_augmentation(cfg):
    aug = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=cfg["brightness_limit"],
            contrast_limit=cfg["contrast_limit"],
            p=0.3
        ),
        A.HueSaturationValue(
            hue_shift_limit=(-cfg["hue_shift_limit"], cfg["hue_shift_limit"]),
            sat_shift_limit=(-cfg["sat_shift_limit"], cfg["sat_shift_limit"]),
            val_shift_limit=(-cfg["val_shift_limit"], cfg["val_shift_limit"]),
            p=0.4
        ),
        A.ChannelShuffle(p=0.1),
        A.InvertImg(p=0.05),
        A.CoarseDropout(
            num_holes_range=tuple(cfg["coarse_dropout"]["num_holes_range"]),
            hole_height_range=tuple(cfg["coarse_dropout"]["hole_height_range"]),
            hole_width_range=tuple(cfg["coarse_dropout"]["hole_width_range"]),
            fill=0,
            p=cfg["coarse_dropout"]["p"]
        ),
        A.GaussianBlur(blur_limit=(3,5), p=cfg["blur_p"]),
        A.GaussNoise(std_range=(0.1,0.5), p=cfg["noise_p"]),
        A.ImageCompression(quality_range=(40,90), p=cfg["compression_p"]),
        A.Affine(
            scale=(0.9,1.1),
            translate_percent=(-0.05,0.05),
            rotate=(-10,10),
            shear=(-5,5),
            p=cfg["affine_p"]
        )
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    bbox_params=A.BboxParams(format="pascal_voc")
    )
    return aug


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習コード")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # モデル保存フォルダ
    save_dir = os.path.join("trains", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    train(
        cfg=cfg,
        save_dir=save_dir,
    )
    