import os
import copy
import yaml
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from schedulefree import AdamWScheduleFree
import itertools

from src.model import SwinTransformerHead
from src.loss import KLDivLoss
from src.dataloader import PersonPositionDataset
from src.depth import DepthFeatureExtractor
from src.augumentation import create_augmentation
from src.utils import load_config, setup_logger
    

def train(
    cfg:dict,
    save_dir:str,
):
    # config 
    input_size = int(cfg["in_size"])
    target_std = float(cfg["targetstd"])
    batch_size = int(cfg["batch"])
    epochs = int(cfg["epoch"])
    DEVICE1 = f"cuda:{int(cfg["DEVICE1"])}"
    DEVICE2 = f"cuda:{int(cfg["DEVICE2"])}"
    # pretrained model
    pretrain_model = cfg.get("pretrain_model") if (cfg.get("pretrain_model") and os.path.isfile(cfg.get("pretrain_model"))) else None
    
    seed = 42
    generator = torch.Generator().manual_seed(seed)

    # Augmentation
    augmentation = create_augmentation()
  
    loss_function = KLDivLoss()
    
    logger = setup_logger(save_dir)
    logger.info("Logger initialized.")

    writer = SummaryWriter(log_dir=save_dir)
    
    logger.info("Loading Dataset...")
    dataset = PersonPositionDataset("data/train", input_size=input_size, target_std=target_std, augmentation=augmentation)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Depth Model
    depthmodel_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = cfg["DepthAnythingV2_encoder"]
    depth_model = DepthFeatureExtractor(**depthmodel_configs[encoder], device=DEVICE2)
    
    depth_model.load_state_dict(torch.load(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_model = depth_model.to(DEVICE2).eval()
    
    # model
    model = SwinTransformerHead(in_size=input_size, in_channel=depthmodel_configs[encoder]['features'], out_channel=2, device=DEVICE1)
    model = model.to(DEVICE1) 

    # optimizer
    optimizer = AdamWScheduleFree(model.parameters())

    start_epoch = 0
    best_train_loss = float("inf")
    best_val_loss = float("inf")
                             
    if pretrain_model is not None and os.path.isfile(pretrain_model):
        ckpt = torch.load(pretrain_model, map_location='cpu')
        model.load_state_dict(ckpt["model"]) 
        optimizer.load_state_dict(ckpt["optimizer"]) 
        start_epoch = ckpt.get("epoch", -1) + 1        
        best_train_loss = ckpt.get("best_train", float("inf"))
        logger.info(f"Loaded pretrained model: {pretrain_model}")

    best_model_weights = None

    if isinstance(optimizer, AdamWScheduleFree):
        logger.info("Schedule-Free calibration start")
        model.train()
        optimizer.eval()
        with torch.no_grad():
            for (images, _, _, _) in tqdm(itertools.islice(train_loader, 50), total=50, desc="Calibration"):
                images = images.to(DEVICE2)
                feature, _ = depth_model(images)
                feature = feature.to(DEVICE1)                
                outputs = model(feature)
        model.eval()
        logger.info("Schedule-Free calibration done")
    
    
    for epoch in range(start_epoch,epochs):
        model.train()
        optimizer.train()
        running_loss = 0.0
        # if best_train_loss < 2e-4:
        # if epoch == epochs-2:
        #     train_dataset.random_crop = False
        #     valid_dataset.random_crop = False

        for batch, (images, _, heatmap, _) in enumerate(tqdm(train_loader, desc=f"Train : Epoch {epoch + 1}/{epochs}", unit="batch")):
            images = images.to(DEVICE2)
            with torch.no_grad():
                feature, _ = depth_model(images)
            feature = feature.to(DEVICE1)
            # pos = pos.to(DEVICE1)
            heatmap = heatmap.to(DEVICE1)
            optimizer.zero_grad()
            
            outputs = model(feature)
            loss = loss_function(outputs, heatmap)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            logger.info(f"Best model updated (train_loss={best_train_loss:.6f})")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode(): 
            for batch, (images, _, heatmap, _) in enumerate(tqdm(valid_loader, desc=f"Validation : Epoch {epoch + 1}/{epochs}", unit="batch")):
                images = images.to(DEVICE2)
                feature, _ = depth_model(images)
                feature = feature.to(DEVICE1)
                heatmap = heatmap.to(DEVICE1)
                outputs = model(feature)
                val_loss += loss_function(outputs, heatmap).item()

        avg_val_loss = val_loss / len(valid_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch+1)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"Best model updated (val_loss={best_val_loss:.6f})")

        ckpt = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": optimizer.state_dict(),   # ← AdamWのm/vなどが入る
            "epoch": epoch,                        # 次回は epoch+1 から
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "in_size": (depthmodel_configs[encoder]['features'], input_size, input_size)
        }
        logger.info(f"model saved (val_loss={best_val_loss:.6f})")
        torch.save(
            ckpt,
            os.path.join(save_dir, f"model_epoch{epoch+1}.pt")
        )

        torch.cuda.empty_cache()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習コード")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()

    assert os.path.exists(args.config), f"config {args.config} is not exists."
    
    cfg = load_config(args.config)
    cfg["dataset"] = "train"

    # モデル保存フォルダ
    save_dir = os.path.join("runs/trains", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    train(cfg=cfg, save_dir=save_dir)
    