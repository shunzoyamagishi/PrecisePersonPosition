
import argparse
import numpy as np
import cv2
import os
import glob
import json
from PIL import Image
import shutil
import sys
import torch

sys.path.append(os.path.dirname(__file__))

from persondetection import PersonDetectionPipeline

ROOT_DIR = "data/train"
MOTION = "SoccerPlayer_walk"
UNITYPROJECT_PATH = "SoccerPlayer" # 自分の環境によって変更、あるいは引数で指定してください。

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_files(root_dir:str, motion:str, unityproject_path:str):
    assert not os.path.isdir(os.path.join(root_dir, motion)), f"{os.path.join(root_dir, motion)} have already existed."
    
    for i in range(3):
        for j in range(3):
            for k in range(12):
                if not os.path.isdir(os.path.join(root_dir, motion, f"{i*100+j*20+k:03}")):
                    print(f"Create Folder : {i*100+j*20+k:03}")
                    os.makedirs(os.path.join(root_dir, motion, f"{i*100+j*20+k:03}"))
                
                l = 0
                while os.path.exists(f"{unityproject_path}/Recordings/Image Sequence_{i*100+j*20+k:03}_{l:04}.jpg"):
                    print(f"Move file : Image Sequence_{i*100+j*20+k:03}_{l:04}.jpg")
                    if os.path.exists(os.path.join(root_dir, motion, f"{i*100+j*20+k:03}", f"Image Sequence_{i*100+j*20+k:03}_{l:04}.jpg")):
                        os.remove(os.path.join(root_dir, motion, f"{i*100+j*20+k:03}", f"Image Sequence_{i*100+j*20+k:03}_{l:04}.jpg"))
                    shutil.move(f"{unityproject_path}/Recordings/Image Sequence_{i*100+j*20+k:03}_{l:04}.jpg", os.path.join(root_dir, motion, f"{i*100+j*20+k:03}"))
                l += 1

def create_json(root_dir:str, motion:str):
    print("Create json files...")
    with open('./data/cameras.json', 'r') as json_open:
        cams = json.load(json_open)

    dirs = os.listdir(os.path.join(root_dir, motion))
    dirs = [
        f for f in os.listdir(os.path.join(root_dir, motion)) if os.path.isdir(os.path.join(root_dir, motion, f))
    ]
    dirs.sort()

    for dir in dirs:
        datas = []
        current_dir = os.path.join(root_dir,motion,dir)
        files = sorted(glob.glob(f"{current_dir}/*.jpg"))

        for f, file in enumerate(files):       
            data = {'img_name': os.path.basename(file)}
            datas.append(data)
            
        json_write = {'imgs':datas,
                    'Camera Parameter':cams[dir]
                    }
        
        with open(os.path.join(current_dir, "data.json"), 'w') as f:
            json.dump(json_write, f, indent=2)

def main(root_dir:str, motion:str):
    cfg = {
        "detector": "RTDETR",
        "detector_threshold": 0.3,
        "pose_estimator": "ViTPose",
    }
    model = PersonDetectionPipeline(cfg=cfg, device=DEVICE)
    motion_path = os.path.join(root_dir, motion)
    for video_folder in sorted(os.listdir(motion_path)):
        video_path = os.path.join(motion_path, video_folder)
        with open(os.path.join(video_path, "data.json"), 'r',) as json_data:
            data = json.load(json_data)
        rvec = np.array(data["Camera Parameter"]['rvec'])
        tvec = np.array(data["Camera Parameter"]['tvec'])
        K = np.array(data["Camera Parameter"]['Camera Matrix'])
        D = np.array(data["Camera Parameter"]['Distortion Coefficients'])
        pts = cv2.projectPoints(np.array([[0,0,0]],dtype=np.float32), rvec, tvec, K, D)[0][0,0]

        print(video_folder)
        bbox = np.array([None])
        
        for i, img in enumerate(data['imgs']):
            image = Image.open(os.path.join(video_path, img['img_name']))
            detect, kp, _ = model(image)
            if detect is None:
                if bbox[0] == None:
                    bbox = np.array([image.size[0]//2 - 250, image.size[1]//2 - 250, image.size[0]//2 + 250, image.size[1]//2 + 250])
                else:
                    pass
            else:
                id = detect.confidence.argmax()

                bbox = detect.xyxy[id]
                conf = detect.confidence[id]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                square = bbox.astype(np.uint32).copy()

                pad_w = pad_h = min_pad = 120 #200 #40 #120
                ratio_wh = (1,1)

                if (w+2*pad_w)/(h+2*pad_h) > ratio_wh[0]/ratio_wh[1]:
                    pad_h = round(ratio_wh[1]*(w+2*pad_w)/(2*ratio_wh[0]) - h/2)
                elif (w+2*pad_w)/(h+2*pad_h) < ratio_wh[0]/ratio_wh[1]:
                    pad_w = round(ratio_wh[0]*(h+2*pad_h)/(2*ratio_wh[1]) - w/2)

                square[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_w, pad_w], 0, image.size[0])
                square[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_h, pad_h], 0, image.size[1])
                square = expand_bbox_to_include_point(
                    square=square,
                    pt=(pts[0], pts[1]),
                    img_w=image.size[0],
                    img_h=image.size[1],
                    keep_ratio=True,
                    ratio_wh=(1,1),      # 正方形維持（必要に応じて変更）
                    min_margin=80        # 余白が欲しければ増やす
                )
            img['bbox'] = bbox.tolist()
            img['square'] = square.tolist()
            img['pos'] = (float(pts[0]), float(pts[1]))
            if kp is None:
                kp_vector = None
                print(img['img_name'], ":Not detected")
            else:
                # ind = np.argmax(kp.confidence.sum(axis=1))
                kp_vector = np.hstack([kp.xy[id], kp.confidence[id:id+1].T])
                kp_vector = kp_vector.tolist()
                print(f"{img['img_name']}:Detected Conf={float(kp.confidence.mean(axis=1)[id])}")
            img['ViTPose'] = kp_vector        
        with open(os.path.join(video_path, "data.json"), 'w') as f:
            json.dump(data, f, indent=2)

def expand_bbox_to_include_point(
    square: np.ndarray,            # [x1,y1,x2,y2]
    pt: tuple[float, float],     # (px,py)
    img_w: int, img_h: int,
    keep_ratio: bool = True,
    ratio_wh: tuple[int, int] = (1, 1),  # 幅:高さ
    min_margin: int = 0                  # 点からの最低余白(px)
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, square.tolist())
    px, py = float(pt[0]), float(pt[1])

    # 1) まず点を含むまで外向きに拡大（縮小や平行移動はしない）
    #    点が square の外なら、対応する側を広げる（min_marginも確保）
    if px < x1 + min_margin:
        x1 = int(np.floor(px - min_margin))
    if px > x2 - min_margin:
        x2 = int(np.ceil(px + min_margin))
    if py < y1 + min_margin:
        y1 = int(np.floor(py - min_margin))
    if py > y2 - min_margin:
        y2 = int(np.ceil(py + min_margin))

    # 画像境界でクリップ（はみ出しは許容、ただし最終的にクリップ）
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)

    # 2) アスペクト比を保つ場合は、短い辺を拡大のみで合わせる
    if keep_ratio:
        w = x2 - x1; h = y2 - y1
        # 目標比
        rw, rh = ratio_wh
        # 現在の比と比較し、必要に応じて不足次元を"拡大のみ"で満たす
        # (a) 幅基準で高さが不足
        target_h_from_w = int(np.ceil(w * rh / rw))
        # (b) 高さ基準で幅が不足
        target_w_from_h = int(np.ceil(h * rw / rh))

        if target_h_from_w > h:
            # 高さ不足 → 上下に拡大（できる範囲で上にも下にも広げる。足りなければ片側をさらに拡大）
            add = target_h_from_w - h
            up = add // 2
            down = add - up
            y1_new = max(0, y1 - up)
            y2_new = min(img_h, y2 + down)
            # クリップのせいで不足したら、残りを反対側に追加（拡大のみ）
            lack_up = up - (y1 - y1_new)
            lack_down = down - (y2_new - y2)
            if lack_up > 0:
                y2_new = min(img_h, y2_new + lack_up)
            if lack_down > 0:
                y1_new = max(0, y1_new - lack_down)
            y1, y2 = y1_new, y2_new

        # 再計算
        w = x2 - x1; h = y2 - y1
        target_w_from_h = int(np.ceil(h * rw / rh))
        if target_w_from_h > w:
            # 幅不足 → 左右に拡大
            add = target_w_from_h - w
            left = add // 2
            right = add - left
            x1_new = max(0, x1 - left)
            x2_new = min(img_w, x2 + right)
            lack_left = left - (x1 - x1_new)
            lack_right = right - (x2_new - x2)
            if lack_left > 0:
                x2_new = min(img_w, x2_new + lack_left)
            if lack_right > 0:
                x1_new = max(0, x1_new - lack_right)
            x1, x2 = x1_new, x2_new

    # 最終的に整数＆画像内へ
    x1 = int(np.clip(x1, 0, img_w))
    y1 = int(np.clip(y1, 0, img_h))
    x2 = int(np.clip(x2, x1+1, img_w))
    y2 = int(np.clip(y2, y1+1, img_h))

    return np.array([x1, y1, x2, y2], dtype=np.int32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unityからのデータから自動アノテーションしてデータセットを作成します。")
    parser.add_argument("motion", help="モーション名（例: SoccerPlayer_walk）")
    parser.add_argument("--root_dir", default=ROOT_DIR, help="データセットのルートディレクトリ名前（例: test）")
    parser.add_argument("--unityproject_path", default=UNITYPROJECT_PATH, help="Unityプロジェクトのパス（例: SoccerPlayer）")
    parser.add_argument("--reannotate", action="store_true", help="既存のアノテーションを再作成する")
    args = parser.parse_args()
    args.root_dir = os.path.join("data", args.root_dir)
    # example usage:
    # python -m data.datamaker SoccerPlayer_walk --root_dir "test" --unityproject_path "mnt/e/UnityProjects/SoccerPlayer"
    # python -m data.datamaker BasketballPlayer_chestpassright --root_dir "test" --reannotate

    if not args.reannotate:
        move_files(
            root_dir=args.root_dir,
            motion=args.motion,
            unityproject_path=args.unityproject_path
        )
    create_json(
        root_dir=args.root_dir,
        motion=args.motion
    )
    main(
        root_dir=args.root_dir,
        motion=args.motion
    )