import cv2
import numpy as np
import torch
from torchvision import transforms
import torch
import torch.nn as nn

from model_simple import SwinTransformerUnet_T, SwinTransformerUnet_S
import argparse
import os
import re
from src.utils import add_padding, calc_max
# デバイスを指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
softmax = nn.Softmax(dim=1)
mode = {
            "t":SwinTransformerUnet_T,
            "s":SwinTransformerUnet_S,
        }

def predict(img:np.ndarray, model:nn.Module):

    img_originalsize  = (img.shape[1],img.shape[0])
    img_pad, pad_w, pad_h = add_padding(img, model.aspect)
    img_paddedsize  = (img_pad.shape[1],img_pad.shape[0])
    img_resize = cv2.resize(img_pad, model.imgsize)
    

    image_tensor = transform(img_resize)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        predict = model(image_tensor)
        predict = softmax(predict)

    pred_result = predict[0][0].to('cpu').detach().numpy()
    pred_result = cv2.resize(pred_result, img_paddedsize)
    pred_result = pred_result[pad_h:pad_h+img_originalsize[1], pad_w:pad_w+img_originalsize[0]]

    peak = calc_max(pred_result)
    
    return pred_result, peak


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='推論のコード')
    parser.add_argument('image_path', help='モデルの入力画像のパス')
    parser.add_argument('model_path', help='swintransformerの学習済みモデルのパス')
    args = parser.parse_args()

    assert os.path.exists(args.image_path), f"input image {args.image_path} is not exists."
    assert os.path.exists(args.model_path), f"pretrain model {args.image_path} is not exists."

    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    modelname = os.path.splitext(os.path.basename(args.model_path))[0]
    # 正規表現で括弧内の数字を取得
    _size = re.findall(r'\d+', modelname)
    swinmodel = modelname.split("_")[-1]
    # 取得した文字列をint型に変換
    in_size = (int(_size[0]), int(_size[1]))

    model = mode[swinmodel](imgsize=in_size, num_classes=2)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    
    pred_result, peak = predict(img, model)
    print(peak)