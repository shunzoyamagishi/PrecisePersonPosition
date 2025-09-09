import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from Depth_Anything_V2.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2, DPTHead
from Depth_Anything_V2.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose


"""

    This code is based on:
    Depth-Anything-V2 (https://github.com/DepthAnything/Depth-Anything-V2)
    Licensed under CC-BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

    This implementation includes modifications by Shunzo Yamagishi.

"""


class DPTDecorder(DPTHead):
    def __init__(
            self, 
            in_channels, 
            features=256, 
            use_bn=False, 
            out_channels=[256, 512, 1024, 1024], 
            use_clstoken=False
        ):
            super(DPTDecorder, self).__init__(in_channels, features=features, use_bn=use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
            

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = F.interpolate(path_1, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)

        depth = self.scratch.output_conv1(path_1)
        depth = F.interpolate(depth, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        depth = self.scratch.output_conv2(depth)
        
        return out, depth
    
    
class DepthFeatureExtractor(DepthAnythingV2):
    def __init__(
            self, 
            encoder='vitl', 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False,
            device='cuda'
        ):
            super(DepthFeatureExtractor, self).__init__(encoder=encoder, features=features, out_channels=out_channels, use_bn=use_bn, use_clstoken=use_clstoken)
            self.depth_head = DPTDecorder(in_channels=self.pretrained.embed_dim, features=features, use_bn=use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
            for p in self.pretrained.parameters():
                p.requires_grad = False

            self.device = device
            self.to(self.device)


    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        features, depth = self.depth_head(features, patch_h, patch_w)

        depth = F.relu(depth)
                
        return features, depth
    

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=224):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        _, depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    