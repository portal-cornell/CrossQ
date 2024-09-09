# TODO (yuki): Have to change the path to work with the folder strucuture in CrossQ
#   Probably a better way to do this

from vlm_reward.utils.SemanticGuidedHumanMatting.model.model import HumanMatting
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from loguru import logger

import gc

class HumanSegmentationModel(nn.Module):
    def __init__(self, rank, weight_path='~/research/language_irl/pretrained_checkpoints/SGHM-ResNet50.pth'):
        super().__init__()
        
        model = HumanMatting(backbone='resnet50')
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model = nn.DataParallel(model).cuda(rank).eval()
            model.load_state_dict(torch.load(weight_path))
        else:
            state_dict = torch.load(weight_path, map_location="cpu")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.eval()
        self.model=model
        self.device = device
        self.infer_size = 1280

    def forward_logits(self, img):
        """
        img: torch.Tensor, (3, H, W) or (B, 3, H, W)
        """
        with torch.no_grad():
            if len(img.size()) == 3:
                _, h, w = img.size()
                img = img[None, :, :, :].to(self.device)
            else:
                _, _, h, w = img.size()
                img = img.to(self.device)

            if w >= h:
                rh = self.infer_size
                rw = int(w / h * self.infer_size)
            else:
                rw = self.infer_size
                rh = int(h / w * self.infer_size)
            rh = rh - rh % 64
            rw = rw - rw % 64    

            input_tensor = F.interpolate(img, size=(rh, rw), mode='bilinear')

            pred = self.model(input_tensor) 
            
            pred_segment = pred['segment']
            pred_segment = F.interpolate(pred_segment, size=(h, w), mode='bilinear')

            # Make sure this tensor which is GPU-Memory intensive is deleted and frees up its memory
            del input_tensor
            torch.cuda.empty_cache()
            gc.collect()

        return pred_segment

    def forward(self, img, thresh=.5):
        pred_segment = self.forward_logits(img)

        return pred_segment > thresh