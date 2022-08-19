import torch
import numpy as np
from lib.utils.misc import NestedTensor

class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray, amask_arr_box : np.ndarray = None):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0).clamp(0.0, 1.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)

        if amask_arr_box is None:
            return NestedTensor(img_tensor_norm, amask_tensor)
        else:
            amask_arr_box = torch.from_numpy(amask_arr_box).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
            return NestedTensor(img_tensor_norm, amask_tensor, amask_arr_box)

class Preprocessor_wo_mask(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr : np.ndarray= None):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0).clamp(0.0, 1.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        if amask_arr is not None:
            amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
            return img_tensor_norm, amask_tensor

        return img_tensor_norm