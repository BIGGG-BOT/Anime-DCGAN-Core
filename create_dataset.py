import os
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid
import torch.utils.data.dataset as Dataset


class My_dataset(Dataset.Dataset):

    def __init__(self, path, transform):

        self.path = path
        loc_list = os.listdir(self.path)
        self.loc_list = loc_list
        self.tranform = transform

    def __getitem__(self, index):

        loc_data = os.path.join(self.path, self.loc_list[index])
        data = Image.open(loc_data)
        data = np.array(data)
        data = Image.fromarray(data)  # 若导入图像不为PIL格式则需要转换
        data = self.tranform(data)
        return data

    def __len__(self):

        return len(self.loc_list)


def save_img(tensor, fp):

    grid = make_grid(tensor)
    ndarr = (grid.mul(0.5).add_(0.5)).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)

