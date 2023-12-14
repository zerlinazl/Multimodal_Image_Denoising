# import sys
# from pathlib import Path
# from importlib import import_module

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
# import params as args
import torchvision
import torchvision.transforms as T
import os
from torch.utils.data import DataLoader
from uvcgan.consts import USE_META

SPLIT_TRAIN = 'train'
SPLIT_VAL   = 'val'
SPLIT_TEST  = 'test'


class custom_dataset(Dataset):
    def __init__(self, dataset, path, domain, split=SPLIT_TRAIN, **kwargs):
        super().__init__()
        self._path = os.path.join(path, split, domain)
        self._files = sorted(os.listdir(self._path))

    def __len__(self):
        return len(self._files)

    # def get_meta(filename):
    #     # 0126_006_S6_00400_00200_4400_L.PNG
    #     f = filename.split(".")[0].split("_") # get all relevant fields
    #     # numerical data
    #     iso = float(f[3])
    #     shutter = float(f[4])
    #     temp = float(f[5])
    #     meta = [iso, shutter, temp]
    #     return meta

    def __getitem__(self, idx):    

        fpath = self._path + "/" + self._files[idx]
        # meta = get_meta(self._files[idx])

        filename = self._files[idx]
        f = filename.split(".")[0].split("_") # get all relevant fields
        # numerical data
        iso = float(f[3])
        shutter = float(f[4])
        temp = float(f[5])
        meta = torch.Tensor([iso, shutter, temp])

        img = Image.open(fpath).convert("RGB")
        t = T.Compose([T.ToTensor(), T.RandomCrop(256)])
        img = t(img)

        # print(meta)
        
        if USE_META:
            return img, meta
        else:
            return img
    