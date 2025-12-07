import torch
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_file, transform=None, loader=default_loader):
        self.csv = pd.read_csv(csv_file)
        self.root = root_file
        self.transform = transform
        self.loader = loader
        self.imgs = []

        for idx in range(len(self.csv['filename'])):
            self.imgs.append((self.csv['filename'][idx], int(self.csv['label'][idx])))

    def __getitem__(self, item):
        path, label = self.imgs[item]
        path = self.root + path
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
