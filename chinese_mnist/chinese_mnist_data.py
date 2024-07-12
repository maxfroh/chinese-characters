import os
import pandas as pd
import numpy as np
import torch
import torch.utils
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch import nn, Tensor
from torch.utils.data import Dataset

class ChineseMNISTDataset(Dataset):
    map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, 
        '八': 8, '九': 9, '十': 10, '百': 11, '千': 12, '万': 13, '亿': 14,
        0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七',
        8: '八', 9: '九', 10: '十', 11: '百', 12: '千', 13: '万', 14: '亿'
    }
    
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self._get_path(idx) 
        image:Tensor = read_image(img_path)
        image = image.type(torch.float32)
        # label is chinese character
        label = self.img_labels.iloc[idx, 4]
        label = idx_labels_map[label]
        label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor)
        if self.transform:
            image = self.transform
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _get_path(self, idx):
        path = f'input_{self.img_labels.iloc[idx, 0]}_{self.img_labels.iloc[idx, 1]}_{self.img_labels.iloc[idx, 2]}.jpg'
        return os.path.join(self.img_dir, path)
    
    
class ChineseMNISTNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=0)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 15),
        )
        
    def forward(self, x:Tensor):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits