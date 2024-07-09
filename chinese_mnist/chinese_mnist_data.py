import os
import pandas as pd
from torchvision.io import read_image
from torch import nn
from torch.utils.data import Dataset

class ChineseMNISTDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self._get_path(idx) 
        image = read_image(img_path)
        # label is chinese character
        label = self.img_labels.iloc[idx, 4]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _get_path(self, idx):
        path = f'input_{self.img_labels.iloc[idx, 0]}_{self.img_labels.iloc[idx, 1]}_{self.img_labels.iloc[idx, 2]}.jpg'
        return os.path.join(self.img_dir, path)
    
    
class ChineseMNISTNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 15),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits