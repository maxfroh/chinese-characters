import torch, os
import pandas as pd
from pathlib import Path
from chinese_mnist_model import ChineseMNISTDataset
from torch import Tensor
from torchvision.io import ImageReadMode, read_image, write_jpeg
from torchvision.transforms import v2, InterpolationMode, ToTensor
import matplotlib.pyplot as plt
BASE_DIM = 64
    
def add_to_dict(new_dict:dict[list[int|str]], suite_id, sample_id, code, value, character, transform_idx):
    new_dict['suite_id'].append(suite_id)
    new_dict['sample_id'].append(sample_id)
    new_dict['code'].append(code)
    new_dict['transform_idx'].append(transform_idx)
    new_dict['value'].append(value)
    new_dict['character'].append(character)

resize_sizes = [BASE_DIM, BASE_DIM // 1.25, BASE_DIM // 1.5, BASE_DIM // 1.75]
annotations_file = '..\\data\\chinese_mnist\\chinese_mnist.csv'
new_annotations_file = '..\\data\\chinese_mnist_extended\\chinese_mnist_extended.csv'
old_img_dir = '..\\data\\chinese_mnist\\data'
new_img_dir = '..\\data\\chinese_mnist_extended\\data'
new_dict = {'suite_id':[],'sample_id':[],'code':[],'transform_idx':[],'value':[],'character':[]}

img_labels = pd.read_csv(annotations_file)

for i in range(len(img_labels)):
    row = img_labels.loc[i]
    suite_id = row.loc['suite_id']
    sample_id = row.loc['sample_id']
    code = row.loc['code']
    value = row.loc['value']
    character = row.loc['character']
    transform_idx = 0
    img_path = os.path.join(old_img_dir, f'input_{suite_id}_{sample_id}_{code}.jpg')
    new_path = Path(os.path.join(new_img_dir, f'input_{suite_id}_{sample_id}_{code}_{transform_idx}.jpeg'))
    original_img = read_image(path=img_path, mode=ImageReadMode.GRAY)
    write_jpeg(input=original_img, filename=new_path)
    add_to_dict(new_dict, suite_id, sample_id, code, value, character, 0)
    for size_idx in range(len(resize_sizes)):
        for distortion in [0, 0.2, 0.5]:
            transform = v2.Compose([
                    v2.RandomRotation(degrees=(-15, 15)),
                    v2.RandomPerspective(distortion_scale=distortion, p=1),
                    v2.CenterCrop(size=resize_sizes[size_idx]),
                    v2.Resize(size=(BASE_DIM)),
                ])
            transform_idx += 1
            new_path = Path(os.path.join(new_img_dir, f'input_{suite_id}_{sample_id}_{code}_{transform_idx}.jpeg'))
            transformed_img = transform(original_img)
            write_jpeg(input=transformed_img, filename=new_path)
            add_to_dict(new_dict, suite_id, sample_id, code, value, character, transform_idx)

pd.DataFrame(new_dict).to_csv(new_annotations_file, index=False)
print(len(new_dict['character']))