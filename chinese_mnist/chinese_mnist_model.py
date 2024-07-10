import torch, numpy
from torch import nn
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from chinese_mnist_data import ChineseMNISTDataset, ChineseMNISTNN

labels_map = {
        0: '零',
        1: '一',
        2: '二',
        3: '三',
        4: '四',
        5: '五',
        6: '六',
        7: '七',
        8: '八',
        9: '九',
        10: '十',
        11: '百',
        12: '千',
        13: '万',
        14: '亿'
    }

idx_labels_map = {
    '零': 0,
    '一': 1,
    '二': 2,
    '三': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '七': 7,
    '八': 8,
    '九': 9,
    '十': 10,
    '百': 11,
    '千': 12,
    '万': 13,
    '亿': 14
}

def main():
    plt.rc('font', family='Microsoft YaHei')
    print("Started")
    dataset = ChineseMNISTDataset(
        '../data/chinese_mnist/chinese_mnist.csv',
        '../data/chinese_mnist/data',
        None,
        None,
        )
    
    # plot(training_data=dataset)

    non_testing_size = int(len(dataset) * 0.8)
    testing_size = len(dataset) - non_testing_size
    training_size = non_testing_size * 0.8
    validation_size = non_testing_size - training_size
    # non_testing_data, testing_data = random_split(dataset, [non_testing_size, testing_size])
    # training_data, validation_data = random_split(non_testing_data, [training_size, validation_size])
        
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    model = ChineseMNISTNN().to(device)

    print(f"Using {device} device")
    print(model)
    
    # X = torch.rand(1, 64, 64, device=device)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    

def plot(training_data):
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        print(label)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

if __name__ == '__main__':
    main()
