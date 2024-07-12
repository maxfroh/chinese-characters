import torch
import numpy
from torch import nn
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import random_split, DataLoader
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


def train(dataloader: DataLoader, model: ChineseMNISTNN, loss_func, optimizer, batch_size, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader, model: ChineseMNISTNN, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (accuracy, test_loss)


def main():
    # plt.rc('font', family='Microsoft YaHei')
    print("Started")
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    
    dataset = ChineseMNISTDataset(
        '..\\data\\chinese_mnist\\chinese_mnist.csv',
        '..\\data\\chinese_mnist\\data',
        None,
        None,
    )

    batch_size = 64
    training_proportion = 0.8
    training_data, testing_data = random_split(dataset, [training_proportion, (1-training_proportion)])
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    epochs = 20
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
    test_accuracies = torch.zeros(len(learning_rates))
    test_losses = torch.zeros(len(learning_rates))
    models:list[ChineseMNISTNN] = []

    for i, learning_rate in enumerate(learning_rates):
        print(f"\nLearning rate {learning_rate}\n")
        model = ChineseMNISTNN().to(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train(train_dataloader, model, loss_func, optimizer, batch_size, device)
            test_acc, test_loss = test(test_dataloader, model, loss_func, device)
            test_accuracies[i] = test_acc
            test_losses[i] = test_loss
        models[i] = model

    for i in range(len(learning_rates)):
        torch.save(models[i].state_dict(), f'chinese_mnist_model_{learning_rates[i]}.pth')
    print("Done!")
    print("Results:")
    for i in range(learning_rates):
        print(f"\tLearning rate {learning_rates[i]}:\n\t\tAccuracy: {test_accuracies[i]}\n\t\tLoss: {test_losses[i]}")

    # X = torch.rand(1, 64, 64, device=device)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")


def plot(training_data):
    figure = plt.figure(figsize=(8, 8))
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
