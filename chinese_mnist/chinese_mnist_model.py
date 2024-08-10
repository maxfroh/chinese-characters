import torch, time
import numpy as np
from torch import nn
from torchvision.transforms import ToTensor, Lambda, v2, InterpolationMode
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from chinese_mnist_data import ChineseMNISTDataset, ChineseMNISTNN

BASE_DIM = 64

def plot(training_loss, testing_loss, version, learning_rate, batch_size):
    plt.figure(figsize=(10,5))
    plt.title("Training and Testing Loss")
    plt.plot(training_loss, label="train")
    plt.plot(testing_loss, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f'model_{learning_rate}_{batch_size}_{version}.png')

def train(dataloader: DataLoader, model: ChineseMNISTNN, loss_func, optimizer, batch_size, device):
    size = len(dataloader.dataset)
    model.train()
    loss = 0
    for batch, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % ((size/batch_size) // 10 - 1) == 0:
            loss, current = loss.item(), batch * batch_size + len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


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
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    return (accuracy, test_loss)


def main():
    print("Started")
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    
    dataset = ChineseMNISTDataset(
        '..\\data\\chinese_mnist_extended\\chinese_mnist_extended.csv',
        '..\\data\\chinese_mnist_extended\\data',
    )
    
    version = 'v4'

    batch_sizes = [32, 64, 128]
    training_proportion = 0.8
    training_data, testing_data = random_split(dataset, [training_proportion, (1-training_proportion)])
    
    epochs = 60
    learning_rates = [1e-3, 1e-4, 1e-5]
    train_losses = np.zeros((len(learning_rates), len(batch_sizes), epochs))
    test_accuracies = np.zeros((len(learning_rates), len(batch_sizes), epochs))
    test_losses = np.zeros((len(learning_rates), len(batch_sizes), epochs))
    models:list[list[ChineseMNISTNN]] = [[None] * len(batch_sizes) for i in range(len(learning_rates))]
    times:list[list[float]] = np.zeros((len(learning_rates), len(batch_sizes)))

    for i, learning_rate in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
            start_time = time.time()
            print(f"\nLearning rate {learning_rate}, Batch size {batch_size}\n")
            model = ChineseMNISTNN().to(device)
            loss_func = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
            for epoch in range(epochs):
                epoch_time = time.time()
                print(f"Epoch {epoch+1}\n-------------------------------")
                train_loss = train(train_dataloader, model, loss_func, optimizer, batch_size, device)
                test_acc, test_loss = test(test_dataloader, model, loss_func, device)
                train_losses[i][j][epoch] = train_loss
                test_accuracies[i][j][epoch] = test_acc
                test_losses[i][j][epoch] = test_loss
                print(f"Epoch {epoch+1} took {time.time()-epoch_time}s to run\n")
            models[i][j] = model
            total_time = time.time() - start_time
            times[i][j] = total_time
            plot(train_losses[i][j], test_losses[i][j], version, learning_rate, batch_size)

    for i in range(len(learning_rates)):
        for j in range(len(batch_sizes)):
            torch.save(models[i][j].state_dict(), f'chinese_mnist_model_{learning_rates[i]}_{batch_sizes[j]}_{version}.pth')
    print("Done!")
    print("Results:")
    final_str = ""
    for i in range(len(learning_rates)):
        for j in range(len(batch_sizes)):
            string = f"\tLearning rate {learning_rates[i]} and Batch size {batch_sizes[j]}:\n\t\tAccuracy: {test_accuracies[i][j][epochs-1]}\n\t\tLoss: {test_losses[i][j][epochs-1]}\n\t\t Time: {times[i][j]} \n\t\t Average Time per Epoch: {times[i][j] / epochs}"
            print(string)
            final_str += string + "\n\t\t\t\t--------\n"
    with open(f"{version}_results.txt", "w") as file:
        file.write(final_str)

if __name__ == '__main__':
    main()
