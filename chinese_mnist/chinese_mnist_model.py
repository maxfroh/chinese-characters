import torch, time
from torch import nn
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from chinese_mnist_data import ChineseMNISTDataset, ChineseMNISTNN

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

        if batch % 18 == 0:
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

    epochs = 100
    learning_rates = [1e-2, 1e-3, 1e-4]
    test_accuracies = torch.zeros(len(learning_rates))
    test_losses = torch.zeros(len(learning_rates))
    models:list[ChineseMNISTNN] = []
    times:list[float] = []

    for i, learning_rate in enumerate(learning_rates):
        start_time = time.time()
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
        models.append(model)
        total_time = time.time() - start_time
        times.append(total_time)

    for i in range(len(learning_rates)):
        torch.save(models[i].state_dict(), f'chinese_mnist_model_{learning_rates[i]}_v3.pth')
    print("Done!")
    print("Results:")
    for i in range(len(learning_rates)):
        print(f"\tLearning rate {learning_rates[i]}:\n\t\tAccuracy: {test_accuracies[i]}\n\t\tLoss: {test_losses[i]}\n\t\t Time: {times[i]}")

if __name__ == '__main__':
    main()
