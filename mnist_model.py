from torch import nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from source import random_voice_attack
import pandas as pd

BATCH_SIZE = 100
LR = 0.001
EPOCHS = 6


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 10000)
        self.fc2 = nn.Linear(10000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class LightNet(nn.Module):
    def __init__(self):
        super(LightNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


def get_data(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=True)
    train = DataLoader(MyDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=False)
    test = DataLoader(MyDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False)
    return train, test


def training(train, test):
    model = DeepNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train loss": [], "test loss": [], "train acc": [], "test acc": []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for i, (x, y) in enumerate(train):
            out = model(x)
            loss = criterion(out, y)
            history["train loss"].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = y.size(0)
            _, predicted = torch.max(out.data, 1)
            correct = (predicted == y).sum().item()
            history["train acc"].append(correct / total)
            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch, EPOCHS, i + 1, total, loss.item(), (correct / total) * 100))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test:
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        print("Test Accuracy: {:.2f}".format(correct / total))
    torch.save(model.state_dict(), "mnist_deep_weights.pt")


class TripleNet(nn.Module):
    def __init__(self):
        super(TripleNet, self).__init__()
        self.m1 = ConvNet()
        self.m2 = ConvNet()
        self.m3 = ConvNet()

    def forward(self, x):
        return (self.m1(x) + self.m2(x) + self.m3(x)) / 3


if __name__ == "__main__":
    data = pd.read_csv("train.csv").values
    x, y = data[:, 1:], data[:, 0]
    x = x.reshape(len(x), 1, 28, 28)
    train, test = get_data(x, y)
    training(train, test)
