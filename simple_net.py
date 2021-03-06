import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import numpy as np
import seaborn as sns

epochs = 1
batch_size = 32

class Net(nn.Module):
    def __init__(self, dropout=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        return F.sigmoid(self.fc2(x))


def train(epoch, model, optimizer, loss_vector, log_interval=100):
    model.train()
    val_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        val_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    val_loss /= len(train_loader)
    loss_vector.append(val_loss)

def validate(model, loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            val_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(validation_loader.dataset), accuracy))

def format_results(epochs, lr, momentum, decay, dropout, losst, lossv, accv):
    results = np.append(np.array([np.arange(1, epochs+1)]).T, np.array([losst]).T, axis=1)
    results = np.append(results, np.array([lossv]).T, axis=1)
    results = np.append(results, np.array([accv]).T, axis=1)
    results = np.append(results, np.full((epochs, 1), lr, dtype=np.float64), axis=1)
    results = np.append(results, np.full((epochs, 1), momentum, float), axis=1)
    results = np.append(results, np.full((epochs, 1), decay, float), axis=1)
    results = np.append(results, np.full((epochs, 1), dropout, float), axis=1)

    return results

def run_network(epochs, model, optimizer):
    losst, lossv, accv = [], [], []

    for epoch in range(1, epochs+1):
        train(epoch, model, optimizer, losst)
        validate(model, lossv, accv)

    return losst, lossv, accv

transform = transforms.Compose([
                transforms.ToTensor()
                ])

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True,
            transform=transform),
            batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False,
            transform=transform),
            batch_size=batch_size, shuffle=False)

def run_test(m, lr, momentum, decay, dropout):
    model = m(dropout)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    r = run_network(epochs, model, optimizer)
    s = format_results(epochs, lr, momentum, decay, dropout, r[0], r[1], r[2])
    return s

run_test(Net, .001, .9, 0, 0.2)
