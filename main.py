import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import seaborn as sns

epochs = 10
batch_size = 32

# See https://github.com/pytorch/vision/issues/168
class ChunkSampler(Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 40000
NUM_VAL = 10000

class SigNet(nn.Module):
    def __init__(self, dropout=0.2):
        super(SigNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)

class ReluNet(nn.Module):
    def __init__(self, dropout=0.2):
        super(ReluNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)

class TwoLayerNet(nn.Module):
    def __init__(self, dropout=0.2):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)

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
                epoch, batch_idx * len(data), 40000,
                100. * batch_idx / len(train_loader), loss.item()))
    val_loss /= len(train_loader)
    loss_vector.append(val_loss)

def validate(model, loader, loss_vector, accuracy_vector, test_name='Validation'):
    model.eval()
    val_loss, correct, num_samples = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            val_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            num_samples += pred.size(0)

    val_loss /= num_samples 
    loss_vector.append(val_loss)

    accuracy = 100. * correct / num_samples
    accuracy_vector.append(accuracy)

    print('\n' + test_name + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, num_samples, accuracy))

    return val_loss, correct, num_samples, accuracy

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
        validate(model, validation_loader, lossv, accv)

    return losst, lossv, accv

def run_test(m, lr, momentum, decay, dropout):
    model = m(dropout)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    r = run_network(epochs, model, optimizer)
    s = format_results(epochs, lr, momentum, decay, dropout, r[0], r[1], r[2])
    return s

def run_model_test(m, lr, momentum, decay, dropout):
    losst, lossv, accv = [], [], []
    model = m(dropout)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    run_network(epochs, model, optimizer)
    return validate(model, test_loader, lossv, accv, test_name='Test')

transform = transforms.Compose([
                transforms.ToTensor()
                ])

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True,
            transform=transform),
            sampler=ChunkSampler(NUM_TRAIN, 0),
            batch_size=batch_size)

validation_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True,
            transform=transform),
            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN),
            batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False,
            transform=transform),
            batch_size=batch_size, shuffle=False)

def two_layer_tuning():
    results = np.empty((0,8))
    for lr in [.1, .01, .001]:
        s = run_test(TwoLayerNet, lr, .5, 0, 0)
        results = np.append(results, s, axis=0)
    np.savetxt('twolayer_lr_results.csv', results)

    results = np.empty((0,8))
    for momentum in [.1, 0.5, 0.9]:
        s = run_test(TwoLayerNet, .0001, momentum, 0, 0.2)
        results = np.append(results, s, axis=0)
    np.savetxt('twolayer_momentum_results.csv', results)

    results = np.empty((0,8))
    for dropout in [0.0, 0.2, 0.5]:
        s = run_test(TwoLayerNet, .0001, .9, 0, dropout)
        results = np.append(results, s, axis=0)
    np.savetxt('twolayer_dropout_results.csv', results)

def single_layer_tuning():
    # Run both sigmoid and relu activation functions through a range of learning rates
    # Sigmoid network
    results = np.empty((0,8))
    for lr in [.1, .01, .001, .0001]:
        s = run_test(SigNet, lr, 0.5, 0, 0.2)
        results = np.append(results, s, axis=0)
    np.savetxt('sigmoid_results.csv', results)

    # Relu network
    results = np.empty((0,8))
    for lr in [.1, .01, .001, .0001]:
        s = run_test(ReluNet, lr, 0.5, 0, 0.2)
        results = np.append(results, s, axis=0)
    np.savetxt('relu_results.csv', results)

    # Compare different values of momentum
    results = np.empty((0,8))
    for momentum in [.25, 0.5, 0.75, 0.9]:
        s = run_test(ReluNet, .001, momentum, 0, 0.2)
        results = np.append(results, s, axis=0)
    np.savetxt('momentum_results.csv', results)

    # Compare different dropout rates
    results = np.empty((0, 8))
    for decay in [0, .1, .25, .5]:
        s = run_test(ReluNet, .001, .9, decay, 0.2)
        results = np.append(results, s, axis=0)
    np.savetxt('decay_results.csv', results)

    # Compare dropouts
    results = np.empty((0, 8))
    for dropout in [0, .2, .5, .75]:
        s = run_test(ReluNet, .001, .9, 0, dropout)
        results = np.append(results, s, axis=0)
    np.savetxt('dropout_results.csv', results)

def single_layer_test(model_class):
    r = run_model_test(model_class, .001, 0.9, 0, 0.2)
    np.savetxt('single_layer_test.csv', np.array(r))

def two_layer_test():
    r = run_model_test(TwoLayerNet, .0001, .9, 0, 0.2)
    np.savetxt('single_layer_test.csv', np.array(r))

single_layer_tuning()
single_layer_test(SigNet)
single_layer_test(ReluNet)

two_layer_tuning()
two_layer_test()
