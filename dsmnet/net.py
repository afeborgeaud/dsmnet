import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """Implements LeNet neural network.

    The input image should be of dimensions (nc, 32, 32).
    The activation function used is tanh, to modelize input
    data with positive and negative amplitudes.

    Args:
        nc (int): number of channels (default is 1)
        no (int): number

    """

    def __init__(self, nc=1, no=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(nc, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, no)
        self = self.double()

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNetD3(nn.Module):
    """Implements LeNet neural network.

    The input image should be of dimensions (nc, 84, 60).
    The activation function used is tanh, to modelize input
    data with positive and negative amplitudes.
    The convolutions are all valid.

    Args:
        nc (int): number of channels (default is 1)
        no (int): number

    """

    def __init__(self, nc=1, no=10):
        super(LeNetD3, self).__init__()
        self.conv1 = nn.Conv2d(nc, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 7 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, no)
        self = self.double()

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        x = x.view(-1, 32 * 7 * 4)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

