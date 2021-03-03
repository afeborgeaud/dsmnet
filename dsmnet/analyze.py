from net import LeNetD3
from dataset import RSDataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # dataset
    path_x = '../tests/X.npy'
    path_y = '../tests/Y.npy'
    dataset = RSDataset(path_x, path_y)
    no = dataset[0][1].shape[0]
    trainloader = DataLoader(dataset, batch_size=32,
                             shuffle=True)

    # net
    net = LeNetD3(1, no)
    net.load_state_dict(torch.load('./dsmnet.pth'))

    input, target = next(trainloader)

    out1 = net.conv1(input)

    print(out1.shape)

    plt.imshow(out1)
