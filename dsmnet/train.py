from nn import LeNet
from dataset import RSDataset
import torch.optim as optim
import torch.nn as nn
import torch

def train(net, lr=0.001, momentum=0.9):
    trainset = RSDataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    for epoch in range(2):

        running_loss = 0.
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1,
                                                          running_loss / 2000))

    path = './dsmnet.pth'
    torch.save(net.state_dict(), path)
