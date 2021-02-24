from net import LeNetD3
from dataset import RSDataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt


def train(net, dataloader, losses):
    net.train()
    running_loss = 0.
    count = 0
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    losses.append(running_loss / count)


def test(net, dataloader, criterion, losses):
    net.eval()
    data = next(iter(dataloader))
    inputs, targets = data
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())


if __name__ == '__main__':
    # dataset
    path_x = '../tests/X.npy'
    path_y = '../tests/Y.npy'
    dataset = RSDataset(path_x, path_y)
    no = dataset[0][1].shape[0]
    ns = len(dataset)
    n_train = int(0.9 * ns)
    n_val = ns - n_train
    trainset, valset = random_split(dataset, [n_train, n_val])
    trainloader = DataLoader(trainset, batch_size=30,
                             shuffle=True)
    testloader = DataLoader(valset, batch_size=n_train,
                            shuffle=False, drop_last=False)

    # model
    net = LeNetD3(nc=1, no=no)

    # optimizer
    lr = 0.01
    momentum = 0.9
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # train
    n_epoch = 300
    train_losses = []
    test_losses = []
    for epoch in range(1, n_epoch + 1):
        train(net, trainloader, train_losses)
        test(net, testloader, criterion, test_losses)

        if epoch % 20 == 0:
            print('{} train_loss: {:.3e}'.format(epoch, train_losses[-1]))

    # net.load_state_dict(torch.load('./dsmnet.pth'))

    path = './dsmnet.pth'
    torch.save(net.state_dict(), path)

    fig, ax = plt.subplots(1)
    ax.plot(train_losses, label='train')
    ax.plot(test_losses, label='test')
    ax.set(xlabel='epoch', ylabel='loss')
    plt.legend()
    fig.savefig('./convergence.pdf', bbox_inches='tight')
    plt.close(fig)
