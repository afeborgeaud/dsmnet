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


def train(net, dataloader, losses, epoch, writer):
    net.train()
    running_loss = 0.
    count = 0
    for i, data in enumerate(dataloader, 0):
        inputs, targets = data

        optimizer.zero_grad()

        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    writer.add_scalar('training loss',
                      loss.item(),
                      epoch * len(trainloader) + i)
    writer.add_figure('prediction vs. actuals',
                      plot_inputs_pred(net, inputs, targets),
                      global_step=epoch * len(trainloader) + i)
            
    if epoch % 20 == 0:
        print('{} train_loss: {:.3e}'.format(epoch, loss.item()))


def test(net, dataloader, criterion, losses):
    net.eval()
    data = next(iter(dataloader))
    inputs, targets = data
    outputs, _ = net(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())

def target_to_plot_arr(target):
    y = target.detach().numpy()
    y[0] *= 0.5
    y[1] = 200 + y[1] * 190
    vs = np.array([0., 0., y[0], y[0]])
    h = np.array([400., y[1], y[1], 0.])
    return vs, h

def plot_inputs_pred(net, inputs, targets):
    outputs, _ = net(inputs)
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))
    for i in range(axes.shape[1]):
        vs_pred, h_pred = target_to_plot_arr(outputs[i])
        vs_target, h_target = target_to_plot_arr(targets[i])

        axes[0, i].plot(vs_target, h_target, '-k', label='target')
        axes[0, i].plot(vs_pred, h_pred, '-r', label='pred')
        axes[0, i].set(xlabel='dVs (km/s)',
                       xlim=[-0.5, 0.5],
                       ylim=[0, 400])
        axes[0, i].legend()

        imshow(inputs[i], ax=axes[1, i])
        axes[1, i].set(xlabel='Distance - 70 (deg)')

    axes[0, 0].set(ylabel='Height above CMB (km)')
    axes[1, 0].set(ylabel='Time + 20 before S (s)')

    return fig


def imshow(img, ax=None):
    img = img.mean(dim=0)
    npimg = img.numpy()
    if ax is not None:
        ax.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(npimg, cmap='Greys')


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
    trainloader = DataLoader(trainset, batch_size=32,
                             shuffle=True)
    testloader = DataLoader(valset, batch_size=n_train,
                            shuffle=False, drop_last=False)

    # model
    net = LeNetD3(nc=1, no=no)

    # tensorboard writer
    writer = SummaryWriter('runs/dsmset_experiment_1')

    dataiter = iter(trainloader)
    inputs, targets = dataiter.next()

    img_grid = torchvision.utils.make_grid(inputs)
    imshow(img_grid)
    writer.add_image('Record section images', img_grid)

    writer.add_graph(net, inputs)


    # optimizer
    lr = 0.01
    momentum = 0.9
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # train
    n_epoch = 500
    train_losses = []
    test_losses = []
    for epoch in range(1, n_epoch + 1):
        train(net, trainloader, train_losses, epoch, writer)
        test(net, testloader, criterion, test_losses)

    writer.close()

    # net.load_state_dict(torch.load('./dsmnet.pth'))

    path = './dsmnet.pth'
    torch.save(net.state_dict(), path)

    # fig, ax = plt.subplots(1)
    # ax.plot(train_losses, label='train')
    # ax.plot(test_losses, label='test')
    # ax.set(xlabel='epoch', ylabel='loss')
    # plt.legend()
    # fig.savefig('./convergence.pdf', bbox_inches='tight')
    # plt.close(fig)
