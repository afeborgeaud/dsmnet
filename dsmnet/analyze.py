from net import LeNetD3
from dataset import RSDataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def imshow_grid(x):
    """
    Arguments:
        x (tensor): tensor with ndim=3

    Returns:
        Figure
        array of Axes
    """

    nc = x.shape[0]
    nrow = int(np.sqrt(nc))
    ncol = int(nc / nrow)
    if nc % nrow != 0:
        ncol += 1

    nx = x.shape[2]
    ny = x.shape[1]
    fig_width = 10
    fig_height = fig_width * nrow / ncol * ny / nx * 1.01
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    gs = fig.add_gridspec(nrow, ncol, wspace=0, hspace=0)
    axes = gs.subplots()

    for i, ax in enumerate(np.ravel(axes)):
        if i < nc:
            ax.imshow(x[i], cmap='Spectral')

    for ax in np.ravel(axes):
        ax.set(xticks=[], yticks=[])

    return fig, axes


if __name__ == '__main__':
    # dataset
    path_x = '../tests/X.npy'
    path_y = '../tests/Y.npy'
    dataset = RSDataset(path_x, path_y)
    no = dataset[0][1].shape[0]
    trainloader = DataLoader(dataset, batch_size=len(dataset),
                             shuffle=False)

    # net
    net = LeNetD3(1, no)
    net.load_state_dict(torch.load('./dsmnet.pth'))

    input, target = next(iter(trainloader))

    output, hiddens = net(input)

    for i_input in range(20):
        figname = '../tests/figures/input_{}.pdf'.format(i_input+1)
        fig, axes = imshow_grid(input[i_input].detach())
        fig.savefig(figname, bbox_inches='tight')
        plt.close(fig)

        rootname = '../tests/figures/output_conv'
        for i in range(len(hiddens)):
            fig, axes = imshow_grid(hiddens[i][i_input].detach())
            figname = rootname + '{:d}_{:d}.pdf'.format(i+1, i_input+1)
            fig.savefig(figname, bbox_inches='tight')
            plt.close(fig)
