import torch
import torch.nn as nn
from schnetpack.datasets import *
from schnetpack.nn.activations import shifted_softplus


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


class ShiftedSoftplus(nn.Module):
    def __init__(self, inplace=False):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, input):
        return shifted_softplus(input)


def l1_loss(y_pred, y_true):
    # TODO should use weight for each types?
    loss = torch.abs(y_true - y_pred)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)
    loss = torch.sum(loss)

    return loss


def smooth_l1_loss(delta=1.):
    def loss_fn(y_pred, y_true):
        loss = torch.abs(y_true - y_pred)
        loss = torch.where(loss < delta, 0.5 * (loss ** 2), delta * loss - 0.5 * (delta ** 2))
        loss = loss.mean(dim=0)
        loss = torch.log(loss)
        loss = torch.sum(loss)
        return loss

    return loss_fn


def log_cosh_loss(y_pred, y_true):
    loss = torch.cosh(y_pred - y_true)
    loss = torch.log(loss)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)
    loss = torch.sum(loss)
    return loss
