import numpy as np
import torch
from torch import Tensor
from typing import Optional, List, Tuple, Union
import torch.nn.functional as F

def clamp_image(x, eps, clamp_min=0, clamp_max=1):
    min_x = torch.clamp(x-eps, min=clamp_min)
    max_x = torch.clamp(x+eps, max=clamp_max)
    x_center = 0.5 * (max_x + min_x)
    x_beta = 0.5 * (max_x - min_x)
    return x_center, x_beta


def head_from_bounds(min_x,max_x):
    x_center = 0.5 * (max_x + min_x)
    x_betas = 0.5 * (max_x - min_x)
    return x_center, x_betas


class AbstractElement:
    def __init__(self) -> None:
        pass

    def __sub__(self, other) -> "AbstractElement":
        raise NotImplementedError

    def max_center(self) -> Tensor:
        raise NotImplementedError

    def conv2d(self,weight, bias, stride, padding, dilation, groups) -> "AbstractElement":
        raise NotImplementedError

    def upsample(self, size, mode, align_corners, consolidate_errors) -> "AbstractElement":
        raise NotImplementedError

    def linear(self, weight, bias) -> "AbstractElement":
        raise NotImplementedError

    def size(self) -> "AbstractElement":
        raise NotImplementedError

    def view(self, shape_tuple) -> "AbstractElement":
        raise NotImplementedError

    def normalize(self) -> "AbstractElement":
        raise NotImplementedError

    def clone(self) -> "AbstractElement":
        raise NotImplementedError

    def relu(self, deepz_lambda, bounds) -> "AbstractElement":
        raise NotImplementedError


def get_neg_pos_comp(x: Tensor) -> Tuple[Tensor, Tensor]:
    neg_comp = torch.where(x < 0, x, torch.zeros_like(x))
    pos_comp = torch.where(x >= 0, x, torch.zeros_like(x))
    return neg_comp, pos_comp

def construct_C(n_class, target):
    bs = target.shape[0]
    device = target.device
    C = -torch.eye(n_class, device=device).repeat(bs, 1, 1)
    C[np.arange(bs), :, target] += 1.
    C = C[torch.Tensor(np.arange(n_class)).repeat(bs, 1).to(device) != target.unsqueeze(1)].view(bs, -1, n_class)
    return C


