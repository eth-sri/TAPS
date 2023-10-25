import torch
import torch.nn as nn
import numpy as np
import json
import os
from AIDomains.abstract_layers import BatchNorm1d, BatchNorm2d, Linear, Conv2d, Sequential

def clamp_image(x, eps):
    min_x = torch.clamp(x-eps, min=0)
    max_x = torch.clamp(x+eps, max=1)
    x_center = 0.5 * (max_x + min_x)
    x_betas = 0.5 * (max_x - min_x)
    return x_center, x_betas



def compute_l1(net, l1_reg):
    ret = 0
    for layer in net.blocks:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            for param_name, param_value in layer.named_parameters():
                if 'bias' in param_name:
                    continue
                ret += param_value.abs().sum()
    return l1_reg * ret

class Scheduler:
    def __init__(self, start_epoch, end_epoch, start_value, end_value, mode="linear", c=0.25, e=4, s=500):
        assert end_epoch >= start_epoch
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_value = start_value
        self.end_value = end_value
        self.mode = mode
        self.c = c
        self.e = e
        self.s = s
        assert e >= 1, "please choose an exponent >= 1"
        assert 0 < c < 0.5, "please choose c in the range (0,0.5)"

    def getcurrent(self, epoch):
        if epoch < self.start_epoch:
            return self.start_value
        if epoch >= self.end_epoch:
            return self.end_value

        if self.mode == "linear":
            current = self.start_value + (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch) * \
                  (self.end_value - self.start_value)
        elif self.mode == "smooth":
            c = self.c # portion of the mid point
            e = self.e
            width = self.end_epoch - self.start_epoch
            mid_epoch = int(c * width) + self.start_epoch
            d = self.end_value - self.start_value
            t = (mid_epoch - self.start_epoch) ** (e - 1)
            alpha = d / ((self.end_epoch - mid_epoch) * e * t + (mid_epoch - self.start_epoch) * t)
            mid_value = self.start_value + alpha * (mid_epoch - self.start_epoch) ** e
            exp_value = self.start_value + alpha * float(epoch - self.start_epoch) ** e
            linear_value = min(mid_value + (self.end_value - mid_value) * (epoch - mid_epoch) / (self.end_epoch - mid_epoch), self.end_value)
            current = exp_value if epoch <= mid_epoch else linear_value
        elif self.mode == "step":
            n_steps = int((self.end_epoch - self.start_epoch)/self.s)
            delta = (self.end_value -self.start_value) / n_steps
            current = np.ceil((epoch-self.start_epoch+0.1)/(self.end_epoch-self.start_epoch)*n_steps)*delta + self.start_value
        else:
            raise NotImplementedError
        return current


class Statistics:

    def __init__(self):
        self.n = 0
        self.avg = 0.0

    def update(self, x, num:int=1):
        self.avg = self.avg * (self.n / (self.n + num)) + x * num / (self.n + num)
        self.n += num

    @staticmethod
    def get_statistics(k):
        return [Statistics() for _ in range(k)]

def write_perf_to_json(perf_dict, save_root, filename:str="monitor.json"):
    filepath = os.path.join(save_root, filename)
    with open(filepath, "w") as f:
        json.dump(perf_dict, f, indent=4)

def load_perf_from_json(load_root, filename:str="monitor.json"):
    filepath = os.path.join(load_root, filename)
    if not os.path.isfile(filepath):
        print(filepath, "does not exist!")
        return None
    with open(filepath, "r") as f:
        perf_dict = json.load(f)
    return perf_dict

def get_model_param_stat(net, tol=1e-10, ndigits=4):
    d = dict()
    dead_count = 0
    total_count = 0
    min_value, max_value = 1e10, -1e10
    for param in net.parameters():
        dead_count += (param.abs() <= tol).sum().item()
        total_count += param.numel()
        min_param, max_param = param.min().item(), param.max().item()
        min_value, max_value = min(min_value, min_param), max(max_value, max_param)
    d['dead_ratio'] = round(dead_count / total_count, ndigits=ndigits)
    d['min_value'] = round(min_value, ndigits=ndigits)
    d['max_value'] = round(max_value, ndigits=ndigits)
    return d

def pertub_model_param(model, noise_rate=1e-4):
    state_dict = model.state_dict()
    for key, weight in state_dict.items():
        noise = (torch.rand_like(weight) - 0.5) * noise_rate
        state_dict[key] = weight + noise
    model.load_state_dict(state_dict)

def fuse_BN2d_to_Conv2d(BN2d, Conv2d):
    '''
    Adapted from: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
    '''
    w_conv = Conv2d.weight.clone().view(Conv2d.out_channels, -1)
    bv = torch.sqrt(BN2d.eps + BN2d.running_var)
    w_bn = torch.diag(BN2d.weight.data / bv)
    Conv2d.weight.data = torch.matmul(w_bn, w_conv).view(Conv2d.weight.shape)
    b_bn = BN2d.bias.data - (BN2d.weight.data * BN2d.running_mean) / bv
    Conv2d.bias.data = torch.matmul(w_bn, Conv2d.bias.data) + b_bn
    return Conv2d

def fuse_BN1d_to_Linear(BN1d, Linear):
    bn_mean, bn_var, bn_weight, bn_bias = BN1d.running_mean.data, BN1d.running_var.data, BN1d.weight.data, BN1d.bias.data
    W = bn_weight / torch.sqrt(bn_var + BN1d.eps)
    b = - W * bn_mean + bn_bias
    W = torch.diag(W)
    Linear.weight.data = torch.matmul(W, Linear.weight.data)
    Linear.bias.data = torch.matmul(W, Linear.bias.data) + b
    return Linear

def fuse_BN(net, start_from:int=0):
    '''
    Merge the BatchNorm into its parent layer: 
    Linear + BN1d -> Linear
    Conv2d + BN2d -> Conv2d
    '''
    layers = []
    for i, layer in enumerate(net):
        if i < start_from:
            layers.append(layer)
            continue
        
        if isinstance(layer, BatchNorm1d):
            pr_layer = layers[-1]
            assert isinstance(pr_layer, Linear), "BN1d should follow a Linear layer."
            transformed_layer = fuse_BN1d_to_Linear(layer, pr_layer)
            layers[-1] = transformed_layer
        elif isinstance(layer, BatchNorm2d):
            pr_layer = layers[-1]
            assert isinstance(pr_layer, Conv2d), "BN2d should follow a Conv2d layer."
            transformed_layer = fuse_BN2d_to_Conv2d(layer, pr_layer)
            layers[-1] = transformed_layer
        else:
            layers.append(layer)
    net = Sequential(*layers)
    net.output_dim = layers[-1].output_dim
    return net