"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import os
import pickle
import numpy as np
import scipy as sp
from scipy.sparse import csr_array
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from gurobipy import GRB, Model, LinExpr, QuadExpr
import gurobipy as gp
import sys, inspect
import multiprocessing
import math
import sys
import time
import warnings
# from utils import evaluate_cstr, check_timeout, check_timeleft
import torch.nn.functional as F


currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(currentdir,".."))

from AIDomains.abstract_layers import Linear, ReLU, Flatten, Conv2d, AvgPool2d, Entropy, MaxPool2d, Bias, Sequential, Normalization, _BatchNorm, BatchNorm1d, BatchNorm2d


def get_layers(net):
    if isinstance(net, Sequential):
        net = [get_layers(x) for x in net.layers]
        net = [x for y in net for x in (y if isinstance(y,list) else [y])]
    return net

def build_milp_model(net, x_abs, x, end_layer_id=None, partial_milp=-1, max_milp_neurons=-1):
    model = Model("milp")
    model.setParam('OutputFlag', 0)
    model.setParam(GRB.Param.FeasibilityTol, 1e-4)

    layers = get_layers(net)

    n_layers = len(layers) if end_layer_id is None else min(len(layers), end_layer_id+1)

    layers_milp_support = [MaxPool2d, ReLU]
    milp_activation_layers = [i for i, x in enumerate(layers) if any([isinstance(x, y) for y in layers_milp_support])]

    ### Determine which layers, if any to encode with MILP
    if partial_milp < 0:
        partial_milp = len(milp_activation_layers)
    first_milp_layer = len(layers) if partial_milp == 0 else milp_activation_layers[-min(partial_milp, len(milp_activation_layers))]

    curr = x
    curr_flattened = curr.view((1, -1))
    n_inputs = curr_flattened.numel()
    abs_flat = x_abs.view((1, -1))
    lb, ub = abs_flat.concretize()
    lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()


    # ### Encode input as zonotope
    # if x_abs.errors is not None:
    #     n_errors = abs_flat.errors.size()[0]
    #     errors = [model.addVar(-1.0, 1.0, vtype=GRB.CONTINUOUS, name='error_{}'.format(j)) for j in range(n_errors)]

    # print("Start input encoding...")
    # t1 = time.time()
    # neuron_vars = {}
    # layer_idx = "input"
    # neuron_vars[layer_idx] = []
    # for j in range(n_inputs):
    #     neuron_vars[layer_idx] += [model.addVar(lb=lb[0, j], ub=ub[0, j], vtype=GRB.CONTINUOUS, name=f'input_{j}')]
    #     neuron_vars[layer_idx][-1].setAttr("Start", curr_flattened[0, j])
    #     expr = LinExpr()
    #     expr += abs_flat.head[0, j].item()
    #     expr += LinExpr(abs_flat.errors[:, 0, j].detach().cpu().numpy().tolist(), errors)
    #     model.addConstr(expr, GRB.EQUAL, neuron_vars[layer_idx][j])
    # print(f"Input encoding time: {time.time()-t1}")

    t1 = time.time()
    neuron_vars = {}
    layer_idx = "input"
    var = model.addMVar(shape=n_inputs, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f'n_{layer_idx}')
    var.start = curr_flattened[0].cpu().detach().numpy()
    neuron_vars[layer_idx] = var.tolist()
    # print(f"Input encoding time: {time.time()-t1}")

    # print("Start encoding...")
    ## Encode Network
    pr_lb, pr_ub = lb, ub
    pr_layer_idx = layer_idx
    layer_idx_prefix = ""
    for layer_id in range(n_layers):
        layer_idx = "%s%d" % (layer_idx_prefix, layer_id)
        layer = layers[layer_id]
        input_shape = tuple(x.shape[1:])
        x = layer(x)
        x_abs = layer(x_abs)
        n_milp_neurons = max_milp_neurons if layer_id >= first_milp_layer else 0

        t1 = time.time()
        pr_lb, pr_ub = add_layer_to_model(layer, model, neuron_vars, x, x_abs, pr_lb, pr_ub, layer_idx, pr_layer_idx, input_shape, n_milp_neurons)
        # print(f"Layer {layer_id} Build Time: {time.time()-t1:.2f}")

        pr_layer_idx = layer_idx
        # model.reset()
        # model.update()
        # model.optimize()
        # assert model.status == 2, "Model is infeasible"

    return model, neuron_vars


def set_model_objective(model, output_neurons, target_idx, adv_idx, threshold_min=0, feasibility_problem=False, binary=False):
    if binary:
        if feasibility_problem:
            model.addConstr((-1 + 2 * target_idx) * output_neurons[0], GRB.LESS_EQUAL,
                            (-1 + 2 * target_idx) * threshold_min, name="class_false")
        else:
            model.setObjective((-1 + 2 * target_idx) * output_neurons[0], GRB.MINIMIZE)
            model._threshold_min = (-1 + 2 * target_idx) * threshold_min

    else:
        if feasibility_problem:
            model.addConstr(output_neurons[target_idx] - output_neurons[adv_idx],
                            GRB.LESS_EQUAL, threshold_min, name="class_false")
        else:
            model.setObjective(output_neurons[target_idx] - output_neurons[adv_idx],
                               GRB.MINIMIZE)
            model._threshold_min = threshold_min


def add_layer_to_model(layer, model, neuron_vars, x, x_abs, pr_lb, pr_ub, layer_idx, pr_layer_idx, in_shape, n_milp_neurons=0):
    out_shape = tuple(x.shape[1:])

    x_flat = x.view((1, -1)).detach().cpu().numpy()
    x_abs_flat = x_abs.view((1, -1))

    lb, ub = x_abs_flat.concretize()

    if layer.bounds is not None:
        input_lb, input_ub = layer.bounds
        pr_lb = np.maximum(pr_lb, input_lb.view((1, -1)).detach().cpu().numpy())
        pr_ub = np.minimum(pr_ub, input_ub.view((1, -1)).detach().cpu().numpy())

    # neuron_vars[lidx_str] = []
    lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()

    translate_layer(model, neuron_vars, layer, pr_layer_idx, layer_idx, lb[0], ub[0], pr_lb[0], pr_ub[0], n_milp_neurons, in_shape, out_shape, feasible_activation=x_flat[0])
    return lb, ub


def translate_layer(model, neuron_vars, layer, pr_layer_idx, layer_idx, lb, ub, pr_lb, pr_ub, n_milp_neurons, in_shape, out_shape,
                    feasible_activation=None):
    if isinstance(layer, Linear):
        weight, bias = layer.weight.data.cpu().detach().numpy(), layer.bias.data.cpu().detach().numpy()
        handle_affine(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, in_shape, out_shape, feasible_activation)

    elif isinstance(layer, ReLU):
        handle_relu(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, pr_lb, pr_ub, in_shape, out_shape, n_milp_neurons, feasible_activation)

    elif isinstance(layer, Flatten):
        neuron_vars[layer_idx] = neuron_vars[pr_layer_idx]

    elif isinstance(layer, Conv2d):
        weight, bias = layer.weight.cpu().detach().numpy(), layer.bias.cpu().detach().numpy()
        filter_size, stride, pad, dilation = layer.kernel_size, layer.stride, layer.padding, layer.dilation
        assert dilation == (1,1), "Dilation != 1 not implemented"

        if isinstance(pad, int):
            pad_top, pad_left = pad, pad
        elif len(pad)>=2:
            pad_top, pad_left, = pad[0], pad[1]

        handle_conv2d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, filter_size,
                    stride, pad_top, pad_left, in_shape, out_shape,  feasible_activation)

    elif isinstance(layer, Normalization):
        mean, sigma = layer.mean.cpu().detach().numpy(), layer.sigma.cpu().detach().numpy()
        bias = (-mean / sigma).flatten()
        weight = np.diag(1 / sigma.flatten()).reshape(mean.size,mean.size,1,1)

        handle_conv2d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, [1,1],
                    [1,1], 0, 0, in_shape, out_shape,  feasible_activation)

    elif isinstance(layer, _BatchNorm):
        mean, var = layer.running_mean.cpu().detach().numpy(), layer.running_var.cpu().detach().numpy()
        weight, bias = layer.weight.cpu().detach().numpy(), layer.bias.cpu().detach().numpy()
        c = weight / np.sqrt(var + layer.eps)
        bias = - c * mean + bias

        if isinstance(layer, BatchNorm2d):
            weight = np.diag(c).reshape(mean.size,mean.size,1,1)
            handle_conv2d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, [1,1],
                    [1,1], 0, 0, in_shape, out_shape, feasible_activation)
        elif isinstance(layer, BatchNorm1d):
            # weight = np.diag(c)
            weight = sp.sparse.diags(c)
            handle_affine(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, in_shape, out_shape, feasible_activation)

    elif isinstance(layer, AvgPool2d):
        kernel_size, stride, pad = layer.kernel_size, layer.stride, layer.padding
        if isinstance(pad, int):
            pad_top, pad_left = pad, pad
        elif len(pad)>=2:
            pad_top, pad_left = pad[0], pad[1]

        handle_avgpool2d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, kernel_size, stride, pad_top, pad_left,
                        in_shape, out_shape, feasible_activation)
    else:
        print('unknown layer type: ', layer)
        assert False


def milp_callback(model, where):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        if obj_bound > 0.01:
            model.terminate()
        if obj_best < -0.01:
            model.terminate()


def milp_callback_adex(model, where):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        sol_count = model.cbGet(GRB.Callback.MIP_SOLCNT)
        if obj_bound > 0.01:
            model.terminate()
        if obj_best < -0.50 and sol_count > 0:
            model.terminate()


def lp_callback(model, where):
    # pass
    if where == GRB.Callback.SIMPLEX:
        obj_best = model.cbGet(GRB.Callback.SPX_OBJVAL)
        if model.cbGet(
                GRB.Callback.SPX_PRIMINF) == 0 and obj_best < -0.01:  # and model.cbGet(GRB.Callback.SPX_DUALINF) == 0:
            print("Used simplex terminate")
            model.terminate()
    if where == GRB.Callback.BARRIER:
        obj_best = model.cbGet(GRB.Callback.SPX_OBJVAL)
        if model.cbGet(
                GRB.Callback.BARRIER_PRIMINF) == 0 and obj_best < -0.01:  # and model.cbGet(GRB.Callback.BARRIER_DUALINF) == 0
            model.terminate()
            print("Used barrier terminate")

def handle_conv1d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, kernel_size:int, stride:int, pad_left:int, in_shape, out_shape, feasible_activation=None):
    assert len(out_shape) == 2 and len(in_shape) == 2, "Should have two-dimensional in/out."
    neuron_vars[layer_idx] = []
    out_ch, out_w = out_shape
    in_ch, in_w = in_shape
    in_neurons = in_ch * in_w
    out_neurons = out_ch * out_w

    for j in range(out_neurons):
        var_name = f"x_{layer_idx}_{j}"
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
        if feasible_activation is not None:
            var.start = feasible_activation[j]
        neuron_vars[layer_idx].append(var)

    for out_z in range(out_ch):
        for out_x in range(out_w):
            out_ind = out_z * out_w + out_x
            expr = LinExpr()
            expr += -1 * neuron_vars[layer_idx][out_ind]
            for in_z in range(in_ch):
                for x_shift in range(kernel_size):
                    in_x = out_x * stride + x_shift - pad_left
                    if in_x < 0 or in_x >= in_w:
                        continue
                    in_ind = in_z * in_w + in_x
                    if in_ind >= in_neurons:
                        continue
                    expr.addTerms(weight[out_z][in_z][x_shift], neuron_vars[pr_layer_idx][in_ind])
            
            expr.addConstant(bias[out_z])
            model.addConstr(expr, GRB.EQUAL, 0)

# def conv_to_matrix(weight, bias, stride, padding, image_shape, output_shape):
#     '''
#     Apply Conv2d on the eye matrix to get the matrix form of the conv2d
#     '''
#     identity = torch.eye(np.prod(image_shape).item()).reshape(-1, *image_shape)
#     output = F.conv2d(identity, torch.from_numpy(weight), None, stride, padding)
#     W = output.reshape(-1, np.prod(output_shape).item()).T
#     b = torch.stack([torch.ones(output_shape[1:]) * bi for bi in bias])
#     b = b.reshape(np.prod(output_shape).item())
#     return W.numpy(), b.numpy()

# def handle_conv2d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, kernel_size, stride, pad_top, pad_left, in_shape, out_shape, feasible_activation=None):
#     assert len(out_shape) == 3 and len(in_shape) == 3, "Should have three-dimensional in/out."
#     neuron_vars[layer_idx] = []
#     out_ch, out_h, out_w = out_shape
#     in_ch, in_h, in_w = in_shape
#     in_hw = in_h * in_w
#     out_hw = out_h * out_w
#     in_neurons = in_ch * in_hw
#     out_neurons = out_ch * out_hw

#     var = model.addMVar(shape=out_neurons, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f'n_{layer_idx}')
#     if feasible_activation is not None:
#         var.start = feasible_activation
#     neuron_vars[layer_idx] = var.tolist()

#     W, b = conv_to_matrix(weight, bias, stride, (pad_top, pad_left), in_shape, out_shape)
#     input_var = neuron_vars[pr_layer_idx]
#     input_var = gp.MVar(input_var)
#     t1 = time.time()
#     model.addConstr(W @ input_var + b == var)
#     print(time.time()-t1)


def handle_conv2d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, kernel_size, stride, pad_top, pad_left, in_shape, out_shape, feasible_activation=None):
    assert len(out_shape) == 3 and len(in_shape) == 3, "Should have three-dimensional in/out."
    neuron_vars[layer_idx] = []
    out_ch, out_h, out_w = out_shape
    in_ch, in_h, in_w = in_shape
    in_hw = in_h * in_w
    out_hw = out_h * out_w
    in_neurons = in_ch * in_hw
    out_neurons = out_ch * out_hw

    for j in range(out_neurons):
        var_name = f"x_{layer_idx}_{j}"
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
        if feasible_activation is not None:
            var.start = feasible_activation[j]
        neuron_vars[layer_idx].append(var)

    for out_z in range(out_ch):
        for out_y in range(out_h):
            for out_x in range(out_w):
                out_ind = out_z * out_hw + out_y * out_w + out_x

                expr = LinExpr()
                expr += -1 * neuron_vars[layer_idx][out_ind]
                for in_z in range(in_ch):
                    for x_shift in range(kernel_size[0]):
                        for y_shift in range(kernel_size[1]):
                            in_x = out_x * stride[0] + x_shift - pad_top
                            in_y = out_y * stride[1] + y_shift - pad_left
                            if (in_y < 0 or in_y >= in_h):
                                continue
                            if (in_x < 0 or in_x >= in_w):
                                continue
                            in_ind = in_z * in_hw + in_y * in_w + in_x
                            if (in_ind >= in_neurons):
                                continue
                            expr.addTerms(weight[out_z][in_z][y_shift][x_shift], neuron_vars[pr_layer_idx][in_ind])

                expr.addConstant(bias[out_z])
                model.addConstr(expr, GRB.EQUAL, 0)


def handle_avgpool2d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, kernel_size, stride, pad_top, pad_left,
                     in_shape, out_shape, feasible_activation):
    neuron_vars[layer_idx] = []
    out_ch, out_h, out_w = out_shape
    in_ch, in_h, in_w = in_shape

    for ch in range(out_ch):
        for y in range(0, out_h):
            for x in range(0, out_w):
                new_idx = ch * (out_h * out_w) + y * out_w + x
                out_var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[0, new_idx].item(), ub=ub[0, new_idx].item(),
                                    name='n_{}_{}'.format(layer_idx, new_idx))
                out_var.setAttr("Start", layer_idx[0, new_idx])
                neuron_vars[layer_idx].append(out_var)
                expr = 0
                for ky in range(0, kernel_size):
                    for kx in range(0, kernel_size):
                        old_x = -pad_left + x * stride + kx
                        old_y = -pad_top + y * stride + ky
                        if old_x < 0 or old_y < 0 or old_x >= in_w or old_y >= in_h:
                            continue
                        old_idx = ch * (in_h * in_w) + old_y * (in_w) + old_x
                        expr += neurons[pr_layer_idx][old_idx] / (kernel_size * kernel_size)
                model.addConstr(expr, GRB.EQUAL, out_var)


def handle_padding(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, pad_top, pad_left, in_shape, out_shape, feasible_activation):
    return handle_avgpool2d(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, 1, 1, pad_top, pad_left,
                       in_shape, out_shape, feasible_activation)


def handle_maxpool(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, lb_prev, ub_prev, kernel_size, stride, pad_top, pad_left,
                       in_shape, out_shape, use_milp=False, feasible_activation=None):
    neuron_vars[layer_idx] = []
    out_ch, out_h, out_w = out_shape
    in_ch, in_h, in_w = in_shape
    in_hw = in_h * in_w
    out_hw = out_h * out_w
    in_neurons = in_ch * in_hw
    out_neurons = in_ch * in_hw

    if use_milp:
        binary_vars = []

    for ch in range(out_ch):
        for y in range(0, out_h):
            for x in range(0, out_w):
                new_idx = ch * (out_h * out_w) + y * out_w + x
                out_var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[0, new_idx].item(), ub=ub[0, new_idx].item(),
                                    name='n_{}_{}'.format(layer_idx, new_idx))
                out_var.setAttr("Start", layer_idx[0, new_idx])
                neuron_vars[layer_idx].append(out_var)
                if use_milp:
                    var_name = 'b_{}_{}'.format(layer_idx, j)
                    var = model.addVar(vtype=GRB.BINARY, name=var_name)
                    binary_vars.append(var)

                max_u = float("-inf")
                max_l = float("-inf")
                sum_l = 0.0
                max_l_var = 0
                max_u_var = 0
                pool_map = []
                l = 0

                for ky in range(0, kernel_size):
                    for kx in range(0, kernel_size):
                        old_x = -pad_left + x * stride + kx
                        old_y = -pad_top + y * stride + ky
                        if old_x < 0 or old_y < 0 or old_x >= in_w or old_y >= in_h:
                            continue
                        old_idx = ch * (in_h * in_w) + old_y * (in_w) + old_x

                        pool_map.append(old_idx)
                        lb_i = lb_prev[old_idx]
                        ub_i = ub_prev[old_idx]
                        sum_l = sum_l + lb_i
                        if ub_i > max_u:
                            max_u = ub_i
                            max_u_var = pool_cur_dim
                        if lb_i > max_l:
                            max_l = lb_i
                            max_l_var = pool_cur_dim
                        l = l + 1

                if use_milp:
                    binary_expr = LinExpr()
                    for l in range(len(pool_map)):
                        src_index = pool_map[l]
                        src_var = neuron_vars[pr_layer_idx][src_index]
                        binary_var = binary_vars[src_index]
                        
                        if (ub_prev[src_index] < max_l):
                            # Variable is dominated
                            continue
    
                        # y >= x
                        expr = out_var - src_var
                        model.addConstr(expr, GRB.GREATER_EQUAL, 0)
    
                        # y <= x + (1-a)*(u_{rest}-l)
                        max_u_rest = float("-inf")
                        for j in range(len(pool_map)):
                            if j == l:
                                continue
                            if (ub_prev[j] > max_u_rest):
                                max_u_rest = ub_prev[j]
    
                        cst = max_u_rest - lb_prev[l]
                        expr = out_var - src_var + cst * binary_var
                        model.addConstr(expr, GRB.LESS_EQUAL, cst)
    
                        # indicator constraints
                        model.addGenConstrIndicator(binary_var, True, out_var - src_var,
                                                    GRB.EQUAL, 0.0)
                        binary_expr += binary_var
    
                    # only one indicator can be true
                    model.addConstr(binary_expr, GRB.EQUAL, 1)
    
                else:
                    flag = True
                    for l in range(len(pool_map)):
                        if pool_map[l] == max_l_var:
                            continue
                        ub_i = ub_prev[pool_map[l]]
                        if ub_i >= max_l:
                            flag = False
                            #no variable dominates
                            break
                    if flag:
                        # one variable dominates all others
                        src_var = max_l_var + src_counter
                        expr = out_var - neuron_vars[layer_idx][max_l_var]
                        model.addConstr(expr, GRB.EQUAL, 0)
                    else:
                        # No one variable dominates all other
                        add_expr = LinExpr()
                        add_expr += -out_var
                        for l in range(len(pool_map)):
                            src_index = pool_map[l]
                            # y >= x
                            expr = out_var - neuron_vars[pr_layer_idx][src_index]
                            model.addConstr(expr, GRB.GREATER_EQUAL, 0)
    
                            add_expr += neuron_vars[pr_layer_idx][src_index]
                        model.addConstr(add_expr, GRB.GREATER_EQUAL, sum_l - max_l)
    return maxpool_counter

# def handle_affine(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, in_shape, out_shape, feasible_activation=None):
#     neuron_vars[layer_idx] = []
    
#     in_n = in_shape[-1]
#     out_n = out_shape[-1]

#     # output of matmult
#     for j in range(out_n):
#         var_name = 'n_{}_{}'.format(layer_idx, j)
#         var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
#         if feasible_activation is not None:
#             var.start = feasible_activation[j]
#         neuron_vars[layer_idx].append(var)

#     for j in range(out_n):
#         expr = LinExpr()
#         expr += -1 * neuron_vars[layer_idx][j]
#         # matmult constraints
#         for k in range(in_n):
#             expr.addTerms(weight[j][k], neuron_vars[pr_layer_idx][k])
#         expr.addConstant(bias[j])
#         model.addConstr(expr, GRB.EQUAL, 0)

def handle_affine(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, weight, bias, in_shape, out_shape, feasible_activation=None):
    in_n = in_shape[-1]
    out_n = out_shape[-1]

    # create matrix var
    var = model.addMVar(shape=out_n, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f'n_{layer_idx}')
    if feasible_activation is not None:
        var.start = feasible_activation
    neuron_vars[layer_idx] = var.tolist()

    # define constraints
    # weight.shape = (out_n, in_n), bias.shape = (out_n, )
    input_var = neuron_vars[pr_layer_idx]
    input_var = gp.MVar(input_var)
    model.addConstr(weight @ input_var + bias == var)
    # weight_padded = np.concatenate([weight, -np.eye(out_n, dtype=int)], axis=1)
    # vars = input_var + neuron_vars[layer_idx]
    # model.addMConstr(weight_padded, vars, '=', -bias)

def handle_bias(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, bias, in_shape, out_shape, feasible_activation=None):
    neuron_vars[layer_idx] = []

    in_n = in_shape[-1]
    out_n = out_shape[-1]

    # output of matmult
    for j in range(out_n):
        var_name = 'n_{}_{}'.format(layer_idx, j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[j], ub=ub[j], name=var_name)
        if feasible_activation is not None:
            var.start = feasible_activation[j]
        neuron_vars[layer_idx].append(var)

    for j in range(num_neurons_affine):
        num_in_neurons = len(weights[j])

        expr = -1 * neuron_vars[layer_idx][j] + neuron_vars[pr_layer_idx][j] + biases[j]
        model.addConstr(expr, GRB.EQUAL, 0)


# def handle_residual(model, var_list, branch1_counter, branch2_counter, lbi, ubi, feasible_activation=None):
#     num_neurons_affine = len(lbi)
#     start_idx = len(var_list)
# 
#     # output of matmult
#     for j in range(num_neurons_affine):
#         var_name = "x" + str(start_idx + j)
#         var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub=ubi[j], name=var_name)
#         var_list.append(var)
# 
#     if feasible_activation is not None:
#         for j, var in enumerate(var_list[-num_neurons_affine:]):
#             var.start = feasible_activation[j]
# 
#     for j in range(num_neurons_affine):
#         # num_in_neurons = len(weights[j])
# 
#         expr = LinExpr()
#         expr += -1 * var_list[start_idx + j]
#         # matmult constraints
#         # for k in range(num_in_neurons):
#         expr += var_list[branch1_counter + j]
#         expr += var_list[branch2_counter + j]
#         expr.addConstant(0)
#         model.addConstr(expr, GRB.EQUAL, 0)
#     return start_idx


def handle_relu(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, lb_prev, ub_prev, in_shape, out_shape, partial_milp_neurons=0, feasible_activation=None):
    neuron_vars[layer_idx] = []
    num_neurons = np.prod(in_shape)
    
    ### Determine which neuorons to encode with MILP and which with LP (triangle)
    cross_over_idx = list(np.nonzero(np.array(lb_prev) * np.array(ub_prev) < 0)[0])
    width = np.array(ub_prev) - np.array(lb_prev)
    cross_over_idx = sorted(cross_over_idx, key=lambda x: -width[x])
    milp_encode_idx = cross_over_idx[:partial_milp_neurons] if partial_milp_neurons >= 0 else cross_over_idx  # cross_over_idx if use_milp else cross_over_idx[:partial_milp_neurons]
    temp_idx = np.ones(lb_prev.size, dtype=bool)
    temp_idx[milp_encode_idx] = False
    lp_encode_idx = np.arange(num_neurons)[temp_idx]
    assert len(lp_encode_idx) + len(milp_encode_idx) == num_neurons

    ### Add binary variables to model
    binary_vars = []
    if len(milp_encode_idx) > 0:
        binary_guess = None if feasible_activation is None else (feasible_activation > 0).astype(np.int) # Initialize binary variables with a feasible solution
        for i, j in enumerate(milp_encode_idx):
            var_name = f"b_{layer_idx}_{j}"
            var_bin = model.addVar(vtype=GRB.BINARY, name=var_name)
            if binary_guess is not None:
                var_bin.start = binary_guess[j]
            binary_vars.append(var_bin)

    # Add ReLU output variables
    if feasible_activation is not None:
        feas_act_post = np.maximum(feasible_activation, 0.) # Initialize output variables with a feasible solution
    for j in range(num_neurons):
        var_name = f"x_{layer_idx}_{j}"
        upper_bound = max(0.0, ub_prev[j])
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=upper_bound, name=var_name)
        if feasible_activation is not None:
            var.start = feas_act_post[j]
        neuron_vars[layer_idx].append(var)

    ### Add MILP encoding
    if len(milp_encode_idx) > 0:
        for i, j in enumerate(milp_encode_idx):
            var_bin = binary_vars[i]
            var_in = neuron_vars[pr_layer_idx][j]
            var_out = neuron_vars[layer_idx][j]

            if (ub_prev[j] <= 0):
                # stabely inactive
                expr = var_out
                model.addConstr(expr, GRB.EQUAL, 0)
            elif (lb_prev[j] >= 0):
                # stabely active
                expr = var_out - var_in
                model.addConstr(expr, GRB.EQUAL, 0)
            else:
                # y <= x - l(1-a)
                expr = var_out - var_in - lb_prev[j] * var_bin
                model.addConstr(expr, GRB.LESS_EQUAL, -lb_prev[j])

                # y >= x
                expr = var_out - var_in
                model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                # y <= u . a
                expr = var_out - ub_prev[j] * var_bin
                model.addConstr(expr, GRB.LESS_EQUAL, 0)

                # y >= 0
                # expr = var_out
                # model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                # indicator constraint
                model.addGenConstrIndicator(var_bin, True, var_in, GRB.GREATER_EQUAL, 0.0)

    ### Add LP encoding
    if len(lp_encode_idx) > 0:
        for j in lp_encode_idx:
            var_in = neuron_vars[pr_layer_idx][j]
            var_out = neuron_vars[layer_idx][j]
            if ub_prev[j] <= 0:
                expr = var_out
                model.addConstr(expr, GRB.EQUAL, 0)
            elif lb_prev[j] >= 0:
                expr = var_out - var_in
                model.addConstr(expr, GRB.EQUAL, 0)
            else:
                # y >= 0 (already encoded in range of output variable)
                # expr = var_out
                # model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                # y >= x
                expr = var_out - var_in
                model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                # y <= (x-l) * u/(u-l) 
                expr = (ub_prev[j] - lb_prev[j]) * var_out - (var_in - lb_prev[j]) * ub_prev[j]
                model.addConstr(expr, GRB.LESS_EQUAL, 0)


def handle_sign(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, lb_prev, ub_prev, in_shape, out_shape, 
                use_milp=False, feasible_activation=None):
    neuron_vars[layer_idx] = []
    num_neurons = np.prod(in_shape)
    
    ### Determine which neuorons to encode with MILP and which with LP (triangle)
    cross_over_idx = list(np.nonzero(np.array(lb_prev) * np.array(ub_prev) < 0)[0])
    width = np.array(ub_prev) - np.array(lb_prev)
    cross_over_idx = sorted(cross_over_idx, key=lambda x: -width[x])
    milp_encode_idx = cross_over_idx[:partial_milp_neurons] if partial_milp_neurons >= 0 else cross_over_idx  # cross_over_idx if use_milp else cross_over_idx[:partial_milp_neurons]
    temp_idx = np.ones(lb_prev.size, dtype=bool)
    temp_idx[milp_encode_idx] = False
    lp_encode_idx = np.arange(num_neurons)[temp_idx]
    assert len(lp_encode_idx) + len(milp_encode_idx) == num_neurons

    ### Add binary variables to model
    binary_vars = []
    if len(milp_encode_idx) > 0:
        binary_guess = None if feasible_activation is None else (feasible_activation > 0).astype(np.int) # Initialize binary variables with a feasible solution
        for i, j in enumerate(milp_encode_idx):
            var_name = f"b_{layer_idx}_{j}"
            var_bin = model.addVar(vtype=GRB.BINARY, name=var_name)
            if binary_guess is not None:
                var_bin.start = binary_guess[j]
            binary_vars.append(var_bin)

    # Add Sign output variables
    if feasible_activation is not None:
        feas_act_post = np.sign(feasible_activation, 0.)  # Initialize output variables with a feasible solution
    for j in range(num_neurons):
        var_name = f"x_{layer_idx}_{j}"
        upper_bound = min(max(0.0, ub[j]), 1.0)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=upper_bound, name=var_name)
        if feasible_activation is not None:
            var.start = feas_act_post[j]
        neuron_vars[layer_idx].append(var)

    ### Add MILP encoding
    if len(milp_encode_idx) > 0:
        for i, j in enumerate(milp_encode_idx):
            var_bin = binary_vars[i]
            var_in = neuron_vars[pr_layer_idx][j]
            var_out = neuron_vars[layer_idx][j]

            if (ub[j] <= 0):
                # stabely inactive
                expr = var_out
                model.addConstr(expr, GRB.EQUAL, 0)
            elif (lb[j] >= 0):
                # stabely active
                expr = var_out
                model.addConstr(expr, GRB.EQUAL, 1.0)
            else:
                # x >= l(1-a)
                expr = var_in + lb[j] * var_bin
                model.addConstr(expr, GRB.GREATER_EQUAL, lb[j])

                # x <= u.a
                expr = var_in - ub[j] * var_bin
                model.addConstr(expr, GRB.LESS_EQUAL, 0)

                # y = a
                expr = var_out - var_bin
                model.addConstr(expr, GRB.EQUAL, 0)

                # indicator constraint
                model.addGenConstrIndicator(var_bin, True, var_in, GRB.GREATER_EQUAL, 0.0)

    ### Add LP encoding
    if len(lp_encode_idx) > 0:
        raise NotImplementedError

    return neuron_counter


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def handle_sigmoid(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, lb_prev, ub_prev, in_shape, out_shape, 
                use_milp=False, feasible_activation=None):
    neuron_vars[layer_idx] = []
    num_neurons = np.prod(in_shape)

    for j in range(num_neurons):
        var_name = f"x_{layer_idx}_{j}"
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=sigmoid(lb[j]), ub=sigmoid(lb[j]), name=var_name)
        var_list.append(var)

    return num_neurons + neuron_counter


def handle_tanh(model, neuron_vars, layer_idx, pr_layer_idx, lb, ub, lb_prev, ub_prev, in_shape, out_shape,
                use_milp=False, feasible_activation=None):
    neuron_vars[layer_idx] = []
    num_neurons = np.prod(in_shape)

    for j in range(num_neurons):
        var_name = f"x_{layer_idx}_{j}"
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=math.tanh(lb[j]), ub=math.tanh(ub[j]), name=var_name)
        var_list.append(var)

    return num_neurons + neuron_counter


# class Cache:
#     model = None
#     out_vars = []
#     lbi = None
#     ubi = None
#
#
# def solver_call(ind):
#     ### Call solver to compute neuronwise bounds in parallel
#     if Cache.terminate_time is not None and Cache.terminate_time - time.time() < 1:
#         return Cache.lbi[ind], Cache.ubi[ind], False, 0
#
#     model = Cache.model.copy()
#     model.setParam(GRB.Param.TimeLimit, max(0, min(Cache.time_limit, np.inf if Cache.terminate_time is None else (
#                 Cache.terminate_time - time.time()))))
#     runtime = 0
#
#     obj = LinExpr()
#     obj += out_vars
#     # print (f"{ind} {model.getVars()[Cache.output_counter+ind].VarName}")
#
#     model.setObjective(obj, GRB.MINIMIZE)
#     model.reset()
#     model.optimize()
#     runtime += model.RunTime
#     soll = Cache.lbi[ind] if model.SolCount == 0 else model.objbound
#     # print (f"{ind} {model.status} lb ({Cache.lbi[ind]}, {soll}) {model.RunTime}s")
#     sys.stdout.flush()
#
#     model.setObjective(obj, GRB.MAXIMIZE)
#     model.setParam(GRB.Param.TimeLimit, max(0, min(Cache.time_limit, np.inf if Cache.terminate_time is None else (
#                 Cache.terminate_time - time.time()))))
#     model.reset()
#     model.optimize()
#     runtime += model.RunTime
#     solu = Cache.ubi[ind] if model.SolCount == 0 else model.objbound
#     # print (f"{ind} {model.status} ub ({Cache.ubi[ind]}, {solu}) {model.RunTime}s")
#     sys.stdout.flush()
#
#     soll = max(soll, Cache.lbi[ind])
#     solu = min(solu, Cache.ubi[ind])
#
#     addtoindices = (soll > Cache.lbi[ind]) or (solu < Cache.ubi[ind])
#
#     return soll, solu, addtoindices, runtime


# def get_bounds_for_layer_with_milp(nn, LB_N0, UB_N0, layerno, abs_layer_count, output_size, nlb, nub, relu_groups,
#                                    use_milp, candidate_vars, timeout, is_nchw, gpu_model=False, start_time=None,
#                                    feasible_activation=None):
#     lbi = nlb[abs_layer_count]
#     ubi = nub[abs_layer_count]
#
#     widths = [u - l for u, l in zip(ubi, lbi)]
#     candidate_vars = sorted(candidate_vars, key=lambda k: widths[k])
#     # if start_time is not None and config.timeout_complete is not None:
#     #     candidate_vars = candidate_vars[:int(max(0, np.floor((start_time + config.timeout_complete * 0.8 - time.time())/timeout)))]
#     #     if len(candidate_vars) == 0:
#     #         return lbi, ubi, []
#
#     counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, relu_groups, layerno + 1, use_milp,
#                                             partial_milp=-1 if use_milp else 0, max_milp_neurons=-1,
#                                             gpu_model=gpu_model,
#                                             is_nchw=is_nchw, feasible_activation=feasible_activation)
#     resl = [0] * len(lbi)
#     resu = [0] * len(ubi)
#     indices = []
#
#     model.setParam(GRB.Param.TimeLimit, timeout)
#     model.setParam(GRB.Param.Threads, 2)
#     output_counter = counter
#
#     model.update()
#     model.reset()
#
#     NUMPROCESSES = config.numproc
#     Cache.model = model
#     Cache.output_counter = output_counter
#     Cache.lbi = lbi
#     Cache.ubi = ubi
#     Cache.time_limit = timeout
#     Cache.terminate_time = None if start_time is None or config.timeout_complete is None else start_time + config.timeout_complete * 0.8
#
#     refined = [False] * len(lbi)
#
#     for v in candidate_vars:
#         refined[v] = True
#         # print (f"{v} {deltas[v]} {widths[v]} {deltas[v]/widths[v]}")
#     with multiprocessing.Pool(NUMPROCESSES) as pool:
#         solver_result = pool.map(solver_call, candidate_vars)
#
#     for (l, u, addtoindices, runtime), ind in zip(solver_result, candidate_vars):
#         resl[ind] = l
#         resu[ind] = u
#
#         if (l > u):
#             print(f"unsound {ind}")
#
#         if addtoindices:
#             indices.append(ind)
#
#     for i, flag in enumerate(refined):
#         if not flag:
#             resl[i] = lbi[i]
#             resu[i] = ubi[i]
#
#     for i in range(abs_layer_count):
#         for j in range(len(nlb[i])):
#             if (nlb[i][j] > nub[i][j]):
#                 print("fp unsoundness detected ", nlb[i][j], nub[i][j], i, j)
#
#     return resl, resu, sorted(indices)

def get_bound_with_milp(net, abs_input, adv_input, target, n_class, partial_milp=-1, max_milp_neurons=-1, timeout=200, time_relax_step=0, verbose:bool=True):
    '''
    Get the exact lower bound of y_target - y_adv
    '''
    input_size = adv_input.shape

    t1 = time.time()
    model, neuron_vars = build_milp_model(net, abs_input, adv_input, end_layer_id=None, partial_milp=partial_milp, max_milp_neurons=max_milp_neurons)

    layers = get_layers(net)
    output_layer_idx = str(len(layers)-1)
    exact_bounds = torch.zeros(n_class-1)
    for adv_label in range(n_class):
        if adv_label == target:
            continue
        obj = neuron_vars[output_layer_idx][target] - neuron_vars[output_layer_idx][adv_label]
        model.setObjective(obj, GRB.MINIMIZE)
        for cutoff in [GRB.INFINITY]:
            model.reset()
            model.setParam(GRB.Param.TimeLimit, timeout)
            model.setParam(GRB.Param.Cutoff, cutoff)
            model.optimize()
            if model.status not in [3, 4]:  # status 3 and 4 indicate an infeasible model
                # no infeasibility reported.
                break
            else:
                warnings.warn("Infeasible model encountered. Trying to increase the Cutoff parameter to recover.")
        else:
            # all values led to an infeasible model
            assert model.status not in [3, 4], f"Infeasible model encountered. Model status {model.status}"

        obj_bound = f"{model.objbound:.4f}" if hasattr(model, "objbound") else "failed"
        obj_val = f"{model.objval:.4f}" if hasattr(model, "objval") else "failed"
        if verbose:
            print(f"MILP model status: {model.Status}, Obj val/bound for adv label {adv_label}: {obj_val}/{obj_bound}, Final solve time: {model.Runtime:.3f}")
        offset = int(adv_label > target)
        exact_bounds[adv_label - offset] = -float(obj_val)
    return exact_bounds

def verify_network_with_milp(net, abs_input, adv_input, target, n_class, partial_milp=-1, max_milp_neurons=-1, timeout=200, time_relax_step=0, verbose:bool=True):
    input_size = adv_input.shape

    t1 = time.time()
    model, neuron_vars = build_milp_model(net, abs_input, adv_input, end_layer_id=None, partial_milp=partial_milp, max_milp_neurons=max_milp_neurons)
    print(f"Model Building Time: {time.time()-t1:.3f}")
    
    layers = get_layers(net)

    adv_examples = []
    non_adv_examples = []
    adv_val = []
    non_adv_val = []
    output_layer_idx = str(len(layers)-1)
    for adv_label in range(n_class):
        if adv_label == target:
            continue
        adv_label_safe = False
        obj = neuron_vars[output_layer_idx][target] - neuron_vars[output_layer_idx][adv_label]
        model.setObjective(obj, GRB.MINIMIZE)

        # In some cases it occurs that Gurobi reports an infeasible model
        # probably due to numerical difficulties (c.f. https://github.com/eth-sri/eran/issues/74).
        # These can be resolved (in the cases considered) by increasing the Cutoff parameter.
        # The code below tries to recover from an infeasible model by increasing the default cutoff
        # a few times.
        # 0.01 is the default cutoff value
        for cutoff in [0.01, 0.1, GRB.INFINITY]:
            model.reset()
            model.setParam(GRB.Param.TimeLimit, timeout)
            model.setParam(GRB.Param.Cutoff, cutoff)
            model.optimize(milp_callback)
            if model.status not in [3, 4]:  # status 3 and 4 indicate an infeasible model
                # no infeasibility reported.
                break
            else:
                warnings.warn("Infeasible model encountered. Trying to increase the Cutoff parameter to recover.")
        else:
            # all values led to an infeasible model
            assert model.status not in [3, 4], f"Infeasible model encountered. Model status {model.status}"

        obj_bound = f"{model.objbound:.4f}" if hasattr(model, "objbound") else "failed"
        obj_val = f"{model.objval:.4f}" if hasattr(model, "objval") else "failed"
        if verbose:
            print(f"MILP model status: {model.Status}, Obj val/bound for adv label {adv_label}: {obj_val}/{obj_bound}, Final solve time: {model.Runtime:.3f}")

        if model.Status in [1]:
            pass  # model loaded but not run ==> out of time
        elif model.Status == 6 or model.objbound > 0:
            adv_label_safe = True
            if model.solcount > 0:
                non_adv_examples.append(torch.tensor([x.X for x in neuron_vars["input"]]).view(input_size))
                non_adv_val.append(model.objval)
        elif model.solcount > 0:
            adv_examples.append(torch.tensor([x.X for x in neuron_vars["input"]]).view(input_size))
            adv_val.append(model.objval)

        if not adv_label_safe:
            if len(adv_examples) > 0:
                return False, adv_examples, adv_val
            else:
                return False, None, None
        else:
            # looks likely to be verified, so we increase the budget
            timeout += time_relax_step
    if len(non_adv_examples) > 0:
        return True, non_adv_examples, non_adv_val
    else:
        return True, None, None


# def refine_input_with_constr(model, var_list, counter, lbi, ubi, or_list, start_time, time_out_ind, time_out_total,
#                              input_idxs=None):
#     # model = model.copy()
#     for i, is_greater_tuple in enumerate(or_list):
#         obj_constr = obj_from_is_greater_tuple(is_greater_tuple, var_list, counter)
#         model.addConstr(obj_constr, GRB.LESS_EQUAL, 0, name=f"Adex_Obj_{i:d}")
#     runtime = 0
#
#     lbi_new = lbi.copy()
#     ubi_new = ubi.copy()
#
#     old_cutoff = model.getParamInfo(GRB.Param.Cutoff)[2]
#
#     if input_idxs is None:
#         input_idxs = range(len(ubi))
#
#     for idx in input_idxs:
#         if check_timeleft(config, start_time, min(time_out_ind, time_out_total - runtime)) < 1:
#             break
#         obj = LinExpr()
#         obj += var_list[idx]
#         model.setObjective(obj, GRB.MINIMIZE)
#         model.setParam(GRB.Param.Cutoff, GRB.INFINITY)
#         model.setParam(GRB.Param.TimeLimit,
#                        check_timeleft(config, start_time, min(time_out_ind, time_out_total - runtime)))
#         model.reset()
#         model.optimize()
#         if hasattr(model, "objbound"):
#             lbi_new[idx] = model.objbound
#         elif model.status == 2 and hasattr(model, "objval"):
#             lbi_new[idx] = model.objval
#         elif model.status == 3:
#             break
#         else:
#             lbi_new[idx] = lbi[idx]
#         runtime += model.RunTime
#
#         model.setParam(GRB.Param.TimeLimit,
#                        check_timeleft(config, start_time, min(time_out_ind, time_out_total - runtime)))
#         model.setObjective(obj, GRB.MAXIMIZE)
#         model.setParam(GRB.Param.Cutoff, -GRB.INFINITY)
#         model.reset()
#         model.optimize()
#         runtime += model.RunTime
#         if hasattr(model, "objbound"):
#             ubi_new[idx] = model.objbound
#         elif model.status == 2 and hasattr(model, "objval"):
#             ubi_new[idx] = model.objval
#         elif model.status == 3:
#             break
#         else:
#             ubi_new[idx] = ubi[idx]
#         # print (f"{ind} {model.status} ub ({Cache.ubi[ind]}, {solu}) {model.RunTime}s")
#         # sys.stdout.flush()
#     if model.status == 3:
#         for i, is_greater_tuple in enumerate(or_list):
#             model.remove(model.getConstrByName(f"Adex_Obj_{i:d}"))
#         model.reset()
#         model.optimize()
#         if model.status == 3:
#             assert False, "model remained infeasible"
#         else:
#             model.reset()
#             return None, None
#     model.setParam(GRB.Param.Cutoff, old_cutoff)
#     for i, is_greater_tuple in enumerate(or_list):
#         model.remove(model.getConstrByName(f"Adex_Obj_{i:d}"))
#     model.reset()
#
#     lbi_new = np.maximum(lbi, lbi_new)
#     ubi_new = np.minimum(ubi, ubi_new)
#     return lbi_new, ubi_new


# def evaluate_model(model, var_list, counter, input_len, is_greater_tuple, start_time, time_out, verbosity=2,
#                    callback=None):
#     or_result = False
#     sol = None
#     model.reset()
#     obj = obj_from_is_greater_tuple(is_greater_tuple, var_list, counter)
#     model.setObjective(obj, GRB.MINIMIZE)
#
#     model.setParam(GRB.Param.TimeLimit, check_timeleft(config, start_time, time_out))
#     if callback is None:
#         model.optimize()
#     else:
#         model.optimize(callback)
#
#     obj_bound_num = model.objbound if hasattr(model, "objbound") else None
#
#     obj_bound = f"{model.objbound:.4f}" if hasattr(model, "objbound") else "failed"
#     obj_val = f"{model.objval:.4f}" if hasattr(model, "objval") else "failed"
#     if verbosity >= 2:
#         print(
#             f"Model status: {model.Status}, MILP: {model.IsMIP}, Obj val/bound for constraint {is_greater_tuple}: {obj_val}/{obj_bound}, Final solve time: {model.Runtime:.3f}")
#
#     if model.SolCount > 0:
#         sol = model.x[0:input_len]
#
#     if model.Status == 6:
#         # Cutoff active
#         or_result = True
#     elif model.Status in [2, 9, 11] and obj_bound_num is not None:
#         # positive objbound => sound against adv_label
#         or_result = obj_bound_num > 0
#     elif model.Status not in [2, 6, 9, 11]:
#         if verbosity >= 1:
#             print("Model was not successful status is", model.Status)
#         assert model.Status not in [3, 4], "Infeasible model encountered"
#
#     return or_result, None if or_result else sol, obj_bound_num
#
#
# def evaluate_models(model_lp, var_list_lp, counter_lp, input_len, constraints, terminate_on_failure, model_p_milp=None,
#                     var_list_p_milp=None, counter_p_milp=None, eval_instance=None, find_adex=False, verbosity=2,
#                     start_time=None, regression=False, bound_comp=False):
#     ### Evaluate an LP and partial MILP model for one of the refine domains
#
#     adex_list = []
#     and_result = True
#     updated_constraint = constraints.copy()
#     model_bounds = {}
#     adex_found = False
#
#     for _ in range(len(constraints)):
#         if adex_found and (not bound_comp): break
#         or_list = updated_constraint.pop(0)
#         # OR-Constraint
#         or_result = False
#         ### first do standard lp for all elements in or list
#         for is_greater_tuple in or_list:
#             if adex_found and (not bound_comp): break
#             if check_timeout(config, start_time, 0.1):
#                 continue
#
#             or_result, adex, obj_bound = evaluate_model(model_lp, var_list_lp, counter_lp, input_len, is_greater_tuple,
#                                                         start_time, config.timeout_final_lp, verbosity)
#             model_bounds[is_greater_tuple] = (obj_bound)
#
#             if adex is not None and not regression:
#                 if eval_instance is not None:
#                     cex_label, cex_out = eval_instance(adex)
#                     adex_found = not evaluate_cstr(constraints, cex_out)
#                 if eval_instance is None or adex_found:
#                     adex_list.append(adex)
#             if (adex_found or or_result) and (not bound_comp):
#                 break
#
#         ### try and find adex with lp
#         if find_adex and not or_result and not adex_found and len(constraints) == 1 and len(
#                 constraints[0]) > 1 and eval_instance is not None:
#             if check_timeout(config, start_time, 0.1):
#                 continue
#             _, adex, _ = evaluate_model(model_lp, var_list_lp, counter_lp, input_len, constraints[0], start_time,
#                                         config.timeout_final_lp, verbosity)
#             if adex is not None:
#                 if eval_instance is not None:
#                     cex_label, cex_out = eval_instance(adex)
#                     adex_found = not evaluate_cstr(constraints, cex_out)
#                 if eval_instance is None or adex_found:
#                     adex_list.append(adex)
#
#         ### do partial milp for elements in or list
#         if model_p_milp is not None and (not or_result) and (not adex_found):
#             ### if prio is to find an adex, try that with partial milp, before doing a proof with partial milp
#             if len(or_list) > 1 and not check_timeout(config, start_time, 0.1) and len(constraints) == 1 and len(
#                     constraints[0]) > 1 and eval_instance is not None and not regression:
#                 _, adex, _ = evaluate_model(model_p_milp, var_list_p_milp, counter_p_milp, input_len, constraints[0],
#                                             start_time, config.timeout_final_milp, verbosity)
#                 if adex is not None:
#                     if eval_instance is not None:
#                         cex_label, cex_out = eval_instance(adex)
#                         adex_found = not evaluate_cstr(constraints, cex_out)
#                     if eval_instance is None or adex_found:
#                         adex_list.append(adex)
#
#             if (not or_result) and (not adex_found):
#                 for is_greater_tuple in or_list:
#                     if check_timeout(config, start_time, 0.1): continue
#                     or_result, adex, obj_bound = evaluate_model(model_p_milp, var_list_p_milp, counter_p_milp,
#                                                                 input_len,
#                                                                 is_greater_tuple, start_time, config.timeout_final_milp,
#                                                                 verbosity, callback=(
#                             milp_callback_adex if find_adex else (None if regression else milp_callback)))
#                     if obj_bound is not None:
#                         model_bounds[is_greater_tuple] = float(obj_bound) if model_bounds[
#                                                                                  is_greater_tuple] is None else np.maximum(
#                             model_bounds[is_greater_tuple], float(obj_bound))
#
#                     if adex is not None and not regression:
#                         if eval_instance is not None:
#                             cex_label, cex_out = eval_instance(adex)
#                             adex_found = not evaluate_cstr(constraints, cex_out)
#                         if eval_instance is None or adex_found:
#                             adex_list.append(adex)
#                     if (adex_found or or_result) and (not bound_comp):
#                         break
#
#             ### try and find adex with partial milp
#             if find_adex and len(or_list) == 1 and (not or_result) and (not adex_found) and len(
#                     constraints) == 1 and len(constraints[0]) > 1 and eval_instance is not None:
#                 if check_timeout(config, start_time, 0.1):
#                     continue
#                 _, adex, _ = evaluate_model(model_p_milp, var_list_p_milp, counter_p_milp, input_len, constraints[0],
#                                             start_time, config.timeout_final_milp, verbosity, )
#                 if adex is not None:
#                     if eval_instance is not None:
#                         cex_label, cex_out = eval_instance(adex)
#                         adex_found = not evaluate_cstr(constraints, cex_out)
#                     if eval_instance is None or adex_found:
#                         adex_list.append(adex)
#
#         if not or_result:
#             and_result = False
#             updated_constraint.append(or_list)
#             if (adex_found or terminate_on_failure) and (not bound_comp):
#                 break
#
#     return and_result, updated_constraint, adex_list, model_bounds