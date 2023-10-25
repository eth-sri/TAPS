import torch
import torch.nn.functional as F
from AIDomains.abstract_layers import Linear, Conv2d, ReLU, _BatchNorm
from AIDomains.zonotope import HybridZonotope

def compute_L1_reg(abs_net):
    loss = 0
    cnt = 0
    for module in abs_net.modules():
        if isinstance(module, Linear) or isinstance(module, Conv2d):
            loss = loss + module.weight.abs().sum() + module.bias.abs().sum()
            cnt += 1
    loss /= (cnt + 1e-8)
    return loss

def compute_L2_reg(abs_net):
    loss = 0
    for module in abs_net.modules():
        if isinstance(module, Linear) or isinstance(module, Conv2d):
            loss = loss + (module.weight **2).sum() + (module.bias **2).sum()
    return loss    

def compute_L2_reg(abs_net):
    loss = 0
    for module in abs_net.modules():
        if isinstance(module, Linear) or isinstance(module, Conv2d):
            loss = loss + (module.weight **2).sum() + (module.bias **2).sum()
    return loss

def compute_vol_reg(abs_net, x, eps, bound_tol:float=0, recompute_box:bool=False, min_reg_eps=0, max_reg_eps=0.4, start_from:int=0):
    '''L = the area of relaxation triangles'''
    reg = 0
    reg_eps = max(min_reg_eps, min(eps, max_reg_eps))
    if recompute_box:
        abs_net.reset_bounds()
        x = torch.clamp(x + 2 * (torch.rand_like(x, device=x.device) - 0.5) * (eps - reg_eps), min=0, max=1)
        x_abs = HybridZonotope.construct_from_noise(x, reg_eps, "box")
        abs_out = abs_net(x_abs)
        abs_out.concretize()
    cnt = 0
    for i, module in enumerate(abs_net.modules()):
        if i < start_from:
            continue
        if isinstance(module, ReLU):
            lower, upper = module.bounds
            # cross_mask = (lower <= 0) & (upper > 0)
            # reg += ((-lower)[cross_mask] * upper[cross_mask]).sum() / lower.numel()
            reg += ((-lower - bound_tol).clamp(min=0) * (upper - bound_tol).clamp(min=0)).mean()
            # unstable_lb_tol_exceed, unstable_ub_tol_exceed = ((-lower - bound_tol > 0) & (upper > 0)).float().mean().item(), ((upper - bound_tol > 0) & (lower < 0)).float().mean().item()
            inactive_neuron, active_neuron = (upper < 0).float().mean().item(), (lower > 0).float().mean().item()
            cnt += 1
    reg = reg / cnt
    return reg, inactive_neuron, active_neuron

# def compute_fast_reg(model, eps, max_eps, reference = 0.5, reg_lambda=0.5):
#     reg = torch.zeros(()).to(model[-1].weight.device)
#     relu_layers = [layer for layer in model if isinstance(layer, ReLU)]
#     first_layer = [layer for layer in model if not isinstance(layer, Normalization)][0]

#     if first_layer.bounds is None:
#         return reg

#     reg_tightness, reg_std, reg_relu = (reg.clone() for _ in range(3))

#     input_radius = ((first_layer.bounds[1] - first_layer.bounds[0]) / 2).mean()
#     relu_cnt = len(relu_layers)
#     for layer in relu_layers:
#         lb, ub = layer.bounds
#         center = (ub + lb) / 2
#         radius = ((ub - lb) / 2).mean()
#         mean_ = center.mean()
#         std_ = center.std()            

#         reg_tightness += F.relu(reference - input_radius / radius.clamp(min=1e-12)) / reference
#         reg_std += F.relu(reference - std_) / reference

#         # L_{relu}
#         mask_act, mask_inact = lb > 0, ub < 0
#         mean_act = (center * mask_act).mean()
#         mean_inact = (center * mask_inact).mean()
#         delta = (center - mean_)**2
#         var_act = (delta * mask_act).sum()
#         var_inact = (delta * mask_inact).sum()

#         mean_ratio = mean_act / -mean_inact
#         var_ratio = var_act / var_inact
#         mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
#         var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
#         reg_relu_ = (F.relu(reference - mean_ratio) + F.relu(reference - var_ratio)) / reference
#         if not torch.isnan(reg_relu_) and not torch.isinf(reg_relu_):
#             reg_relu += reg_relu_

#     reg = (reg_tightness + reg_relu) / relu_cnt
#     reg *= reg_lambda * (1 - eps / max_eps)

#     return reg

def compute_fast_reg(abs_net, eps, tol=0.5):
    '''
    Ref: https://github.com/shizhouxing/Fast-Certified-Robust-Training/blob/addac383f6fac58d1bae8a231cf0ac9dab405a06/regularization.py

    loss = loss_tightness + loss_relu
    '''
    loss_tightness, loss_relu = 0, 0
    input_lower, input_upper = abs_net[1].bounds # net[0] is the normalization layer
    input_tightness = ((input_upper - input_lower) / 2).mean()
    cnt = 0
    for module in abs_net.modules():
        if isinstance(module, ReLU):
            # L_tightness
            lower, upper = module.bounds
            center = (upper + lower) / 2
            diff = ((upper - lower) / 2)
            tightness = diff.mean()
            mean_ = center.mean()

            loss_tightness += F.relu(tol - input_tightness / tightness.clamp(min=1e-12)) / tol

            mask_act, mask_inact = lower>0, upper<0
            mean_act = (center * mask_act).mean()
            mean_inact = (center * mask_inact).mean()
            delta = (center - mean_)**2
            var_act = (delta * mask_act).sum()
            var_inact = (delta * mask_inact).sum()

            mean_ratio = mean_act / -mean_inact
            var_ratio = var_act / var_inact
            mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
            var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
            loss_relu_ = (F.relu(tol - mean_ratio) + F.relu(tol - var_ratio)) / tol
            if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                loss_relu += loss_relu_ 
            cnt += 1
            
    loss = (loss_tightness + loss_relu) / cnt
    return loss