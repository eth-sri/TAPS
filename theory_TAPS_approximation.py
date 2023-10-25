import torch
import torch.nn as nn
from args_factory import get_args
from loaders import get_loaders
from utils import Scheduler, Statistics
from PARC_networks import get_network, fuse_BN_wrt_Flatten, add_BN_wrt_Flatten
from torch_model_wrapper import BoxModelWrapper, SmallBoxModelWrapper
import os
from utils import write_perf_to_json, load_perf_from_json, fuse_BN
from tqdm import tqdm
import random
import numpy as np
from regularization import compute_fast_reg, compute_vol_reg, compute_L1_reg
import time
from datetime import datetime
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d
from AIDomains.wrapper import propagate_abs
from AIDomains.zonotope import HybridZonotope
from AIDomains.ai_util import construct_C
from attacks import adv_whitebox

from MILP_Encoding.milp_utility import get_bound_with_milp

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    perf_dict["start_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    verbose = False

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args, shuffle_test=False)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device)

    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    if args.load_model:
        net.load_state_dict(torch.load(args.load_model))

    TAPS_model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes=(3,4))
    SABR_model_wrapper = SmallBoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, eps_shrinkage=0.4)


    os.makedirs(args.save_dir, exist_ok=True)

    eps = args.test_eps
    all_bounds = {"IBP":[], "TAPS":[], "PGD":[], "MILP":[], "SABR":[], }
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            for i in range(len(x)):
                input = x[i:i+1]
                target = y[i:i+1]
                abs_input = HybridZonotope.construct_from_noise(input, eps, domain="box")
                IBP_bound, _ = propagate_abs(net, "box", abs_input, target)
                TAPS_bound = TAPS_model_wrapper.get_cert_performance((input-eps).clamp(min=0), (input+eps).clamp(max=1), target, eps, num_steps=20, use_vanilla_ibp=False, return_bound=True)

                pgd_input = adv_whitebox(net, input, target, (input-eps).clamp(min=0), (input+eps).clamp(max=1), device, num_steps=50, ODI_num_steps=0, restarts=3, lossFunc="margin")
                C = construct_C(net.output_dim[-1], target)
                PGD_bound = net(pgd_input).unsqueeze(2)
                # convert to pseudo bounds
                PGD_bound = torch.matmul(C, PGD_bound).squeeze(2)
                PGD_bound = torch.cat([torch.zeros(size=(PGD_bound.shape[0],1), dtype=PGD_bound.dtype, device=PGD_bound.device), -PGD_bound], dim=1)
                
                MILP_bound = get_bound_with_milp(net, abs_input, input, target, n_class, verbose=False)
                IBP_bound = IBP_bound[0, 1:].cpu().numpy()
                TAPS_bound = TAPS_bound[0, 1:].cpu().numpy()
                PGD_bound = PGD_bound[0, 1:].cpu().numpy()
                MILP_bound = MILP_bound.numpy()

                SABR_bound = SABR_model_wrapper.compute_cert_loss(eps, input, target, num_steps=10, use_vanilla_ibp=True, return_bound=True)[0, 1:].cpu().numpy()

                all_bounds["IBP"].append(IBP_bound)
                all_bounds["TAPS"].append(TAPS_bound)
                all_bounds["PGD"].append(PGD_bound)
                all_bounds["MILP"].append(MILP_bound)
                all_bounds["SABR"].append(SABR_bound)

        for key in all_bounds.keys():
            all_bounds[key] = np.concatenate(all_bounds[key])
            np.save(f"{os.path.join(args.save_dir, key)}", all_bounds[key])

def main():
    args = get_args()
    run(args)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(123)
    main()
