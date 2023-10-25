import torch
import torch.nn as nn
import os
import numpy as np
import time
from args_factory import get_args
from loaders import get_loaders
from networks import get_network, fuse_BN_wrt_Flatten, remove_BN_wrt_Flatten
from model_wrapper import BoxModelWrapper, BasicModelWrapper, MNBaBDeepPolyModelWrapper
from attacks import adv_whitebox
from AIDomains.zonotope import HybridZonotope
from AIDomains.abstract_layers import Sequential, Flatten
from AIDomains.concrete_layers import Normalization as PARC_normalize
from AIDomains.wrapper import propagate_abs
from AIDomains.wrapper import construct_C

from utils import write_perf_to_json, load_perf_from_json, fuse_BN

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('mn-bab/')
sys.path.append('mn-bab/ELINA/python_interface/')
try:
    from src.mn_bab_verifier import MNBaBVerifier
    from src.abstract_layers.abstract_network import AbstractNetwork
    from src.utilities.argument_parsing import get_config_from_json
    from src.utilities.config import make_config
    from src.utilities.loading.network import freeze_network
    from src.concrete_layers.normalize import Normalize as mnbab_normalize
    from src.verification_instance import VerificationInstance
    from src.utilities.initialization import seed_everything as mnbab_seed
except:
    raise ModuleNotFoundError("MN-BaB not installed or not found.")


def verify_with_mnbab(net, mnbab_verifier, x, y, eps, norm_mean, norm_std, device, mnbab_config, num_classes:int=10, tolerate_error:bool=False):
    is_verified = np.zeros(len(x), dtype=bool)
    is_undecidable = np.zeros(len(x), dtype=bool)
    is_attacked = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        try:
            net.reset_input_bounds()
            net.reset_output_bounds()
            net.reset_optim_input_bounds()
            input = x[i:i+1]
            label = y[i:i+1]
            input_lb = (input - eps).clamp(min=0)
            input_ub = (input + eps).clamp(max=1)
            # normalize the input here
            input = (input - norm_mean) / norm_std
            input_lb = (input_lb - norm_mean) / norm_std
            input_ub = (input_ub - norm_mean) / norm_std
            with torch.enable_grad():
                inst = VerificationInstance.create_instance_for_batch_ver(net, mnbab_verifier, input, input_lb, input_ub, int(label), mnbab_config, num_classes)
                inst.run_instance()
            if inst.is_verified:
                is_verified[i] = 1
                print("mnbab verifies a new one!")
            if not inst.is_verified and inst.adv_example is None:
                is_undecidable[i] = 1
                print("mnbab cannot decide!")
            if inst.adv_example is not None:
                is_attacked[i] = 1
                print("mnbab finds an adex!")
            inst.free_memory()
        except Exception as e:
            if tolerate_error:
                print("mnbab error! Either GPU/CPU memory overflow.")
                is_undecidable[i] = 1
                continue
            else:
                raise e
    return is_verified, is_undecidable, is_attacked
        

def update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, certify_start_time, previous_time, batch_idx, test_loader):
    perf_dict = {
        'num_cert_ibp':num_cert_ibp, 
        'num_nat_accu':num_nat_accu, 
        'num_cert_dpb':num_cert_dp_box,
        'num_cert_mnbab':num_mnbab_verified,
        'num_total':num_total, 
        'num_adv_attacked':num_adv_attacked,
        'nat_accu': num_nat_accu / num_total,
        'ibp_cert_rate': num_cert_ibp / num_total,
        'dpb_cert_rate': num_cert_dp_box / num_total,
        'mnbab_cert_rate': num_mnbab_verified / num_total,
        'adv_unattacked_rate': (num_nat_accu - num_adv_attacked) / num_total,
        "total_cert_rate": (num_cert_ibp + num_cert_dp_box + num_mnbab_verified) / num_total,
        "total_time": time.time() - certify_start_time + previous_time,
        "batch_remain": len(test_loader) - batch_idx - 1
        }
    postfix = "" if args.start_idx is None else f"{args.start_idx}-{args.end_idx}"
    write_perf_to_json(perf_dict, save_root, filename=f"cert{postfix}.json")
    write_perf_to_json(args.__dict__, save_root, filename=f"cert_args{postfix}.json")
    return perf_dict

def transform_abs_into_torch(abs_net, torch_net):
    '''
    load the params in the abs_net into torch net
    '''
    abs_state = abs_net.state_dict()
    torch_state = {}
    for key, value in abs_state.items():
        key = key.lstrip("layers.")
        if key == "0.sigma":
            key = "0.std"
        torch_state[key] = value

    torch_net.load_state_dict(torch_state)
    return torch_net

def switch_normalization_version(torch_net):
    '''
    Using the normalization layer defined in MN-BaB instead
    '''
    for i, layer in enumerate(torch_net):
        if isinstance(layer, PARC_normalize):
            mnbab_layer = mnbab_normalize(layer.mean, layer.std, channel_dim=1)
            torch_net[i] = mnbab_layer
    return torch_net


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_file = get_config_from_json(args.mnbab_config)
    mnbab_config = make_config(**config_file)

    loaders, input_size, input_channel, n_class = get_loaders(args, shuffle_test=False) # shuffle test set so that we get reasonable estimation of the statistics even when we don't finish the full test.
    input_dim = (input_channel, input_size, input_size)

    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    torch_net = get_network(args.net, args.dataset, device)
    torch_net.eval()
    net = Sequential.from_concrete_network(torch_net, input_dim, disconnect=True)
    net.eval()

    if not os.path.isfile(args.load_model):
        raise ValueError(f"There is no such file {args.load_model}.")
    save_root = os.path.dirname(args.load_model)


    net.load_state_dict(torch.load(args.load_model, map_location=device))
    print(f"Loaded {args.load_model}")

    net = fuse_BN_wrt_Flatten(net, device, remove_all=True)
    torch_net = remove_BN_wrt_Flatten(torch_net, device, remove_all=True)
    print(net)


    model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), (input_channel, input_size, input_size), device, args)
    model_wrapper.summary_accu_stat = False
    model_wrapper.robust_weight = 0
    
    model_wrapper.net.eval()

    eps = args.test_eps
    print("Certifying for eps:", eps)
    num_cert_ibp, num_nat_accu, num_cert_dp_box, num_total, num_adv_attacked, num_mnbab_verified = 0, 0, 0, 0, 0, 0

    previous_time = 0
    if args.load_certify_file:
        # warning: loading from existing file is not always trustable. If a certification is terminated by ctrl-C, the gurobi only shows an info and returns, resulting in a false negative. If the interuption is not too much, then the result should be only very slightly worse.
        perf_dict = load_perf_from_json(save_root, args.load_certify_file)
        if perf_dict is not None:
            num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, previous_time = perf_dict['num_cert_ibp'], perf_dict['num_nat_accu'], perf_dict['num_cert_dpb'],perf_dict['num_cert_mnbab'], perf_dict['num_total'], perf_dict['num_adv_attacked'], perf_dict["total_time"]

    model_wrapper.net.set_dim(torch.zeros((test_loader.batch_size, *input_dim), device='cuda'))

    # prepare mn-bab model
    # mnbab use a different normalization class
    torch_net = transform_abs_into_torch(net, torch_net)
    # torch_net = switch_normalization_version(torch_net)
    mnbab_net = AbstractNetwork.from_concrete_module(
        torch_net[1:], mnbab_config.input_dim
    ).to(device) # remove normalization layer, which would be done directly to its input
    freeze_network(mnbab_net)
    mnbab_verifier = MNBaBVerifier(mnbab_net, device, mnbab_config.verifier)


    certify_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if args.load_certify_file and batch_idx*args.test_batch < num_total:
                continue
            if args.start_idx:
                if batch_idx*args.test_batch < args.start_idx:
                    continue
                elif batch_idx*args.test_batch >= args.end_idx:
                    break
            print("Batch id:", batch_idx)
            model_wrapper.net = model_wrapper.net.to(device)
            x, y = x.to(device), y.to(device)
            # 1. try to verify with IBP 
            (loss, nat_loss, cert_loss), (nat_accu, cert_accu), (is_nat_accu, is_cert_accu) = model_wrapper.compute_model_stat(x, y, eps)
            num_nat_accu += is_nat_accu.sum().item()
            num_cert_ibp += is_cert_accu.sum().item()
            num_total += len(x)
            print(f"Batch size: {len(x)}, Nat accu: {is_nat_accu.sum().item()}, IBP cert: {is_cert_accu.sum().item()}")

            # only consider classified correct and not IBP verified below
            x = x[is_nat_accu & (~is_cert_accu)]
            y = y[is_nat_accu & (~is_cert_accu)]
            if len(x) == 0:
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, certify_start_time, previous_time, batch_idx, test_loader)
                continue

            # 2. try to attack with pgd
            is_adv_attacked = torch.zeros(len(x), dtype=torch.bool)
            x_adv = adv_whitebox(model_wrapper.net, x, y, (x-eps).clamp(min=0), (x+eps).clamp(max=1), device, lossFunc='pgd', ODI_num_steps=0, restarts=5, num_steps=args.test_steps)
            y_adv = model_wrapper.net(x_adv).argmax(dim=1)
            is_adv_attacked[(y_adv != y)] = 1
            num_adv_attacked += is_adv_attacked.sum().item()

            # only consider not adv attacked below
            x = x[~is_adv_attacked]
            y = y[~is_adv_attacked]
            if len(x) == 0:
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, certify_start_time, previous_time, batch_idx, test_loader)
                continue

            # 3. try to verify with dp_box
            data_abs = HybridZonotope.construct_from_noise(x, eps, "box")
            dpb, pesudo_label = propagate_abs(model_wrapper.net, "deeppoly_box", data_abs, y)
            is_dpb_cert = (dpb.argmax(1) == pesudo_label)
            num_cert_dp_box += is_dpb_cert.sum().item()
            x = x[~is_dpb_cert]
            y = y[~is_dpb_cert]
            if len(x) == 0:
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, certify_start_time, previous_time, batch_idx, test_loader)
                continue

            # 4. try to verify with MN-BaB
            is_verified, is_undecidable, is_mnbab_attacked = verify_with_mnbab(mnbab_net, mnbab_verifier, x, y, eps, torch_net[0].mean, torch_net[0].std, device, mnbab_config, n_class, tolerate_error=args.tolerate_error)
            num_mnbab_verified += is_verified.sum().item()
            x = x[is_undecidable]
            y = y[is_undecidable]
            num_adv_attacked += is_mnbab_attacked.sum().item()

            perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, certify_start_time, previous_time, batch_idx, test_loader)


        perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, certify_start_time, previous_time, batch_idx, test_loader)
        postfix = "" if args.start_idx is None else f"{args.start_idx}-{args.end_idx}"
        write_perf_to_json(perf_dict, save_root, filename=f"complete_cert{postfix}.json")


        

def main():
    args = get_args()
    run(args)

if __name__ == '__main__':
    mnbab_seed(123)
    main()


