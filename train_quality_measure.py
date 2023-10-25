import torch
from torch_model_wrapper import BoxModelWrapper
from mix_train import test_loop, seed_everything
from PARC_networks import get_network
from loaders import get_loaders
from AIDomains.abstract_layers import Sequential
from args_factory import get_args
import torch.nn as nn
import os
from utils import write_perf_to_json

def test_under_given_split(net, eps, test_loader, input_dim, device, block_sizes, args):
    '''
    Test the model under the given split of TAPS
    '''
    model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes)
    model_wrapper.current_pgd_weight = 1
    nat_accu, TAPS_accu = test_loop(model_wrapper, eps, test_loader, device, args)
    return TAPS_accu

def test_all_split(net, eps, test_loader, input_dim, device, args):
    all_classifier_size = [4, 8, 11, 14, 17, 20]
    all_block_sizes = [[len(net)-i, i] for i in all_classifier_size]
    result = {}
    for bs, cs in zip(all_block_sizes, all_classifier_size):
        print("Testing for classifier size:", cs)
        result[cs] = test_under_given_split(net, eps, test_loader, input_dim, device, bs, args)
    return result


dir_eps_map = {
    "eps2.255": 2/255,
    "eps8.255": 8/255,
    "eps0.1": 0.1,
    "eps0.3": 0.3
}

if __name__ == "__main__":
    seed_everything(123)
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    test_loader = loaders[-1]
    input_dim = (input_channel, input_size, input_size)
    net = get_network(args.net, args.dataset, device)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)

    # first load a model, then test under given split
    TAPS_root = f"final_TAPS_models/{args.dataset}"
    IBP_root = f"reproduced_IBP_models/{args.dataset}"
    all_results = {}
    for eps_str in os.listdir(IBP_root):
        eps = dir_eps_map[eps_str]
        results = {}
        all_results[eps_str] = results

        IBP_model_dir = os.path.join(IBP_root, eps_str, "box_trained/cnn_7layer_bn/alpha5.0/fast_reg")
        print("Testing", IBP_model_dir)
        net.load_state_dict(torch.load(os.path.join(IBP_model_dir, "model.ckpt")))
        IBP_result = test_all_split(net, eps, test_loader, input_dim, device, args)
        results["IBP"] = IBP_result
        write_perf_to_json(all_results, "./train_quality_results", f"{args.dataset}_result.json")

        TAPS_eps_dir = os.path.join(TAPS_root, eps_str)
        for bs_str in os.listdir(TAPS_eps_dir):
            bs = int(bs_str.split("_")[-1])
            TAPS_model_dir = os.path.join(TAPS_eps_dir, bs_str)
            print("Testing", TAPS_model_dir)
            net.load_state_dict(torch.load(os.path.join(TAPS_model_dir, "model.ckpt")))
            TAPS_result = test_all_split(net, eps, test_loader, input_dim, device, args)
            results[bs_str] = TAPS_result
            write_perf_to_json(all_results, "./train_quality_results", f"{args.dataset}_result.json")





