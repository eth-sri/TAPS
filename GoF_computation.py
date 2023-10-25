from utils import load_perf_from_json
import os
import numpy as np

root = "./final_TAPS_models"
datasets = ["mnist", "cifar10"]

def get_TAPS_believed_and_real(dir):
    d = load_perf_from_json(dir, "monitor.json")
    believed = d["test_cert_accu"]
    d = load_perf_from_json(dir, "complete_cert.json")
    certified = d["total_cert_rate"]
    adv = d["adv_unattacked_rate"]
    return believed, certified, adv

for dataset in datasets:
    believed_list, certified_list, adv_list = [], [], []
    p1 = os.path.join(root, dataset)
    for eps_str in os.listdir(p1):
        p2 = os.path.join(p1, eps_str)
        for cs in os.listdir(p2):
            dir = os.path.join(p2, cs)
            believed, certified, adv = get_TAPS_believed_and_real(dir)
            believed_list.append(believed)
            certified_list.append(certified)
            adv_list.append(adv)

    believed_list = np.array(believed_list)
    certified_list = np.array(certified_list)
    adv_list = np.array(adv_list)
    print("-"*10, dataset, "-"*10)
    print("TAPS vs Certified")
    print("{:.4f}".format(np.corrcoef(believed_list, certified_list)[0, 1]))
    print("{:.4f}".format((believed_list - certified_list).mean()), "{:.4f}".format((believed_list - certified_list).std()))
    print("TAPS vs Adversarial")
    print("{:.4f}".format(np.corrcoef(believed_list, adv_list)[0, 1]))
    print("{:.4f}".format((believed_list - adv_list).mean()), "{:.4f}".format((believed_list - adv_list).std()))

