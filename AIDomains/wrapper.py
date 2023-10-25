
import torch

from AIDomains.deeppoly import DeepPoly, backward_deeppoly, forward_deeppoly
from AIDomains.ai_util import AbstractElement, construct_C


def propagate_abs(net_abs, domain, data_abs, y):
    net_abs.reset_bounds()
    C = construct_C(net_abs.output_dim[-1], y)

    if domain == "box":
        out_box = net_abs(data_abs, C=C)
        lb = out_box.concretize()[0]
    elif domain == "deeppoly_box":
        out_box = net_abs(data_abs, C=C)
        abs_dp_element = DeepPoly(expr_coef=C)
        lb, ub = backward_deeppoly(net_abs, len(net_abs.layers) - 1, abs_dp_element, it=0, use_lambda=False,
                                   use_intermediate=True,
                                   abs_inputs=data_abs)
    elif domain == "deeppoly":
        lb, ub = forward_deeppoly(net_abs, data_abs, expr_coef=C, recompute_bounds=True, use_intermediate=True)

    lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
    fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
    return -lb_padded, fake_labels
