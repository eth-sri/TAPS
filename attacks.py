
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
# import ot
import torch.nn.functional as F


def margin_loss(logits, y, device):
    logit_org = logits.gather(1, y.view(-1, 1))
    y_target = (logits - torch.eye(10, device=device)[y] * 9999).argmax(1, keepdim=True)
    logit_target = logits.gather(1, y_target)
    loss = -logit_org + logit_target
    loss = loss.view(-1)
    return loss

# def feature_scatter_loss(logits_adv, logits_nat):
#         batch_size = logits_nat.size(0)
#         m, n = batch_size, batch_size

#         ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_nat, logits_adv, None, None, 0.01, m, n)

#         return ot_loss


class step_lr_scheduler:
    def __init__(self, initial_step_size, gamma=0.1, interval=10):
        self.initial_step_size = initial_step_size
        self.gamma = gamma
        self.interval = interval
        self.current_step = 0

    def step(self, k=1):
        self.current_step += k

    def get_lr(self):
        if isinstance(self.interval, int):
            return self.initial_step_size * self.gamma**(np.floor(self.current_step/self.interval))
        else:
            phase = len([x for x in self.interval if self.current_step>=x])
            return self.initial_step_size * self.gamma**(phase)


def adv_whitebox(model, X, y, specLB, specUB, device, num_steps=200, step_size=0.2,
                  ODI_num_steps=10, ODI_step_size=1., lossFunc="margin", restarts=1, train=True, retain_graph:bool=False):
    out_X = model(X).detach()
    adex = X.detach().clone()
    adex_found = torch.zeros(X.shape[0], dtype=bool, device=X.device)
    best_loss = torch.ones(X.shape[0], dtype=bool, device=X.device)*(-np.inf)
    gama_lambda_orig = 10

    with torch.enable_grad():
        for _ in range(restarts):
            # if adex_found.all():
            #     break

            X_pgd = Variable(X.data, requires_grad=True).to(device)
            randVector_ = torch.ones_like(model(X_pgd)).uniform_(-1, 1)
            random_noise = torch.ones_like(X_pgd).uniform_(-0.5, 0.5)*(specUB-specLB)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
            X_pgd.data[adex_found] = adex.data[adex_found]

            lr_scale = torch.max((specUB-specLB)/2)
            if num_steps >= 50:
                lr_scheduler = step_lr_scheduler(step_size, gamma=0.1, interval=[np.ceil(0.5*num_steps), np.ceil(0.8*num_steps), np.ceil(0.9*num_steps)])
            elif num_steps >= 20:
                lr_scheduler = step_lr_scheduler(step_size, gamma=0.1, interval=[np.ceil(0.7*num_steps)])
            else:
                lr_scheduler = step_lr_scheduler(step_size, gamma=0.2, interval=10)
            # lr_scheduler = step_lr_scheduler(step_size, gamma=0.1, interval=[np.ceil(1 * num_steps)])
            gama_lambda = gama_lambda_orig

            for i in range(ODI_num_steps + num_steps+1):
                opt = optim.SGD([X_pgd], lr=1e-3)
                opt.zero_grad()

                with torch.enable_grad():
                    out = model(X_pgd)

                    adex_found[~torch.argmax(out.detach(), dim=1).eq(y)] = True
                    adex[adex_found] = X_pgd[adex_found].detach()
                    if adex_found.all() or (i == ODI_num_steps + num_steps):
                        break

                    regularization = 0.0
                    if i < ODI_num_steps:
                        loss = (out * randVector_).sum(-1)
                    elif lossFunc == 'pgd':
                        loss = nn.CrossEntropyLoss(reduction="none")(out, y)
                    elif lossFunc == "margin":
                        loss = margin_loss(out, y, device)
                    elif lossFunc == "GAMA":
                        out = torch.softmax(out, 1)
                        loss = margin_loss(out, y, device)
                        regularization = (gama_lambda * (out_X-out)**2).sum(dim=1)
                        if num_steps > 50:
                            gama_lambda *= 0.9
                        else:
                            gama_lambda = max((1 - (i-ODI_num_steps)/(num_steps*0.8)), 0) * gama_lambda_orig
                    # elif lossFunc == "fs":
                    #     out_X = model(X)
                    #     loss = feature_scatter_loss(out, out_X)

                    if i >= ODI_num_steps:
                        improvement_idx = loss > best_loss
                        best_loss[improvement_idx] = loss[improvement_idx].detach()
                        if train:
                            adex[improvement_idx] = X_pgd[improvement_idx].detach()

                        # BN = list(model.children())[2]
                        # print(BN.running_mean)

                loss = loss + regularization
                if not train:
                    loss[adex_found] = 0.0
                loss.sum().backward(retain_graph=retain_graph)

                if i < ODI_num_steps:
                    eta = ODI_step_size * lr_scale * X_pgd.grad.data.sign()
                else:
                    eta = lr_scheduler.get_lr() * lr_scale * X_pgd.grad.data.sign()
                    lr_scheduler.step()
                X_pgd = Variable(torch.minimum(torch.maximum(X_pgd.data + eta, specLB), specUB), requires_grad=True)
    return adex