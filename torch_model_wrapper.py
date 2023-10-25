import torch
import numpy as np
import math
import time
import torch.nn as nn
from attacks import adv_whitebox

from AIDomains.zonotope import HybridZonotope
import AIDomains.abstract_layers as abs_layers
from AIDomains.wrapper import propagate_abs
from AIDomains.ai_util import construct_C


def project_to_bounds(x, lb, ub):
    # requires x.shape[1:] == lb.shape[1:] and lb.shape[0] == 1
    return torch.max(torch.min(x, ub), lb)

class BasicModelWrapper(nn.Module):
    '''
    Implements standard training procedure
    '''
    def __init__(self, net, loss_fn, input_dim, args):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.args = args
        self.input_dim = input_dim
        self.grad_cleaner = torch.optim.SGD(self.net.parameters(), lr=1) # will only call zero_grad on it

    def forward(self, x):
        return self.net(x)

    def Get_Performance(self, x, y, use_model=None):
        if use_model is None:
            outputs = self.forward(x)
        else:
            outputs = use_model(x)
        loss = self.loss_fn(outputs, y)
        accu, pred_correct = self._Get_Accuracy(outputs, y)
        return loss, accu, pred_correct

    def _Get_Accuracy(self, outputs, y):
        assert len(outputs) == len(y), 'prediction and label should match.'
        pred_correct = torch.argmax(outputs, dim=1) == y
        num_correct = torch.sum(pred_correct)
        return num_correct / len(y), pred_correct

    def _set_BN(self, BN_layers, update_stat:bool=None, enforce_eval:bool=None):
        '''
        # Note that this only influence the volatile BNs, i.e. BNs that are invoked by PGD steps.
        '''
        if enforce_eval == True:
            for layer in BN_layers:
                layer.eval()
        
        if update_stat is not None:
            for layer in BN_layers:
                layer.update_stat = update_stat


class PGDModelWrapper(BasicModelWrapper):
    '''
    Implements PGD training
    '''
    def __init__(self, net, loss_fn, input_dim, device, args):
        super().__init__(net, loss_fn, input_dim, args)
        self.device = device
        self.BNs = [layer for layer in self.net if isinstance(layer, abs_layers._BatchNorm)]
    
    def common_step(self, x, y, eps, num_steps:int, use_vanilla_ibp:bool=None, summary_accu_stat:bool=True, enforce_eval:bool=False):
        '''
        Ignore use_vanilla_ibp
        '''
        self.current_eps = eps
        self._set_BN(self.BNs, update_stat=True, enforce_eval=enforce_eval)
        nat_loss, nat_accu, is_nat_accu = self.Get_Performance(x, y)
        self._set_BN(self.BNs, update_stat=False, enforce_eval=enforce_eval)
        xadv = adv_whitebox(self.net, x, y, (x-eps).clamp(min=0), (x+eps).clamp(max=1), self.device, num_steps, ODI_num_steps=0, lossFunc="pgd")
        yadv = self.net(xadv)
        adv_accu, is_adv_accu = self._Get_Accuracy(yadv, y)
        adv_loss = self.loss_fn(yadv, y)

        loss = (1 - self.args.cert_weight) * nat_loss + self.args.cert_weight * adv_loss

        if summary_accu_stat:
            return (loss, nat_loss, adv_loss), (nat_accu.item(), adv_accu.item())
        else:
            return (loss, nat_loss, adv_loss), (is_nat_accu, is_adv_accu)

class BoxModelWrapper(BasicModelWrapper):
    '''
    Implements vanilla IBP and TAPS-IBP
    '''
    def __init__(self, net, loss_fn, input_dim, device, args, block_sizes=None, store_box_bounds:bool=False, min_eps_pgd:float=0):
        super().__init__(net, loss_fn, input_dim, args)
        self.current_eps = self.args.test_eps
        self.device = device
        self.net_blocks_abs = self.split_net_to_blocks(block_sizes)
        self.volatile_BNs = [layer for layer in self.net_blocks_abs[-1] if isinstance(layer, abs_layers._BatchNorm)]
        self.all_BNs = [layer for layer in self.net if isinstance(layer, abs_layers._BatchNorm)]
        self.current_pgd_weight = 1
        self.store_box_bounds = store_box_bounds
        self.detach_ibp = False
        self.min_eps_pgd = min_eps_pgd
        self.soft_thre = args.soft_thre
        self.tol = 1e-5
        if args.relu_shrinkage is not None:
            for layer in self.net:
                if isinstance(layer, abs_layers.ReLU):
                    layer.relu_shrinkage = args.relu_shrinkage
            print(f"Set ReLU shrinkage to {args.relu_shrinkage}")

    def split_net_to_blocks(self, block_sizes):
        if block_sizes is None:
            return [self.net, ]
        assert len(self.net) == sum(block_sizes), f"Provided block splits have {sum(block_sizes)} layers, but the net has {len(self.net)} layers."
        assert len(block_sizes) == 2, f"Some functions assume there are only two blocks: the first uses IBP, the second uses PGD."

        start = 0
        blocks = []
        for size in block_sizes:
            end = start + size
            abs_block = abs_layers.Sequential(*self.net[start:end])
            abs_block.output_dim = abs_block[-1].output_dim
            blocks.append(abs_block)
            start = end
        return blocks

    def get_IBP_bounds(self, abs_net, input_lb, input_ub, y=None):
        '''
        If y is specified, then use final layer elision trick and only provide pseudo bounds;
        '''
        x_abs = HybridZonotope.construct_from_bounds(input_lb, input_ub, domain='box')
        if y is None:
            abs_out = abs_net(x_abs)
            out_lb, out_ub = abs_out.concretize()
            if not self.store_box_bounds:
                abs_net.reset_bounds()
            return out_lb, out_ub
        else:
            pseudo_bound, pseudo_labels = propagate_abs(abs_net, "box", x_abs, y)
            if not self.store_box_bounds:
                abs_net.reset_bounds()
            return pseudo_bound, pseudo_labels


    def get_cert_performance(self, lb, ub, y, eps, num_steps, use_vanilla_ibp, return_bound:bool=False):

        # propagate the bound block-wisely
        def propagate_with_estimation(lb, ub, y, num_steps, use_single_estimator:bool):
            for block_id, block in enumerate(self.net_blocks_abs):
                if block_id + 1 < len(self.net_blocks_abs):
                    lb, ub = self.get_IBP_bounds(block, lb, ub)
                else:
                    # prepare PGD bounds, Box bounds for y_i - y_t
                    comb_w = self.current_pgd_weight
                    if self.current_eps >= self.min_eps_pgd and comb_w > 0:
                        if use_single_estimator:
                            with torch.no_grad():
                                pgd_input = adv_whitebox(block, (lb+ub)/2, y, lb, ub, self.device, lossFunc="pgd", num_steps=20, retain_graph=(len(self.volatile_BNs)>0))
                            C = construct_C(block.output_dim[-1], y) if y is not None else None
                            PGD_bound = block(pgd_input).unsqueeze(2)
                            # convert to pseudo bounds
                            PGD_bound = torch.matmul(C, PGD_bound).squeeze(2)
                            PGD_bound = torch.cat([torch.zeros(size=(PGD_bound.shape[0],1), dtype=PGD_bound.dtype, device=PGD_bound.device), -PGD_bound], dim=1) # shape: (batch_size, n_class)
                        else:
                            PGD_bound = self._get_estimated_bounds(block, lb, ub, None, num_steps, y)
                        Box_bound, pseudo_labels = self.get_IBP_bounds(block, lb, ub, y)
                        surrogate_bound = comb_w * PGD_bound + (1-comb_w) * Box_bound if comb_w<=1-1e-6 else PGD_bound
                    elif self.current_eps < self.min_eps_pgd or comb_w == 0:
                        Box_bound, pseudo_labels = self.get_IBP_bounds(block, lb, ub, y)
                        surrogate_bound = Box_bound
                    
            return surrogate_bound, Box_bound, pseudo_labels

        def propagate_without_estimation(lb, ub, y):
            pseudo_bound, pseudo_labels = self.get_IBP_bounds(self.net, lb, ub, y)
            return pseudo_bound, pseudo_labels

        loss = 0
        if use_vanilla_ibp:
            pseudo_bound, pseudo_labels = propagate_without_estimation(lb, ub, y)
            loss = self.loss_fn(pseudo_bound, pseudo_labels)
        else:
            pseudo_bound, Box_bound, pseudo_labels = propagate_with_estimation(lb, ub, y, num_steps, self.args.use_single_estimator)
            if self.args.no_ibp_reg:
                loss = self.loss_fn(pseudo_bound, pseudo_labels)
            else:
                alpha = self.args.alpha_box
                loss = GradExpander.apply(self.loss_fn(pseudo_bound, pseudo_labels), alpha) * self.loss_fn(Box_bound, pseudo_labels)

        if return_bound:
            return pseudo_bound
        is_cert_accu = torch.eq(pseudo_bound.argmax(1), pseudo_labels)
        return loss, is_cert_accu.sum() / len(is_cert_accu), is_cert_accu

    def _get_estimated_bounds(self, block, input_lb, input_ub, dim_to_estimate, num_steps, y=None):
        '''
        If dim_to_estimate is None, then estimate all dims
        '''
        C = construct_C(block.output_dim[-1], y) if y is not None else None
        if self.detach_ibp:
            input_lb = input_lb.detach()
            input_ub = input_ub.detach()

        with torch.no_grad():
            if isinstance(block[-1], abs_layers.ReLU):
                assert False # we only estimate for the final block, should not be called
                # RelU does not change pivotal points but stops gradient when a neuron is deactivated
                pts_list = self._get_pivotal_points(abs_layers.Sequential(*block[:-1]), input_lb, input_ub, dim_to_estimate, num_steps, C)
            else:
                pts_list = self._get_pivotal_points(block, input_lb, input_ub, dim_to_estimate, num_steps, C)

        bounds_list = []
        for pts in pts_list:
            # Establish gradient link between pivotal points and bound
            # # 1. projection link: binary gradient, 1 iff the coordinate bound is reached. Sparse gradient, not training well (the final pseudo rate of the trained model is even lower than IBP trained models).
            # pts = project_to_bounds(pts, input_lb.unsqueeze(1), input_ub.unsqueeze(1))
            # # 2. linear link: linear gradient. Problem when ub=lb: the gradient almost only flow through lb, as coef == 0. Performs better than projection.
            # with torch.no_grad():
            #     coef = (pts - input_lb.unsqueeze(1)) / (input_ub - input_lb).unsqueeze(1).clamp(min=1e-10)
            # pts = (1-coef) * input_lb.unsqueeze(1) + coef * input_ub.unsqueeze(1)
            # # 3. rectified linear link
            pts = torch.transpose(pts, 0, 1)
            pts = RectifiedLinearGradientLink.apply(input_lb.unsqueeze(0), input_ub.unsqueeze(0), pts, self.args.soft_thre, self.tol)
            pts = torch.transpose(pts, 0, 1)
            # # 4. Rectified cubic link. A smooth comstruction in that its gradient has better Lipschitzness compared to RectifiedLinear construction. Not much performance gain compared to 3, so we don't use it (Ockham's Razor principle).
            # pts = torch.transpose(pts, 0, 1)
            # pts = RectifiedPolyGradientLink.apply(input_lb.unsqueeze(0), input_ub.unsqueeze(0), pts, self.args.soft_thre)
            # pts = torch.transpose(pts, 0, 1)
            bounds_list.append(self._get_bound_estimation_from_pts(block, pts, dim_to_estimate, C))

        if C is None:
            assert False # should not be called as we estimate for the final block
            # Should have lb and ub
            return bounds_list
        else:
            # Should only have one bound when C is not None
            return bounds_list[0]


    def _get_bound_estimation_from_pts(self, block, pts, dim_to_estimate, C=None):
        '''
        only return estimated bounds for dims need to be estimated;
        '''
        if C is None:
            # pts shape (batch_size, num_pivotal, *shape_in[1:])
            out_pts = block(pts.reshape(-1, *pts.shape[2:]))
            out_pts = out_pts.reshape(*pts.shape[:2], -1)
            dim_to_estimate = dim_to_estimate.unsqueeze(1)
            dim_to_estimate = dim_to_estimate.expand(dim_to_estimate.shape[0], out_pts.shape[1], dim_to_estimate.shape[2])
            out_pts = torch.gather(out_pts, dim=2, index=dim_to_estimate) # shape: (batch_size, num_pivotal, num_pivotal)
            estimated_bounds = torch.diagonal(out_pts, dim1=1, dim2=2) # shape: (batch_size, num_pivotal)
        else:
            # # main idea: convert the 9 adv inputs into one batch to compute the bound at the same time; involve many reshaping
            batch_C = C.unsqueeze(1).expand(-1, pts.shape[1], -1, -1).reshape(-1, *(C.shape[1:])) # may need shape adjustment
            batch_pts = pts.reshape(-1, *(pts.shape[2:]))
            out_pts = block(batch_pts, C=batch_C)
            out_pts = out_pts.reshape(*(pts.shape[:2]), *(out_pts.shape[1:]))
            out_pts = - out_pts # the out is the lower bound of yt - yi, transform it to the upper bound of yi - yt
            # the out_pts should be in shape (batch_size, n_class - 1, n_class - 1)
            ub = torch.diagonal(out_pts, dim1=1, dim2=2) # shape: (batch_size, n_class - 1)
            estimated_bounds = torch.cat([torch.zeros(size=(ub.shape[0],1), dtype=ub.dtype, device=ub.device), ub], dim=1) # shape: (batch_size, n_class)

        return estimated_bounds


    def _get_pivotal_points_one_batch(self, block, lb, ub, num_steps, C):

        num_pivotal = block.output_dim[-1] - 1 # only need to estimate n_class - 1 dim for the final output

        def init_pts(input_lb, input_ub):
            rand_init = input_lb.unsqueeze(1) + (input_ub-input_lb).unsqueeze(1)*torch.rand(input_lb.shape[0], num_pivotal, *input_lb.shape[1:], device=self.device)
            return rand_init
        
        def select_schedule(num_steps):
            if num_steps >= 20 and num_steps <= 50:
                lr_decay_milestones = [int(num_steps*0.7)]
            elif num_steps > 50 and num_steps <= 80:
                lr_decay_milestones = [int(num_steps*0.4), int(num_steps*0.7)]
            elif num_steps > 80:
                lr_decay_milestones = [int(num_steps*0.3), int(num_steps*0.6), int(num_steps*0.8)]
            else:
                lr_decay_milestones = []
            return lr_decay_milestones

        # TODO: move this to args factory
        lr_decay_milestones = select_schedule(num_steps)
        lr_decay_factor = 0.1
        init_lr = max(0.2, 2/num_steps)

        retain_graph = True if len(self.volatile_BNs) > 0 else False
        pts = init_pts(lb, ub)
        variety = (ub - lb).unsqueeze(1).detach()
        best_estimation = -1e5*torch.ones(pts.shape[:2], device=pts.device)
        best_pts = torch.zeros_like(pts)
        with torch.enable_grad():
            for re in range(self.args.restarts):
                lr = init_lr
                pts = init_pts(lb, ub)
                for it in range(num_steps+1):
                    pts.requires_grad = True
                    estimated_pseudo_bound = self._get_bound_estimation_from_pts(block, pts, None, C=C)
                    improve_idx = estimated_pseudo_bound[:, 1:] > best_estimation
                    best_estimation[improve_idx] = estimated_pseudo_bound[:, 1:][improve_idx].detach()
                    best_pts[improve_idx] = pts[improve_idx].detach()
                    # wants to maximize the estimated bound
                    if it != num_steps:
                        loss = - estimated_pseudo_bound.sum()
                        loss.backward(retain_graph=retain_graph)
                        new_pts = pts - pts.grad.sign() * lr * variety
                        pts = project_to_bounds(new_pts, lb.unsqueeze(1), ub.unsqueeze(1)).detach()
                        if (it+1) in lr_decay_milestones:
                            lr *= lr_decay_factor
        return best_pts

    def _get_pivotal_points(self, block, input_lb, input_ub, dim_to_estimate, num_steps, C=None):
        '''
        This assumes the block net is fixed in this procedure. If a BatchNorm is involved, freeze its stat before calling this function.
        '''
        assert C is not None # Should only estimate for the final block
        lb, ub = input_lb.clone().detach(), input_ub.clone().detach()

        pt_list = []
        # split into batches
        bs = self.args.estimation_batch
        lb_batches = [lb[i*bs:(i+1)*bs] for i in range(math.ceil(len(lb) / bs))]
        ub_batches = [ub[i*bs:(i+1)*bs] for i in range(math.ceil(len(ub) / bs))]
        C_batches = [C[i*bs:(i+1)*bs] for i in range(math.ceil(len(C) / bs))]
        for lb_one_batch, ub_one_batch, C_one_batch in zip(lb_batches, ub_batches, C_batches):
            pt_list.append(self._get_pivotal_points_one_batch(block, lb_one_batch, ub_one_batch, num_steps, C_one_batch))
        pts = torch.cat(pt_list, dim=0)
        return [pts, ]

    def compute_nat_loss_and_set_BN(self, eps, x, y, enforce_eval):
        self.current_eps = eps
        self._set_BN(self.volatile_BNs, update_stat=True, enforce_eval=enforce_eval)
        nat_loss, nat_accu, is_nat_accu = self.Get_Performance(x, y)
        self._set_BN(self.volatile_BNs, update_stat=False, enforce_eval=enforce_eval)
        return nat_loss, nat_accu, is_nat_accu

    def compute_cert_loss(self, eps, x, y, num_steps, use_vanilla_ibp):
        lb = torch.clamp(x - eps, min=0, max=1) # shape: (batch_size, n_channel, width, height)
        ub = torch.clamp(x + eps, min=0, max=1)
        cert_loss, cert_accu, is_cert_accu = self.get_cert_performance(lb, ub, y, eps, num_steps, use_vanilla_ibp)
        self.grad_cleaner.zero_grad() # clean the grad from PGD propagation
        return cert_loss, cert_accu, is_cert_accu

    def format_return(self, loss, nat_loss, nat_accu, is_nat_accu, cert_loss, cert_accu, is_cert_accu, summary_accu_stat):
        if summary_accu_stat:
            return (loss, nat_loss, cert_loss), (nat_accu.item(), cert_accu.item())
        else:
            return (loss, nat_loss, cert_loss), (is_nat_accu, is_cert_accu)  

    def common_step(self, x, y, eps, num_steps, use_vanilla_ibp:bool, summary_accu_stat:bool=True, enforce_eval:bool=False):
        nat_loss, nat_accu, is_nat_accu = self.compute_nat_loss_and_set_BN(eps, x, y, enforce_eval)
        cert_loss, cert_accu, is_cert_accu = self.compute_cert_loss(eps, x, y, num_steps, use_vanilla_ibp)
        loss = (1 - self.args.cert_weight) * nat_loss + self.args.cert_weight * cert_loss
        return self.format_return(loss, nat_loss, nat_accu, is_nat_accu, cert_loss, cert_accu, is_cert_accu, summary_accu_stat)


class AdvBoundGradientLink(torch.autograd.Function):
    '''
    The template class for gradient link between adversarial inputs and the input bounds
    '''
    @staticmethod
    def forward(ctx, lb, ub, x, c:float, tol:float):
        ctx.save_for_backward(lb, ub, x)
        ctx.c = c
        ctx.tol = tol
        return x
    
    @staticmethod
    def backward(ctx, grad_x):
        raise NotImplementedError

class RectifiedLinearGradientLink(AdvBoundGradientLink):
    '''
    Estabilish Rectified linear gradient link between the input bounds and the input point.
    Note that this is not a valid gradient w.r.t. the forward function
    Take ub as an example: 
        For dims that x[dim] \in [lb, ub-c*(ub-lb)], the gradient w.r.t. ub is 0. 
        For dims that x[dim] == ub, the gradient w.r.t. ub is 1.
        For dims that x[dim] \in [ub-c*(ub-lb), ub], the gradient is linearly interpolated between 0 and 1.
    
    x should be modified to shape (batch_size, *bound_dims) by reshaping.
    bounds should be of shape (1, *bound_dims)
    '''
    @staticmethod
    def backward(ctx, grad_x):
        lb, ub, x = ctx.saved_tensors
        c, tol = ctx.c, ctx.tol
        slackness = c * (ub - lb)
        # handle grad w.r.t. ub
        thre = (ub - slackness)
        Rectifiedgrad_mask = (x >= thre)
        grad_ub = (Rectifiedgrad_mask * grad_x * (x - thre).clamp(min=0.5*tol) / slackness.clamp(min=tol)).sum(dim=0, keepdim=True)
        # handle grad w.r.t. lb
        thre = (lb + slackness)
        Rectifiedgrad_mask = (x <= thre)
        grad_lb = (Rectifiedgrad_mask * grad_x * (thre - x).clamp(min=0.5*tol) / slackness.clamp(min=tol)).sum(dim=0, keepdim=True)
        # we don't need grad w.r.t. x and param
        return grad_lb, grad_ub, None, None, None

class RectifiedPolyGradientLink(AdvBoundGradientLink):
    '''
    Similar to Rectified Linear gradient link, but replace the piecewise linear connection by -2x^3+3x^2 for x in [0,1]
    '''
    @staticmethod
    def backward(ctx, grad_x):
        lb, ub, x = ctx.saved_tensors
        c, tol = ctx.c, ctx.tol
        slackness = c * (ub - lb)
        # handle grad w.r.t. ub
        thre = (ub - slackness)
        Rectifiedgrad_mask = (x >= thre)
        normalized_diff = (x - thre).clamp(min=0.5*tol) / slackness.clamp(min=tol)
        poly_diff = -2 * normalized_diff**3 + 3*normalized_diff**2
        grad_ub = (Rectifiedgrad_mask * grad_x * poly_diff).sum(dim=0, keepdim=True)
        # handle grad w.r.t. lb
        thre = (lb + slackness)
        Rectifiedgrad_mask = (x <= thre)
        normalized_diff = (thre - x).clamp(min=0.5*tol) / slackness.clamp(min=tol)
        poly_diff = -2 * normalized_diff**3 + 3 * normalized_diff**2
        grad_lb = (Rectifiedgrad_mask * grad_x * poly_diff).sum(dim=0, keepdim=True)
        # we don't need grad w.r.t. x and param
        return grad_lb, grad_ub, None, None, None


class GradExpander(torch.autograd.Function):
    '''
    Multiply the gradient by alpha
    '''
    @staticmethod
    def forward(ctx, x, alpha:float=1):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_x):
        return ctx.alpha * grad_x, None


class SmallBoxModelWrapper(BoxModelWrapper):
    '''
    Implements SABR and TAPS-SABR
    '''
    def __init__(self, net, loss_fn, input_dim, device, args, block_sizes=None, store_box_bounds:bool=False, min_eps_pgd:float=0, eps_shrinkage:float=1, **kwargs):
        super().__init__(net, loss_fn, input_dim, device, args, block_sizes, store_box_bounds, min_eps_pgd)
        self.eps_shrinkage = eps_shrinkage
        assert 0 < eps_shrinkage < 1, "lambda must be in (0, 1); If lambda = 1, then this is exactly IBP, please use Box wrapper instead for efficiency."
        print("Using small box with shrinkage:", self.eps_shrinkage)
    
    def compute_cert_loss(self, eps, x, y, num_steps, use_vanilla_ibp, return_bound:bool=False):
        with torch.no_grad():
            lb_box, ub_box = (x-eps).clamp(min=0), (x+eps).clamp(max=1)
            retain_graph = True if len(self.all_BNs) > 0 else False
            adex = adv_whitebox(self.net, x, y, lb_box, ub_box, self.device, num_steps=10, ODI_num_steps=0, lossFunc="pgd", retain_graph=retain_graph)
            eff_eps = (ub_box - lb_box) / 2 * self.eps_shrinkage
            x_new = torch.clamp(adex, lb_box+eff_eps, ub_box-eff_eps)
            lb_new, ub_new = (x_new - eff_eps), (x_new + eff_eps)

        if return_bound:
            return self.get_cert_performance(lb_new, ub_new, y, eff_eps, num_steps, use_vanilla_ibp, return_bound=True)
        cert_loss, cert_accu, is_cert_accu = self.get_cert_performance(lb_new, ub_new, y, eff_eps, num_steps, use_vanilla_ibp)
        self.grad_cleaner.zero_grad() # clean the grad from PGD propagation
        return cert_loss, cert_accu, is_cert_accu



# grad accumulation wrapper
class GradAccuBoxModelWrapper(BoxModelWrapper):
    def __init__(self, net, loss_fn, input_dim, device, args, block_sizes=None, store_box_bounds:bool=False, min_eps_pgd:float=0, **kawrgs):
        super().__init__(net, loss_fn, input_dim, device, args, block_sizes, store_box_bounds, min_eps_pgd)
        self.accu_batch_size = self.args.grad_accu_batch
        self.named_grads = {} # used to keep the grad of each accumulation batch
        for key, p in self.net.named_parameters():
            self.named_grads[key] = 0.0
        print("Using gradient accumulation with batch size:", self.accu_batch_size)

    def common_step(self, x, y, eps, num_steps, use_vanilla_ibp: bool, summary_accu_stat: bool = True, enforce_eval: bool = False):
        if not self.net.training:
            return super().common_step(x, y, eps, num_steps, use_vanilla_ibp, summary_accu_stat, enforce_eval)
        nat_loss, nat_accu, is_nat_accu = self.compute_nat_loss_and_set_BN(eps, x, y, enforce_eval)
        # split into batches
        num_accu_batches = math.ceil(len(x) / self.accu_batch_size)
        is_cert_accu = []
        cert_loss = []
        retain_graph = True if len(self.all_BNs) > 0 else False
        # TODO: during accumulation, memory cost gradully steps up?
        for i in range(num_accu_batches):
            batch_x = x[i*self.accu_batch_size:(i+1)*self.accu_batch_size]
            batch_y = y[i*self.accu_batch_size:(i+1)*self.accu_batch_size]
            batch_cert_loss, batch_cert_accu, batch_is_cert_accu = self.compute_cert_loss(eps, batch_x, batch_y, num_steps, use_vanilla_ibp)
            is_cert_accu.append(batch_is_cert_accu)
            cert_loss.append(batch_cert_loss.item())
            self.grad_cleaner.zero_grad()
            batch_cert_loss.backward(retain_graph=retain_graph)
            for key, p in self.net.named_parameters():
                if p.grad is not None:
                    self.named_grads[key] += p.grad
        is_cert_accu = torch.cat(is_cert_accu)
        self.grad_cleaner.zero_grad()
        for key, p in self.net.named_parameters():
            if not isinstance(self.named_grads[key], float):
                p.grad = self.named_grads[key] / num_accu_batches
                self.named_grads[key] = 0.0
        cert_loss = torch.mean(torch.tensor(cert_loss))
        cert_accu = is_cert_accu.sum() / len(is_cert_accu)
        loss = cert_loss.clone().to(x.device)
        loss.requires_grad=True # dummy, backward will do nothing
        return self.format_return(loss, nat_loss, nat_accu, is_nat_accu, cert_loss, cert_accu, is_cert_accu, summary_accu_stat)

class GradAccuSmallBoxModelWrapper(SmallBoxModelWrapper, GradAccuBoxModelWrapper):
    def __init__(self, net, loss_fn, input_dim, device, args, block_sizes=None, store_box_bounds:bool=False, min_eps_pgd:float=0, eps_shrinkage:float=1):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, block_sizes=block_sizes, store_box_bounds=store_box_bounds, min_eps_pgd=min_eps_pgd, eps_shrinkage=eps_shrinkage)