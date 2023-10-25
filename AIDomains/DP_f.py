"""
Based on DeepPoly_f from DiffAI (https://github.com/eth-sri/diffai/blob/master/ai.py)
"""
import numpy as np
import torch
import torch.nn.functional as F
from .ai_util import clamp_image, head_from_bounds, AbstractElement
from .zonotope import HybridZonotope
from typing import Optional, List, Tuple, Union
from torch import Tensor
import warnings

from .zonotope import HybridZonotope
from .ai_util import clamp_image, head_from_bounds, AbstractElement, get_neg_pos_comp

try:
    from gurobipy import GRB, Model, LinExpr
    GUROBI_AVAILABLE = True
except:
    GUROBI_AVAILABLE = False
    warnings.warn("GUROBI not available, no inclusion checks for DeepPoly")


class DeepPoly_f(AbstractElement):
    def __init__(self, x_l_coef: Tensor, x_u_coef: Tensor, x_l_bias: Optional[Tensor]=None,
                 x_u_bias: Optional[Tensor]=None, inputs: Optional["AbstractElement"]=None) -> None:
        super(DeepPoly_f,self).__init__()
        dtype = x_l_coef.dtype
        device = x_l_coef.device
        self.x_l_coef = x_l_coef
        self.x_u_coef = x_u_coef
        self.x_l_bias = torch.zeros(x_l_coef.shape[1:], device=device, dtype=dtype) if x_l_bias is None else x_l_bias
        self.x_u_bias = torch.zeros(x_l_coef.shape[1:], device=device, dtype=dtype) if x_u_bias is None else x_u_bias

        self.inputs = inputs
        self.domain = "DPF"

    @classmethod
    def construct_from_noise(cls, x: Tensor, eps: Union[float,Tensor], data_range: Tuple[float,float]=None, dtype:Optional[str]=None, domain:Optional[str]=None) -> "DeepPoly_f":
        # compute center and error terms from input, perturbation size and data range
        assert domain is None or domain == "DPF"
        if dtype is None:
            dtype = x.dtype
        if data_range is None:
            data_range = (-np.inf, np.inf)
        assert data_range[0] < data_range[1]
        x_center, x_beta = clamp_image(x, eps, data_range[0], data_range[1])
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(x_center, HybridZonotope.construct(x_center, x_beta, domain="box"))

    @classmethod
    def construct_constant(cls, x: Tensor, inputs:Optional["HybridZonotope"]=None, dtype:Optional[str]=None, domain:Optional[str]=None) -> "DeepPoly_f":
        # compute center and error terms from input, perturbation size and data range
        assert domain is None or domain == "DPF"
        if dtype is None:
            dtype = x.dtype
        x_l_coef = torch.zeros([inputs.head[0].numel() if inputs is not None else 1, *x.shape], dtype=dtype, device=x.device)
        x_u_coef = torch.zeros([inputs.head[0].numel() if inputs is not None else 1, *x.shape], dtype=dtype, device=x.device)
        return cls(x_l_coef, x_u_coef, x, x, inputs=inputs)

    @classmethod
    def construct_from_bounds(cls, min_x: Tensor, max_x: Tensor, dtype: torch.dtype=torch.float32, domain:Optional[str]=None) -> "DeepPoly_f":
        assert domain is None or domain == "DPF"
        assert min_x.shape == max_x.shape
        x_center, x_beta = head_from_bounds(min_x, max_x)
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(x_center, HybridZonotope.construct(x_center, x_beta, domain="box"))

    @classmethod
    def construct_form_zono(cls, input_zono: "HybridZonotope") -> "DeepPoly_f":
        x_l_coef = input_zono.errors.clone()
        x_u_coef = input_zono.errors.clone()
        x_l_bias = input_zono.head.clone()
        x_u_bias = input_zono.head.clone()
        base_box = HybridZonotope.construct_from_noise(torch.zeros_like(input_zono.head), eps=1, domain="box", data_range=(-1,1))
        return cls(x_l_coef, x_u_coef, x_l_bias, x_u_bias, inputs=base_box)

    @staticmethod
    def construct(x:Tensor, inputs:"HybridZonotope"):
        k = int(np.prod(np.array(x.size()[1:])))
        x_l_coef = torch.stack(x.shape[0] * [torch.eye(k).view(-1, *x.size()[1:])], dim=1).to(x.device)
        x_u_coef = x_l_coef.clone().detach()
        return DeepPoly_f(x_l_coef, x_u_coef, inputs=inputs)

    def dim(self):
        return self.x_l_coef.dim()-1

    @staticmethod
    def cat(x: List["DeepPoly_f"], dim: Union[Tensor, int] = 0) -> "DeepPoly_f":
        assert all([x[0].inputs == y.inputs for y in x])

        x_l_coef = torch.cat([x_i.x_l_coef for x_i in x], dim=dim+1)
        x_u_coef = torch.cat([x_i.x_u_coef for x_i in x], dim=dim+1)
        x_l_bias = torch.cat([x_i.x_l_bias for x_i in x], dim=dim)
        x_u_bias = torch.cat([x_i.x_u_bias for x_i in x], dim=dim)

        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, x[0].inputs)

    @staticmethod
    def stack(x: List["DeepPoly_f"], dim: Union[Tensor, int] = 0) -> "DeepPoly_f":
        assert all([x[0] == y for y in x])

        x_l_coef = torch.stack([x_i.x_l_coef for x_i in x], dim=dim)
        x_u_coef = torch.stack([x_i.x_u_coef for x_i in x], dim=dim)
        x_l_bias = torch.stack([x_i.x_l_bias for x_i in x], dim=dim)
        x_u_bias = torch.stack([x_i.x_u_bias for x_i in x], dim=dim)

        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, x[0].inputs)

    def size(self, idx=None):
        size_tmp = self.x_l_coef.shape[1:]
        if idx is None:
            return size_tmp
        else:
            return size_tmp(idx)

    def view(self, size):
        input_terms = self.x_l_coef.shape[0]
        return DeepPoly_f(self.x_l_coef.view(input_terms,*size), self.x_u_coef.view(input_terms,*size),
                          self.x_l_bias.view(*size), self.x_u_bias.view(*size), self.inputs)


    def contains(self, other: "DeepPoly_f", verbose:Optional[bool]=False) -> Tuple[bool, float]:
        return other.contained(self, verbose=verbose)

    def contained(self, other: "DeepPoly_f", verbose:Optional[bool]=False) -> Tuple[bool, float]:
        assert self.shape == other.shape, "Can only check inclusion for models of same dimension"
        assert self.inputs.shape == other.inputs.shape, "Can only check inclusion for models of same input dimension"
        assert self.shape[0]==1
        # self is the inner polyhedra 
        # other is the outer polyhedra

        # based on Sadraddini et al. (2018), Lemma 1, Equation 9
        # by consturction DP has qx=qy=2 constraints per dimension n
        model = Model("Containment DeepPoly")
        out_size = self.x_l_bias.numel() # n in Sadraddini
        in_size = self.inputs.head.numel()

        # additionally we have constraints in the input space
        # and thus require containment or equiality in the input space

        model.setParam("OutputFlag", 0)
        model.setParam(GRB.Param.FeasibilityTol, 1e-5)
        var_list = []

        # Encode input
        def add_input_encoding_to_model(inputs: "HybridZonotope", model: "Model", var_list: list, pre: str) -> Tuple["Model", list, int]:
            assert inputs.shape[0] == 1 # only one batch element

            # add input error terms
            flat_inputs = inputs.flatten()[0]
            num_error_terms = 0 if self.inputs.errors is None else self.inputs.errors.shape[0]
            pos_box_terms = 0 
            pos_zono_terms = len(var_list)
            # add zonotope error terms
            for j in range(num_error_terms):
                var_name = pre + "_e_" + str(j)
                var = model.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=var_name)
                var_list.append(var)
            lb, ub = flat_inputs.concretize()
            # add box error terms
            if flat_inputs.beta is not None:
                pos_box_terms = len(var_list)
                for i in range(in_size):
                    var_name = pre + "_b_" + str(i)
                    var = model.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=var_name)
                    var_list.append(var)

            pos_in_vars = len(var_list)
            # describe input values as a vector i, in terms of its error terms
            for i in range(in_size):
                var_name = pre + "_i_" + str(i)
                var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[i], ub=ub[i], name=var_name)
                var_list.append(var)
                expr = LinExpr()
                #expr += -1 * var                            # new variable
                if flat_inputs.beta is not None:
                    expr += flat_inputs.beta[i] * var_list[pos_box_terms + i]   # box error component
                if flat_inputs.errors is not None:
                    expr.addTerms(flat_inputs.errors[:,i], var_list[pos_zono_terms:pos_zono_terms+num_error_terms]) # zonotope terms
                expr.addConstant(flat_inputs.head[i])
                model.addConstr(expr, GRB.EQUAL, var)

            return model, var_list, pos_in_vars

        model, var_list, pos_in_vars_s = add_input_encoding_to_model(self.inputs, model, var_list, 's')

        if self.inputs != other.inputs:
            contained_in = other.inputs.contains(self.inputs)[0]
        else:
            contained_in = True

        if not contained_in:
            if verbose: print("Inputs are were not contained")
            return False, -np.inf

        flat_self = self.flatten()[0]
        lb, ub = flat_self.concretize()
        pos_in_vars_x = len(var_list)
        # represent the variable of self as a vector x (as in Sadraddini)
        for i in range(out_size):
            var_name = "x_" + str(i)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[i], ub=ub[i], name=var_name)
            var_list.append(var)
            lb_expr = LinExpr()
            lb_expr.addTerms(flat_self.x_l_coef[:, i], var_list[pos_in_vars_s:pos_in_vars_s+in_size])
            lb_expr.addConstant(flat_self.x_l_bias[i])
            model.addConstr(lb_expr, GRB.LESS_EQUAL, var)
            ub_expr = LinExpr()
            ub_expr.addTerms(flat_self.x_u_coef[:, i], var_list[pos_in_vars_s:pos_in_vars_s+in_size])
            ub_expr.addConstant(flat_self.x_u_bias[i])
            model.addConstr(lb_expr, GRB.GREATER_EQUAL, var)


        # the Sadraddini check requires us that for each dimension of x,
        # the corresponding bounds of the outer zonotope (other) hold
        flat_other = other.flatten()[0]
        contained = True
        worst_violation = -np.inf
        for i in range(out_size):
           # check lower bound in i-th dimension
            obj = LinExpr()
            # to obtain the maximal attainble lower bound y_lb we maximize the linear lower bound of y s on the support of x
            obj.addTerms(flat_other.x_l_coef[:, i], var_list[pos_in_vars_s:pos_in_vars_s+in_size])
            obj.addConstant(flat_other.x_l_bias[i])
            # and subtract the actual value of x_i
            obj -= var_list[pos_in_vars_x + i]
            # obj is y_lb - x_i
            model.setObjective(obj, GRB.MAXIMIZE)
            # if the maximum <= 0, we know that x_i will be conatined in the outer zonotope (for that dimension)
            model.update()
            model.reset()
            model.optimize()
            if model.Status != 2:
                assert False
            else:
                if verbose: print("checking lower-bound for dim " + str(i) + ": " + str(model.objval))
                worst_violation = max(worst_violation, model.objval)
                contained = model.objval <= 0
            if not contained:
                if not contained:
                    if verbose:
                        fn = "/tmp/check_lower_bound_" + str(i) + ".lp"
                        model.write(fn)
                        print("mismatch in lower-bound for dim" + str(i))
                        with open(fn, 'r') as f:
                            print(f.read())
                break

            # check upper bound in i-th dimension
            # same idea as the lower bound, just signs flipped
            obj = LinExpr()
            obj.addTerms(-flat_other.x_u_coef[:, i], var_list[pos_in_vars_s:pos_in_vars_s+in_size])
            obj.addConstant(-flat_other.x_u_bias[i])
            obj += var_list[pos_in_vars_x + i]
            # obj is x_i - y_ob
            model.setObjective(obj, GRB.MAXIMIZE)
            model.update()
            model.reset()
            model.optimize()
            model.write("check_upper_bound_" + str(i) + ".lp")
            if model.Status != 2:
                assert False
            else:
                if verbose: print("checking upper-bound for dim " + str(i) + ": " + str(model.objval))
                worst_violation = max(worst_violation, model.objval)
                contained = model.objval <= 0
            if not contained:
                if verbose:
                    fn = "/tmp/check_upper_bound_" + str(i) + ".lp"
                    model.write(fn)
                    print("mismatch in upper-bound for dim" + str(i))
                    with open(fn, 'r') as f:
                        print(f.read())
                break
        return contained, worst_violation

    @property
    def shape(self):
        return self.x_l_bias.shape

    @property
    def device(self):
        return self.x_l_bias.device

    @property
    def dtype(self):
        return self.x_l_bias.dtype

    def flatten(self):
        return self.view([*self.shape[:1],-1])

    def normalize(self, mean: Tensor, sigma: Tensor) -> "DeepPoly_f":
        return (self - mean) / sigma

    def __sub__(self, other: Union[Tensor, float, int, "DeepPoly_f"]) -> "DeepPoly_f":
        if isinstance(other, torch.Tensor) or isinstance(other, float) or isinstance(other, int):
            return DeepPoly_f(self.x_l_coef, self.x_u_coef, self.x_l_bias-other, self.x_u_bias-other, self.inputs)
        elif isinstance(other, DeepPoly_f):
            assert self.inputs == other.inputs
            return DeepPoly_f(self.x_l_coef-other.x_u_coef, self.x_u_coef-other.x_l_coef, self.x_l_bias-other.x_u_bias, self.x_u_bias-other.x_l_bias, self.inputs)
        else:
            assert False, 'Unknown type of other object'

    def __neg__(self) -> "DeepPoly_f":
        return DeepPoly_f(-self.x_u_coef, -self.x_l_coef, -self.x_u_bias, -self.x_l_bias, self.inputs)

    def __add__(self, other: Union[Tensor, float, int, "DeepPoly_f"]) -> "DeepPoly_f":
        if isinstance(other, torch.Tensor) or isinstance(other, float) or isinstance(other, int):
            return DeepPoly_f(self.x_l_coef, self.x_u_coef, self.x_l_bias+other, self.x_u_bias+other, self.inputs)
        elif isinstance(other, DeepPoly_f):
            assert self.inputs == other.inputs
            return DeepPoly_f(self.x_l_coef + other.x_l_coef, self.x_u_coef + other.x_u_coef, self.x_l_bias + other.x_l_bias, self.x_u_bias + other.x_u_bias, self.inputs)
        else:
            assert False, 'Unknown type of other object'

    def __truediv__(self, other: Union[Tensor, float, int]) -> "DeepPoly_f":
        if isinstance(other, torch.Tensor):
            assert (other!=0).all()
            x_l_coef = torch.where(other >= 0, self.x_l_coef / other, self.x_u_coef / other)
            x_u_coef = torch.where(other >= 0, self.x_u_coef / other, self.x_l_coef / other)
            x_l_bias = torch.where(other >= 0, self.x_l_bias / other, self.x_u_bias / other)
            x_u_bias = torch.where(other >= 0, self.x_u_bias / other, self.x_l_bias / other)
        elif isinstance(other, float) or isinstance(other, int):
            assert other!=0
            x_l_coef = self.x_l_coef / other if other >= 0 else self.x_u_coef / other
            x_u_coef = self.x_u_coef / other if other >= 0 else self.x_l_coef / other
            x_l_bias = self.x_l_bias / other if other >= 0 else self.x_u_bias / other
            x_u_bias = self.x_u_bias / other if other >= 0 else self.x_l_bias / other
        else:
            assert False, 'Unknown type of other object'
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def __rmul__(self, other: Union[Tensor, float, int]) -> "DeepPoly_f":
        return self.__mul__(other)

    def __mul__(self, other: Union[Tensor, float, int]) -> "DeepPoly_f":
        if isinstance(other, torch.Tensor):
            x_l_coef = torch.where(other >= 0, self.x_l_coef * other, self.x_u_coef * other)
            x_u_coef = torch.where(other >= 0, self.x_u_coef * other, self.x_l_coef * other)
            x_l_bias = torch.where(other >= 0, self.x_l_bias * other, self.x_u_bias * other)
            x_u_bias = torch.where(other >= 0, self.x_u_bias * other, self.x_l_bias * other)
            return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)
        elif isinstance(other, int) or isinstance(other, float):
            x_l_coef = self.x_l_coef * other if other >= 0 else self.x_u_coef * other
            x_u_coef = self.x_u_coef * other if other >= 0 else self.x_l_coef * other
            x_l_bias = self.x_l_bias * other if other >= 0 else self.x_u_bias * other
            x_u_bias = self.x_u_bias * other if other >= 0 else self.x_l_bias * other
            return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)
        else:
            assert False, 'Unknown type of other object'

    def __getitem__(self, indices) -> "DeepPoly_f":
        if not isinstance(indices, tuple):
            indices = tuple([indices])
        return DeepPoly_f(self.x_l_coef[(slice(None,None,None),*indices)], self.x_u_coef[(slice(None,None,None),*indices)],
                          self.x_l_bias[indices], self.x_u_bias[indices], self.inputs)

    def clone(self) -> "DeepPoly_f":
        return DeepPoly_f(self.x_l_coef.clone(), self.x_u_coef.clone(),
                          self.x_l_bias.clone(), self.x_u_bias.clone(), self.inputs)

    def detach(self) -> "DeepPoly_f":
        return DeepPoly_f(self.x_l_coef.detach(), self.x_u_coef.detach(),
                          self.x_l_bias.detach(), self.x_u_bias.detach(), self.inputs)

    def max_center(self) -> Tensor:
        return self.x_u_bias.max(dim=1)[0].unsqueeze(1)

    def avg_pool2d(self, kernel_size: int, stride:int) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        x_l_coef = F.avg_pool2d(self.x_l_coef.view(-1,*self.x_l_coef.shape[2:]), kernel_size, stride)
        x_l_coef = x_l_coef.view(*n_in_dims,*x_l_coef.shape[1:])
        x_u_coef = F.avg_pool2d(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), kernel_size, stride)
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.avg_pool2d(self.x_l_bias, kernel_size, stride)
        x_u_bias = F.avg_pool2d(self.x_u_bias, kernel_size, stride)
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def global_avg_pool2d(self) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        x_l_coef = F.adaptive_avg_pool2d(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]),1)
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.adaptive_avg_pool2d(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]),1)
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.adaptive_avg_pool2d(self.x_l_bias,1)
        x_u_bias = F.adaptive_avg_pool2d(self.x_u_bias,1)
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def conv2d(self, weight:Tensor, bias:Tensor, stride:int, padding:int, dilation:int, groups:int) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(weight)

        x_l_coef = F.conv2d(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos, None, stride, padding, dilation, groups) \
                   + F.conv2d(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg, None, stride, padding, dilation, groups)
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.conv2d(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos, None, stride, padding, dilation, groups) \
                   + F.conv2d(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg, None, stride, padding, dilation, groups)
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.conv2d(self.x_l_bias, weight_pos, bias, stride, padding, dilation, groups) \
                   + F.conv2d(self.x_u_bias, weight_neg, None, stride, padding, dilation, groups)
        x_u_bias = F.conv2d(self.x_u_bias, weight_pos, bias, stride, padding, dilation, groups) \
                   + F.conv2d(self.x_l_bias, weight_neg, None, stride, padding, dilation, groups)
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def linear(self, weight:Tensor, bias:Union[Tensor,None]=None, C:Union[Tensor,None]=None) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(weight)

        x_l_coef = F.linear(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos, None) \
                   + F.linear(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg, None)
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.linear(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos, None) \
                   + F.linear(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg, None)
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.linear(self.x_l_bias, weight_pos, bias) \
                   + F.linear(self.x_u_bias, weight_neg, None)
        x_u_bias = F.linear(self.x_u_bias, weight_pos, bias) \
                   + F.linear(self.x_l_bias, weight_neg, None)
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def matmul(self, other: Tensor) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(other)

        x_l_coef = torch.matmul(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos) \
                   + torch.matmul(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg)
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = torch.matmul(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), weight_pos) \
                   + torch.matmul(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), weight_neg)
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = torch.matmul(self.x_l_bias, weight_pos) \
                   + torch.matmul(self.x_u_bias, weight_neg)
        x_u_bias = torch.matmul(self.x_u_bias, weight_pos) \
                   + torch.matmul(self.x_l_bias, weight_neg)
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def rev_matmul(self, other: Tensor) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        weight_neg, weight_pos = get_neg_pos_comp(other)

        x_l_coef = torch.matmul(weight_pos, self.x_l_coef.view(-1, *self.x_l_coef.shape[2:])) \
                   + torch.matmul(weight_neg, self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]))
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = torch.matmul(weight_pos, self.x_u_coef.view(-1, *self.x_l_coef.shape[2:])) \
                   + torch.matmul(weight_neg, self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]))
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = torch.matmul(weight_pos, self.x_l_bias) \
                   + torch.matmul(weight_neg, self.x_u_bias)
        x_u_bias = torch.matmul(weight_pos, self.x_u_bias) \
                   + torch.matmul(weight_neg, self.x_l_bias)
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def batch_norm(self, bn) -> "DeepPoly_f":
        view_dim_list = [1, -1] + (self.x_l_bias.dim()-2)*[1]
        self_stat_dim_list = [0, 2, 3] if self.x_l_bias.dim()==4 else [0]
        if bn.training and (bn.current_var is None or bn.current_mean is None):
            if bn.running_mean is not None and bn.running_var is not None:
                bn.current_mean = bn.running_mean
                bn.current_var = bn.running_var
            else:
                bn.current_mean = (0.5*(self.x_l_bias + self.x_u_bias)).mean(dim=self_stat_dim_list).detach()
                bn.current_var = (0.5*(self.x_l_bias + self.x_u_bias)).var(unbiased=False, dim=self_stat_dim_list).detach()

        c = (bn.weight / torch.sqrt(bn.current_var + bn.eps))
        b = (-bn.current_mean*c + bn.bias)

        out_dp = self*c.view(*view_dim_list) + b.view(*view_dim_list)
        return out_dp

    def relu(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None, init_lambda:Optional[bool]=False) -> Tuple["DeepPoly_f",Union[Tensor,None]]:
        lb, ub = self.concretize()
        assert (ub-lb>=0).all()

        dtype = lb.dtype
        D = 1e-8

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        is_cross = (lb < 0) & (ub > 0)

        lambda_u = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).to(dtype))
        lambda_l = torch.where(ub < -lb, torch.zeros_like(lb), torch.ones_like(lb))
        lambda_l = torch.where(is_cross, lambda_l, (lb >= 0).to(dtype))

        # lambda_l = torch.where(is_cross, lambda_u, (lb >= 0).to(dtype))

        if deepz_lambda is not None:
            if ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()) and not init_lambda:
                lambda_l = deepz_lambda
            else:
                deepz_lambda.data = lambda_l.data.detach().requires_grad_(True)

        mu_l = torch.zeros_like(lb)
        mu_u = torch.where(is_cross, -lb * lambda_u, torch.zeros_like(lb))  # height of upper bound intersection with y axis

        x_l_bias = mu_l + lambda_l * self.x_l_bias
        x_u_bias = mu_u + lambda_u * self.x_u_bias
        lambda_l, lambda_u = lambda_l.unsqueeze(0), lambda_u.unsqueeze(0)
        x_l_coef = self.x_l_coef * lambda_l
        x_u_coef = self.x_u_coef * lambda_u

        DP_out = DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

        assert (DP_out.concretize()[1] - DP_out.concretize()[0] >= 0).all()
        # return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs), deepz_lambda
        return DP_out, deepz_lambda

    def log(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None) -> Tuple["HybridZonotope",Union[Tensor,None]]:
        lb, ub = self.concretize()
        D = 1e-15

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        assert (lb >= 0).all()

        is_tight = (ub == lb)

        lambda_l = (ub.log() - (lb+D).log()) / (ub - lb + D)
        if deepz_lambda is not None:
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = ((lambda_l - (1/ub)) / (1/(lb+D)-1/ub)).detach().requires_grad_(True)
            lambda_u = deepz_lambda * (1 / (lb+D)) + (1 - deepz_lambda) * (1 / ub)
        else:
            lambda_u = lambda_l

        mu_l = ub.log() - lambda_l * ub
        mu_u = -lambda_u.log() - 1

        lambda_l = torch.where(is_tight, torch.zeros_like(lambda_l), lambda_l)
        lambda_u = torch.where(is_tight, torch.zeros_like(lambda_l), lambda_u)
        mu_l = torch.where(is_tight, ub.log(), mu_l)
        mu_u = torch.where(is_tight, ub.log(), mu_u)

        x_l_bias = mu_l + lambda_l * self.x_l_bias
        x_u_bias = mu_u + lambda_u * self.x_u_bias
        lambda_l, lambda_u = lambda_l.unsqueeze(0), lambda_u.unsqueeze(0)
        x_l_coef = self.x_l_coef * lambda_l
        x_u_coef = self.x_u_coef * lambda_u
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs), deepz_lambda

    def exp(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None) -> Tuple["HybridZonotope",Union[Tensor,None]]:
        lb, ub = self.concretize()
        D = 1e-15

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        is_tight = (ub == lb)

        lambda_u = (ub.exp() - lb.exp()) / (ub - lb + D)
        if deepz_lambda is not None:
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = ((lambda_u -lb.exp()) / (ub.exp()-lb.exp()+D)).detach().requires_grad_(True)
            lambda_l = deepz_lambda * torch.min(ub.exp(), (lb + 1 - 0.05).exp()) + (1 - deepz_lambda) * lb.exp()
        else:
            lambda_l = lambda_u
        lambda_l = torch.min(lambda_l, (lb + 1 - 0.01).exp())

        mu_l = lambda_l - lambda_l.log() * lambda_l
        mu_u = lb.exp() - lb * lambda_u

        lambda_l = torch.where(is_tight, torch.zeros_like(lambda_l), lambda_l)
        lambda_u = torch.where(is_tight, torch.zeros_like(lambda_l), lambda_u)
        mu_l = torch.where(is_tight, lb.exp(), mu_l)
        mu_u = torch.where(is_tight, lb.exp(), mu_u)

        x_l_bias = mu_l + lambda_l * self.x_l_bias
        x_u_bias = mu_u + lambda_u * self.x_u_bias
        lambda_l, lambda_u = lambda_l.unsqueeze(0), lambda_u.unsqueeze(0)
        x_l_coef = self.x_l_coef * lambda_l
        x_u_coef = self.x_u_coef * lambda_u
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs), deepz_lambda

    def upsample(self, size:int, mode:str, align_corners:bool, consolidate_errors:bool=True) -> "DeepPoly_f":
        n_in_dims = self.x_l_coef.shape[0:2]

        x_l_coef = F.interpolate(self.x_l_coef.view(-1, *self.x_l_coef.shape[2:]), size=size, mode=mode, align_corners=align_corners)
        x_l_coef = x_l_coef.view(*n_in_dims, *x_l_coef.shape[1:])
        x_u_coef = F.interpolate(self.x_u_coef.view(-1, *self.x_l_coef.shape[2:]), size=size, mode=mode, align_corners=align_corners)
        x_u_coef = x_u_coef.view(*n_in_dims, *x_u_coef.shape[1:])

        x_l_bias = F.interpolate(self.x_l_bias, size=size, mode=mode, align_corners=align_corners)
        x_u_bias = F.interpolate(self.x_u_bias, size=size, mode=mode, align_corners=align_corners)
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def unsqueeze(self, dim):
        return DeepPoly_f(self.x_l_coef.unsqueeze(dim+1), self.x_u_coef.unsqueeze(dim+1), self.x_l_bias.unsqueeze(dim),
                          self.x_u_bias.unsqueeze(dim), self.inputs)

    def squeeze(self, dim):
        return DeepPoly_f(self.x_l_coef.squeeze(dim+1), self.x_u_coef.squeeze(dim+1), self.x_l_bias.squeeze(dim),
                          self.x_u_bias.squeeze(dim), self.inputs)

    def add(self, other:"DeepPoly_f") -> "DeepPoly_f":
        x_l_coef = self.x_l_coef + other.x_l_coef
        x_u_coef = self.x_u_coef + other.x_u_coef
        x_l_bias = self.x_l_bias + other.x_l_bias
        x_u_bias = self.x_u_bias + other.x_u_bias
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def sum(self, dim: int) -> "DeepPoly_f":
        x_l_coef = self.x_l_coef.sum(dim+1).unsqueeze(dim+1)
        x_u_coef = self.x_u_coef.sum(dim+1).unsqueeze(dim+1)
        x_l_bias = self.x_l_bias.sum(dim).unsqueeze(dim)
        x_u_bias = self.x_u_bias.sum(dim).unsqueeze(dim)
        return DeepPoly_f(x_l_coef, x_u_coef, x_l_bias, x_u_bias, self.inputs)

    def concretize(self) -> Tuple[Tensor,Tensor]:
        input_lb, input_ub = self.inputs.concretize()
        input_shape = input_lb.shape
        input_lb = input_lb.flatten(start_dim=1).transpose(1, 0).view([-1,input_shape[0]]+(self.x_l_coef.dim()-2)*[1])
        input_ub = input_ub.flatten(start_dim=1).transpose(1, 0).view([-1,input_shape[0]]+(self.x_l_coef.dim()-2)*[1])
        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)

        lb = self.x_l_bias + (neg_x_l_coef * input_ub).sum(0) + (pos_x_l_coef * input_lb).sum(dim=0)
        ub = self.x_u_bias + (neg_x_u_coef * input_lb).sum(0) + (pos_x_u_coef * input_ub).sum(dim=0)
        return lb, ub

    def avg_width(self) -> Tensor:
        lb, ub = self.concretize()
        return (ub - lb).mean()

    def is_greater(self, i:int, j:int, threshold_min:Union[Tensor,float]=0) -> Tuple[Tensor,bool]:
        input_lb, input_ub = self.inputs.concretize()
        b_dim = input_lb.shape[0]
        dims = list(range(1, input_lb.dim()))
        dims.append(0)
        input_lb, input_ub = input_lb.permute(dims), input_ub.permute(dims) #dim0, ... dimn, batch_dim, 
        input_lb, input_ub = input_lb.view(-1, b_dim), input_ub.view(-1, b_dim) #dim, batch_dim
        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef[:,:,i]-self.x_u_coef[:,:,j])
        neg_x_l_coef, pos_x_l_coef = neg_x_l_coef.view(-1, b_dim), pos_x_l_coef.view(-1, b_dim)
        delta = self.x_l_bias[:,i] - self.x_u_bias[:,j] + (neg_x_l_coef * input_ub).sum(0) + (pos_x_l_coef * input_lb).sum(dim=0)
        return delta, delta > threshold_min

    def verify(self, targets: Tensor, threshold_min:Union[Tensor,float]=0, corr_only:bool=False) -> Tuple[Tensor,Tensor,Tensor]:
        n_class = self.x_l_bias.size()[1]
        device = self.x_l_bias.device
        verified = torch.zeros(targets.size(), dtype=torch.uint8).to(device)
        verified_corr = torch.zeros(targets.size(), dtype=torch.uint8).to(device)
        if n_class == 1:
            # assert len(targets) == 1
            verified_list = torch.cat([self.concretize()[1] < threshold_min, self.concretize()[0] >= threshold_min], dim=1)
            verified[:] = torch.any(verified_list, dim=1)
            verified_corr[:] = verified_list.gather(dim=1,index=targets.long().unsqueeze(dim=1)).squeeze(1)
            threshold = torch.cat(self.concretize(),1).gather(dim=1, index=(1-targets).long().unsqueeze(dim=1)).squeeze(1)
        else:
            threshold = np.inf * torch.ones(targets.size(), dtype=torch.float).to(device)
            for i in range(n_class):
                if corr_only and i not in targets:
                    continue
                isg = torch.ones(targets.size(), dtype=torch.uint8).to(device)
                print(isg.shape)
                margin = np.inf * torch.ones(targets.size(), dtype=torch.float).to(device)
                for j in range(n_class):
                    if i != j and isg.any():
                        margin_tmp, ok = self.is_greater(i, j, threshold_min)
                        margin = torch.min(margin, margin_tmp)
                        isg = isg & ok.byte()
                verified = verified | isg
                verified_corr = verified_corr | (targets.eq(i).byte() & isg)
                threshold = torch.where(targets.eq(i).byte(), margin, threshold)
        return verified, verified_corr, threshold

    def get_wc_logits(self, targets:Tensor, use_margins:bool=False)->Tensor:
        n_class = self.size(-1)
        device = self.head.device

        if use_margins:
            def get_c_mat(n_class, target):
                return torch.eye(n_class, dtype=torch.float32)[target].unsqueeze(dim=0) \
                       - torch.eye(n_class, dtype=torch.float32)
            if n_class > 1:
                c = torch.stack([get_c_mat(n_class,x) for x in targets], dim=0)
                self = -(self.unsqueeze(dim=1) * c.to(device)).sum(dim=2, reduce_dim=True)
        batch_size = targets.size()[0]
        lb, ub = self.concretize()
        if n_class == 1:
            wc_logits = torch.cat([ub, lb],dim=1)
            wc_logits = wc_logits.gather(dim=1, index=targets.long().unsqueeze(1))
        else:
            wc_logits = ub.clone()
            wc_logits[np.arange(batch_size), targets] = lb[np.arange(batch_size), targets]
        return wc_logits

    def ce_loss(self, targets:Tensor) -> Tensor:
        wc_logits = self.get_wc_logits(targets)
        if wc_logits.size(1) == 1:
            return F.binary_cross_entropy_with_logits(wc_logits.squeeze(1), targets.float(), reduction="none")
        else:
            return F.cross_entropy(wc_logits, targets.long(), reduction="none")
