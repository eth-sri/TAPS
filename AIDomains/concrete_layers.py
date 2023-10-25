import torch
import torch.nn as nn

class Bias(nn.Module):
    def __init__(self, in_dim=None, bias=None):
        super().__init__()
        assert in_dim is not None or bias is not None
        in_dim = list(bias.shape) if in_dim is None else in_dim
        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]
        if bias is not None:
            self.bias = bias
        else:
            self.bias = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        return x + self.bias


class Scale(nn.Module):
    def __init__(self, in_dim=None, scale=None):
        super().__init__()
        assert in_dim is not None
        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]
        if scale is not None:
            self.scale = scale
        else:
            self.scale = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        return x * self.scale


class Normalization(nn.Module):
    def __init__(self, in_dim=None, mean=None, std=None):
        super().__init__()
        assert in_dim is not None
        self.mean = torch.nn.Parameter(torch.tensor(0.) if mean is None else torch.tensor(mean, dtype=torch.float), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor(0.) if std is None else torch.tensor(std, dtype=torch.float), requires_grad=False)

        if len(in_dim) in [3, 4]:
            self.mean.data = self.mean.data.view(1, -1, 1, 1)
            self.std.data = self.std.data.view(1, -1, 1, 1)
        elif len(in_dim) in [1, 2]:
            self.mean.data = self.mean.data.view(1, -1)
            self.std.data = self.std.data.view(1, -1)
        else:
            assert False

        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]

    def forward(self, x):
        return (x - self.mean) / self.std


class DeNormalization(nn.Module):
    def __init__(self, in_dim=None, mean=None, std=None):
        super().__init__()
        assert in_dim is not None
        self.mean = torch.nn.Parameter(torch.tensor(0.) if mean is None else torch.tensor(mean, dtype=torch.float), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor(0.) if std is None else torch.tensor(std, dtype=torch.float), requires_grad=False)

        if len(in_dim) in [3, 4]:
            self.mean.data = self.mean.data.view(1, -1, 1, 1)
            self.std.data = self.std.data.view(1, -1, 1, 1)
        elif len(in_dim) in [1, 2]:
            self.mean.data = self.mean.data.view(1, -1)
            self.std.data = self.std.data.view(1, -1)
        else:
            assert False

        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]

    def forward(self, x):
        return x * self.std + self.mean