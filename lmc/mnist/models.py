"""MLP variants compared in the MNIST study.

Every model has the layout of Lim et al. (2024): each hidden layer is
Linear -> LayerNorm -> activation, and the output layer is a bare Linear.

  MLP              plain network, GELU activation
  WAsymMLP         part of each weight row frozen to fixed Gaussian values (W-Asym)
  SigmaAsymMLP     GELU replaced by the fixed random gate sigmoid(Cx) * x (sigma-Asym)
  ScaleQuotientMLP each weight row normalised and multiplied by a trainable gain

Each linear layer exposes `effective_weight`, the matrix the forward pass uses.
Metrics that compare weights use it, so that frozen entries, row norms and
gains are treated the same way for every architecture.

Initialisation: PlainLinear uses torch's default
(weights and biases uniform), while WAsymLinear and ScaleQuotientLinear start
their biases at zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmc import stable_seed


class PlainLinear(nn.Linear):
    @property
    def effective_weight(self) -> torch.Tensor:
        return self.weight


class WAsymLinear(nn.Module):
    """W_eff = M * W + (1 - M) * kappa * G.

    M has `n_frozen` zeros in each row, at positions drawn independently for each
    row; G ~ N(0, 1). M and G are buffers seeded only by the layer index, so A and B
    share them and they are the same for every seed. Frozen positions get zero
    gradient because the forward pass multiplies W by M.
    """

    def __init__(self, in_dim: int, out_dim: int, n_frozen: int, kappa: float, layer_index: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        g = torch.Generator().manual_seed(stable_seed(layer_index, 999))
        mask = torch.ones(out_dim, in_dim)
        for row in range(out_dim):
            mask[row, torch.randperm(in_dim, generator=g)[:n_frozen]] = 0.0
        self.register_buffer("mask", mask)

        g = torch.Generator().manual_seed(stable_seed(layer_index, 777))
        self.register_buffer("fill", kappa * torch.randn(out_dim, in_dim, generator=g))

    @property
    def effective_weight(self) -> torch.Tensor:
        return self.mask * self.weight + (1 - self.mask) * self.fill

    def forward(self, x):
        return F.linear(x, self.effective_weight, self.bias)


class ScaleQuotientLinear(nn.Module):
    """W_eff = g * W / ||W||_row: weight normalisation with a per-row gain g.

    g starts at 1, so effective rows start with unit norm (about 0.58 under
    torch's default Linear initialisation).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.gain = nn.Parameter(torch.ones(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def effective_weight(self) -> torch.Tensor:
        norm = self.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return self.weight / norm * self.gain.unsqueeze(1)

    def forward(self, x):
        return F.linear(x, self.effective_weight, self.bias)


class FiGLU(nn.Module):
    """The FiGLU gate of Lim et al. (2024): x -> sigmoid(C x) * x.

    C ~ N(0, 1), unscaled, as in Lim et al.'s MNIST code. It is a frozen buffer seeded only
    by the layer index, so A and B share it and it is the same for every seed.
    """

    def __init__(self, dim: int, layer_index: int):
        super().__init__()
        g = torch.Generator().manual_seed(stable_seed(layer_index, 555))
        self.register_buffer("C", torch.randn(dim, dim, generator=g))

    def forward(self, x):
        return torch.sigmoid(F.linear(x, self.C)) * x


class _MLP(nn.Module):
    def __init__(self, dims, make_linear, make_act):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        n_hidden_layers = len(dims) - 2
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(make_linear(i, d_in, d_out))
            if i < n_hidden_layers:
                self.norms.append(nn.LayerNorm(d_out))
                self.acts.append(make_act(i, d_out))

    def forward(self, x):
        for lin, norm, act in zip(self.layers[:-1], self.norms, self.acts):
            x = act(norm(lin(x)))
        return self.layers[-1](x)


def _dims(in_dim, hidden, n_classes, n_hidden_layers):
    assert n_hidden_layers >= 1
    return [in_dim] + [hidden] * n_hidden_layers + [n_classes]


class MLP(_MLP):
    def __init__(self, in_dim, hidden, n_classes, n_hidden_layers):
        super().__init__(_dims(in_dim, hidden, n_classes, n_hidden_layers),
                         lambda i, d_in, d_out: PlainLinear(d_in, d_out), lambda i, d: nn.GELU())


class ScaleQuotientMLP(_MLP):
    def __init__(self, in_dim, hidden, n_classes, n_hidden_layers):
        super().__init__(_dims(in_dim, hidden, n_classes, n_hidden_layers),
                         lambda i, d_in, d_out: ScaleQuotientLinear(d_in, d_out), lambda i, d: nn.GELU())


class SigmaAsymMLP(_MLP):
    def __init__(self, in_dim, hidden, n_classes, n_hidden_layers):
        super().__init__(_dims(in_dim, hidden, n_classes, n_hidden_layers),
                         lambda i, d_in, d_out: PlainLinear(d_in, d_out), lambda i, d: FiGLU(d, layer_index=i))


def w_asym_schedule(n_hidden_layers: int, hidden: int) -> list[tuple[int, float]]:
    """(n_frozen, kappa) for each linear layer, input layer first.

    Three hidden layers of width 512 is Lim et al.'s MLP-MNIST setting (their
    Table 10). Two hidden layers use an adaptation of it to that depth;
    other shapes use a generic fallback.
    """
    n_linear = n_hidden_layers + 1
    if n_hidden_layers == 3 and hidden >= 512:
        n_frozen, kappa = [64, 64, 64, 256], [1.0, 1.0, 0.5, 0.25]
    elif n_hidden_layers == 2:
        n_frozen, kappa = [64, 64, min(128, hidden - 1)], [1.0, 0.5, 0.25]
    else:
        n_frozen = [min(64, hidden - 1)] * n_linear
        n_frozen[-1] = min(hidden - 1, hidden // 2)
        kappa = [1.0] * n_linear
        kappa[-1] = 0.25
        kappa[-2] = 0.5
    return list(zip(n_frozen, kappa))


class WAsymMLP(_MLP):
    def __init__(self, in_dim, hidden, n_classes, n_hidden_layers):
        schedule = w_asym_schedule(n_hidden_layers, hidden)

        def make_linear(i, d_in, d_out):
            n_frozen, kappa = schedule[i]
            return WAsymLinear(d_in, d_out, n_frozen, kappa, layer_index=i)

        super().__init__(_dims(in_dim, hidden, n_classes, n_hidden_layers), make_linear, lambda i, d: nn.GELU())


ARCHITECTURES = {
    "mlp": MLP,
    "w_asym": WAsymMLP,
    "sigma_asym": SigmaAsymMLP,
    "scale_quotient": ScaleQuotientMLP,
}


def build_model(arch: str, hidden: int, n_hidden_layers: int, seed: int, in_dim=784, n_classes=10):
    """Seed torch's global generator with `seed`, then build the model (this sets its initial weights)."""
    torch.manual_seed(seed)
    return ARCHITECTURES[arch](in_dim, hidden, n_classes, n_hidden_layers)
