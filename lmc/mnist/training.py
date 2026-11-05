"""Training procedures for pairs (and K-tuples) of networks.

All procedures use AdamW (lr 1e-3, weight decay 0.03) and cross-entropy, as in
Lim et al.'s MLP-MNIST script. `gen` is a torch.Generator that sets the batch
order. Procedures that train networks one after another draw successive epochs
from it, so the second network sees a different batch order from the first.
"""

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F
from torch.func import functional_call

from lmc import stable_seed
from lmc.mnist.data import batches


def mean_params(nets) -> dict[str, torch.Tensor]:
    """Parameter-wise mean of `nets` (same architecture), keyed by parameter name."""
    params = [dict(n.named_parameters()) for n in nets]
    return {k: sum(p[k] for p in params) / len(params) for k in params[0]}


def _midpoint_logits(nets, x):
    """Logits of the network whose parameters are the mean of `nets`' parameters.

    Gradients flow back into every network whose parameters require grad. Buffers
    (frozen masks, fills, gates) are identical across nets and taken from the first.
    """
    params = mean_params(nets) | dict(nets[0].named_buffers())
    return functional_call(nets[0], params, (x,))


def _fit(nets, loss_fn, data, cfg, gen, device):
    """Train the parameters of `nets` jointly with one AdamW optimiser for cfg.epochs.
    `loss_fn(x, y)` returns the scalar loss for one mini-batch."""
    opt = torch.optim.AdamW([p for n in nets for p in n.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    for n in nets:
        n.train()
    for _ in range(cfg.epochs):
        for x, y in batches(data.x_train, data.y_train, cfg.batch_size, gen):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(x, y).backward()
            opt.step()


def train_independent(model, data, cfg, gen, device):
    """Ordinary training of one network."""
    _fit([model], lambda x, y: F.cross_entropy(model(x), y), data, cfg, gen, device)


def train_shared_init(a, b, data, cfg, gen, device):
    """Control for TNA-PFN: B starts as an exact copy of A, then both train
    independently with different batch orders."""
    b.load_state_dict(a.state_dict())
    train_independent(a, data, cfg, gen, device)
    train_independent(b, data, cfg, gen, device)


def train_tna_pfn(a, b, data, cfg, gen, device, mask_ratio=0.4):
    """TNA-PFN (Li et al., 2024): shared initialisation, plus a fixed subset of
    weights that receives no gradient.

    In every weight matrix a fraction `mask_ratio` of entries, chosen by a draw
    seeded with the parameter's name (so identical for A and B, and for every
    seed), has its gradient zeroed. The choice is stored on the layer as the buffer
    `grad_mask` (1 = receives gradient). A and B then train independently with
    different batch orders, so this differs from `train_shared_init` only by the
    gradient mask.

    AdamW's decoupled weight decay still shrinks the masked entries, by the same
    factor in both networks, so they stay equal between A and B.
    """
    b.load_state_dict(a.state_dict())
    for name, p in a.named_parameters():
        if p.dim() < 2:
            continue
        module_name, _, param_name = name.rpartition(".")
        g = torch.Generator().manual_seed(stable_seed(name))
        grad_mask = (torch.rand(p.shape, generator=g) > mask_ratio).float().to(p.device)
        for net in (a, b):
            layer = net.get_submodule(module_name)
            layer.register_buffer("grad_mask", grad_mask)
            getattr(layer, param_name).register_hook(lambda grad, m=grad_mask: grad * m)
    train_independent(a, data, cfg, gen, device)
    train_independent(b, data, cfg, gen, device)


def train_joint_midpoint(a, b, data, cfg, gen, device, lam=2.0):
    """A and B trained together, with one optimiser and the same batches, on

        L = CE(f_A) + CE(f_B) + lam * CE(f_mid),  theta_mid = (theta_A + theta_B) / 2.
    """
    def loss(x, y):
        total = F.cross_entropy(a(x), y) + F.cross_entropy(b(x), y)
        if lam != 0:
            total = total + lam * F.cross_entropy(_midpoint_logits([a, b], x), y)
        return total
    _fit([a, b], loss, data, cfg, gen, device)


def train_frozen_anchor(a, b, data, cfg, gen, device, lam=2.0):
    """A is trained normally and frozen; B is then trained on

        L = CE(f_B) + lam * CE(f_mid).

    Each network gets cfg.epochs of training, one after the other.
    """
    train_independent(a, data, cfg, gen, device)
    for p in a.parameters():
        p.requires_grad_(False)
    _fit([b], lambda x, y: F.cross_entropy(b(x), y) + lam * F.cross_entropy(_midpoint_logits([a, b], x), y),
         data, cfg, gen, device)
    for p in a.parameters():
        p.requires_grad_(True)


def train_midpoint_clique(nets, data, cfg, gen, device, lam=2.0):
    """K networks trained together with a midpoint term for every pair:

        L = sum_i CE(f_i) + lam * sum_{i<j} CE(f_mid(i, j)).

    The midpoint terms are summed, not averaged, so the coupling grows with the
    number of pairs, K(K-1)/2. For K = 2 this is `train_joint_midpoint`.
    """
    pairs = list(combinations(range(len(nets)), 2))

    def loss(x, y):
        total = sum(F.cross_entropy(n(x), y) for n in nets)
        return total + lam * sum(F.cross_entropy(_midpoint_logits([nets[i], nets[j]], x), y) for i, j in pairs)
    _fit(nets, loss, data, cfg, gen, device)
