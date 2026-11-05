"""The masked one-hidden-layer MLP and the pair experiment used by every fill-study run.

The network is x -> W2 relu(W1 x), with no biases and no normalisation. Only W1
is masked: W1 = M * W + (1 - M) * F, with M (1 = trainable, 0 = frozen) and the
fill F held as buffers. Both networks of a pair use the same M and F (except in
the unshared-fill control, `run_pair(fill_b=...)`); only their initial weights
differ.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmc.fills.catalog import FILLS

N_ALPHA = 11  # points on the interpolation path; index 5 is alpha = 0.5

# Training settings. The 4-4-3 network is used by enumeration, sweep and hybrid;
# the larger networks by width, grid, grid_wide and controls.
SMALL_NET = dict(in_dim=4, hidden=4, n_classes=3, steps=200, lr=5e-3, n_seeds=5)
LARGER_NET = dict(n_classes=10, steps=100, lr=5e-3, n_seeds=10)
SMALL_PATTERNS = [(1, 1, 0, 1), (1, 2, 0, 1), (0, 1, 2, 3)]  # row i of W1 freezes input pattern[i]

# Seeds of the mask and fill draws. Design "fixed": one draw per cell, shared by all
# seed pairs. Design "resampled": seed pair s gets its own draw.
BASE_MASK_SEED, BASE_FILL_SEED = 99, 10_000
UNSHARED_FILL_SEED = 20_000  # network B's own fill in the unshared-fill control


def design_seeds(design: str, seed: int) -> tuple[int, int]:
    """(mask seed, fill seed) for seed pair `seed` under a design."""
    if design == "fixed":
        return BASE_MASK_SEED, BASE_FILL_SEED
    if design == "resampled":
        return BASE_MASK_SEED + 1 + seed, BASE_FILL_SEED + 1 + seed
    raise ValueError(f"unknown design: {design}")


def sweep_fill_seed(fill_name: str) -> int:
    """Fill seed for the 4-4-3 sweep: one draw per fill, shared by all masks and seed pairs.
    It depends on the position of the fill in FILLS, so FILL_FAMILIES must not be reordered."""
    return BASE_FILL_SEED + FILLS.index(fill_name)


class MaskedMLP(nn.Module):
    def __init__(self, mask: torch.Tensor, fill: torch.Tensor, n_classes: int):
        super().__init__()
        hidden, in_dim = mask.shape
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, n_classes, bias=False)
        self.register_buffer("mask", mask.float())
        self.register_buffer("fill", fill.float())

    @property
    def w1(self):
        return self.mask * self.fc1.weight + (1 - self.mask) * self.fill

    def forward(self, x):
        return self.fc2(F.relu(F.linear(x, self.w1)))


def random_mask(in_dim: int, hidden: int, n_frozen: int, seed: int) -> torch.Tensor:
    """(hidden, in_dim) mask with `n_frozen` frozen entries (zeros) in each row, at
    positions drawn independently for each row.

    With the same seed, masks are nested: a smaller n_frozen freezes a subset of the
    entries a larger one freezes, and a smaller `hidden` gives the first rows of a
    larger one.
    """
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(hidden, in_dim)
    for row in range(hidden):
        mask[row, torch.randperm(in_dim, generator=g)[:n_frozen]] = 0.0
    return mask


def drop_one_mask(pattern, in_dim: int) -> torch.Tensor:
    """Mask with one frozen entry per row: row i freezes input pattern[i]."""
    mask = torch.ones(len(pattern), in_dim)
    mask[torch.arange(len(pattern)), torch.as_tensor(pattern)] = 0.0
    return mask


def train(net, X, y, steps: int, lr: float):
    """Full-batch AdamW without weight decay."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    net.train()
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(net(X), y).backward()
        opt.step()
    net.eval()


@torch.no_grad()
def _loss_and_acc(net, params, X, y):
    logits = torch.func.functional_call(net, params, (X,))
    return F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item()


@torch.no_grad()
def _first_layer_stats(a, b):
    """frozen_energy_share  mean over rows of ||frozen part of the row||^2 / ||row||^2, A and B averaged
    w1_cos               mean cosine between corresponding rows of A's and B's W1, no permutation
    w1_cos_trainable     the same, over trainable entries only
    """
    def share(net):
        frozen = ((1 - net.mask) * net.fill).pow(2).sum(1)
        return (frozen / net.w1.pow(2).sum(1).clamp_min(1e-12)).mean().item()
    return {
        "frozen_energy_share": 0.5 * (share(a) + share(b)),
        "w1_cos": F.cosine_similarity(a.w1, b.w1, dim=1).mean().item(),
        "w1_cos_trainable": F.cosine_similarity(a.mask * a.fc1.weight, b.mask * b.fc1.weight, dim=1).mean().item(),
    }


def run_pair(mask, fill, data, n_classes, steps, lr, seed_a, seed_b, full_curve=True, fill_b=None) -> dict:
    """Train two networks from different initialisations and measure the barrier between them.

    data = (X_train, y_train, X_test, y_test); losses and accuracies are on the test set.
    midpoint_barrier = L(0.5) - (L(0) + L(1)) / 2. With full_curve=True the loss is
    evaluated at N_ALPHA points, which also gives
    max_barrier = max over alpha of L(alpha) - [(1 - alpha) L(0) + alpha L(1)].

    fill_b (unshared-fill control only): network B gets its own fill. The fills are
    then interpolated along with the weights, so the midpoint's frozen entries are
    (F_A + F_B) / 2, which for two independent Gaussian draws has half their variance.
    """
    X_train, y_train, X_test, y_test = data
    torch.manual_seed(seed_a)
    a = MaskedMLP(mask, fill, n_classes)
    train(a, X_train, y_train, steps, lr)
    torch.manual_seed(seed_b)
    b = MaskedMLP(mask, fill if fill_b is None else fill_b, n_classes)
    train(b, X_train, y_train, steps, lr)

    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    if fill_b is None:
        shared = dict(a.named_buffers())
    else:
        pa |= dict(a.named_buffers())
        pb |= dict(b.named_buffers())
        shared = {}
    alphas = torch.linspace(0, 1, N_ALPHA).tolist() if full_curve else [0.0, 0.5, 1.0]
    losses, accs = [], []
    for alpha in alphas:
        params = {k: (1 - alpha) * pa[k] + alpha * pb[k] for k in pa} | shared
        loss, acc = _loss_and_acc(a, params, X_test, y_test)
        losses.append(loss)
        accs.append(acc)
    mid = len(alphas) // 2
    rec = {
        "midpoint_barrier": losses[mid] - 0.5 * (losses[0] + losses[-1]),
        "loss_a": losses[0], "loss_b": losses[-1],
        "acc_a": accs[0], "acc_b": accs[-1], "acc_mid": accs[mid],
    }
    if full_curve:
        chord = [(1 - alpha) * losses[0] + alpha * losses[-1] for alpha in alphas]
        rec["max_barrier"] = max(loss - c for loss, c in zip(losses, chord))
        rec["losses"] = losses
    rec.update(_first_layer_stats(a, b))
    return rec
