"""Evaluation, interpolation barriers, permutation alignment and similarity measures."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call


# --------------------------------------------------------------------------- evaluation

@torch.no_grad()
def logits_of(model, x, params=None, chunk=5000):
    """Logits on x, optionally with `params` substituted for the model's parameters."""
    model.eval()
    out = []
    for start in range(0, len(x), chunk):
        xb = x[start:start + chunk]
        out.append(model(xb) if params is None else functional_call(model, params, (xb,)))
    return torch.cat(out)


def expected_calibration_error(probs, y, n_bins=15):
    """Equal-width-bin ECE (Guo et al., 2017). Bins are [lo, hi); the last bin includes 1.0."""
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    for i in range(n_bins):
        upper = conf <= edges[i + 1] if i == n_bins - 1 else conf < edges[i + 1]
        in_bin = (conf >= edges[i]) & upper
        if in_bin.any():
            ece += in_bin.float().mean() * (correct[in_bin].mean() - conf[in_bin].mean()).abs()
    return float(ece)


def evaluate(model, x, y, params=None):
    """Test cross-entropy, accuracy, 15-bin ECE and Brier score (squared error summed
    over the 10 classes, so between 0 and 2)."""
    logits = logits_of(model, x, params).cpu()
    probs = logits.softmax(1)
    return {
        "loss": float(F.cross_entropy(logits, y)),
        "acc": float((logits.argmax(1) == y).float().mean()),
        "ece": expected_calibration_error(probs, y),
        "brier": float(((probs - F.one_hot(y, probs.shape[1]).float()) ** 2).sum(1).mean()),
    }


# --------------------------------------------------------------------------- interpolation

def interpolated_params(a, b, alpha):
    """(1 - alpha) * theta_A + alpha * theta_B for parameters; buffers taken from A."""
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    out = {k: ((1 - alpha) * pa[k] + alpha * pb[k]).detach() for k in pa}
    out.update(dict(a.named_buffers()))
    return out


def barrier_curve(a, b, x, y, n_points=25):
    """Test loss and accuracy at `n_points` evenly spaced alpha in [0, 1] on the line from A to B.

    midpoint_barrier: L(0.5) - (L(0) + L(1)) / 2
    max_barrier:      max over alpha of L(alpha) - [(1 - alpha) L(0) + alpha L(1)]
    """
    assert n_points % 2 == 1, "an odd number of points puts one at alpha = 0.5"
    alphas = np.linspace(0, 1, n_points)
    losses, accs = [], []
    for alpha in alphas:
        r = evaluate(a, x, y, interpolated_params(a, b, float(alpha)))
        losses.append(r["loss"])
        accs.append(r["acc"])
    losses, accs = np.array(losses), np.array(accs)
    chord = (1 - alphas) * losses[0] + alphas * losses[-1]
    return {
        "alphas": alphas.tolist(), "losses": losses.tolist(), "accs": accs.tolist(),
        "midpoint_barrier": float(losses[n_points // 2] - 0.5 * (losses[0] + losses[-1])),
        "max_barrier": float((losses - chord).max()),
    }


# --------------------------------------------------------------------------- alignment

@torch.no_grad()
def align_hidden_units(a, b):
    """Permute B's hidden units to match A's. Returns (aligned copy of B, permutations).

    For each hidden layer, first to last: rows of the effective weight matrices
    (each unit's incoming weights) are compared by cosine similarity and matched
    with the Hungarian algorithm. The permutation is applied to B's rows, bias,
    gain (if any) and LayerNorm parameters, and to the columns of the next layer,
    before the next layer is matched. The output layer is not permuted.

    This is a simplified form of the weight matching of Ainsworth et al. (2023),
    which also uses each unit's outgoing weights and repeats the sweep over layers
    until no permutation changes.

    Only valid for architectures where permuting hidden units leaves the function
    unchanged (MLP, ScaleQuotientMLP).
    """
    b = copy.deepcopy(b)
    perms = []
    for ell in range(len(a.layers) - 1):
        wa = F.normalize(a.layers[ell].effective_weight, dim=1)
        wb = F.normalize(b.layers[ell].effective_weight, dim=1)
        _, cols = linear_sum_assignment(-(wa @ wb.T).cpu().numpy())
        perm = torch.as_tensor(cols, device=wa.device)
        perms.append(perm.cpu())
        layer, nxt, norm = b.layers[ell], b.layers[ell + 1], b.norms[ell]
        for t in (layer.weight, layer.bias, getattr(layer, "gain", None), norm.weight, norm.bias):
            if t is not None:
                t.copy_(t[perm])
        nxt.weight.copy_(nxt.weight[:, perm])
    return b, perms


def count_moved_units(perms) -> int:
    """Number of hidden units the permutations move, summed over layers."""
    return int(sum((p != torch.arange(len(p))).sum() for p in perms))


# --------------------------------------------------------------------------- similarity
#
# "Trainable entries" are the weight entries that receive gradient. Left out are
# W-Asym's frozen positions (`mask` == 0), where the effective weight holds the fill
# shared by A and B and the raw parameter is never used, and TNA-PFN's
# gradient-masked entries (`grad_mask` == 0), which stay equal in A and B.

def _trainable_entries(module):
    """Boolean mask of the module's weight entries that are trained, or None if all are."""
    for name in ("mask", "grad_mask"):
        m = getattr(module, name, None)
        if m is not None:
            return m.bool()
    return None


@torch.no_grad()
def row_cosine(a, b) -> float:
    """Mean cosine between corresponding rows of A's and B's hidden-layer effective
    weights, over trainable entries, without any permutation."""
    total, n = 0.0, 0
    for la, lb in zip(a.layers[:-1], b.layers[:-1]):
        wa, wb = la.effective_weight, lb.effective_weight
        keep = _trainable_entries(la)
        if keep is not None:
            wa, wb = wa * keep, wb * keep
        total += F.cosine_similarity(wa, wb, dim=1).sum().item()
        n += wa.shape[0]
    return total / n


def trainable_vector(model) -> torch.Tensor:
    """All trainable parameter entries of the model, flattened."""
    parts = []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            keep = _trainable_entries(module) if name == "weight" else None
            parts.append((p if keep is None else p[keep]).detach().flatten())
    return torch.cat(parts)


def param_cosine(a, b) -> float:
    """Cosine between the trainable parameter vectors of A and B."""
    va, vb = trainable_vector(a), trainable_vector(b)
    return float(va @ vb / (va.norm() * vb.norm()).clamp_min(1e-12))


def param_distance(a, b) -> float:
    """Euclidean distance between the trainable parameter vectors of A and B."""
    return float((trainable_vector(a) - trainable_vector(b)).norm())


def logit_cka(a, b, x) -> float:
    """Linear CKA between the two networks' logits on x (a function-space similarity)."""
    fa = logits_of(a, x).cpu().double()
    fb = logits_of(b, x).cpu().double()
    fa = fa - fa.mean(0)
    fb = fb - fb.mean(0)
    return float((fb.T @ fa).norm() ** 2 / ((fa.T @ fa).norm() * (fb.T @ fb).norm()))
