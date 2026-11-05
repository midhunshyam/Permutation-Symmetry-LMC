"""Train one pair (or K-tuple) of networks and measure everything reported about it.

Record fields are described in results/README.md.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from itertools import combinations

import torch

from lmc.mnist import metrics, training
from lmc.mnist.models import build_model


@dataclass(frozen=True)
class Protocol:
    hidden: int
    n_hidden_layers: int
    epochs: int
    n_seeds: int
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.03
    shuffle_seed: int = 42

    def training_config(self) -> dict:
        """The settings that determine training (everything except n_seeds)."""
        return {k: v for k, v in asdict(self).items() if k != "n_seeds"}


PROTOCOLS = {
    "smoke": Protocol(hidden=128, n_hidden_layers=1, epochs=2, n_seeds=2),     # pipeline check
    "main": Protocol(hidden=256, n_hidden_layers=2, epochs=6, n_seeds=8),      # every reported number
    "lim_full": Protocol(hidden=512, n_hidden_layers=3, epochs=25, n_seeds=8),  # Lim et al.'s size; not run
}

# method -> (architecture, training procedure)
METHODS = {
    "standard":       ("mlp", "independent"),
    "w_asym":         ("w_asym", "independent"),
    "sigma_asym":     ("sigma_asym", "independent"),
    "scale_quotient": ("scale_quotient", "independent"),
    "shared_init":    ("mlp", "shared_init"),
    "tna_pfn":        ("mlp", "tna_pfn"),
    "joint_midpoint": ("mlp", "joint_midpoint"),
    "frozen_anchor":  ("mlp", "frozen_anchor"),
}
USES_LAMBDA = {"joint_midpoint", "frozen_anchor"}

# Permuting hidden units is an exact symmetry only for these architectures.
# W-Asym masks and sigma-Asym gates are tied to unit positions, so they are not aligned.
PERMUTATION_SYMMETRIC = {"mlp", "scale_quotient"}


def network_seed(seed: int, i: int) -> int:
    """Initialisation seed of network i (0 = A, 1 = B, ...) in pair or clique number `seed`."""
    return 100 * seed + i + 1


def _train(procedure, a, b, data, cfg, gen, device, lam):
    if procedure == "independent":
        training.train_independent(a, data, cfg, gen, device)
        training.train_independent(b, data, cfg, gen, device)
    elif procedure == "shared_init":
        training.train_shared_init(a, b, data, cfg, gen, device)
    elif procedure == "tna_pfn":
        training.train_tna_pfn(a, b, data, cfg, gen, device)
    elif procedure == "joint_midpoint":
        training.train_joint_midpoint(a, b, data, cfg, gen, device, lam=lam)
    elif procedure == "frozen_anchor":
        training.train_frozen_anchor(a, b, data, cfg, gen, device, lam=lam)
    else:
        raise ValueError(f"unknown training procedure: {procedure}")


def run_pair(method, seed, cfg: Protocol, data, device, lam=2.0) -> dict:
    """Train networks A and B with `method` and measure them, their midpoint and the path between them."""
    arch, procedure = METHODS[method]
    a = build_model(arch, cfg.hidden, cfg.n_hidden_layers, seed=network_seed(seed, 0)).to(device)
    b = build_model(arch, cfg.hidden, cfg.n_hidden_layers, seed=network_seed(seed, 1)).to(device)
    gen = torch.Generator().manual_seed(cfg.shuffle_seed + seed)

    t0 = time.time()
    _train(procedure, a, b, data, cfg, gen, device, lam)
    train_seconds = time.time() - t0

    x, y = data.x_test.to(device), data.y_test.cpu()
    rec = {
        "method": method, "seed": seed, "lam": lam if procedure in USES_LAMBDA else None,
        "config": cfg.training_config(), "train_seconds": train_seconds,
        "a": metrics.evaluate(a, x, y), "b": metrics.evaluate(b, x, y),
        "midpoint": metrics.evaluate(a, x, y, metrics.interpolated_params(a, b, 0.5)),
        "curve": metrics.barrier_curve(a, b, x, y),
        "row_cosine": metrics.row_cosine(a, b),
        "param_cosine": metrics.param_cosine(a, b),
        "param_distance": metrics.param_distance(a, b),
        "logit_cka": metrics.logit_cka(a, b, x),
        "aligned": None,
    }
    if arch in PERMUTATION_SYMMETRIC:
        b_aligned, perms = metrics.align_hidden_units(a, b)
        rec["aligned"] = {
            "curve": metrics.barrier_curve(a, b_aligned, x, y),
            "row_cosine": metrics.row_cosine(a, b_aligned),
            "n_units_moved": metrics.count_moved_units(perms),
            "n_hidden_units": sum(len(p) for p in perms),
        }
    return rec


def run_clique(K, seed, cfg: Protocol, data, device, lam=2.0) -> dict:
    """K MLPs trained as a midpoint clique; barrier of every pair and accuracy of the K-way average.

    Networks 0 and 1 have the same initialisations and batch order as pair `seed`
    of the joint_midpoint method, so K = 2 reproduces it.
    """
    nets = [build_model("mlp", cfg.hidden, cfg.n_hidden_layers, seed=network_seed(seed, i)).to(device)
            for i in range(K)]
    gen = torch.Generator().manual_seed(cfg.shuffle_seed + seed)
    t0 = time.time()
    training.train_midpoint_clique(nets, data, cfg, gen, device, lam=lam)
    train_seconds = time.time() - t0

    x, y = data.x_test.to(device), data.y_test.cpu()
    average = {k: v.detach() for k, v in training.mean_params(nets).items()}
    return {
        "K": K, "seed": seed, "lam": lam, "config": cfg.training_config(), "train_seconds": train_seconds,
        "endpoint_accs": [metrics.evaluate(n, x, y)["acc"] for n in nets],
        "average_acc": metrics.evaluate(nets[0], x, y, average)["acc"],
        "pair_barriers": [metrics.barrier_curve(nets[i], nets[j], x, y)["midpoint_barrier"]
                          for i, j in combinations(range(K), 2)],
    }
