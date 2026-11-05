"""Synthetic Gaussian-mixture classification tasks.

Both functions return (X_train, y_train, X_test, y_test) with balanced classes.
"""

from __future__ import annotations

import torch


def small_mixture(seed: int = 42, n_train: int = 600, n_test: int = 300, noise_std: float = 0.55):
    """3 classes in R^4. The centres differ mainly along inputs 0 and 1, slightly along 2 and 3."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([
        [2.0, 0.0, 0.4, -0.3],
        [-1.0, 1.7, -0.5, 0.5],
        [-1.0, -1.7, 0.3, 0.7],
    ])
    return (*_draw(centers, noise_std, n_train, g), *_draw(centers, noise_std, n_test, g))


def mixture(in_dim: int, n_classes: int = 10, seed: int = 42, n_train: int = 2000, n_test: int = 1000,
            radius: float = 3.0, noise_std: float = 0.7):
    """`n_classes` classes in R^in_dim, with centres drawn uniformly on a sphere of the given radius."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, in_dim, generator=g)
    centers = centers / centers.norm(dim=1, keepdim=True) * radius
    return (*_draw(centers, noise_std, n_train, g), *_draw(centers, noise_std, n_test, g))


def _draw(centers, noise_std, n_total, g):
    n_classes, dim = centers.shape
    per_class = n_total // n_classes
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(torch.randn(per_class, dim, generator=g) * noise_std + centers[c])
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)
