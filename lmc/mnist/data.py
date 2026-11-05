"""MNIST held in memory as flat, normalised tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import random_split
from torchvision import datasets

MEAN, STD = 0.1307, 0.3081


@dataclass
class MNISTSplits:
    x_train: torch.Tensor  # (54000, 784)
    y_train: torch.Tensor
    x_val: torch.Tensor    # (6000, 784); held out, not used by the study
    y_val: torch.Tensor
    x_test: torch.Tensor   # (10000, 784)
    y_test: torch.Tensor


def load_mnist(root, split_seed: int = 42) -> MNISTSplits:
    """Load MNIST (downloading it to `root` if needed), normalise it, and split the
    60,000 training images at random into 54,000 train and 6,000 validation.
    """
    train = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)

    def prep(ds):
        x = ds.data.float().div(255.0).sub(MEAN).div(STD).flatten(1)
        return x, ds.targets.clone()

    x, y = prep(train)
    x_test, y_test = prep(test)
    n_train = int(0.9 * len(x))
    train_idx, val_idx = random_split(
        range(len(x)), [n_train, len(x) - n_train],
        generator=torch.Generator().manual_seed(split_seed),
    )
    tr, va = list(train_idx.indices), list(val_idx.indices)
    return MNISTSplits(x[tr], y[tr], x[va], y[va], x_test, y_test)


def batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator):
    """One epoch of shuffled mini-batches. The order is set entirely by `generator`."""
    order = torch.randperm(len(x), generator=generator)
    for start in range(0, len(x), batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]
