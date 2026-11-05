"""The fills: values held at the frozen entries of W1 = M * W + (1 - M) * F.

`make_fill(name, mask, seed)` returns a fill F with the shape of `mask`
(hidden x in_dim). Only `gaussian_equalised` depends on the mask's pattern; the
others depend only on its shape.

Scale. gaussian, rademacher, laplace, gaussian_sparse_50 and gaussian_equalised
have unit variance per entry (gaussian_equalised exactly, per row, at the frozen
entries). gaussian_kappa_025 (standard deviation 0.25) and kaiming (sqrt(2 / in_dim))
are deliberately smaller. Every other non-zero fill is rescaled so that each row
has L2 norm sqrt(in_dim), the root-mean-square norm of an N(0, 1) row; ones,
hadamard and square_wave have that norm already.

Basis fills (orthogonal_q, hadamard, dct, chebyshev, legendre, signed_permutation)
take `hidden` rows from an in_dim x in_dim basis, chosen at random, so they need
hidden <= in_dim; hadamard also needs in_dim to be a power of 2.

chebyshev and dct are the same basis: T_k at the Chebyshev nodes is
cos(pi k (j + 0.5) / n), the DCT-II basis up to a per-row constant that row
normalisation removes. With the same seed they return the same matrix.
pe_sin and pe_cos are the first `hidden` rows of the DST-IV and DCT-IV bases, so
their rows are orthogonal too. They, pe_sincos, pe_tanh, triangle_wave and
square_wave involve no random draw.
"""

from __future__ import annotations

import math

import torch

FILL_FAMILIES = {
    "constant": ["zero", "ones"],
    "i.i.d. light-tailed": ["gaussian", "gaussian_kappa_025", "uniform", "rademacher", "kaiming"],
    "i.i.d. heavy-tailed": ["laplace", "lognormal"],
    "orthogonal basis": ["orthogonal_q", "hadamard", "dct", "signed_permutation"],
    "sinusoidal": ["pe_sin", "pe_cos", "pe_sincos", "pe_tanh"],
    "waveform": ["triangle_wave", "square_wave"],
    "polynomial": ["chebyshev", "legendre"],
    "patterned Gaussian": ["gaussian_lowrank", "gaussian_banded", "gaussian_pe_modulated", "gaussian_sparse_50",
                           "gaussian_equalised"],
}
FILLS = [name for members in FILL_FAMILIES.values() for name in members]
PLUS_MINUS_ONE = {"rademacher", "hadamard", "square_wave"}
BASIS_FILLS = {"orthogonal_q", "hadamard", "dct", "chebyshev", "legendre", "signed_permutation"}


def _rows_to_norm(F, norm):
    return F / F.norm(dim=1, keepdim=True).clamp_min(1e-12) * norm


def _grid(n_rows, n_cols):
    i = torch.arange(n_rows, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(n_cols, dtype=torch.float64).unsqueeze(0)
    return i, j


def _sylvester(n):
    if n & (n - 1):
        raise ValueError(f"hadamard fill needs in_dim to be a power of 2, got {n}")
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def equalise_frozen_energy(fill: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rescale each row's frozen entries so that their squared norm equals the number
    of frozen entries in the row, as it is for any +-1 fill."""
    frozen = 1 - mask
    energy = (fill * frozen).pow(2).sum(1, keepdim=True)
    return fill * torch.sqrt(frozen.sum(1, keepdim=True) / energy.clamp_min(1e-12))


def make_fill(name: str, mask: torch.Tensor, seed: int) -> torch.Tensor:
    hidden, in_dim = mask.shape
    if name in BASIS_FILLS and hidden > in_dim:
        raise ValueError(f"{name} fill needs hidden <= in_dim, got {hidden} > {in_dim}")
    g = torch.Generator().manual_seed(seed)
    row_norm = math.sqrt(in_dim)
    i, j = _grid(hidden, in_dim)

    def basis_rows(B):  # `hidden` rows of an in_dim x in_dim basis, in random order
        return B[torch.randperm(in_dim, generator=g)][:hidden]

    def dst4(rows):  # sin(pi (i + 1/2)(j + 1/2) / in_dim)
        return torch.sin(math.pi * (rows + 0.5) * (j + 0.5) / in_dim)

    if name == "zero":
        return torch.zeros(hidden, in_dim)
    if name == "ones":
        return torch.ones(hidden, in_dim)
    if name == "gaussian":
        return torch.randn(hidden, in_dim, generator=g)
    if name == "gaussian_kappa_025":
        return 0.25 * torch.randn(hidden, in_dim, generator=g)
    if name == "uniform":
        return _rows_to_norm(torch.rand(hidden, in_dim, generator=g) * 2 - 1, row_norm)
    if name == "rademacher":
        return torch.randint(0, 2, (hidden, in_dim), generator=g).float() * 2 - 1
    if name == "kaiming":
        return math.sqrt(2.0 / in_dim) * torch.randn(hidden, in_dim, generator=g)
    if name == "laplace":  # Laplace(0, 1/sqrt(2)): unit variance
        U = torch.rand(hidden, in_dim, generator=g) - 0.5
        return -(1 / math.sqrt(2)) * U.sign() * torch.log1p(-2 * U.abs())
    if name == "lognormal":  # centred exp(N(0, 0.25))
        F = torch.exp(0.5 * torch.randn(hidden, in_dim, generator=g)) - math.exp(0.125)
        return _rows_to_norm(F, row_norm)
    if name == "orthogonal_q":  # rows of a random orthogonal matrix
        Q, _ = torch.linalg.qr(torch.randn(in_dim, in_dim, generator=g))
        return Q.T[:hidden] * row_norm
    if name == "hadamard":
        return basis_rows(_sylvester(in_dim))
    if name in ("dct", "chebyshev"):
        k, jj = _grid(in_dim, in_dim)
        B = torch.cos(math.pi * k * (jj + 0.5) / in_dim)
        return basis_rows(_rows_to_norm(B, row_norm).float())
    if name == "signed_permutation":  # one entry of +-sqrt(in_dim) per row, in distinct columns
        cols = torch.randperm(in_dim, generator=g)[:hidden]
        signs = torch.randint(0, 2, (hidden,), generator=g).float() * 2 - 1
        F = torch.zeros(hidden, in_dim)
        F[torch.arange(hidden), cols] = signs
        return F * row_norm
    if name == "legendre":  # P_k on equispaced points in [-1, 1], Bonnet recursion
        x = -1.0 + 2.0 * (torch.arange(in_dim, dtype=torch.float64) + 0.5) / in_dim
        P = [torch.ones_like(x), x]
        for k in range(1, in_dim - 1):
            P.append(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1))
        return basis_rows(_rows_to_norm(torch.stack(P[:in_dim]), row_norm).float())
    if name == "pe_sin":
        return _rows_to_norm(dst4(i), row_norm).float()
    if name == "pe_cos":
        return _rows_to_norm(torch.cos(math.pi * (i + 0.5) * (j + 0.5) / in_dim), row_norm).float()
    if name == "pe_tanh":
        return _rows_to_norm(torch.tanh(2 * dst4(i)), row_norm).float()
    if name == "pe_sincos":  # transformer positional encoding (row = position), base in_dim
        F = torch.zeros(hidden, in_dim, dtype=torch.float64)
        pos = torch.arange(1, hidden + 1, dtype=torch.float64)
        for k in range(in_dim // 2):
            F[:, 2 * k] = torch.sin(pos / float(in_dim) ** (2 * k / in_dim))
            F[:, 2 * k + 1] = torch.cos(pos / float(in_dim) ** (2 * k / in_dim))
        if in_dim % 2:
            F[:, -1] = torch.sin(pos / float(in_dim) ** ((in_dim - 1) / in_dim))
        return _rows_to_norm(F, row_norm).float()
    if name == "triangle_wave":  # frequency (2i + 1) / (2 in_dim), phase 1/4
        x = ((2 * i + 1) / (2 * in_dim) * j + 0.25) % 1.0
        return _rows_to_norm(1.0 - 4.0 * (x - 0.5).abs(), row_norm).float()
    if name == "square_wave":  # sign of pe_sin; row 0 is constant +1
        return torch.where(dst4(i) >= 0, 1.0, -1.0).float()
    if name == "gaussian_lowrank":  # rank 1: every row is a multiple of one Gaussian vector
        u = torch.randn(hidden, generator=g)
        v = torch.randn(in_dim, generator=g)
        return _rows_to_norm(torch.outer(u, v), row_norm)
    if name == "gaussian_banded":  # Gaussian on |i - j| <= 1, zero elsewhere
        band = ((i - j).abs() <= 1).float()
        return _rows_to_norm(torch.randn(hidden, in_dim, generator=g) * band, row_norm)
    if name == "gaussian_pe_modulated":  # Gaussian times the pe_sin pattern
        envelope = dst4(i).float()
        return _rows_to_norm(torch.randn(hidden, in_dim, generator=g) * envelope, row_norm)
    if name == "gaussian_sparse_50":  # each entry zero with probability 1/2; survivors scaled to unit variance
        keep = (torch.rand(hidden, in_dim, generator=g) < 0.5).float()
        return torch.randn(hidden, in_dim, generator=g) * keep * math.sqrt(2.0)
    if name == "gaussian_equalised":  # Gaussian with equal frozen energy in every row
        return equalise_frozen_energy(torch.randn(hidden, in_dim, generator=g), mask)
    raise ValueError(f"unknown fill: {name}")
