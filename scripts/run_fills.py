"""Run the fill-study experiments; one JSON record per trained pair.

    python scripts/run_fills.py enumeration  # 4-4-3 net: all 256 one-per-row masks, zero vs Gaussian fill
    python scripts/run_fills.py sweep        # 4-4-3 net: all 26 fills on 3 masks
    python scripts/run_fills.py hybrid       # 4-4-3 net: k of the 4 frozen entries Gaussian, the rest zero
    python scripts/run_fills.py width        # in_dim 64, 16 frozen entries per row, hidden 4..64
    python scripts/run_fills.py grid         # hidden 32: in_dim {32..256} x n_frozen {4..32}, 11 fills
    python scripts/run_fills.py grid_wide    # hidden 64 and 128: in_dim {64..256} x n_frozen {8..32}, 7 fills
    python scripts/run_fills.py controls     # hidden 32: unmasked; Gaussian fill not shared by A and B

Writes results/fills/<experiment>.jsonl. Pairs already in that file are skipped, so
an interrupted run can be restarted with the same command.

grid and grid_wide run every cell (in_dim, n_frozen) under two designs (see design_seeds):
  fixed      one mask and fill draw per cell, used by all 10 seed pairs; the seed
             pairs differ only in their initial weights
  resampled  each seed pair draws its own mask and fill (fills without a random
             draw, such as pe_sin and square_wave, change only through the mask)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmc import stable_seed  # noqa: E402
from lmc.fills.catalog import FILLS, make_fill  # noqa: E402
from lmc.fills.data import mixture, small_mixture  # noqa: E402
from lmc.fills.experiment import (BASE_FILL_SEED, BASE_MASK_SEED, LARGER_NET, SMALL_NET,  # noqa: E402
                                  SMALL_PATTERNS, UNSHARED_FILL_SEED, design_seeds, drop_one_mask,
                                  random_mask, run_pair, sweep_fill_seed)

GRID_IN_DIMS = [32, 64, 128, 256]
GRID_N_FROZEN = [4, 8, 16, 32]
GRID_FILLS = ["zero", "gaussian", "chebyshev", "laplace", "hadamard", "rademacher",
              "square_wave", "legendre", "pe_sin", "orthogonal_q", "gaussian_equalised"]
GRID_WIDE_FILLS = ["gaussian", "hadamard", "rademacher", "square_wave", "chebyshev", "laplace",
                   "gaussian_equalised"]


# ----------------------------------------------------------------------------- job lists
# Each job is (key, run): `key` identifies the pair and is written into its record;
# `run()` trains the pair and returns the measurements.

def _small_net_kwargs(**extra):
    """Training arguments of run_pair for the 4-4-3 network."""
    return dict(n_classes=SMALL_NET["n_classes"], steps=SMALL_NET["steps"], lr=SMALL_NET["lr"], **extra)


def _larger_net_kwargs(**extra):
    """Training arguments of run_pair for the larger networks."""
    return dict(n_classes=LARGER_NET["n_classes"], steps=LARGER_NET["steps"], lr=LARGER_NET["lr"], **extra)


def jobs_enumeration():
    data = small_mixture()
    in_dim, hidden = SMALL_NET["in_dim"], SMALL_NET["hidden"]
    common = _small_net_kwargs(data=data, full_curve=False)
    for seed in range(SMALL_NET["n_seeds"]):  # own seeds: not paired with the masked runs
        yield ({"fill": "unmasked", "seed": seed},
               lambda seed=seed: run_pair(torch.ones(hidden, in_dim), torch.zeros(hidden, in_dim),
                                          seed_a=1000 + 2 * seed, seed_b=1001 + 2 * seed, **common))
    for index, pattern in enumerate(itertools.product(range(in_dim), repeat=hidden)):
        mask = drop_one_mask(pattern, in_dim)
        fills = {"zero": torch.zeros(hidden, in_dim),
                 "gaussian": torch.randn(hidden, in_dim, generator=torch.Generator().manual_seed(50_000 + index))}
        for fill_name, fill in fills.items():
            for seed in range(SMALL_NET["n_seeds"]):
                yield ({"pattern": list(pattern), "fill": fill_name, "seed": seed},
                       lambda mask=mask, fill=fill, seed=seed:
                       run_pair(mask, fill, seed_a=2 * seed, seed_b=2 * seed + 1, **common))


def jobs_sweep():
    common = _small_net_kwargs(data=small_mixture())
    for pattern in SMALL_PATTERNS:
        mask = drop_one_mask(pattern, SMALL_NET["in_dim"])
        for fill_name in FILLS:
            fill = make_fill(fill_name, mask, seed=sweep_fill_seed(fill_name))
            for seed in range(SMALL_NET["n_seeds"]):
                yield ({"pattern": list(pattern), "fill": fill_name, "seed": seed},
                       lambda mask=mask, fill=fill, seed=seed:
                       run_pair(mask, fill, seed_a=2 * seed, seed_b=2 * seed + 1, **common))


def jobs_hybrid():
    """k of the 4 frozen entries hold Gaussian values and the rest are zero; every
    subset of each size k = 0..4 is run."""
    common = _small_net_kwargs(data=small_mixture())
    in_dim, hidden = SMALL_NET["in_dim"], SMALL_NET["hidden"]
    for pattern in SMALL_PATTERNS:
        mask = drop_one_mask(pattern, in_dim)
        name = "".join(map(str, pattern))
        gaussian = torch.randn(hidden, in_dim, generator=torch.Generator().manual_seed(stable_seed("hybrid", name)))
        frozen = list(enumerate(pattern))
        for k in range(len(frozen) + 1):
            for subset_index, positions in enumerate(itertools.combinations(frozen, k)):
                fill = torch.zeros(hidden, in_dim)
                for row, col in positions:
                    fill[row, col] = gaussian[row, col]
                for seed in range(SMALL_NET["n_seeds"]):
                    key = {"pattern": list(pattern), "n_gaussian": k, "subset_index": subset_index, "seed": seed}
                    yield (key, lambda mask=mask, fill=fill, seed=seed:
                           run_pair(mask, fill, seed_a=2 * seed, seed_b=2 * seed + 1, **common))


def _cell_jobs(hidden, in_dim, n_frozen, fills, designs, data):
    common = _larger_net_kwargs(data=data)
    for design in designs:
        for fill_name in fills:
            for seed in range(LARGER_NET["n_seeds"]):
                mask_seed, fill_seed = design_seeds(design, seed)
                key = {"design": design, "hidden": hidden, "in_dim": in_dim, "n_frozen": n_frozen,
                       "fill": fill_name, "seed": seed}
                yield (key, lambda fill_name=fill_name, mask_seed=mask_seed, fill_seed=fill_seed, seed=seed:
                       _masked_pair(in_dim, hidden, n_frozen, fill_name, mask_seed, fill_seed, seed, common))


def _masked_pair(in_dim, hidden, n_frozen, fill_name, mask_seed, fill_seed, seed, common):
    mask = random_mask(in_dim, hidden, n_frozen, seed=mask_seed)
    return run_pair(mask, make_fill(fill_name, mask, fill_seed), seed_a=2 * seed, seed_b=2 * seed + 1, **common)


def jobs_width():
    data = mixture(64)
    for hidden in [4, 8, 16, 32, 64]:
        yield from _cell_jobs(hidden, 64, 16, ["zero", "gaussian", "chebyshev"], ["fixed"], data)


def jobs_grid():
    for in_dim in GRID_IN_DIMS:
        data = mixture(in_dim)
        for n_frozen in GRID_N_FROZEN:
            if n_frozen < in_dim:
                yield from _cell_jobs(32, in_dim, n_frozen, GRID_FILLS, ["fixed", "resampled"], data)


def jobs_grid_wide():
    for in_dim in [64, 128, 256]:
        data = mixture(in_dim)
        for hidden in [64, 128]:
            if hidden > in_dim:  # basis fills need hidden <= in_dim
                continue
            for n_frozen in [8, 16, 32]:
                yield from _cell_jobs(hidden, in_dim, n_frozen, GRID_WIDE_FILLS, ["fixed", "resampled"], data)


def jobs_controls():
    """Two controls at hidden 32, fixed design:

    unmasked           no frozen entries (n_frozen = 0); 10 seed pairs per in_dim
    gaussian_unshared  Gaussian fill, but B gets an independent draw instead of A's
    """
    common = _larger_net_kwargs()
    for in_dim in GRID_IN_DIMS:
        data = mixture(in_dim)
        for seed in range(LARGER_NET["n_seeds"]):
            key = {"design": "fixed", "hidden": 32, "in_dim": in_dim, "n_frozen": 0, "fill": "unmasked", "seed": seed}
            yield (key, lambda data=data, in_dim=in_dim, seed=seed:
                   run_pair(torch.ones(32, in_dim), torch.zeros(32, in_dim), data=data,
                            seed_a=2 * seed, seed_b=2 * seed + 1, **common))
        for n_frozen in GRID_N_FROZEN:
            if n_frozen >= in_dim:
                continue
            mask = random_mask(in_dim, 32, n_frozen, seed=BASE_MASK_SEED)
            fill_a = make_fill("gaussian", mask, BASE_FILL_SEED)  # the shared fill of the grid's fixed design
            fill_b = make_fill("gaussian", mask, UNSHARED_FILL_SEED)
            for seed in range(LARGER_NET["n_seeds"]):
                key = {"design": "fixed", "hidden": 32, "in_dim": in_dim, "n_frozen": n_frozen,
                       "fill": "gaussian_unshared", "seed": seed}
                yield (key, lambda data=data, mask=mask, fill_a=fill_a, fill_b=fill_b, seed=seed:
                       run_pair(mask, fill_a, data=data, seed_a=2 * seed, seed_b=2 * seed + 1, fill_b=fill_b, **common))


EXPERIMENTS = {"enumeration": jobs_enumeration, "sweep": jobs_sweep, "hybrid": jobs_hybrid,
               "width": jobs_width, "grid": jobs_grid, "grid_wide": jobs_grid_wide, "controls": jobs_controls}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=list(EXPERIMENTS))
    parser.add_argument("--out", default=str(ROOT / "results" / "fills"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.out) / f"{args.experiment}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = list(EXPERIMENTS[args.experiment]())
    key_fields = set().union(*(key for key, _ in jobs))

    def key_string(record):
        return json.dumps({k: record[k] for k in key_fields if k in record}, sort_keys=True)

    done = set()
    if out.exists():
        done = {key_string(json.loads(line)) for line in out.read_text().splitlines() if line.strip()}
    todo = [(key, run) for key, run in jobs if key_string(key) not in done]
    print(f"{args.experiment}: {len(todo)} of {len(jobs)} pairs to train -> {out}", flush=True)
    t0 = time.time()
    with out.open("a") as fh:
        for i, (key, run) in enumerate(todo, 1):
            fh.write(json.dumps({**key, **run()}) + "\n")
            fh.flush()
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
