"""Run the MNIST experiments; one JSON record per trained pair (or K-tuple).

    python scripts/run_mnist.py methods --protocol main   # every method, n_seeds pairs each
    python scripts/run_mnist.py lambda  --protocol main   # joint midpoint training, midpoint weight lambda swept
    python scripts/run_mnist.py clique  --protocol main   # midpoint cliques of K = 2..5 networks

Writes results/mnist/<experiment>_<protocol>.jsonl. Runs already in that file are
skipped, so an interrupted job can be restarted with the same command.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmc.mnist.data import load_mnist  # noqa: E402
from lmc.mnist.experiment import METHODS, PROTOCOLS, run_clique, run_pair  # noqa: E402

LAMBDAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
CLIQUE_SIZES = [2, 3, 4, 5]


def _json_safe(v):
    """NaN and infinity become null, so the file is strict JSON."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _completed(path, key_fn):
    if not path.exists():
        return set()
    return {key_fn(json.loads(line)) for line in path.read_text().splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["methods", "lambda", "clique"])
    parser.add_argument("--protocol", default="main", choices=sorted(PROTOCOLS))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS),
                        help="methods to run (methods experiment only)")
    parser.add_argument("--device", default="cpu", help="cpu, mps or cuda; the reported runs used cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "results" / "mnist"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cfg = PROTOCOLS[args.protocol]
    device = torch.device(args.device)
    data = load_mnist(args.data)
    out = Path(args.out) / f"{args.experiment}_{args.protocol}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.experiment == "methods":
        jobs = [(m, s) for m in args.methods for s in range(cfg.n_seeds)]
        done = _completed(out, lambda r: (r["method"], r["seed"]))
        run = lambda m, s: run_pair(m, s, cfg, data, device)  # noqa: E731
    elif args.experiment == "lambda":
        jobs = [(lam, s) for lam in LAMBDAS for s in range(cfg.n_seeds)]
        done = _completed(out, lambda r: (r["lam"], r["seed"]))
        run = lambda lam, s: run_pair("joint_midpoint", s, cfg, data, device, lam=lam)  # noqa: E731
    else:
        n_clique_seeds = max(1, cfg.n_seeds // 2)  # half as many seeds: cliques are slower to train
        jobs = [(K, s) for K in CLIQUE_SIZES for s in range(n_clique_seeds)]
        done = _completed(out, lambda r: (r["K"], r["seed"]))
        run = lambda K, s: run_clique(K, s, cfg, data, device)  # noqa: E731

    todo = [job for job in jobs if job not in done]
    print(f"{args.experiment} / {args.protocol}: {len(todo)} of {len(jobs)} runs to do -> {out}", flush=True)
    for i, job in enumerate(todo, 1):
        rec = run(*job)
        with out.open("a") as fh:
            fh.write(json.dumps(_json_safe(rec), allow_nan=False) + "\n")
        if "curve" in rec:
            barrier = rec["curve"]["midpoint_barrier"]
            result = "midpoint barrier " + ("non-finite" if not math.isfinite(barrier) else f"{barrier:+.4f}")
        else:
            result = f"worst pair barrier {max(rec['pair_barriers']):+.4f}"
        print(f"[{i}/{len(todo)}] {job}: {result}  ({rec['train_seconds']:.0f} s training)", flush=True)


if __name__ == "__main__":
    main()
