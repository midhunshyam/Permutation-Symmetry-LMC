"""Tables and figures for the MNIST study.

    python scripts/analyze_mnist.py [--protocol main]

Reads results/mnist/{methods,lambda,clique}_<protocol>.jsonl and writes
results/mnist/summary_<protocol>.md and figures/mnist_*.png (with a _<protocol>
suffix for protocols other than main). Values are mean ± SEM over seed pairs.
Comparisons between methods are paired on the seed. q-values are Benjamini–Hochberg
adjusted: per column in the comparison against Standard, and over all rows in
each other table.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmc import plotting  # noqa: E402
from lmc.report import Report, fmt_diff, fmt_mean_sem, fmt_p, load_jsonl, p_clause  # noqa: E402
from lmc.stats import benjamini_hochberg, mean_sem, paired_ttest  # noqa: E402

MNIST_DIR = ROOT / "results" / "mnist"
ORDER = ["standard", "w_asym", "sigma_asym", "scale_quotient", "shared_init", "tna_pfn", "joint_midpoint",
         "frozen_anchor"]
LABEL = {"standard": "Standard", "w_asym": "W-Asym", "sigma_asym": "σ-Asym", "scale_quotient": "ScaleQuotient",
         "shared_init": "Shared init", "tna_pfn": "TNA-PFN", "joint_midpoint": "Joint midpoint",
         "frozen_anchor": "Frozen anchor"}
COLOR_SLOT = {"w_asym": 0, "sigma_asym": 1, "scale_quotient": 2, "shared_init": 3, "tna_pfn": 4,
              "joint_midpoint": 6, "frozen_anchor": 5}
GROUPS = [("Architectures", ["standard", "w_asym", "sigma_asym", "scale_quotient"]),
          ("Training procedures", ["shared_init", "tna_pfn", "joint_midpoint", "frozen_anchor"])]
OTHER_COMPARISONS = [("tna_pfn", "shared_init"), ("frozen_anchor", "joint_midpoint")]


def color(t, method):
    return t.reference if method == "standard" else t.series[COLOR_SLOT[method]]


def endpoint_mean(r, metric):
    return 0.5 * (r["a"][metric] + r["b"][metric])


def aligned_barrier(r):
    return None if r["aligned"] is None else r["aligned"]["curve"]["midpoint_barrier"]


def figure_name(base, protocol):
    return base if protocol == "main" else f"{base}_{protocol}"


def paired_on_seed(runs_x, runs_y, fn):
    """Paired t-test of fn(run) between two {seed: run} dicts, over the seeds both contain."""
    seeds = sorted(set(runs_x) & set(runs_y))
    return paired_ttest([fn(runs_x[s]) for s in seeds], [fn(runs_y[s]) for s in seeds])


# ----------------------------------------------------------------------------- methods

def report_methods(rep, recs, protocol):
    runs = defaultdict(dict)  # method -> seed -> record
    for r in recs:
        runs[r["method"]][r["seed"]] = r
    present = [m for m in ORDER if m in runs]
    others = [m for m in present if m != "standard"]
    cfg = recs[0]["config"]
    n_pairs = sorted({len(runs[m]) for m in present})

    def values(m, fn):
        return [fn(runs[m][s]) for s in sorted(runs[m])]

    rep.heading(f"Protocol `{protocol}`")
    layers = f"{cfg['n_hidden_layers']} hidden layer{'s' if cfg['n_hidden_layers'] > 1 else ''}"
    rep.paragraph(f"{layers} of width {cfg['hidden']}, {cfg['epochs']} epochs, batch {cfg['batch_size']}, AdamW "
                  f"(lr {cfg['lr']}, weight decay {cfg['weight_decay']}), "
                  f"{' or '.join(map(str, n_pairs))} seed pairs per method. "
                  "Evaluated on the 10,000 MNIST test images.")

    rep.heading("Accuracy and barrier", 3)
    rep.paragraph("Accuracies in %. Endpoint acc is the mean of A's and B's. 'Barrier after alignment' is the "
                  "midpoint barrier after permuting B's hidden units to match A's (only for architectures where "
                  "a permutation leaves the network's function unchanged).")
    rep.table(["method", "endpoint acc", "midpoint acc", "midpoint barrier", "max barrier", "barrier after alignment"],
              [[LABEL[m], fmt_mean_sem(values(m, lambda r: endpoint_mean(r, "acc")), percent=True),
                fmt_mean_sem(values(m, lambda r: r["midpoint"]["acc"]), percent=True),
                fmt_mean_sem(values(m, lambda r: r["curve"]["midpoint_barrier"])),
                fmt_mean_sem(values(m, lambda r: r["curve"]["max_barrier"])),
                fmt_mean_sem(values(m, aligned_barrier))] for m in present])

    compared_metrics = [("midpoint barrier", lambda r: r["curve"]["midpoint_barrier"], 4),
                        ("midpoint acc (pp)", lambda r: 100 * r["midpoint"]["acc"], 2),
                        ("endpoint acc (pp)", lambda r: 100 * endpoint_mean(r, "acc"), 2)]
    if "standard" in runs:
        rep.heading("Paired comparisons against Standard", 3)
        rep.paragraph(f"Difference = method − Standard, paired on seed. q: BH over the {len(others)} methods "
                      "within each column.")
        tests = {}
        for name, fn, _ in compared_metrics:
            column = [paired_on_seed(runs[m], runs["standard"], fn) for m in others]
            benjamini_hochberg(column)
            tests[name] = dict(zip(others, column))
        rep.table(["method"] + [h for name, _, _ in compared_metrics for h in (name, "q")],
                  [[LABEL[m]] + [c for name, _, d in compared_metrics
                                 for c in (fmt_diff(tests[name][m], d), fmt_p(tests[name][m].q))]
                   for m in others])

    extra = [(x, y, name, d, paired_on_seed(runs[x], runs[y], fn))
             for x, y in OTHER_COMPARISONS if x in runs and y in runs for name, fn, d in compared_metrics]
    if extra:
        benjamini_hochberg([e[-1] for e in extra])
        rep.paragraph("Other comparisons (q: BH over the rows of this table):")
        rep.table(["comparison", "metric", "difference", "p", "q"],
                  [[f"{LABEL[x]} − {LABEL[y]}", name, fmt_diff(p, d), fmt_p(p.p), fmt_p(p.q)]
                   for x, y, name, d, p in extra], text_columns=2)

    rep.heading("Calibration of the midpoint", 3)
    rep.paragraph("ECE: 15-bin expected calibration error, in %. ΔECE = midpoint ECE − mean endpoint ECE, "
                  "paired on seed; q: BH over methods. Brier: squared error summed over classes (0–2).")
    tests = [paired_ttest(values(m, lambda r: r["midpoint"]["ece"]), values(m, lambda r: endpoint_mean(r, "ece")))
             for m in present]
    benjamini_hochberg(tests)
    rep.table(["method", "endpoint ECE", "midpoint ECE", "ΔECE", "q", "endpoint Brier", "midpoint Brier"],
              [[LABEL[m], fmt_mean_sem(values(m, lambda r: endpoint_mean(r, "ece")), percent=True),
                fmt_mean_sem(values(m, lambda r: r["midpoint"]["ece"]), percent=True), fmt_diff(p, 2, scale=100),
                fmt_p(p.q), fmt_mean_sem(values(m, lambda r: endpoint_mean(r, "brier"))),
                fmt_mean_sem(values(m, lambda r: r["midpoint"]["brier"]))] for m, p in zip(present, tests)])

    rep.heading("Similarity of the two networks", 3)
    rep.paragraph("Parameter cosine, distance and row cosine leave out weight entries that receive no gradient "
                  "(W-Asym's frozen positions, TNA-PFN's gradient-masked entries); alignment uses the full effective "
                  "weights. Parameter cosine and "
                  "distance: between the parameter vectors of A and B. Row cosine: mean cosine between "
                  "corresponding hidden-unit weight rows; 'aligned' is after the permutation. Units moved: share of "
                  "hidden units the alignment permutation moves. Logit CKA: linear CKA between the two networks' "
                  "test-set logits.")

    def moved(r):
        return None if r["aligned"] is None else 100 * r["aligned"]["n_units_moved"] / r["aligned"]["n_hidden_units"]

    rep.table(["method", "parameter cosine", "‖θ_A − θ_B‖", "row cosine", "row cosine, aligned", "units moved (%)",
               "logit CKA"],
              [[LABEL[m], fmt_mean_sem(values(m, lambda r: r["param_cosine"]), 3),
                fmt_mean_sem(values(m, lambda r: r["param_distance"]), 2),
                fmt_mean_sem(values(m, lambda r: r["row_cosine"]), 3),
                fmt_mean_sem(values(m, lambda r: None if r["aligned"] is None else r["aligned"]["row_cosine"]), 3),
                fmt_mean_sem(values(m, moved), 1),
                fmt_mean_sem(values(m, lambda r: r["logit_cka"]), 3)] for m in present])

    draw_paths(runs, protocol)
    draw_methods(runs, present, protocol)
    return runs


def draw_paths(runs, protocol):
    """Loss above the chord along the interpolation path, one panel per group of methods."""
    def excess(r):
        alphas, losses = np.array(r["curve"]["alphas"]), np.array(r["curve"]["losses"])
        return losses - ((1 - alphas) * losses[0] + alphas * losses[-1])

    def draw(t):
        fig, axes = plotting.plt.subplots(1, 2, figsize=(9.4, 3.7), gridspec_kw={"wspace": 0.28})
        for ax, (title, group) in zip(axes, GROUPS):
            for m in [g for g in group if g in runs]:
                rs = list(runs[m].values())
                alphas = np.array(rs[0]["curve"]["alphas"])
                E = np.array([excess(r) for r in rs])
                mean, sem = E.mean(0), E.std(0, ddof=1) / math.sqrt(len(E))
                ax.fill_between(alphas, mean - sem, mean + sem, color=color(t, m), alpha=0.15, lw=0)
                ax.plot(alphas, mean, color=color(t, m), label=LABEL[m], lw=1.6)
            ax.axhline(0, color=t.axis, lw=1.0, zorder=1)
            ax.set_title(title)
            ax.set_ylabel("test loss above the chord")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(["0\nA", "0.25", "0.5", "0.75", "1\nB"])
            ax.set_xlabel("α")
        axes[0].legend(loc="upper right")
        axes[1].legend(loc="upper left")
        axes[1].set_title("Training procedures (note the y scale)")
        return fig
    plotting.save(draw, figure_name("mnist_paths", protocol))


def draw_methods(runs, present, protocol):
    """Midpoint barrier (as trained and after alignment) and endpoint vs midpoint accuracy, per method."""
    def draw(t):
        fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
        y = np.arange(len(present))[::-1]
        for yi, m in zip(y, present):
            rs = list(runs[m].values())
            c = color(t, m)
            b = np.array([r["curve"]["midpoint_barrier"] for r in rs])
            ax1.scatter(b, np.full(len(b), yi + 0.14), s=11, color=c, alpha=0.5, lw=0, zorder=2)
            ax1.scatter(b.mean(), yi + 0.14, s=46, color=c, edgecolors=t.surface, linewidths=1.2, zorder=4)
            post = [aligned_barrier(r) for r in rs]
            if post[0] is None:
                ax1.text(0.015, yi - 0.2, "alignment n/a", color=t.ink2, fontsize=8.5, va="center", ha="left")
            else:
                ax1.scatter(np.mean(post), yi - 0.2, s=42, facecolors=t.surface, edgecolors=c, linewidths=1.6,
                            zorder=4)
            end = 100 * np.mean([endpoint_mean(r, "acc") for r in rs])
            mid = 100 * np.mean([r["midpoint"]["acc"] for r in rs])
            ax2.plot([end, mid], [yi, yi], color=t.axis, lw=1.4, zorder=1)
            ax2.scatter(end, yi, s=70, marker="|", color=c, linewidths=2.2, zorder=3)
            ax2.scatter(mid, yi, s=46, color=c, edgecolors=t.surface, linewidths=1.2, zorder=4)
        ax1.axvline(0, color=t.axis, lw=1.0)
        ax1.set_yticks(y)
        ax1.set_yticklabels([LABEL[m] for m in present])
        for ax in (ax1, ax2):
            ax.grid(axis="y", visible=False)
        ax1.set_xlabel("midpoint barrier")
        ax1.set_title("As trained and after alignment")
        ax2.set_xlabel("test accuracy (%)")
        ax2.set_title("Endpoints vs midpoint")
        ax1.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.5, color=t.ink2, alpha=0.6, label="one seed pair"),
                            Line2D([], [], marker="o", ls="", ms=6.5, color=t.ink2, label="mean, as trained"),
                            Line2D([], [], marker="o", ls="", ms=6.5, mfc=t.surface, mec=t.ink2, mew=1.6,
                                   label="mean, after alignment")], loc="lower right")
        ax2.legend(handles=[Line2D([], [], marker="|", ls="", ms=9, mew=2.2, color=t.ink2, label="endpoint mean"),
                            Line2D([], [], marker="o", ls="", ms=6.5, color=t.ink2, label="midpoint")],
                   loc="lower left")
        return fig
    plotting.save(draw, figure_name("mnist_methods", protocol))


# ----------------------------------------------------------------------------- lambda and clique

def report_lambda(rep, recs, protocol, method_runs=None):
    by_lam = defaultdict(list)
    for r in recs:
        by_lam[r["lam"]].append(r)
    lams = sorted(by_lam)
    rep.heading("Midpoint weight λ (joint midpoint training)", 3)
    rep.paragraph("L = CE(A) + CE(B) + λ·CE(midpoint), with A and B trained together on the same batches. λ = 0 is "
                  "two networks trained side by side with no coupling; λ < 0 rewards a high midpoint loss. "
                  "'finite' counts seed pairs whose test losses stayed finite.")
    rows = []
    for lam in lams:
        rs = by_lam[lam]
        ok = [r for r in rs if r["curve"]["midpoint_barrier"] is not None]
        rows.append([f"{lam:+g}", f"{len(ok)} / {len(rs)}",
                     fmt_mean_sem([endpoint_mean(r, "acc") for r in ok], percent=True),
                     fmt_mean_sem([r["midpoint"]["acc"] for r in ok], percent=True),
                     fmt_mean_sem([r["curve"]["midpoint_barrier"] for r in ok]),
                     fmt_mean_sem([r["param_cosine"] for r in ok], 3),
                     fmt_mean_sem([r["param_distance"] for r in ok], 2)])
    rep.table(["λ", "finite", "endpoint acc", "midpoint acc", "midpoint barrier", "parameter cosine",
               "‖θ_A − θ_B‖"], rows)
    if method_runs and "standard" in method_runs and 0.0 in by_lam:
        test = paired_on_seed({r["seed"]: r for r in by_lam[0.0]}, method_runs["standard"],
                              lambda r: r["curve"]["midpoint_barrier"])
        rep.paragraph(f"λ = 0 against Standard (methods file), paired on seed: midpoint barrier difference "
                      f"{fmt_diff(test, 4)} ({p_clause(test.p)}).")

    shown = [lam for lam in lams if lam >= 0]  # lambda < 0 drives the barrier above 1e4; reported in the table
    x = np.arange(len(shown))

    def draw(t):
        c = t.series[COLOR_SLOT["joint_midpoint"]]
        fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(8.0, 3.2), gridspec_kw={"wspace": 0.28})
        v = [mean_sem([r["curve"]["midpoint_barrier"] for r in by_lam[lam]]) for lam in shown]
        ax1.errorbar(x, [m for m, _ in v], yerr=[s for _, s in v], color=c, marker="o", ms=6.5, ls="",
                     mec=t.surface, elinewidth=1.4, capsize=0)
        ax1.axhline(0, color=t.axis, lw=1.0, zorder=1)
        ax1.set_ylabel("midpoint barrier")
        v = [mean_sem([100 * endpoint_mean(r, "acc") for r in by_lam[lam]]) for lam in shown]
        ax2.errorbar(x - 0.08, [m for m, _ in v], yerr=[s for _, s in v], color=c, marker="_", ms=12, mew=2.2,
                     ls="", elinewidth=1.4, capsize=0)
        v = [mean_sem([100 * r["midpoint"]["acc"] for r in by_lam[lam]]) for lam in shown]
        ax2.errorbar(x + 0.08, [m for m, _ in v], yerr=[s for _, s in v], color=c, marker="o", ms=6.5, ls="",
                     mec=t.surface, elinewidth=1.4, capsize=0)
        ax2.set_ylabel("test accuracy (%)")
        ax2.legend(handles=[Line2D([], [], marker="_", ls="", ms=12, mew=2.2, color=c, label="endpoint mean"),
                            Line2D([], [], marker="o", ls="", ms=6.5, color=c, mec=t.surface, label="midpoint")],
                   loc="lower right")
        for ax in (ax1, ax2):
            ax.set_xticks(x)
            ax.set_xticklabels([f"{lam:g}" for lam in shown])
            ax.set_xlabel("λ (midpoint weight)")
            ax.grid(axis="x", visible=False)
        plotting.suptitle(fig, t, "Joint midpoint training, λ ≥ 0 (λ < 0 in the table)")
        fig.subplots_adjust(top=0.87, bottom=0.18, left=0.09, right=0.99)
        return fig
    plotting.save(draw, figure_name("mnist_lambda", protocol))


def report_clique(rep, recs):
    by_k = defaultdict(list)
    for r in recs:
        by_k[r["K"]].append(r)
    rep.heading("Midpoint cliques: K networks with a midpoint term for every pair (λ = 2)", 3)
    rep.paragraph("'Average' is the network whose parameters are the mean of all K. Gain = average accuracy − "
                  "mean endpoint accuracy. K = 2 is joint midpoint training with the same seeds.")
    rep.table(["K", "pairs", "seeds", "endpoint acc", "average acc", "gain (pp)", "worst pair barrier",
               "mean pair barrier"],
              [[K, K * (K - 1) // 2, len(by_k[K]),
                fmt_mean_sem([np.mean(r["endpoint_accs"]) for r in by_k[K]], percent=True),
                fmt_mean_sem([r["average_acc"] for r in by_k[K]], percent=True),
                "{:+.2f} ± {:.2f}".format(*mean_sem([100 * (r["average_acc"] - np.mean(r["endpoint_accs"]))
                                                     for r in by_k[K]])),
                fmt_mean_sem([max(r["pair_barriers"]) for r in by_k[K]]),
                fmt_mean_sem([np.mean(r["pair_barriers"]) for r in by_k[K]])] for K in sorted(by_k)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", default="main")
    args = parser.parse_args()
    rep = Report(f"MNIST study: generated tables (`{args.protocol}`)", "scripts/analyze_mnist.py")
    recs = load_jsonl(MNIST_DIR / f"methods_{args.protocol}.jsonl")
    method_runs = report_methods(rep, recs, args.protocol) if recs else None
    recs = load_jsonl(MNIST_DIR / f"lambda_{args.protocol}.jsonl")
    if recs:
        report_lambda(rep, recs, args.protocol, method_runs)
    recs = load_jsonl(MNIST_DIR / f"clique_{args.protocol}.jsonl")
    if recs:
        report_clique(rep, recs)
    rep.write(MNIST_DIR / f"summary_{args.protocol}.md")


if __name__ == "__main__":
    main()
