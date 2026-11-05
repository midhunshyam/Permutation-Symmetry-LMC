"""Tables and figures for the fill study.

    python scripts/analyze_fills.py

Reads results/fills/*.jsonl and writes results/fills/summary.md (one section per
experiment, in the order below) and figures/fills_*.png.

Statistics: t-tests paired by seed (sample SD) with Benjamini–Hochberg
q-values; each function's docstring names the family a correction runs over.
"Detectable" means q < 0.05.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmc import plotting  # noqa: E402
from lmc.fills.catalog import FILL_FAMILIES, PLUS_MINUS_ONE, make_fill  # noqa: E402
from lmc.fills.experiment import SMALL_NET, design_seeds, drop_one_mask, random_mask, sweep_fill_seed  # noqa: E402
from lmc.report import Report, fmt_diff, fmt_float, fmt_p, load_jsonl, p_clause  # noqa: E402
from lmc.stats import benjamini_hochberg, mean_sem, one_sample_ttest, paired_ttest  # noqa: E402

FILL_DIR = ROOT / "results" / "fills"
Q_THRESHOLD = 0.05
DESIGNS = ("fixed", "resampled")
FAMILY_OF = {name: family for family, members in FILL_FAMILIES.items() for name in members}


def load(experiment):
    return load_jsonl(FILL_DIR / f"{experiment}.jsonl")


def endpoint_acc(r):
    return 0.5 * (r["acc_a"] + r["acc_b"])


def pattern_label(pattern):
    return "(" + ",".join(map(str, pattern)) + ")"


FILL_LABEL = {"zero": "zero", "gaussian": "Gaussian", "chebyshev": "Chebyshev",
              "gaussian_unshared": "Gaussian, unshared"}


def fill_color(t, name):
    return {"zero": t.reference, "gaussian": t.series[0], "chebyshev": t.series[1],
            "gaussian_unshared": t.series[2]}[name]


# ============================================================================ 4-4-3 network

def enumeration(rep):
    """Section `enumeration` and figure fills_enumeration."""
    recs = load("enumeration")
    if not recs:
        return
    unmasked = np.array([r["midpoint_barrier"] for r in recs if r["fill"] == "unmasked"])
    unmasked_acc = np.array([endpoint_acc(r) for r in recs if r["fill"] == "unmasked"])
    runs = defaultdict(list)
    for r in recs:
        if r["fill"] != "unmasked":
            runs[(tuple(r["pattern"]), r["fill"])].append(r)
    patterns = sorted({p for p, _ in runs})
    barrier = {f: np.array([np.mean([r["midpoint_barrier"] for r in runs[(p, f)]]) for p in patterns])
               for f in ("zero", "gaussian")}
    acc = {f: np.array([np.mean([endpoint_acc(r) for r in runs[(p, f)]]) for p in patterns])
           for f in ("zero", "gaussian")}
    diff = barrier["zero"] - barrier["gaussian"]
    n_distinct = np.array([len(set(p)) for p in patterns])

    rep.heading("`enumeration`: every mask with one frozen entry per row (4-4-3 network)")
    rep.paragraph(f"{len(patterns)} masks; mask (a,b,c,d) freezes input a in row 0, b in row 1, and so on. 5 seed "
                  "pairs per mask and fill. Each mask's Gaussian fill is its own draw. The unmasked network was "
                  "trained with 5 seed pairs of its own.")
    rep.table(["condition", "mean barrier", "SD", "min", "max", "mean endpoint acc"],
              [[f"unmasked ({len(unmasked)} seed pairs)", fmt_float(unmasked.mean()), fmt_float(unmasked.std(ddof=1)),
                fmt_float(unmasked.min()),
                fmt_float(unmasked.max()), fmt_float(unmasked_acc.mean())]] +
              [[f"{FILL_LABEL[f]} fill (over masks)", fmt_float(barrier[f].mean()),
                fmt_float(barrier[f].std(ddof=1)),
                fmt_float(barrier[f].min()), fmt_float(barrier[f].max()), fmt_float(acc[f].mean())]
               for f in ("zero", "gaussian")])
    unmasked_pairs = sorted((r["seed"], r["acc_a"], r["acc_b"]) for r in recs if r["fill"] == "unmasked")
    rep.paragraph("Unmasked pairs, test accuracy of A and B: " +
                  "; ".join(f"seed {s}: {a:.3f}, {b:.3f}" for s, a, b in unmasked_pairs) + ".")
    w = stats.wilcoxon(diff)
    rep.paragraph(f"Gaussian fill gives the lower barrier on {int((diff > 0).sum())} of {len(diff)} masks; mean "
                  f"difference (zero − Gaussian) {diff.mean():+.3f} (SD {diff.std(ddof=1):.3f}; Wilcoxon signed-rank "
                  f"p = {w.pvalue:.1e}).")
    rep.paragraph("By number of distinct frozen inputs (1 = every row freezes the same input; 4 = all different):")
    rep.table(["distinct inputs", "masks", "barrier, zero", "barrier, Gaussian", "acc, zero", "acc, Gaussian"],
              [[d, int((n_distinct == d).sum()), fmt_float(barrier["zero"][n_distinct == d].mean()),
                fmt_float(barrier["gaussian"][n_distinct == d].mean()), fmt_float(acc["zero"][n_distinct == d].mean()),
                fmt_float(acc["gaussian"][n_distinct == d].mean())] for d in (1, 2, 3, 4)])

    def draw(t):
        fig, ax = plotting.plt.subplots(figsize=(4.8, 4.5))
        lim = max(barrier["zero"].max(), barrier["gaussian"].max(), unmasked.max()) * 1.04
        ax.plot([0, lim], [0, lim], color=t.muted, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.scatter(barrier["gaussian"], barrier["zero"], s=16, color=t.series[0], alpha=0.8,
                   edgecolors=t.surface, linewidths=0.6, zorder=3)
        tick = 0.025 * lim  # the unmasked pairs, as ticks along both axes
        for v in unmasked:
            ax.plot([0, tick], [v, v], color=t.ink, lw=1.6, solid_capstyle="butt", zorder=4, clip_on=False)
            ax.plot([v, v], [0, tick], color=t.ink, lw=1.6, solid_capstyle="butt", zorder=4, clip_on=False)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=t.series[0], mec=t.surface, label="one mask"),
                           Line2D([], [], color=t.muted, lw=1.0, ls=(0, (4, 3)), label="equal barrier"),
                           Line2D([], [], color=t.ink, lw=1.6, solid_capstyle="butt",
                                  label=f"unmasked: {len(unmasked)} seed pairs\n(ticks on both axes)")],
                  loc="lower right")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("midpoint barrier, Gaussian fill")
        ax.set_ylabel("midpoint barrier, zero fill")
        ax.set_title(f"Same mask, two fills ({len(patterns)} masks)")
        return fig
    plotting.save(draw, "fills_enumeration")


def sweep(rep):
    """Section `sweep` and figure fills_sweep."""
    recs = load("sweep")
    if not recs:
        return
    patterns = list(dict.fromkeys(tuple(r["pattern"]) for r in recs))
    fills = list(dict.fromkeys(r["fill"] for r in recs))
    barriers = defaultdict(list)
    for r in recs:
        barriers[(tuple(r["pattern"]), r["fill"])].append(r["midpoint_barrier"])
    mean = {k: float(np.mean(v)) for k, v in barriers.items()}
    delta = {(p, f): mean[(p, f)] - mean[(p, "gaussian")] for p in patterns for f in fills}
    mean_delta = {f: np.mean([delta[(p, f)] for p in patterns]) for f in fills}

    # Each network has only 4 frozen entries; fills that put the same 4 values there give identical runs.
    frozen_values, magnitude = {}, {}
    for p in patterns:
        mask = drop_one_mask(p, SMALL_NET["in_dim"])
        for f in fills:
            F = make_fill(f, mask, sweep_fill_seed(f))
            frozen_values[(p, f)] = tuple(round(float(F[row, col]), 5) for row, col in enumerate(p))
    for f in fills:
        magnitude[f] = np.mean([np.abs(frozen_values[(p, f)]).mean() for p in patterns])

    order = sorted(fills, key=lambda f: mean_delta[f])
    rep.heading("`sweep`: every fill on three masks (4-4-3 network)")
    rep.paragraph("Mean barrier over 5 seed pairs. Δ = fill − Gaussian on the same mask, averaged over the three "
                  "masks. Each fill is one draw, shared by the three masks and all seed pairs.")
    header = ["fill", "family"] + [f"mask {pattern_label(p)}" for p in patterns] + ["mean Δ", "mean \\|frozen value\\|"]
    rep.table(header,
              [[f, FAMILY_OF[f]] + [fmt_float(mean[(p, f)]) for p in patterns] +
               [f"{mean_delta[f]:+.3f}", f"{magnitude[f]:.2f}"] for f in order], text_columns=2)
    rep.paragraph("Fills that place the same four values at the frozen entries, and so give identical runs:")
    for p in patterns:
        groups = defaultdict(list)
        for f in fills:
            groups[frozen_values[(p, f)]].append(f)
        for values, members in groups.items():
            if len(members) > 1:
                rep.add(f"- mask {pattern_label(p)}: {', '.join(members)} (frozen values {values})")
    rep.add()
    rho = stats.spearmanr([magnitude[f] for f in fills], [np.mean([mean[(p, f)] for p in patterns]) for f in fills])
    rep.paragraph(f"Across the {len(fills)} fills, Spearman correlation between mean |frozen value| and mean barrier: "
                  f"ρ = {rho.statistic:+.2f} ({p_clause(rho.pvalue)}).")

    shown = [f for f in order if f != "gaussian"]
    markers = ["o", "s", "^"]

    def draw(t):
        fig, ax = plotting.plt.subplots(figsize=(5.4, 6.6))
        y = np.arange(len(shown))[::-1]
        for j, p in enumerate(patterns):
            ax.scatter([delta[(p, f)] for f in shown], y + 0.3 * (1 - j), s=24, marker=markers[j],
                       facecolors=t.surface, edgecolors=t.ink2, linewidths=1.0, zorder=3,
                       label=f"mask {pattern_label(p)}")
        for yi, f in zip(y, shown):
            ax.plot([mean_delta[f]] * 2, [yi - 0.42, yi + 0.42], color=t.ink, lw=1.6, zorder=2)
        ax.axvline(0, color=t.axis, lw=1.0, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(shown)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(shown) - 0.4)
        ax.set_xlabel("Δ barrier (fill − Gaussian), same mask")
        handles, _ = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color=t.ink, lw=1.6, label="mean of the three masks"))
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=8.3,
                  borderaxespad=0.4, columnspacing=1.2)
        ax.set_title("Each fill vs Gaussian on the 4-4-3 network", pad=40)
        return fig
    plotting.save(draw, "fills_sweep")


def hybrid(rep):
    """Section `hybrid` (no figure)."""
    recs = load("hybrid")
    if not recs:
        return
    patterns = list(dict.fromkeys(tuple(r["pattern"]) for r in recs))
    rep.heading("`hybrid`: some frozen entries Gaussian, the rest zero (4-4-3 network)")
    rep.paragraph("k of the 4 frozen entries hold Gaussian values and the rest are zero. Every subset of size k is "
                  "run (1, 4, 6, 4 and 1 subsets), 5 seed pairs each. Mean barrier ± SEM over subsets × seed pairs.")
    rows = []
    for p in patterns:
        row = [pattern_label(p)]
        for k in range(5):
            m, s = mean_sem([r["midpoint_barrier"] for r in recs if tuple(r["pattern"]) == p and r["n_gaussian"] == k])
            row.append(f"{m:.3f} ± {s:.3f}")
        rows.append(row)
    rep.table(["mask", "k = 0 (zero fill)", "k = 1", "k = 2", "k = 3", "k = 4 (Gaussian)"], rows)


# ============================================================================ larger networks

def width(rep):
    """Section `width` and figure fills_width. BH over the 10 tests in the table."""
    recs = load("width")
    if not recs:
        return
    widths = sorted({r["hidden"] for r in recs})
    fills = ["zero", "gaussian", "chebyshev"]
    barrier = defaultdict(dict)  # (hidden, fill) -> seed -> barrier
    for r in recs:
        barrier[(r["hidden"], r["fill"])][r["seed"]] = r["midpoint_barrier"]

    def test(h, f):
        seeds = sorted(set(barrier[(h, f)]) & set(barrier[(h, "gaussian")]))
        return paired_ttest([barrier[(h, f)][s] for s in seeds], [barrier[(h, "gaussian")][s] for s in seeds])

    tests = {(h, f): test(h, f) for h in widths for f in ("zero", "chebyshev")}
    benjamini_hochberg(list(tests.values()))
    n_seeds = len(barrier[(widths[0], "gaussian")])
    rep.heading("`width`: hidden width 4 to 64 (in_dim 64, 16 frozen entries per row)")
    rep.paragraph(f"Mean barrier over {n_seeds} seed pairs, fixed design. Differences are paired on seed, "
                  f"mean ± SEM; q: BH over the {len(tests)} comparisons in this table.")
    rep.table(["hidden", "zero", "Gaussian", "chebyshev", "zero − Gaussian", "q", "chebyshev − Gaussian", "q",
               "Gaussian endpoint acc"],
              [[h] + [fmt_float(np.mean(list(barrier[(h, f)].values()))) for f in fills] +
               [fmt_diff(tests[(h, "zero")]), fmt_p(tests[(h, "zero")].q),
                fmt_diff(tests[(h, "chebyshev")]), fmt_p(tests[(h, "chebyshev")].q),
                fmt_float(np.mean([endpoint_acc(r) for r in recs if r["hidden"] == h and r["fill"] == "gaussian"]))]
               for h in widths])

    def draw(t):
        fig, ax = plotting.plt.subplots(figsize=(5.2, 3.6))
        for f in fills:
            mean_and_sem = np.array([mean_sem(barrier[(h, f)].values()) for h in widths])
            mean, sem = mean_and_sem[:, 0], mean_and_sem[:, 1]
            c = fill_color(t, f)
            ax.fill_between(widths, mean - sem, mean + sem, color=c, alpha=0.14, lw=0)
            ax.plot(widths, mean, color=c, marker="o", ms=5, mec=t.surface, mew=1.2,
                    label=FILL_LABEL[f])
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels(widths)
        ax.set_xlim(widths[0] * 0.87, widths[-1] * 1.15)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        ax.set_xlabel("hidden width")
        ax.set_ylabel("midpoint barrier")
        ax.set_title(f"Barrier vs hidden width (mean ± SEM, {n_seeds} seed pairs)")
        return fig
    plotting.save(draw, "fills_width")


def cell_tests(recs, design, fills):
    """Paired test of (fill − Gaussian) barrier for every fill and cell (in_dim, n_frozen) of one
    design, at one hidden width.

    BH runs over all fills × cells passed in. Returns ({(fill, cell): TTestResult},
    {(fill, cell): {seed: difference}}).
    """
    assert len({r["hidden"] for r in recs}) == 1, "pass the records of one hidden width"
    barrier = {(r["fill"], r["in_dim"], r["n_frozen"], r["seed"]): r["midpoint_barrier"]
               for r in recs if r["design"] == design}
    cells = sorted({(in_dim, n_frozen) for _, in_dim, n_frozen, _ in barrier})
    tests, diffs = {}, {}
    for f in fills:
        for in_dim, n_frozen in cells:
            seeds = sorted(seed for (name, d, n, seed) in barrier if (name, d, n) == ("gaussian", in_dim, n_frozen)
                           and (f, in_dim, n_frozen, seed) in barrier)
            if len(seeds) < 2:
                continue
            x = [barrier[(f, in_dim, n_frozen, seed)] for seed in seeds]
            y = [barrier[("gaussian", in_dim, n_frozen, seed)] for seed in seeds]
            tests[(f, (in_dim, n_frozen))] = paired_ttest(x, y)
            diffs[(f, (in_dim, n_frozen))] = dict(zip(seeds, np.subtract(x, y)))
    benjamini_hochberg(list(tests.values()))
    return tests, diffs


def pooled_test(diffs, fill):
    """One-sample t-test, over seed pairs, of the fill's difference averaged over all cells.

    The seed pair is the replicate: every cell of a seed pair uses the same seeds,
    and cells with the same in_dim also share data, initial weights, the fill draw
    and nested masks, so cells are not independent.
    """
    per_cell = [v for (f, _), v in diffs.items() if f == fill]
    seeds = sorted(set.intersection(*(set(v) for v in per_cell)))
    return one_sample_ttest([np.mean([v[seed] for v in per_cell]) for seed in seeds])


def grid_table(rep, recs, fills, heading):
    """Fill-vs-Gaussian table for one hidden width, both designs.

    Counts use the per-cell tests (BH over fills × cells of a design). The pooled
    mean Δ is tested over seed pairs, with BH over fills × designs.
    Returns {design: (cell tests, pooled tests)}.
    """
    compared = [f for f in fills if f != "gaussian"]
    out = {}
    for design in DESIGNS:
        tests, diffs = cell_tests(recs, design, compared)
        out[design] = (tests, {f: pooled_test(diffs, f) for f in compared})
    benjamini_hochberg([out[d][1][f] for d in DESIGNS for f in compared])
    rows = []
    for f in compared:
        row = [f]
        for design in DESIGNS:
            tests, pooled = out[design]
            per_cell = [test for (name, _), test in tests.items() if name == f]
            lower = sum(test.q < Q_THRESHOLD and test.mean_diff < 0 for test in per_cell)
            higher = sum(test.q < Q_THRESHOLD and test.mean_diff > 0 for test in per_cell)
            row += [f"{lower} / {higher} / {len(per_cell) - lower - higher}", fmt_diff(pooled[f], 4),
                    fmt_p(pooled[f].q)]
        rows.append(row)
    rep.heading(heading, 3)
    rep.table(["fill"] + [h for d in DESIGNS for h in (f"{d}: lower / higher / n.s.", f"{d}: mean Δ", f"{d}: q")],
              rows)
    return out


def grid(rep):
    """Section `grid` (hidden 32) and figure fills_grid."""
    recs = load("grid")
    if not recs:
        return
    cells = sorted({(r["in_dim"], r["n_frozen"]) for r in recs})
    fills = list(dict.fromkeys(r["fill"] for r in recs))
    n_seeds = len({r["seed"] for r in recs})
    rep.heading("`grid`: hidden width 32, in_dim × frozen entries per row")
    gaussian_mean = {d: np.mean([r["midpoint_barrier"] for r in recs if r["design"] == d and r["fill"] == "gaussian"])
                     for d in DESIGNS}
    rep.paragraph(f"{len(cells)} cells (in_dim ∈ {{32, 64, 128, 256}}, n_frozen ∈ {{4, 8, 16, 32}}, "
                  f"n_frozen < in_dim), {n_seeds} seed pairs per cell, fill and design. Δ = fill − Gaussian "
                  "barrier, paired on seed. lower / higher / n.s.: cells where the fill's barrier is detectably lower "
                  f"or higher than Gaussian's (BH q < {Q_THRESHOLD} over all fills × cells of a design), or neither. "
                  f"Mean Δ: average over cells, ± SEM over the {n_seeds} seed pairs; its q is BH over fills × designs. "
                  "The seed pair is the replicate: every cell of a seed pair uses the same seeds, and cells with the "
                  "same in_dim also share data, initial weights, the fill draw and nested masks. "
                  "Mean Gaussian barrier: " + ", ".join(f"{gaussian_mean[d]:.3f} ({d})" for d in DESIGNS) + ".")
    results = grid_table(rep, recs, fills, "Fill vs Gaussian, both designs")

    rep.heading("First-layer diagnostics and energy CV", 3)
    rep.paragraph("Table: fixed design, mean over cells. frozen energy share: fraction of a W1 row's squared "
                  "norm held in its frozen entries, after training. row cosine: cosine between corresponding "
                  "rows of the two networks' W1, without permutation, over all entries (frozen ones included) or "
                  "over trainable entries only. energy CV: coefficient of variation, across the 32 rows, of the "
                  "squared norm of each row's frozen entries (0 when every row holds the same frozen energy).")
    energy_cv, delta = {}, {}  # (design, fill) -> per-cell values
    for design in DESIGNS:
        in_design = [r for r in recs if r["design"] == design]
        for in_dim, n_frozen in cells:
            in_cell = [r for r in in_design if (r["in_dim"], r["n_frozen"]) == (in_dim, n_frozen)]
            gaussian = np.mean([r["midpoint_barrier"] for r in in_cell if r["fill"] == "gaussian"])
            masks = {seed: random_mask(in_dim, 32, n_frozen, seed=design_seeds(design, seed)[0])
                     for seed in range(n_seeds)}
            for f in fills:
                if f == "zero":
                    continue
                cvs = []
                for seed, mask in masks.items():
                    energy = ((1 - mask) * make_fill(f, mask, design_seeds(design, seed)[1])).pow(2).sum(1).numpy()
                    cvs.append(energy.std() / energy.mean())
                energy_cv.setdefault((design, f), []).append(np.mean(cvs))
                delta.setdefault((design, f), []).append(
                    np.mean([r["midpoint_barrier"] for r in in_cell if r["fill"] == f]) - gaussian)
    fixed = [r for r in recs if r["design"] == "fixed"]

    def fixed_mean(f, field):
        return np.mean([r[field] for r in fixed if r["fill"] == f])

    rep.table(["fill", "barrier", "endpoint acc", "frozen energy share", "row cosine (all entries)",
               "row cosine (trainable)", "energy CV"],
              [[f, fmt_float(fixed_mean(f, "midpoint_barrier")),
                fmt_float(np.mean([endpoint_acc(r) for r in fixed if r["fill"] == f])),
                fmt_float(fixed_mean(f, "frozen_energy_share")), fmt_float(fixed_mean(f, "w1_cos")),
                fmt_float(fixed_mean(f, "w1_cos_trainable")),
                fmt_float(np.mean(energy_cv[("fixed", f)]) if ("fixed", f) in energy_cv else None)]
               for f in fills])
    names = [f for f in fills if f not in ("zero", "gaussian")]
    sentences = []
    for design in DESIGNS:
        rho_fill = stats.spearmanr([np.mean(energy_cv[(design, f)]) for f in names],
                                   [np.mean(delta[(design, f)]) for f in names])
        rho_cell = stats.spearmanr([c for f in names for c in energy_cv[(design, f)]],
                                   [x for f in names for x in delta[(design, f)]])
        sentences.append(f"{design} design: ρ = {rho_fill.statistic:+.2f} across fills ({p_clause(rho_fill.pvalue)}) "
                         f"and ρ = {rho_cell.statistic:+.2f} across the {len(names) * len(cells)} fill × cell "
                         "combinations (descriptive only: combinations within a cell or in_dim are not independent)")
    rep.paragraph(f"Spearman correlation between energy CV and Δ, for the {len(names)} fills other than zero and "
                  "Gaussian (energy CV averaged over the draws each design used): " + "; ".join(sentences) + ".")
    ratio = []
    for in_dim, n_frozen in cells:
        mask = random_mask(in_dim, 32, n_frozen, seed=design_seeds("fixed", 0)[0])
        fill_seed = design_seeds("fixed", 0)[1]
        frozen_energy = {f: ((1 - mask) * make_fill(f, mask, fill_seed)).pow(2).sum().item()
                         for f in ("gaussian", "gaussian_equalised")}
        ratio.append(frozen_energy["gaussian_equalised"] / frozen_energy["gaussian"])
    rep.paragraph("gaussian_equalised rescales the cell's Gaussian draw row by row, which also changes the total "
                  f"frozen energy of the layer: by {100 * (min(ratio) - 1):+.0f}% to {100 * (max(ratio) - 1):+.0f}% "
                  "across the cells of the fixed design.")

    real = [f for f in fills if f not in PLUS_MINUS_ONE | {"gaussian", "zero", "gaussian_equalised"}]
    rep.heading("±1-valued vs real-valued fills", 3)
    rep.paragraph("Cells where a fill is detectably lower or higher than Gaussian (per-cell tests above).")
    rows = []
    for design in DESIGNS:
        tests = results[design][0]
        for group in (sorted(PLUS_MINUS_ONE), real):
            group_tests = [tests[(f, c)] for f in group for c in cells if (f, c) in tests]
            rows.append([design, ", ".join(group), len(group_tests),
                         sum(test.q < Q_THRESHOLD and test.mean_diff < 0 for test in group_tests),
                         sum(test.q < Q_THRESHOLD and test.mean_diff > 0 for test in group_tests)])
    rep.table(["design", "fills", "fill × cell", "lower", "higher"], rows, text_columns=2)

    shown = sorted([f for f in fills if f not in ("gaussian", "zero")],
                   key=lambda f: results["fixed"][1][f].mean_diff)

    def draw(t):
        fig, axes = plotting.plt.subplots(1, 2, figsize=(9.4, 4.7), sharey=True, sharex=True,
                                          gridspec_kw={"wspace": 0.08})
        for ax, design in zip(axes, DESIGNS):
            tests, pooled = results[design]
            for i, f in enumerate(shown[::-1]):
                fill_tests = [tests[(f, c)] for c in cells if (f, c) in tests]
                d = np.array([test.mean_diff for test in fill_tests])
                detectable = np.array([test.q < Q_THRESHOLD for test in fill_tests])
                jitter = (np.arange(len(d)) % 5 - 2) * 0.07
                ax.scatter(d[~detectable], i + jitter[~detectable], s=16, facecolors=t.surface, edgecolors=t.muted,
                           linewidths=0.8, zorder=2)
                ax.scatter(d[detectable], i + jitter[detectable], s=40,
                           color=np.where(d[detectable] < 0, t.div_neg, t.div_pos).tolist(),
                           edgecolors=t.surface, linewidths=0.7, zorder=5)
                ax.plot([pooled[f].mean_diff] * 2, [i - 0.38, i + 0.38], color=t.ink, lw=1.6, zorder=1.5)
            ax.axvline(0, color=t.axis, lw=1.0, zorder=1)
            ax.set_yticks(range(len(shown)))
            ax.set_yticklabels(shown[::-1])
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Δ barrier (fill − Gaussian), per cell")
            ax.set_title("fixed: one mask and fill per cell" if design == "fixed"
                         else "resampled: mask and fill redrawn per seed pair")
        handles = [Line2D([], [], marker="o", ls="", ms=6.5, color=t.div_neg, mec=t.surface, label="lower, q < 0.05"),
                   Line2D([], [], marker="o", ls="", ms=6.5, color=t.div_pos, mec=t.surface, label="higher, q < 0.05"),
                   Line2D([], [], marker="o", ls="", ms=4.5, mfc=t.surface, mec=t.muted, label="not detectable"),
                   Line2D([], [], color=t.ink, lw=1.6, label=f"mean over {len(cells)} cells")]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        plotting.suptitle(fig, t, f"Fill vs Gaussian at hidden width 32 ({len(cells)} cells × {n_seeds} seed pairs)")
        fig.subplots_adjust(left=0.14, right=0.99, top=0.86, bottom=0.2)
        return fig
    plotting.save(draw, "fills_grid")


def grid_wide(rep):
    """Section `grid_wide` (no figure)."""
    recs = load("grid_wide")
    grid_fixed = [r for r in load("grid") if r["design"] == "fixed"]
    if not recs:
        return
    widths = sorted({r["hidden"] for r in recs})
    rep.heading("`grid_wide`: hidden widths " + " and ".join(map(str, widths)))
    rep.paragraph("Cells with in_dim ∈ {64, 128, 256} and n_frozen ∈ {8, 16, 32}, restricted to hidden ≤ in_dim "
                  "(basis fills need it); 10 seed pairs. Same tests and corrections as in `grid`, run separately "
                  "for each width.")
    rows = []
    for h in widths:
        rs = [r for r in recs if r["hidden"] == h]
        cells = sorted({(r["in_dim"], r["n_frozen"]) for r in rs})
        at_h = [r for r in rs if r["design"] == "fixed" and r["fill"] == "gaussian"]
        at_32 = [r for r in grid_fixed if r["fill"] == "gaussian" and (r["in_dim"], r["n_frozen"]) in cells]
        rows.append([h, len(cells), fmt_float(np.mean([r["midpoint_barrier"] for r in at_h])),
                     fmt_float(np.mean([r["midpoint_barrier"] for r in at_32])),
                     fmt_float(np.mean([endpoint_acc(r) for r in at_h])),
                     fmt_float(np.mean([endpoint_acc(r) for r in at_32]))])
    rep.paragraph("Gaussian fill, fixed design, mean over the cells each width covers:")
    rep.table(["hidden", "cells", "barrier", "barrier at hidden 32, same cells", "endpoint acc",
               "endpoint acc at hidden 32"], rows)
    for h in widths:
        rs = [r for r in recs if r["hidden"] == h]
        cells = sorted({(r["in_dim"], r["n_frozen"]) for r in rs})
        grid_table(rep, rs, list(dict.fromkeys(r["fill"] for r in rs)), f"hidden {h} ({len(cells)} cells)")


def controls(rep):
    """Section `controls` and figure fills_controls."""
    ctl = load("controls")
    grid_fixed = [r for r in load("grid") if r["design"] == "fixed"]
    if not ctl or not grid_fixed:
        return
    records = ctl + grid_fixed
    cells = sorted({(r["in_dim"], r["n_frozen"]) for r in grid_fixed})
    conditions = ["zero", "gaussian", "gaussian_unshared"]

    def mean_of(in_dim, n_frozen, fill, fn=lambda r: r["midpoint_barrier"]):
        return np.mean([fn(r) for r in records if (r["in_dim"], r["n_frozen"], r["fill"]) == (in_dim, n_frozen, fill)])

    def unmasked(in_dim, fn=lambda r: r["midpoint_barrier"]):
        return mean_of(in_dim, 0, "unmasked", fn)

    def w1_cos_all(r):
        return r["w1_cos"]

    rep.heading("`controls`: unmasked networks, and a fill not shared by A and B")
    rep.paragraph("Hidden 32, fixed design, 10 seed pairs per entry. 'unmasked': no frozen entries. 'Gaussian, "
                  "unshared': B gets its own Gaussian fill draw instead of A's; the midpoint uses the average of the "
                  "two fills. Frozen fraction = n_frozen / in_dim. Row cosine: between A's and B's W1 rows, without "
                  "permutation, over all entries (frozen ones included). acc: mean endpoint test accuracy.")
    rep.table(["in_dim", "n_frozen", "frozen fraction", "unmasked", "zero", "Gaussian, shared", "Gaussian, unshared",
               "row cosine (all entries), shared", "row cosine (all entries), unshared", "acc, unmasked", "acc, zero",
               "acc, Gaussian shared", "acc, Gaussian unshared"],
              [[in_dim, n_frozen, f"{n_frozen / in_dim:.3f}", fmt_float(unmasked(in_dim)),
                fmt_float(mean_of(in_dim, n_frozen, "zero")), fmt_float(mean_of(in_dim, n_frozen, "gaussian")),
                fmt_float(mean_of(in_dim, n_frozen, "gaussian_unshared")),
                fmt_float(mean_of(in_dim, n_frozen, "gaussian", w1_cos_all)),
                fmt_float(mean_of(in_dim, n_frozen, "gaussian_unshared", w1_cos_all)),
                fmt_float(unmasked(in_dim, endpoint_acc)), fmt_float(mean_of(in_dim, n_frozen, "zero", endpoint_acc)),
                fmt_float(mean_of(in_dim, n_frozen, "gaussian", endpoint_acc)),
                fmt_float(mean_of(in_dim, n_frozen, "gaussian_unshared", endpoint_acc))]
               for in_dim, n_frozen in cells])
    fraction = np.array([n_frozen / in_dim for in_dim, n_frozen in cells])
    rho = stats.spearmanr(fraction, [mean_of(in_dim, n_frozen, "gaussian") for in_dim, n_frozen in cells])
    at_or_above = {f: sum(mean_of(in_dim, n_frozen, f) >= unmasked(in_dim) for in_dim, n_frozen in cells)
                   for f in ("zero", "gaussian_unshared")}
    below = sum(mean_of(in_dim, n_frozen, "gaussian") < unmasked(in_dim) for in_dim, n_frozen in cells)
    zero_cost = max(unmasked(in_dim, endpoint_acc) - mean_of(in_dim, n_frozen, "zero", endpoint_acc)
                    for in_dim, n_frozen in cells)
    rep.paragraph(f"Barrier at or above the unmasked network's (same in_dim): zero fill in {at_or_above['zero']} of "
                  f"{len(cells)} cells, unshared Gaussian in {at_or_above['gaussian_unshared']} of {len(cells)}. "
                  f"Shared Gaussian below it in {below} of {len(cells)}. Shared Gaussian: Spearman correlation between "
                  f"frozen fraction and barrier ρ = {rho.statistic:+.2f} (p = {rho.pvalue:.1e}). Largest endpoint "
                  f"accuracy loss of zero fill against the unmasked network: {100 * zero_cost:.1f} points.")

    offsets = {"zero": 2 ** -0.09, "gaussian": 1.0, "gaussian_unshared": 2 ** 0.09}
    CONTROL_LABEL = {"zero": "zero fill", "gaussian": "Gaussian, shared", "gaussian_unshared": "Gaussian, unshared"}

    def draw(t):
        fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(9.4, 3.7), gridspec_kw={"wspace": 0.25})
        panels = ((ax1, lambda r: r["midpoint_barrier"], 1, "midpoint barrier"),
                  (ax2, endpoint_acc, 100, "endpoint test accuracy (%)"))
        for ax, fn, scale, ylabel in panels:
            band = [scale * unmasked(in_dim, fn) for in_dim, _ in cells]
            ax.axhspan(min(band), max(band), color=t.reference, alpha=0.14, lw=0, zorder=0)
            ax.text(0.99, (min(band) + max(band)) / 2, "unmasked ", transform=ax.get_yaxis_transform(),
                    color=t.ink2, fontsize=8.5, va="center", ha="right")
            for f in conditions:
                values = [scale * mean_of(in_dim, n_frozen, f, fn) for in_dim, n_frozen in cells]
                ax.scatter(fraction * offsets[f], values, s=26, color=fill_color(t, f), edgecolors=t.surface,
                           linewidths=0.8, zorder=3,
                           label=CONTROL_LABEL[f])
            ax.set_xscale("log", base=2)
            ticks = [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]
            ax.set_xticks(ticks)
            ax.set_xticklabels(["1/64", "1/32", "1/16", "1/8", "1/4", "1/2"])
            ax.set_xlabel("fraction of W1 entries frozen")
            ax.set_ylabel(ylabel)
        ax1.set_ylim(bottom=0)
        ax1.set_title("Barrier vs frozen fraction")
        ax2.set_title("Accuracy vs frozen fraction")
        ax1.legend(loc="lower left")
        return fig
    plotting.save(draw, "fills_controls")


def main():
    rep = Report("Fill study: generated tables", "scripts/analyze_fills.py")
    for section in (enumeration, sweep, hybrid, width, grid, grid_wide, controls):
        section(rep)
    rep.write(FILL_DIR / "summary.md")


if __name__ == "__main__":
    main()
