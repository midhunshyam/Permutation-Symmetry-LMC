"""Explanatory diagrams for the README and reports (no experiment data).

    python scripts/make_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmc import plotting  # noqa: E402


def barrier(t):
    """Loss along the straight line between two trained networks, and how the barrier is read off it."""
    a = np.linspace(0, 1, 201)
    L0, L1 = 0.10, 0.12
    chord = L0 + (L1 - L0) * a
    bump = chord + 0.55 * np.sin(np.pi * a) ** 2
    flat = chord - 0.02 * np.sin(np.pi * a)
    fig, ax = plotting.plt.subplots(figsize=(5.6, 3.3))
    ax.plot(a, chord, color=t.muted, lw=1.0, ls=(0, (4, 3)))
    ax.plot(a, bump, color=t.series[1], label="not linearly connected")
    ax.plot(a, flat, color=t.series[0], label="linearly connected")
    for x, y, name in ((0, L0, r"$\theta_A$"), (1, L1, r"$\theta_B$")):
        ax.scatter([x], [y], s=46, color=t.ink, edgecolors=t.surface, linewidths=1.5, zorder=5)
        ax.text(x, y - 0.06, name, ha="center", va="top", color=t.ink, fontsize=10.5)
    mid_chord, mid_bump = chord[100], bump[100]
    ax.add_patch(FancyArrowPatch((0.5, mid_chord), (0.5, mid_bump), arrowstyle="<->", mutation_scale=9,
                                 color=t.ink2, lw=1.1, zorder=4))
    ax.text(0.525, (mid_chord + mid_bump) / 2, "midpoint\nbarrier", color=t.ink2, fontsize=9, va="center")
    ax.text(0.83, chord[166] + 0.025, "chord", color=t.ink2, fontsize=8.5, va="bottom", ha="center")
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 0.78)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])
    ax.set_xlabel(r"$\alpha$ in $(1-\alpha)\,\theta_A + \alpha\,\theta_B$")
    ax.set_ylabel("test loss")
    ax.legend(loc="upper left")
    ax.grid(False)
    return fig


def permutation(t):
    """Swapping two hidden units gives the same function; averaging the two copies does not."""
    fig, axes = plotting.plt.subplots(1, 3, figsize=(9.0, 3.2))
    x_in, x_hidden, x_out = 0.0, 1.0, 2.0
    ys = [2.0, 1.0, 0.0]  # top to bottom: units 1, 2, 3
    colours = [t.series[0], t.series[1], t.series[2]]
    label_color = t.ink if t.name == "light" else t.surface  # dark text on the coloured discs

    def draw_net(ax, title, order):
        """order[k] is the unit (0-based) drawn at position k; a pair (i, j) is a blend of units i and j."""
        for k, unit in enumerate(order):
            edge = t.axis if isinstance(unit, tuple) else colours[unit]
            for yi in ys:
                ax.plot([x_in, x_hidden], [yi, ys[k]], color=edge, lw=1.0, alpha=0.75, zorder=1)
            ax.plot([x_hidden, x_out], [ys[k], 1.0], color=edge, lw=1.0, alpha=0.75, zorder=1)
        for yi in ys:
            ax.add_patch(Circle((x_in, yi), 0.17, color=t.muted, zorder=3))
        for k, unit in enumerate(order):
            if isinstance(unit, tuple):
                ax.add_patch(Wedge((x_hidden, ys[k]), 0.25, 90, 270, color=colours[unit[0]], zorder=3))
                ax.add_patch(Wedge((x_hidden, ys[k]), 0.25, -90, 90, color=colours[unit[1]], zorder=3))
                label = f"{unit[0] + 1}+{unit[1] + 1}"
            else:
                ax.add_patch(Circle((x_hidden, ys[k]), 0.25, color=colours[unit], zorder=3))
                label = str(unit + 1)
            ax.text(x_hidden, ys[k], label, ha="center", va="center", color=label_color, fontsize=8.5,
                    fontweight="bold", zorder=4)
        ax.add_patch(Circle((x_out, 1.0), 0.2, color=t.ink2, zorder=3))
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(-0.75, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9.5, loc="center", fontweight="normal")

    draw_net(axes[0], "Network A", [0, 1, 2])
    draw_net(axes[1], "Network B: A with units 1 and 3 swapped", [2, 1, 0])
    draw_net(axes[2], "Average of A and B", [(0, 2), 1, (2, 0)])
    left, right, avg = axes[0].get_position(), axes[1].get_position(), axes[2].get_position()
    for x, text in (((left.x0 + right.x1) / 2, "A and B compute the same function"),
                    ((avg.x0 + avg.x1) / 2, "units 1 and 3 are blended: the loss rises")):
        fig.text(x, 0.1, text, ha="center", color=t.ink2, fontsize=9)
    return fig


def masked_layer(t):
    """W1 = M * W + (1 - M) * F for a 5 x 8 layer with 2 frozen entries per row."""
    rng = np.random.default_rng(3)
    rows, cols = 5, 8
    M = np.ones((rows, cols))
    for r in range(rows):
        M[r, rng.choice(cols, 2, replace=False)] = 0
    W = rng.normal(0, 0.18, (rows, cols))
    F = rng.normal(0, 1.0, (rows, cols))
    W1 = M * W + (1 - M) * F
    cmap = plotting.diverging_cmap(t)
    fig, axes = plotting.plt.subplots(1, 4, figsize=(9.6, 1.6), gridspec_kw={"wspace": 0.35})
    panels = [("mask $M$", None), ("trainable weights $W$", W), ("fill $F$ (fixed)", F),
              (r"$W_1 = M \odot W + (1-M) \odot F$", W1)]
    for ax, (title, values) in zip(axes, panels):
        if values is None:
            ax.imshow(np.zeros((rows, cols)), cmap=ListedColormap([t.div_mid]), vmin=0, vmax=1)
        else:
            ax.imshow(values, cmap=cmap, vmin=-1.6, vmax=1.6)
        for r in range(rows):
            for c in range(cols):
                if M[r, c] == 0:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=values is None, fc=t.ink2,
                                           ec=t.ink, lw=1.1, clip_on=False, zorder=3))
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color=t.surface, lw=1.5)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.55, cols - 0.45)
        ax.set_ylim(rows - 0.45, -0.55)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=9.5, loc="center", fontweight="normal")
    return fig


if __name__ == "__main__":
    plotting.save(barrier, "diagram_barrier")
    plotting.save(permutation, "diagram_permutation")
    plotting.save(masked_layer, "diagram_masked_layer")
    print("wrote diagrams to", plotting.FIG_DIR)
