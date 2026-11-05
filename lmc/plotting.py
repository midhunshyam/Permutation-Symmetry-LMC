"""Shared figure style. Every figure is drawn twice, for a light and a dark page.

A figure is a function `draw(t) -> matplotlib Figure`, where `t` is a Theme. `save`
writes figures/<name>.png and figures/<name>_dark.png; the Markdown files choose
between them with a <picture> element.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DPI = 200


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str            # titles, primary text
    ink2: str           # labels, secondary text
    muted: str          # tick marks, de-emphasised marks
    grid: str
    axis: str           # axis lines and zero lines
    series: tuple       # categorical slots, always assigned in this order
    reference: str      # baselines and reference marks
    div_neg: str        # diverging pole for "lower than reference"
    div_pos: str        # diverging pole for "higher than reference"
    div_mid: str


LIGHT = Theme(
    "light", surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    reference="#898781", div_neg="#256abf", div_pos="#d03b3b", div_mid="#f0efec",
)
DARK = Theme(
    "dark", surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    reference="#898781", div_neg="#3987e5", div_pos="#e66767", div_mid="#383835",
)


def apply(t: Theme):
    plt.rcParams.update({
        "figure.facecolor": t.surface, "axes.facecolor": t.surface, "savefig.facecolor": t.surface,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 9, "axes.labelsize": 9.5, "legend.fontsize": 8.8,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": t.ink, "axes.labelcolor": t.ink2, "axes.titlecolor": t.ink,
        "xtick.color": t.muted, "ytick.color": t.muted, "xtick.labelcolor": t.ink2, "ytick.labelcolor": t.ink2,
        "axes.edgecolor": t.axis, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t.grid, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 5,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 6,
        "legend.frameon": False, "legend.handlelength": 1.4,
    })


def diverging_cmap(t: Theme):
    return LinearSegmentedColormap.from_list(f"div_{t.name}", [t.div_neg, t.div_mid, t.div_pos])


def suptitle(fig, t: Theme, text: str):
    """Figure-level title, styled like an axes title."""
    fig.suptitle(text, x=0.01, ha="left", fontsize=plt.rcParams["axes.titlesize"],
                 fontweight=plt.rcParams["axes.titleweight"], color=t.ink)


def save(draw, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    for t, suffix in ((LIGHT, ""), (DARK, "_dark")):
        apply(t)
        fig = draw(t)
        fig.savefig(FIG_DIR / f"{name}{suffix}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
