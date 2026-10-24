"""Shared figure style for every report figure (APA-ready: no title inside the image, labelled axes with units,
bar axes from zero, evenly spaced ticks that end on a labelled value, rupee symbol throughout)."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, ORANGE, LIGHT, MID = "#1F4E79", "#ED7D31", "#BDD7EE", "#5B9BD5"   # report palette: blues + one orange accent only
GREY = MID   # kept as an alias so no grey is ever drawn; "secondary" marks are mid-blue
from matplotlib.colors import LinearSegmentedColormap
BLUE_ORANGE = LinearSegmentedColormap.from_list("blue_orange", [ORANGE, "#FFFFFF", NAVY])
FAC = {1: "Public", 2: "Charitable/NGO", 3: "Private"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

def nice_axis(ax, vmax, step, axis="y", headroom=1.08):
    """Axis from 0 to the first multiple of `step` above vmax * headroom, one tick per step."""
    top = step * math.ceil(vmax * headroom / step)
    ticks = np.round(np.arange(0, top + step / 2, step), 6)
    if axis == "y":
        ax.set_ylim(0, top); ax.set_yticks(ticks)
    else:
        ax.set_xlim(0, top); ax.set_xticks(ticks)

def pct_axis(ax):
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))

RUPEE = [(0, "0"), (1e3, "₹1k"), (1e5, "₹1 lakh")]
RUPEE_FULL = [(0, "0"), (100, "₹100"), (1e3, "₹1k"), (1e4, "₹10k"), (1e5, "₹1 lakh"), (1e6, "₹10 lakh")]

def rupee_axis(ax, which="x"):
    """Second axis that labels ln(OOPE + 1) values as rupees."""
    ticks = RUPEE if which == "x" else RUPEE_FULL
    pos, lab = [math.log1p(v) for v, _ in ticks], [l for _, l in ticks]
    if which == "x":
        sec = ax.secondary_xaxis("top"); sec.set_xticks(pos, lab, fontsize=8); sec.set_xlabel("Rupee equivalent", fontsize=9)
        ax.spines["top"].set_visible(True)
    else:
        sec = ax.secondary_yaxis("right"); sec.set_yticks(pos, lab, fontsize=8); sec.set_ylabel("Rupee equivalent", fontsize=9)
        ax.spines["right"].set_visible(True)
    return sec

def save(fig, path):
    fig.savefig(path); plt.close(fig)
