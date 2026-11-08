"""Shared figure style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, ORANGE, LIGHT, MID = "#1F4E79", "#ED7D31", "#BDD7EE", "#5B9BD5"   # palette
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
    """Axis from 0 to a multiple of step."""
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
    """Secondary axis with rupee labels for ln(OOPE + 1)."""
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


# Readable subgroup labels for the interval-coverage figure (codes follow the data dictionary)
SUBGROUP_LABELS = {
    "sector=0": ("Sector", "Rural"), "sector=1": ("Sector", "Urban"),
    "social_group=1": ("Social group", "Scheduled Tribe"), "social_group=2": ("Social group", "Scheduled Caste"),
    "social_group=3": ("Social group", "OBC"), "social_group=9": ("Social group", "Other"),
    "sex=male": ("Sex", "Male"), "sex=female": ("Sex", "Female"), "sex=unknown": ("Sex", "Unknown (imputed)"),
    "hospital=1": ("Hospital", "Public"), "hospital=2": ("Hospital", "Charitable"), "hospital=3": ("Hospital", "Private"),
    "quintile=1": ("MPCE quintile", "Q1 (poorest)"), "quintile=2": ("MPCE quintile", "Q2"),
    "quintile=3": ("MPCE quintile", "Q3"), "quintile=4": ("MPCE quintile", "Q4"), "quintile=5": ("MPCE quintile", "Q5 (richest)"),
}


def coverage_by_group_plot(cov, path, lo=75, hi=85, nominal=80, min_n=30):
    """Horizontal bars: subgroups on the y-axis, conformal 80% interval coverage (%) on the x-axis."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    keys = [k for k in SUBGROUP_LABELS if k in cov and cov[k]["n"] >= min_n]
    vals = [100 * cov[k]["coverage"] for k in keys]
    labels = [f"{SUBGROUP_LABELS[k][0]}: {SUBGROUP_LABELS[k][1]} (n = {cov[k]['n']})" for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(keys))
    ax.axvspan(lo, hi, color=LIGHT, alpha=.45, zorder=0)
    bars = ax.barh(y, vals, .7, color=[NAVY if lo <= v <= hi else ORANGE for v in vals], zorder=3)
    ax.axvline(nominal, color="#555555", lw=1, ls="--", zorder=4)
    for t in ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8):
        t.set_bbox(dict(facecolor="white", edgecolor="none", pad=.6)); t.set_zorder(5)
    groups = [SUBGROUP_LABELS[k][0] for k in keys]           # thin separators between subgroup families
    for i in range(1, len(keys)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - .5, color="#BFBFBF", lw=.6, zorder=1)
    ax.set_yticks(y, labels, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(40, 100); ax.set_xticks(range(40, 101, 10))
    ax.set_xlabel("Test episodes with actual cost inside the 80% interval (%)"); ax.set_ylabel("Subgroup")
    grid(ax, "x")
    handles = [Patch(color=LIGHT, alpha=.45, label=f"{lo}%–{hi}% target band"),
               Line2D([], [], color="#555555", lw=1, ls="--", label=f"Nominal {nominal}%"),
               Patch(color=NAVY, label="Inside target band"), Patch(color=ORANGE, label="Outside target band")]
    ax.legend(handles=handles, frameon=False, ncol=4, fontsize=8.5, loc="lower center", bbox_to_anchor=(.42, 1.0))
    save(fig, path)
