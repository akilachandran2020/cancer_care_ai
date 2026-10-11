"""Step 5 - Figure 1 of the synopsis: survey-weighted health-scheme coverage among hospitalised cancer
patients, by sector, 2017-18 (NSS 75th round) vs 2025 (NSO 80th round).
Needs both episode files from step 1. Episodes of members who died or left the household (serial 91/92)
are excluded because their coverage was not recorded."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PROCESSED, OUTPUTS

ROUNDS = {"2017-18": "cancer_episodes_nss75.csv", "2025": "cancer_episodes_nss80.csv"}

def coverage():
    rows = []
    for label, f in ROUNDS.items():
        path = PROCESSED / f
        if not path.exists():
            raise SystemExit(f"Missing {f} - run 01_build_cancer_episodes.py for both rounds first.")
        d = pd.read_csv(path).dropna(subset=["insured", "weight"])
        for urban, g in d.groupby("sector_urban"):
            rows.append({"survey": label, "sector": "Urban" if urban == 1 else "Rural", "n": len(g),
                         "coverage_pct": round(100 * np.average(g["insured"], weights=g["weight"]), 1)})
    return pd.DataFrame(rows)

def main():
    OUTPUTS.mkdir(exist_ok=True)
    t = coverage()
    t.to_csv(OUTPUTS / "figure1_coverage.csv", index=False)
    print(t.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=300)
    x, w = np.arange(2), 0.34
    for i, (label, colour) in enumerate([("2017-18", "#1F4E79"), ("2025", "#ED7D31")]):
        v = t[t.survey == label].set_index("sector").loc[["Rural", "Urban"], "coverage_pct"].values
        bars = ax.bar(x + (i - 0.5) * w, v, w, color=colour, label=label, zorder=3)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x, ["Rural", "Urban"]); ax.set_ylim(0, 100)
    ax.set_ylabel("Cancer patients covered by a\nhealth-expenditure scheme (%)")
    ax.grid(axis="y", color="#D5DCE6", zorder=0); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(); plt.savefig(OUTPUTS / "figure1_coverage.png")

if __name__ == "__main__":
    main()
