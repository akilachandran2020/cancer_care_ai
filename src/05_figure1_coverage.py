"""Step 5 - Figure 1 of the synopsis: survey-weighted health-scheme coverage among hospitalised cancer
patients, by sector, 2017-18 (NSS 75th round) vs 2025 (NSO 80th round).
Needs both episode files from step 1. Episodes of members who died or left the household (serial 91/92)
are excluded because their coverage was not recorded."""
import numpy as np, pandas as pd
from config import PROCESSED, OUTPUTS
from plot_style import plt, NAVY, ORANGE, grid, pct_axis, save

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
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x, w = np.arange(2), 0.36
    for i, (label, colour) in enumerate([("2017-18", NAVY), ("2025", ORANGE)]):
        v = t[t.survey == label].set_index("sector").loc[["Rural", "Urban"], "coverage_pct"].values
        bars = ax.bar(x + (i - 0.5) * w, v, w, color=colour, label=label.replace("-", "–"), zorder=3)
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_xticks(x, ["Rural", "Urban"]); ax.set_xlabel("Place of residence")
    ax.set_ylabel("Patients covered by a\nhealth-expenditure scheme (%)"); pct_axis(ax); grid(ax)
    ax.legend(title="Survey round", frameon=False, loc="upper left")
    save(fig, OUTPUTS / "figure1_coverage.png")

if __name__ == "__main__":
    main()
