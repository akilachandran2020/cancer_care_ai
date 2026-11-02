"""Step 16 - Extra figures for the final report: RQ2 odds ratios from the final model (unweighted and design-based) and the
RQ4 fall in CHE at different thresholds. Reads outputs/final_models.json, design_based_models.json, price_threshold_sensitivity.json."""
import json
import numpy as np
import matplotlib.pyplot as plt
import plot_style as ps
from config import OUTPUTS

R = OUTPUTS / "report"
NAMES = {"facility_public": "Public hospital", "insured": "Scheme coverage", "outside_state": "Treated outside home state",
         "surgery_received": "Surgery", "log_stay": "ln(length of stay), per log unit", "age": "Age (per year)", "female": "Female",
         "sector_urban": "Urban", "C(mpce_quintile)[T.2]": "MPCE Q2 (vs. Q1)", "C(mpce_quintile)[T.3]": "MPCE Q3 (vs. Q1)",
         "C(mpce_quintile)[T.4]": "MPCE Q4 (vs. Q1)", "C(mpce_quintile)[T.5]": "MPCE Q5 (vs. Q1)", "C(social_group)[T.2]": "SC (vs. ST)",
         "C(social_group)[T.3]": "OBC (vs. ST)", "C(social_group)[T.9]": "Other (vs. ST)"}


def main():
    fm = json.load(open(OUTPUTS / "final_models.json"))["rq2_final"]["terms"]
    db = json.load(open(OUTPUTS / "design_based_models.json"))["rq2"]["final_log_stay"]["listwise_complete_cases"]["params"]
    keys = sorted(NAMES, key=lambda k: fm[k]["or"])
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for off, lab, col, get in ((-.17, "Unweighted (episode-level)", ps.NAVY, lambda k: (fm[k]["or"], fm[k]["ci_low"], fm[k]["ci_high"])),
                               (.17, "Survey-weighted (design-based)", ps.ORANGE, lambda k: (db[k]["or"], db[k]["or_ci_low"], db[k]["or_ci_high"]))):
        for i, k in enumerate(keys):
            o, lo, hi = get(k)
            ax.plot([lo, hi], [i + off] * 2, color=col, lw=1.6); ax.plot(o, i + off, "o", color=col, ms=5, label=lab if i == 0 else None)
    ax.axvline(1, color="#555555", lw=1); ax.set_xscale("log")
    ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys]); ax.set_xlabel("Odds ratio for CHE (log scale, 95% CI)")
    ax.legend(frameon=False, loc="lower right"); ps.grid(ax, "x")
    ps.save(fig, R / "fig17_rq2_final_odds_ratios.png")
    th = json.load(open(OUTPUTS / "price_threshold_sensitivity.json"))["B_threshold_sensitivity"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); x = np.arange(len(th))
    f = [t["fall_points_weighted"] for t in th]; se = [1.96 * t["se_points"] for t in th]
    ax.bar(x, f, color=[ps.NAVY if t["p_design_based"] < .05 else ps.LIGHT for t in th], yerr=se, capsize=4, ecolor="#444444")
    for i, t in enumerate(th):
        ax.text(i, f[i] + se[i] + .5, f"p = {t['p_design_based']:.3f}", ha="center", fontsize=8.5)
    ax.axhline(15, color=ps.ORANGE, ls="--", lw=1.2, label="15-point target"); ax.axhline(0, color="#555555", lw=.8)
    ax.set_xticks(x, [f"{int(100 * t['threshold'])}%" for t in th]); ax.set_xlabel("CHE threshold (share of annual consumption)")
    ax.set_ylabel("Fall in weighted CHE rate,\n2017-18 to 2025 (points)"); ax.legend(frameon=False); ps.grid(ax)
    ps.save(fig, R / "fig18_rq4_threshold_sensitivity.png")
    print("wrote fig17, fig18")


if __name__ == "__main__":
    main()
