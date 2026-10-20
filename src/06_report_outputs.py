"""Step 6 - Aggregate tables and figures for the interim report (cleaning log, EDA, model results).
Run after steps 1 and 3. Writes ONLY aggregate outputs (no episode-level rows), so everything in
outputs/report/ can be published on GitHub without breaching MoSPI's terms of use."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RAW, PROCESSED, OUTPUTS, FILES, CASE_KEY, CANCER_CODE

OUT = OUTPUTS / "report"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY = "#1F4E79", "#ED7D31", "#7F7F7F"
FAC = {1: "Public", 2: "Charitable/NGO", 3: "Private"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png"); plt.close(fig)

LABEL = {"log_oop": "ln(OOPE + 1)", "stay_days": "Length of stay", "age": "Age", "hh_size": "Household size",
         "mpce_quintile": "MPCE quintile", "facility_public": "Public hospital", "surgery_received": "Surgery",
         "insured": "Scheme coverage", "outside_state": "Outside home state", "outside_district": "Outside home district",
         "che": "CHE", "ward_type": "Ward type", "female": "Female", "member_deceased": "Died or left household",
         "sector_urban": "Urban", "social_group": "Social group"}
RUPEE_TICKS = [(0, "0"), (100, "Rs 100"), (1e3, "Rs 1k"), (1e4, "Rs 10k"), (1e5, "Rs 1 lakh"), (1e6, "Rs 10 lakh")]

def rupee_axis(ax, which="x"):
    """Second axis that labels ln(OOPE + 1) values in rupees, so log units are readable."""
    sec = ax.secondary_xaxis("top") if which == "x" else ax.secondary_yaxis("right")
    pos, lab = [np.log1p(v) for v, _ in RUPEE_TICKS], [l for _, l in RUPEE_TICKS]
    (sec.set_xticks if which == "x" else sec.set_yticks)(pos, lab)
    sec.tick_params(labelsize=7, colors="#404040")
    return sec

def wmean(x, w):
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m]))

summary = {}
d = pd.read_csv(PROCESSED / "cancer_episodes_nss75.csv")
summary["n_episodes_2017"] = len(d)

# ---------- Cleaning log ----------
hosp_raw = pd.read_csv(RAW.parent / "nss75" / f"{FILES['hosp']}.csv", low_memory=False)
summary["n_hospitalisation_records_2017"] = len(hosp_raw)
cancer_raw = hosp_raw[pd.to_numeric(hosp_raw["Nature_of_ailment"], errors="coerce") == CANCER_CODE]
summary["duplicate_case_keys_2017"] = int(cancer_raw.astype(str).duplicated(subset=CASE_KEY).sum())
summary["duplicate_full_rows_2017"] = int(cancer_raw.astype(str).duplicated().sum())

cols = ["oop_medical", "oop_total", "stay_days", "age", "female", "insured", "ward_type", "social_group",
        "hh_size", "hh_monthly_consumption", "transport_exp", "other_nonmed_exp", "major_finance", "place_code"]
miss = pd.DataFrame({"n_missing": d[cols].isna().sum(), "pct_missing": (100 * d[cols].isna().mean()).round(2)})
miss.to_csv(OUT / "table_missingness.csv")
summary["member_deceased_n"] = int(d["member_deceased"].sum())
summary["zero_oop_medical_n"] = int((d["oop_medical"] == 0).sum())
summary["zero_oop_by_facility"] = d.loc[d["oop_medical"] == 0, "facility_type"].map(FAC).value_counts().to_dict()
q1, q3 = d["oop_medical"].quantile([.25, .75]); iqr = q3 - q1
summary["outliers_raw_iqr3_n"] = int((d["oop_medical"] > q3 + 3 * iqr).sum())
lq1, lq3 = d["log_oop"].quantile([.25, .75]); liqr = lq3 - lq1
summary["outliers_log_iqr1.5_n"] = int(((d["log_oop"] < lq1 - 1.5 * liqr) | (d["log_oop"] > lq3 + 1.5 * liqr)).sum())
summary["stay_days_over_90_n"] = int((d["stay_days"] > 90).sum())
summary["skew_raw"], summary["skew_log"] = float(d["oop_medical"].skew()), float(d["log_oop"].skew())

# ---------- Descriptives ----------
num = ["oop_medical", "oop_total", "log_oop", "stay_days", "age", "hh_size", "annual_consumption"]
desc = d[num].describe(percentiles=[.25, .5, .75, .9]).T.round(2)
desc["weighted_mean"] = [round(wmean(d[c], d["weight"]), 2) for c in num]
desc.to_csv(OUT / "table_descriptives.csv")
cat = []
for c, lab in [("facility_type", FAC), ("sector_urban", {0: "Rural", 1: "Urban"}), ("female", {0: "Male", 1: "Female"}),
               ("insured", {0: "Not covered", 1: "Covered"}), ("mpce_quintile", None), ("social_group", {1: "ST", 2: "SC", 3: "OBC", 9: "Others"}),
               ("surgery_received", {0: "No", 1: "Yes"}), ("outside_state", {0: "Home state", 1: "Other state"})]:
    for v, g in d.groupby(c):
        cat.append({"variable": c, "level": (lab or {}).get(v, v), "n": len(g), "pct": round(100 * len(g) / len(d), 1),
                    "median_oop": round(g["oop_medical"].median()), "che_pct": round(100 * g["che"].mean(), 1),
                    "che_pct_weighted": round(100 * wmean(g["che"], g["weight"]), 1),
                    "distress_pct": round(100 * g["distress_financing"].mean(), 1)})
pd.DataFrame(cat).to_csv(OUT / "table_by_group.csv", index=False)
summary["che_rate_unweighted"] = float(d["che"].mean())
summary["che_rate_weighted"] = wmean(d["che"], d["weight"])
summary["distress_financing_rate"] = float(d["distress_financing"].mean())
summary["median_oop"], summary["mean_oop"] = float(d["oop_medical"].median()), float(d["oop_medical"].mean())

# ---------- Figures ----------
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
CAP = 500  # Rs thousand; the axis is cut at Rs 5 lakh so the bulk of the distribution is visible
above = int((d["oop_medical"] / 1000 > CAP).sum())
ax[0].hist((d["oop_medical"] / 1000).clip(upper=CAP + 10), bins=np.arange(0, CAP + 20, 10), color=BLUE)
ax[0].set(xlabel="OOPE per episode (Rs thousand; Rs 10k bins)", ylabel="Episodes", title="A. Raw scale", xlim=(0, CAP + 20))
ax[0].annotate(f"{above} episodes above\nRs 5 lakh (max Rs {d['oop_medical'].max() / 1e5:.1f} lakh)\nshown in the last bar",
               xy=(CAP + 5, 5), xytext=(CAP * .55, ax[0].get_ylim()[1] * .55), fontsize=7, arrowprops=dict(arrowstyle="->", lw=.7))
ax[1].hist(d["log_oop"], bins=40, color=BLUE); ax[1].set(xlabel="ln(OOPE + 1)", ylabel="Episodes", title="B. Log scale (all episodes)")
rupee_axis(ax[1], "x")
save(fig, "fig2_oope_distribution")

fig, ax = plt.subplots(figsize=(6, 3.4))
groups = [d.loc[d.facility_type == k, "log_oop"] for k in FAC]
ax.boxplot(groups, tick_labels=[f"{v}\n(n={len(g)})" for v, g in zip(FAC.values(), groups)], patch_artist=True,
           boxprops=dict(facecolor="#BDD7EE"), medianprops=dict(color=ORANGE, lw=2))
ax.set(xlabel="Hospital type", ylabel="ln(OOPE + 1)", title="OOPE by type of hospital, 2017-18")
rupee_axis(ax, "y")
save(fig, "fig3_log_oope_by_facility")

fig, ax = plt.subplots(figsize=(6, 3.4))
q = d.groupby("mpce_quintile")["che"].mean() * 100
b = ax.bar(q.index.astype(str), q.values, color=BLUE)
ax.bar_label(b, fmt="%.1f%%"); ax.set(xlabel="MPCE quintile (1 = poorest)", ylabel="Episodes with CHE (%)", ylim=(0, 100),
                                      title="CHE incidence by household consumption quintile")
save(fig, "fig4_che_by_quintile")

fig, ax = plt.subplots(figsize=(6, 3.4))
t = d.groupby(["facility_type", "insured"])["che"].mean().unstack() * 100
tn = d.groupby(["facility_type", "insured"])["che"].size().unstack()
x = np.arange(len(t)); w = .38
for i, (col, colr, lab) in enumerate([(0.0, GREY, "Not covered"), (1.0, BLUE, "Covered by a scheme")]):
    if col in t:
        bb = ax.bar(x + (i - .5) * w, t[col].values, w, color=colr, label=lab)
        ax.bar_label(bb, labels=[f"{v:.0f}%\nn={k}" for v, k in zip(t[col].values, tn[col].values)], fontsize=7)
ax.set_xticks(x, [FAC[k] for k in t.index]); ax.set(xlabel="Hospital type", ylabel="Episodes with CHE (%)", ylim=(0, 105))
ax.set_title("CHE by hospital type and scheme coverage", pad=24)
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "fig5_che_by_facility_insurance")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
d["stay_band"] = pd.cut(d["stay_days"], [0, 2, 5, 10, 20, 1e9], labels=["1-2", "3-5", "6-10", "11-20", ">20"])
g = d.groupby(["stay_band", "facility_public"], observed=True)["oop_medical"]
s, n = g.median().unstack() / 1000, g.size().unstack()
x, w = np.arange(len(s)), 0.38
# Bars = median per stay band (bands are categories of unequal width, so bars, not a continuous line);
# the marker line on top only guides the eye across bands. n under each bar shows how many episodes back each median.
for i, (col, colr, lab) in enumerate([(1, BLUE, "Public"), (0, ORANGE, "Private/charitable")]):
    if col not in s: continue
    xs = x + (i - 0.5) * w
    ax.bar(xs, s[col].values, w, color=colr, alpha=.85, label=lab)
    ax.plot(xs, s[col].values, marker="o", ms=4, lw=1.2, color="black" if col == 1 else "#7F3F00")
    for xi, v, k in zip(xs, s[col].values, n[col].values):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi, -0.015, f"n={k}", ha="center", va="top", fontsize=6.5, color="#404040", transform=ax.get_xaxis_transform())
ax.set_xticks(x, s.index.astype(str)); ax.tick_params(axis="x", pad=13, length=0)
ax.set_ylim(0, s.max().max() * 1.15)
ax.set(xlabel="Length of stay (days, banded)", ylabel="Median OOPE (Rs thousand)",
       title="Median OOPE by length of stay and hospital type")
ax.legend(frameon=False)
save(fig, "fig6_oope_by_stay")

corr_cols = ["log_oop", "stay_days", "age", "hh_size", "mpce_quintile", "facility_public", "surgery_received", "insured", "outside_state", "che"]
c = d[corr_cols].corr(method="spearman")
c.round(3).to_csv(OUT / "table_spearman.csv")
fig, ax = plt.subplots(figsize=(6.5, 5.5))
lower = c.where(np.tril(np.ones(c.shape, dtype=bool)))   # lower triangle only: each pair shown once
im = ax.imshow(lower, cmap="RdBu_r", vmin=-1, vmax=1)
names_c = [LABEL.get(k, k) for k in c.columns]
ax.set_xticks(range(len(c)), names_c, rotation=45, ha="right"); ax.set_yticks(range(len(c)), names_c)
for i in range(len(c)):
    for j in range(i + 1): ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, shrink=.8, label="Spearman rho"); ax.set_title("Spearman correlations among key variables")
save(fig, "fig7_correlation")

# ---------- Model results (from step 3) ----------
r = json.load(open(OUTPUTS / "results.json"))
sc = pd.DataFrame(r["rq1"]["scores"]).T.round(4); sc.to_csv(OUT / "table_rq1_model_scores.csv")
fig, ax = plt.subplots(figsize=(6, 3.2))
names = {"linear": "Linear (baseline)", "random_forest": "Random forest", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
bb = ax.bar([names[k] for k in sc.index], sc["mae_log"], color=[GREY] + [BLUE] * (len(sc) - 1))
ax.bar_label(bb, fmt="%.3f"); ax.set(xlabel="Model", ylabel="MAE, ln(OOPE + 1)", title="RQ1: five-fold cross-validated MAE by model")
save(fig, "fig8_rq1_mae")
imp = pd.Series(r["rq1"]["mean_abs_shap"]).sort_values().rename(index=LABEL)
fig, ax = plt.subplots(figsize=(6, 4)); ax.barh(imp.index, imp.values, color=BLUE)
ax.set(xlabel="Mean |SHAP| on ln(OOPE + 1)", title="RQ1: XGBoost feature importance"); save(fig, "fig9_shap")
orr = pd.DataFrame({"odds_ratio": r["rq2"]["odds_ratios"], "p": r["rq2"]["p"],
                    "ci_low": r["rq2"].get("ci_low"), "ci_high": r["rq2"].get("ci_high")}).drop(index="Intercept")
orr.round(4).to_csv(OUT / "table_rq2_odds_ratios.csv")
fig, ax = plt.subplots(figsize=(6, 4.5))
PRETTY = {"age": "Age (per year)", "female": "Female", "sector_urban": "Urban", "facility_public": "Public hospital",
          "stay_days": "Length of stay (per day)", "surgery_received": "Surgery received", "outside_state": "Treated outside state",
          "insured": "Covered by a scheme", "C(mpce_quintile)[T.2]": "MPCE Q2 (vs Q1)", "C(mpce_quintile)[T.3]": "MPCE Q3 (vs Q1)",
          "C(mpce_quintile)[T.4]": "MPCE Q4 (vs Q1)", "C(mpce_quintile)[T.5]": "MPCE Q5 (vs Q1)", "C(social_group)[T.2]": "SC (vs ST)",
          "C(social_group)[T.3]": "OBC (vs ST)", "C(social_group)[T.9]": "Others (vs ST)"}
o = orr.sort_values("odds_ratio").rename(index=PRETTY)
cols = [ORANGE if p < .05 else GREY for p in o["p"]]
if o["ci_low"].notna().all():   # 95% confidence intervals (needs results.json from the updated step 3)
    ax.errorbar(o["odds_ratio"], o.index, xerr=[o["odds_ratio"] - o["ci_low"], o["ci_high"] - o["odds_ratio"]],
                fmt="none", ecolor="#A0A0A0", elinewidth=1, capsize=2, zorder=1)
ax.scatter(o["odds_ratio"], o.index, color=cols, zorder=2)
ax.axvline(1, color="black", lw=.8); ax.set_xscale("log")
ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5) if o[["odds_ratio", "ci_low", "ci_high"]].min().min() * .8 <= t <= o[["odds_ratio", "ci_low", "ci_high"]].max().max() * 1.25]
ax.set_xticks(ticks, [f"{t:g}" for t in ticks]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set(xlabel="Odds ratio for CHE with 95% CI (log scale; orange = p < .05)", title="RQ2: logistic regression for CHE"); save(fig, "fig10_rq2_or")

# ---------- 2025 (RQ4) ----------
f80 = PROCESSED / "cancer_episodes_nss80.csv"
if f80.exists():
    d80 = pd.read_csv(f80)
    summary["n_episodes_2025"] = len(d80)
    summary["che_rate_2025_unweighted"] = float(d80["che"].mean())
    summary["che_rate_2025_weighted"] = wmean(d80["che"], d80["weight"])
    summary["median_oop_2025"] = float(d80["oop_medical"].median())
    rows = []
    for lab, dd in [("2017-18", d), ("2025", d80)]:
        for k, g in dd.groupby("facility_type"):
            rows.append({"survey": lab, "facility": FAC.get(k, k), "n": len(g), "che_pct": round(100 * g["che"].mean(), 1),
                         "median_oop": round(g["oop_medical"].median())})
    pd.DataFrame(rows).to_csv(OUT / "table_rq4_by_survey.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.4)); x = np.arange(3); w = .38
    for i, (lab, dd, colr) in enumerate([("2017-18", d, BLUE), ("2025", d80, ORANGE)]):
        v = [100 * dd.loc[dd.facility_type == k, "che"].mean() for k in FAC]
        nn = [int((dd.facility_type == k).sum()) for k in FAC]
        bb = ax.bar(x + (i - .5) * w, v, w, color=colr, label=lab)
        ax.bar_label(bb, labels=[f"{a:.0f}%\nn={b:,}" for a, b in zip(v, nn)], fontsize=7)
    ax.set_xticks(x, FAC.values()); ax.set(xlabel="Hospital type", ylabel="Episodes with CHE (%)", ylim=(0, 105), title="RQ4: CHE by hospital type, 2017-18 vs 2025")
    ax.set_title(ax.get_title(), pad=24); ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0)); save(fig, "fig11_rq4_che_by_survey")

json.dump(summary, open(OUT / "eda_summary.json", "w"), indent=2, default=float)
print(json.dumps(summary, indent=1, default=float))
print(f"Report tables and figures written to {OUT}")
