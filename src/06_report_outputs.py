"""Step 6 - Aggregate tables and figures for the report (cleaning log, EDA, model results), for the PRIMARY
survey round set in config.py (2025 by default). Run after steps 1 (both rounds), 3 and 7.
Writes ONLY aggregate outputs (no episode-level rows), so everything in outputs/report/ can be published on
GitHub without breaching MoSPI's terms of use. Figures follow plot_style.py (no in-image titles, labelled axes)."""
import json
import numpy as np, pandas as pd
from config import RAW, PROCESSED, OUTPUTS, FILES, FILES80, CASE_KEY, HH_KEY80, CANCER_CODE, PRIMARY
from plot_style import plt, NAVY, ORANGE, GREY, LIGHT, MID, BLUE_ORANGE, FAC, grid, nice_axis, pct_axis, rupee_axis, save as _save
from matplotlib.lines import Line2D

OUT = OUTPUTS / "report"; OUT.mkdir(parents=True, exist_ok=True)
save = lambda fig, name: _save(fig, OUT / f"{name}.png")

LABEL = {"log_oop": "ln(OOPE + 1)", "stay_days": "Length of stay", "age": "Age", "hh_size": "Household size",
         "mpce_quintile": "MPCE quintile", "facility_public": "Public hospital", "surgery_received": "Surgery",
         "insured": "Scheme coverage", "outside_state": "Outside home state", "outside_district": "Outside home district",
         "che": "CHE", "ward_type": "Ward type", "female": "Female", "member_deceased": "Died or left household",
         "sector_urban": "Urban", "social_group": "Social group"}

def wmean(x, w):
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m]))

def load(round_):
    f = PROCESSED / f"cancer_episodes_{round_}.csv"
    return pd.read_csv(f) if f.exists() else None

primary = PRIMARY if load(PRIMARY) is not None else "nss75"
d = load(primary)
summary = {"round": primary, "n_episodes": len(d)}

# ---------- Cleaning log ----------
if primary == "nss80":
    hosp_raw = pd.read_csv(RAW.parent / "nss80" / f"{FILES80['hosp']}.csv", low_memory=False)
    ail, key = "b6i5", HH_KEY80 + ["b6i1"]
else:
    hosp_raw = pd.read_csv(RAW.parent / "nss75" / f"{FILES['hosp']}.csv", low_memory=False)
    ail, key = "Nature_of_ailment", CASE_KEY
summary["n_hospitalisation_records"] = len(hosp_raw)
cancer_raw = hosp_raw[pd.to_numeric(hosp_raw[ail], errors="coerce") == CANCER_CODE]
summary["duplicate_case_keys"] = int(cancer_raw.astype(str).duplicated(subset=key).sum())
summary["duplicate_full_rows"] = int(cancer_raw.astype(str).duplicated().sum())

cols = [c for c in ["oop_medical", "oop_total", "stay_days", "ward_type", "surgery_received", "age", "female", "insured",
                    "social_group", "hh_size", "hh_monthly_consumption", "transport_exp", "other_nonmed_exp",
                    "major_finance", "place_code"] if c in d]
miss = pd.DataFrame({"n_missing": d[cols].isna().sum(), "pct_missing": (100 * d[cols].isna().mean()).round(2)})
miss.to_csv(OUT / "table_missingness.csv")
summary["member_deceased_n"] = int(d["member_deceased"].sum())
summary["zero_oop_medical_n"] = int((d["oop_medical"] == 0).sum())
summary["zero_oop_by_facility"] = d.loc[d["oop_medical"] == 0, "facility_type"].map(FAC).value_counts().to_dict()
q1, q3 = d["oop_medical"].quantile([.25, .75]); iqr = q3 - q1
summary["outliers_raw_iqr3_n"] = int((d["oop_medical"] > q3 + 3 * iqr).sum())
summary["stay_days_over_90_n"] = int((d["stay_days"] > 90).sum())
summary["skew_raw"], summary["skew_log"] = float(d["oop_medical"].skew()), float(d["log_oop"].skew())
summary["negative_oop_set_to_zero"] = int(((d["medical_exp"] - d["reimbursed"]) < 0).sum())

# ---------- Descriptives ----------
num = ["oop_medical", "oop_total", "log_oop", "stay_days", "age", "hh_size", "annual_consumption"]
desc = d[num].describe(percentiles=[.25, .5, .75]).T.round(2)
desc["weighted_mean"] = [round(wmean(d[c], d["weight"]), 2) for c in num]
desc["skew"] = d[num].skew().round(2)
desc.to_csv(OUT / "table_descriptives.csv")
cat = []
for c, lab in [("facility_type", FAC), ("ward_type", {1: "Free", 2: "Paying general", 3: "Paying special"}),
               ("sector_urban", {0: "Rural", 1: "Urban"}), ("female", {0: "Male", 1: "Female"}),
               ("insured", {0: "Not covered", 1: "Covered"}), ("mpce_quintile", None),
               ("social_group", {1: "ST", 2: "SC", 3: "OBC", 9: "Others"}), ("surgery_received", {0: "No", 1: "Yes"}),
               ("outside_state", {0: "Home state", 1: "Other state"})]:
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
summary["weighted_mean_oop"] = wmean(d["oop_medical"], d["weight"])

# ---------- Figures (Figure 1 comes from step 5) ----------
# Figure 2: distribution of OOPE, raw and log scale
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
CAP = 500   # ₹ thousand: the raw axis is cut at ₹5 lakh so the bulk of the distribution is visible
above = int((d["oop_medical"] > CAP * 1000).sum())
counts, _, _ = ax[0].hist((d["oop_medical"] / 1000).clip(upper=CAP + 5), bins=np.arange(0, CAP + 11, 10), color=NAVY, zorder=3)
ax[0].set_xlim(0, CAP + 20); ax[0].set_xticks(range(0, CAP + 1, 100)); nice_axis(ax[0], counts.max(), 200)
ax[0].set(xlabel="OOPE per episode (₹ thousand)", ylabel="Number of episodes"); ax[0].set_title("A. Raw scale", loc="left", fontsize=10)
ax[0].annotate(f"{above} episodes above ₹5 lakh\nin the last bar", xy=(CAP + 5, counts[-1]), xytext=(CAP * .45, counts.max() * .55),
               fontsize=8, arrowprops=dict(arrowstyle="->", lw=.7))
counts, _, _ = ax[1].hist(d["log_oop"], bins=np.arange(0, 16.01, .4), color=NAVY, zorder=3)
ax[1].set_xlim(0, 16); ax[1].set_xticks(range(0, 17, 4)); nice_axis(ax[1], counts.max(), 100)
ax[1].set(xlabel="ln(OOPE + 1)", ylabel="Number of episodes"); ax[1].set_title("B. Log scale", loc="left", fontsize=10)
rupee_axis(ax[1], "x")
for a in ax: grid(a)
fig.tight_layout(); save(fig, "fig2_oope_distribution")
summary["episodes_above_5_lakh"] = above

# Figure 3: ln(OOPE + 1) by hospital type
fig, ax = plt.subplots(figsize=(6.2, 3.6))
groups = [d.loc[d.facility_type == k, "log_oop"] for k in FAC]
bp = ax.boxplot(groups, widths=.45, patch_artist=True, medianprops=dict(color=ORANGE, lw=2),
                flierprops=dict(marker="o", ms=4, mfc="none"))
for b in bp["boxes"]: b.set_facecolor(LIGHT)
ax.set_xticks([1, 2, 3], [f"{v}\n(n = {len(g):,})" for v, g in zip(FAC.values(), groups)])
ax.set(xlabel="Hospital type", ylabel="ln(OOPE + 1)"); ax.set_ylim(0, 16); ax.set_yticks(range(0, 17, 2))
rupee_axis(ax, "y"); grid(ax); save(fig, "fig3_log_oope_by_facility")

# Figure 4: CHE by MPCE quintile
fig, ax = plt.subplots(figsize=(6.2, 3.3))
q = d.groupby("mpce_quintile")["che"].mean() * 100
b = ax.bar(q.index, q.values, .6, color=NAVY, zorder=3); ax.bar_label(b, fmt="%.1f%%", padding=2, fontsize=9)
ax.set_xticks(q.index, ["1\n(poorest)", "2", "3", "4", "5\n(richest)"])
ax.set(xlabel="Monthly per-capita consumption expenditure quintile", ylabel="Episodes with CHE (%)"); pct_axis(ax); grid(ax)
save(fig, "fig4_che_by_quintile")

# Figure 5: CHE by hospital type and scheme coverage
fig, ax = plt.subplots(figsize=(6.2, 3.5))
t = d.groupby(["facility_type", "insured"])["che"].mean().unstack() * 100
tn = d.groupby(["facility_type", "insured"])["che"].size().unstack()
x, w = np.arange(len(t)), .36
for i, (col, colr, lab) in enumerate([(0.0, ORANGE, "Not covered"), (1.0, NAVY, "Covered by a scheme")]):
    bb = ax.bar(x + (i - .5) * w, t[col].values, w, color=colr, label=lab, zorder=3)
    ax.bar_label(bb, labels=[f"{v:.0f}%" for v in t[col].values], padding=2, fontsize=8.5)
    for xx, k in zip(x + (i - .5) * w, tn[col].values):
        ax.text(xx, 3, f"n = {k:,}", ha="center", va="bottom", color="white", fontsize=7.5, zorder=4)
ax.set_xticks(x, [FAC[k] for k in t.index]); ax.set(xlabel="Hospital type", ylabel="Episodes with CHE (%)"); pct_axis(ax)
ax.legend(title="Scheme coverage", frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0)); grid(ax)
save(fig, "fig5_che_by_facility_insurance")
summary["che_by_facility_insurance_pct"] = t.round(1).to_dict(); summary["che_by_facility_insurance_n"] = tn.to_dict()

# Figure 6: ln(OOPE + 1) against length of stay (numeric x-axis, LOWESS trend per hospital type)
from statsmodels.nonparametric.smoothers_lowess import lowess
fig, ax = plt.subplots(figsize=(6.8, 3.8))
dd = d[d["stay_days"] <= 30]
rng = np.random.default_rng(42)
for col, colr, lab in [(0, ORANGE, "Private/charitable"), (1, NAVY, "Public")]:
    gg = dd[dd["facility_public"] == col]
    ax.scatter(gg["stay_days"] + rng.uniform(-.25, .25, len(gg)), gg["log_oop"], s=6, color=colr, alpha=.15, linewidths=0, zorder=2)
    fit = lowess(gg["log_oop"], gg["stay_days"], frac=.4, return_sorted=True)
    ax.plot(fit[:, 0], fit[:, 1], color=colr, lw=2.4, label=f"{lab} (trend)", zorder=4)
ax.set_xlim(0, 30); ax.set_xticks(range(0, 31, 5)); ax.set_ylim(0, 16); ax.set_yticks(range(0, 17, 4))
ax.set(xlabel="Length of stay (days)", ylabel="ln(OOPE + 1)")
rupee_axis(ax, "y"); ax.legend(frameon=False, loc="lower right"); grid(ax); save(fig, "fig6_oope_by_stay")
summary["stay_over_30_excluded_from_fig6_n"] = int((d["stay_days"] > 30).sum())
for col in (0, 1):
    gg = dd[dd["facility_public"] == col]; fit = lowess(gg["log_oop"], gg["stay_days"], frac=.4, return_sorted=True)
    summary[f"fig6_trend_public{col}"] = {int(x): round(float(np.expm1(y))) for x, y in fit[[np.abs(fit[:, 0] - k).argmin() for k in (1, 5, 10, 20, 30)]]}

# Figure 7: Spearman correlations
corr_cols = ["log_oop", "stay_days", "age", "hh_size", "mpce_quintile", "facility_public", "surgery_received", "insured", "outside_state", "che"]
c = d[corr_cols].corr(method="spearman"); c.round(3).to_csv(OUT / "table_spearman.csv")
fig, ax = plt.subplots(figsize=(6.4, 5.4))
lower = c.where(np.tril(np.ones(c.shape, dtype=bool)))
im = ax.imshow(lower, cmap=BLUE_ORANGE, vmin=-1, vmax=1)
names_c = [LABEL.get(k, k) for k in c.columns]
ax.set_xticks(range(len(c)), names_c, rotation=45, ha="right"); ax.set_yticks(range(len(c)), names_c)
ax.set(xlabel="Variable", ylabel="Variable")
for i in range(len(c)):
    for j in range(i + 1):
        v = c.iloc[i, j]; v = 0.0 if abs(v) < .005 else v
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > .6 else "black")
for sp in ax.spines.values(): sp.set_visible(False)
fig.colorbar(im, ax=ax, shrink=.8, label="Spearman's ρ", ticks=[-1, -.5, 0, .5, 1]); save(fig, "fig7_correlation")

# ---------- Model results (from step 3) ----------
r = json.load(open(OUTPUTS / "results.json"))
sc = pd.DataFrame(r["rq1"]["scores"]).T.round(4); sc.to_csv(OUT / "table_rq1_model_scores.csv")
names = {"linear": "Linear (baseline)", "random_forest": "Random forest", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.6))
for ax, (k, yl, fmt, step, title) in zip(axs, [("mae_log", "MAE (ln[OOPE + 1] units)", "%.3f", .5, "A. MAE"),
                                              ("rmse_log", "RMSE (ln[OOPE + 1] units)", "%.3f", .5, "B. RMSE"),
                                              ("r2", "R²", "%.2f", .1, "C. R²")]):
    bb = ax.bar(range(len(sc)), sc[k], .65, color=[ORANGE] + [NAVY] * (len(sc) - 1), zorder=3)
    ax.bar_label(bb, fmt=fmt, padding=2, fontsize=8)
    ax.set_xticks(range(len(sc)), [names[i] for i in sc.index], fontsize=8, rotation=40, ha="right", rotation_mode="anchor")
    nice_axis(ax, sc[k].max(), step, headroom=1.12); ax.set(xlabel="Model", ylabel=yl)
    ax.set_title(title, loc="left", fontsize=10); grid(ax)
fig.tight_layout(); save(fig, "fig8_rq1_metrics")

imp = pd.Series(r["rq1"]["mean_abs_shap"]).sort_values().rename(index=LABEL)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
bb = ax.barh(imp.index, imp.values, color=NAVY, zorder=3); ax.bar_label(bb, fmt="%.2f", padding=2, fontsize=8)
ax.set(xlabel="Mean absolute SHAP value (effect on ln[OOPE + 1])", ylabel="Feature")
nice_axis(ax, imp.max(), .2 if imp.max() < 1.5 else .4, axis="x", headroom=1.12); grid(ax, "x"); save(fig, "fig9_shap")

orr = pd.DataFrame({"odds_ratio": r["rq2"]["odds_ratios"], "p": r["rq2"]["p"],
                    "ci_low": r["rq2"]["ci_low"], "ci_high": r["rq2"]["ci_high"]}).drop(index="Intercept")
orr.round(4).to_csv(OUT / "table_rq2_odds_ratios.csv")
PRETTY = {"age": "Age (per year)", "female": "Female", "sector_urban": "Urban", "facility_public": "Public hospital",
          "stay_days": "Length of stay (per day)", "surgery_received": "Surgery received", "outside_state": "Treated outside home state",
          "insured": "Covered by a scheme", "C(mpce_quintile)[T.2]": "MPCE Q2 (vs. Q1)", "C(mpce_quintile)[T.3]": "MPCE Q3 (vs. Q1)",
          "C(mpce_quintile)[T.4]": "MPCE Q4 (vs. Q1)", "C(mpce_quintile)[T.5]": "MPCE Q5 (vs. Q1)", "C(social_group)[T.2]": "SC (vs. ST)",
          "C(social_group)[T.3]": "OBC (vs. ST)", "C(social_group)[T.9]": "Other (vs. ST)"}
o = orr.sort_values("odds_ratio").rename(index=PRETTY)
fig, ax = plt.subplots(figsize=(6.2, 4.6))
for i, (lab, row) in enumerate(o.iterrows()):
    colr = NAVY if row["p"] < .05 else ORANGE
    ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=colr, lw=1.4, zorder=2)
    ax.plot(row["odds_ratio"], i, "o", color=colr, ms=6, zorder=3)
ax.axvline(1, color="black", lw=.8); ax.set_xscale("log")
LOG_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]                 # 1-2-5 sequence
lo_t = max([t for t in LOG_TICKS if t <= o["ci_low"].min()], default=LOG_TICKS[0])
hi_t = min([t for t in LOG_TICKS if t >= o["ci_high"].max()], default=LOG_TICKS[-1])
ticks = [t for t in LOG_TICKS if lo_t <= t <= hi_t]
ax.set_xlim(lo_t, hi_t); ax.set_xticks(ticks, [f"{t:g}" for t in ticks]); ax.minorticks_off()
ax.set_yticks(range(len(o)), o.index); ax.set(xlabel="Odds ratio for CHE (log scale, 95% CI)", ylabel="Predictor"); grid(ax, "x")
ax.legend(handles=[Line2D([], [], marker="o", color=NAVY, label="p < .05"), Line2D([], [], marker="o", color=ORANGE, label="p ≥ .05")],
          frameon=False, loc="lower right")
save(fig, "fig10_rq2_or")

# ---------- RQ4: both rounds ----------
d75, d80 = load("nss75"), load("nss80")
if d75 is not None and d80 is not None:
    rows = []
    for lab, dd in [("2017-18", d75), ("2025", d80)]:
        for k, gg in dd.groupby("facility_type"):
            rows.append({"survey": lab, "facility": FAC.get(k, k), "n": len(gg), "che_pct": round(100 * gg["che"].mean(), 1),
                         "che_pct_weighted": round(100 * wmean(gg["che"], gg["weight"]), 1), "median_oop": round(gg["oop_medical"].median())})
    pd.DataFrame(rows).to_csv(OUT / "table_rq4_by_survey.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.5)); x, w = np.arange(3), .36
    for i, (lab, dd, colr) in enumerate([("2017–18", d75, NAVY), ("2025", d80, ORANGE)]):
        v = [100 * dd.loc[dd.facility_type == k, "che"].mean() for k in FAC]
        nn = [int((dd.facility_type == k).sum()) for k in FAC]
        bb = ax.bar(x + (i - .5) * w, v, w, color=colr, label=lab, zorder=3)
        ax.bar_label(bb, labels=[f"{a:.1f}%" for a in v], padding=2, fontsize=8.5)
        for xx, m_ in zip(x + (i - .5) * w, nn):
            ax.text(xx, 3, f"n = {m_:,}", ha="center", va="bottom", color="white", fontsize=7.5, zorder=4)
    ax.set_xticks(x, FAC.values()); ax.set(xlabel="Hospital type", ylabel="Episodes with CHE (%)"); pct_axis(ax)
    ax.legend(title="Survey round", frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0)); grid(ax)
    save(fig, "fig11_rq4_che_by_survey")

json.dump(summary, open(OUT / "eda_summary.json", "w"), indent=2, default=float)
print(json.dumps(summary, indent=1, default=float))
print(f"Report tables and figures written to {OUT}")
