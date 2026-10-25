"""Step 3 - RQ1 cost prediction, RQ2 CHE drivers, RQ3 public vs private cost, RQ4 CHE 2017-18 vs 2025 and fairness audit."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from config import PROCESSED, OUTPUTS, SEED, PRIMARY
warnings.filterwarnings("ignore")

FEATS = ["age", "female", "sector_urban", "facility_public", "ward_type", "stay_days", "surgery_received",
         "outside_district", "outside_state", "insured", "member_deceased", "mpce_quintile", "social_group", "hh_size"]

CAT_FEATS = ["ward_type", "social_group", "mpce_quintile"]          # dummy-coded for linear regression
RQ2_TERMS = ["age", "female", "sector_urban", "facility_public", "stay_days", "surgery_received",
             "outside_state", "insured", "C(mpce_quintile)", "C(social_group)"]

def rq1(df, feats=FEATS):
    """5-fold cross-validation of linear, random forest, LightGBM and XGBoost; out-of-fold predictions."""
    X, y = df[feats].fillna(df[feats].median()), df["log_oop"]
    Xd = pd.get_dummies(X, columns=[c for c in CAT_FEATS if c in feats], drop_first=True)
    models = {
        "linear": lambda: LinearRegression(),
        "random_forest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
        "xgboost": lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                            colsample_bytree=0.8, random_state=SEED),
    }
    pred = {k: np.zeros(len(df)) for k in models}
    lo, hi = np.zeros(len(df)), np.zeros(len(df))
    qkw = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, random_state=SEED, verbose=-1)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        for k, make in models.items():
            Xa = Xd if k == "linear" else X
            pred[k][te] = make().fit(Xa.iloc[tr], y.iloc[tr]).predict(Xa.iloc[te])
        lo[te] = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **qkw).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        hi[te] = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **qkw).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    err = {k: np.abs(y - p) for k, p in pred.items()}
    scores = {k: {"mae_log": float(e.mean()), "rmse_log": float(np.sqrt(np.mean((y - pred[k]) ** 2))),
                  "r2": float(1 - np.sum((y - pred[k]) ** 2) / np.sum((y - y.mean()) ** 2))} for k, e in err.items()}
    for k in scores:   # MAE in rupees
        scores[k]["mae_rupees"] = float(np.mean(np.abs(np.expm1(y) - np.expm1(pred[k]))))
    best = min((k for k in models if k != "linear"), key=lambda k: scores[k]["mae_log"])
    improvement = (scores["linear"]["mae_log"] - scores[best]["mae_log"]) / scores["linear"]["mae_log"] * 100
    d = err[best] - err["linear"]
    t = stats.ttest_rel(err[best], err["linear"])
    # SHAP from XGBoost on all episodes
    booster = models["xgboost"]().fit(X, y).get_booster()
    contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    importance = dict(sorted(zip(feats, np.abs(contrib).mean(0).round(4)), key=lambda kv: -kv[1]))
    df = df.assign(pred_oop=np.expm1(pred[best]), pi_low=np.expm1(lo), pi_high=np.expm1(hi))
    cover = float(np.mean((y >= lo) & (y <= hi)))
    return df, {"scores": scores, "best_ml": best, "mae_improvement_pct": float(improvement),
                "meets_10pct_target": bool(improvement >= 10), "t": t.statistic, "p": t.pvalue, "dz": d.mean() / d.std(),
                "wilcoxon_p": stats.wilcoxon(d).pvalue, "pi80_coverage": cover, "mean_abs_shap": importance}

def rq2(df, feats=FEATS):
    terms = [t for t in RQ2_TERMS if t.removeprefix("C(").removesuffix(")") in feats]
    m = smf.logit("che ~ " + " + ".join(terms), data=df.dropna(subset=feats)).fit(disp=0)
    ci = np.exp(m.conf_int())                                                  # 95% CI, odds-ratio scale
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = m.model.exog
    vif = {n: float(variance_inflation_factor(X, i)) for i, n in enumerate(m.model.exog_names) if n != "Intercept"}
    events = int(m.model.endog.sum())
    return {"n": int(m.nobs), "odds_ratios": np.exp(m.params).to_dict(), "p": m.pvalues.to_dict(), "pseudo_r2": m.prsquared,
            "ci_low": ci[0].to_dict(), "ci_high": ci[1].to_dict(),
            "auc": float(roc_auc_score(m.model.endog, m.predict())),        # in-sample
            "events": events, "events_per_parameter": events / (len(m.params) - 1), "max_vif": max(vif.values()), "vif": vif}

def rq3(df):
    pub, pvt = df.loc[df["facility_type"] == 1, "log_oop"], df.loc[df["facility_type"] == 3, "log_oop"]
    t = stats.ttest_ind(pub, pvt, equal_var=False)
    v1, v2 = pub.var() / len(pub), pvt.var() / len(pvt)
    welch_df = (v1 + v2) ** 2 / (v1 ** 2 / (len(pub) - 1) + v2 ** 2 / (len(pvt) - 1))   # Welch df
    sd = np.sqrt((pub.var() + pvt.var()) / 2)
    return {"n_public": len(pub), "n_private": len(pvt), "median_oop_public": float(np.expm1(pub.median())),
            "median_oop_private": float(np.expm1(pvt.median())), "welch_t": t.statistic, "welch_df": float(welch_df), "p": t.pvalue,
            "cohen_d": (pvt.mean() - pub.mean()) / sd, "mann_whitney_p": stats.mannwhitneyu(pub, pvt).pvalue}

def load_round(round_):
    f = PROCESSED / f"cancer_episodes_{round_}.csv"
    return pd.read_csv(f).dropna(subset=["log_oop", "annual_consumption"]) if f.exists() else None

def rq4():
    """CHE 2017-18 vs 2025."""
    d75, d80 = load_round("nss75"), load_round("nss80")
    if d75 is not None and d80 is not None:
        both = pd.concat([d75.assign(survey_2025=0), d80.assign(survey_2025=1)], ignore_index=True)
        g = both.groupby("survey_2025")["che"].agg(["sum", "count"])
        z, p = proportions_ztest(g["sum"].values, g["count"].values)
        adj = smf.logit("che ~ survey_2025 + C(mpce_quintile) + sector_urban + facility_public", data=both).fit(disp=0)
        p1, p2 = g.loc[0, "sum"] / g.loc[0, "count"], g.loc[1, "sum"] / g.loc[1, "count"]
        w1 = float(np.average(d75["che"], weights=d75["weight"])); w2 = float(np.average(d80["che"], weights=d80["weight"]))
        return {"test": "CHE 2017-18 vs 2025", "n_2017": int(g.loc[0, "count"]), "n_2025": int(g.loc[1, "count"]),
                "che_2017": p1, "che_2025": p2, "che_2017_weighted": w1, "che_2025_weighted": w2,
                "cohen_h": float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))),
                "cohen_h_weighted": float(2 * np.arcsin(np.sqrt(w1)) - 2 * np.arcsin(np.sqrt(w2))),
                "z": z, "p": p, "adjusted_or_2025": float(np.exp(adj.params["survey_2025"])),
                "adjusted_ci_low": float(np.exp(adj.conf_int().loc["survey_2025", 0])),
                "adjusted_ci_high": float(np.exp(adj.conf_int().loc["survey_2025", 1])),
                "adjusted_p": float(adj.pvalues["survey_2025"])}
    return {"test": "RQ4 not run: build both episode files first (README step 2)"}

def fairness(df):
    df = df.assign(abs_err=np.abs(np.log1p(df["pred_oop"]) - df["log_oop"]),
                   under=(df["pred_oop"] < 0.8 * df["oop_medical"]).astype(int))
    return {k: df.groupby(k)[["abs_err", "under"]].mean().round(3).to_dict() for k in ["sector_urban", "social_group", "female"]}

def main():
    primary = PRIMARY if (PROCESSED / f"cancer_episodes_{PRIMARY}.csv").exists() else "nss75"
    if primary != PRIMARY:
        print(f"cancer_episodes_{PRIMARY}.csv not found: analysing nss75 instead")
    df = load_round(primary)
    df, r1 = rq1(df)
    out = {"round": primary, "n": len(df), "rq1": r1, "rq2": rq2(df), "rq3": rq3(df), "rq4": rq4(), "fairness": fairness(df),
           "che_rate": float(df["che"].mean()), "weighted_che_rate": float(np.average(df["che"], weights=df["weight"]))}
    df.to_csv(OUTPUTS / "episode_predictions.csv", index=False)
    json.dump(out, open(OUTPUTS / "results.json", "w"), indent=2, default=float)
    print(json.dumps({"round": primary, "n": out["n"], "best": r1["best_ml"], "scores": r1["scores"], "rq1_p": r1["p"],
                      "pi80": r1["pi80_coverage"], "rq3_d": out["rq3"]["cohen_d"], "mae_improvement_pct": r1["mae_improvement_pct"],
                      "rq4_p": out["rq4"].get("p"), "che_rate": out["weighted_che_rate"]}, indent=1, default=float))

if __name__ == "__main__":
    main()
