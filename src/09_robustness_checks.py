"""Step 9 - Robustness checks added after the interim review (report Table 18).

A. Stronger linear baseline: adds ln(stay) and a public x ln(stay) interaction (is the RQ1 gain from trees or from a weak baseline?).
B. Admission-time model: RQ1 refit without stay_days (stay is only known at discharge).
C. Fairness audit by sex on recorded sex only (excludes episodes where sex was imputed).
D. Episodes with no person record (patient died or left) vs. the rest: CHE and cost (RQ2 complete-case bias).
E. RQ3 adjusted for case mix: OLS of ln(OOPE + 1) on public vs. private plus covariates, FSU-clustered standard errors.
F. RQ4 design effects and design-adjusted power (reads outputs/design_based_check.json from script 08).
G. RQ1 paired test with FSU-clustered standard errors (out-of-fold errors from the same household cluster are correlated).
H. RQ2 logistic regression with FSU-clustered standard errors (unweighted; survey-weighted estimates are run in SPSS).
I. Sample-size sensitivity: minimum n for each RQ under smaller effect sizes than planned.
Uses the same five folds and seed as 03_models_and_tests.py. Writes outputs/robustness_checks.json (aggregates only).
"""
import importlib, json, math, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from config import PROCESSED, OUTPUTS, SEED, PRIMARY
warnings.filterwarnings("ignore")

_models = importlib.import_module("03_models_and_tests")
FEATS, CAT_FEATS = _models.FEATS, _models.CAT_FEATS          # keep the feature set in one place


def ml_models():
    """Same settings as 03_models_and_tests.rq1."""
    return {
        "random_forest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
        "xgboost": lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                            colsample_bytree=0.8, random_state=SEED),
    }


def linear_design(X, feats, log_stay=False):
    """Dummy-coded design matrix for the linear baseline; optionally adds ln(stay) and public x ln(stay)."""
    Xd = pd.get_dummies(X, columns=[c for c in CAT_FEATS if c in feats], drop_first=True).astype(float)
    if log_stay and "stay_days" in feats:
        Xd["log_stay"] = np.log(X["stay_days"].clip(lower=1))
        Xd["public_x_log_stay"] = X["facility_public"] * Xd["log_stay"]
    return Xd


def cv_predict(df, feats, log_stay=False):
    """Out-of-fold predictions on the same five folds (seed 42) as script 03."""
    X = df[feats].fillna(df[feats].median())
    y = df["log_oop"].to_numpy()
    Xd = linear_design(X, feats, log_stay)
    preds = {"linear": np.zeros(len(df))} | {k: np.zeros(len(df)) for k in ml_models()}
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        preds["linear"][te] = LinearRegression().fit(Xd.iloc[tr], y[tr]).predict(Xd.iloc[te])
        for k, make in ml_models().items():
            preds[k][te] = make().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return y, preds


def score(y, preds):
    """MAE, RMSE, R² and gain over the linear model; paired t-test of the best ML model against linear."""
    out = {}
    for k, p in preds.items():
        out[k] = {"mae_log": float(np.mean(np.abs(y - p))), "rmse_log": float(np.sqrt(np.mean((y - p) ** 2))),
                  "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))}
    base = out["linear"]["mae_log"]
    for k in out:
        out[k]["gain_vs_linear_pct"] = float((base - out[k]["mae_log"]) / base * 100)
    best = min((k for k in out if k != "linear"), key=lambda k: out[k]["mae_log"])
    e_best, e_lin = np.abs(y - preds[best]), np.abs(y - preds["linear"])
    d = e_best - e_lin
    t = stats.ttest_rel(e_best, e_lin)
    return {"scores": out, "best": best, "t": float(t.statistic), "p": float(t.pvalue), "dz": float(d.mean() / d.std(ddof=1)),
            "typical_error_factor_best": float(math.exp(out[best]["mae_log"]))}


def sex_audit(df, abs_err):
    """MAE by recorded sex; episodes with missing sex (imputed for the models) are reported separately."""
    overall = float(abs_err.mean())
    known = df["female"].notna().to_numpy()              # sex is missing when the patient died or left
    res = {"overall_mae": overall, "n_known": int(known.sum()), "n_imputed": int((~known).sum())}
    groups = [("male", known & (df["female"] == 0).to_numpy()), ("female", known & (df["female"] == 1).to_numpy()),
              ("imputed_group", ~known)]
    for lab, m in groups:
        mae = float(abs_err[m].mean()) if m.any() else None
        res[lab] = {"n": int(m.sum()), "mae": mae, "ratio": None if mae is None else mae / overall}
    return res


def died_or_left(df):
    """Episodes with no person record (patient died or left) vs. the rest."""
    g = df["member_deceased"] == 1
    if g.sum() == 0 or (~g).sum() == 0:
        return {"note": "no comparison possible: one group is empty", "n_died_or_left": int(g.sum())}
    w = df["weight"]
    out = {}
    for lab, m in (("died_or_left", g), ("others", ~g)):
        out[lab] = {"n": int(m.sum()), "che_pct": float(100 * df.loc[m, "che"].mean()),
                    "che_weighted_pct": float(100 * np.average(df.loc[m, "che"], weights=w[m])),
                    "median_oop": float(df.loc[m, "oop_medical"].median()),
                    "private_pct": float(100 * (df.loc[m, "facility_type"] == 3).mean())}
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(g, df["che"]))
    out["chi2"], out["chi2_df"], out["p"] = float(chi2), int(dof), float(p)
    out["mann_whitney_p_oop"] = float(stats.mannwhitneyu(df.loc[g, "log_oop"], df.loc[~g, "log_oop"]).pvalue)
    return out


def rq3_adjusted(df):
    """Public vs. private ln(OOPE + 1), unadjusted and adjusted for case mix. Ward type is left out because it lies
    on the path from hospital type to cost (public hospitals mostly use free wards)."""
    d = df[df["facility_type"].isin([1, 3])].copy()
    d["log_stay"] = np.log(d["stay_days"].clip(lower=1))
    f = ("log_oop ~ facility_public + surgery_received + log_stay + age + sector_urban + outside_district + outside_state"
         " + C(mpce_quintile) + C(social_group) + hh_size")
    cl = {"cov_type": "cluster", "cov_kwds": {"groups": d["psu"]}} if "psu" in d else {"cov_type": "HC1"}
    def pack(m):
        b, (lo, hi) = m.params["facility_public"], m.conf_int().loc["facility_public"]
        return {"b": float(b), "ci_low": float(lo), "ci_high": float(hi), "p": float(m.pvalues["facility_public"]),
                "pct_diff_public_vs_private": float(100 * (math.exp(b) - 1)), "r2": float(m.rsquared)}
    return {"n": int(len(d)), "se": "FSU-clustered" if "psu" in d else "HC1", "unadjusted": pack(smf.ols("log_oop ~ facility_public", data=d).fit(**cl)),
            "adjusted": pack(smf.ols(f, data=d).fit(**cl))}


def design_effects(res, p1=.63, p2=.48, alpha=.05, power=.80, meaningful_fall=.15):
    """Design effect, effective n and design-adjusted power for CHE, from the output of 08_design_based_check.py."""
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    n_req = 2 * ((z_a + z_b) / h) ** 2                                   # per survey, simple random sampling
    out = {"n_required_per_survey_srs": n_req}
    for k in ("nss75", "nss80"):
        p, n, se = res[k]["weighted_che"], res[k]["n"], res[k]["se"]
        deff = (se / math.sqrt(p * (1 - p) / n)) ** 2
        out[k] = {"deff": deff, "n_effective": n / deff, "n_required_with_deff": n_req * deff}
    se_diff = res["difference"]["se_points"] / 100
    out["design_power_15pt_fall"] = float(stats.norm.cdf(meaningful_fall / se_diff - z_a))
    out["minimum_detectable_fall_pts"] = float(100 * (z_a + z_b) * se_diff)
    return out


def rq1_clustered(df, abs_err_ml, abs_err_linear):
    """Paired difference in absolute error with a design-based (FSU-clustered, stratified) SE; equal episode weights."""
    chk = importlib.import_module("08_design_based_check")
    d = np.asarray(abs_err_ml) - np.asarray(abs_err_linear)
    t = df[["stratum", "psu"]].assign(d=d, weight=1.0)
    mean, se, dof = chk.design_mean(t, "d")
    srs_se = d.std(ddof=1) / math.sqrt(len(d))
    tval = mean / se
    return {"mean_diff": float(mean), "se_clustered": float(se), "se_srs": float(srs_se), "deff": float((se / srs_se) ** 2),
            "t": float(tval), "df": int(dof), "p": float(2 * stats.t.sf(abs(tval), dof))}


def rq2_clustered(df):
    """RQ2 logistic regression with FSU-clustered SEs on the same complete cases as script 03."""
    cc = df.dropna(subset=FEATS)
    m = smf.logit("che ~ " + " + ".join(_models.RQ2_TERMS), data=cc).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": cc["psu"]})
    ci = np.exp(m.conf_int())
    return {"n": int(m.nobs), "n_clusters": int(cc["psu"].nunique()),
            "terms": {k: {"or": float(np.exp(m.params[k])), "ci_low": float(ci.loc[k, 0]), "ci_high": float(ci.loc[k, 1]),
                          "p": float(m.pvalues[k])} for k in m.params.index if k != "Intercept"}}


def sample_size_sensitivity(alpha=.05, power=.80):
    """Minimum n per research question for the planned and two smaller effect sizes (same formulas as script 02)."""
    from statsmodels.stats.power import TTestPower, TTestIndPower
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    rq2 = lambda o: math.ceil(z ** 2 / (.63 * .37 * math.log(o) ** 2) / (1 - .30))
    h = lambda p2: 2 * math.asin(math.sqrt(.63)) - 2 * math.asin(math.sqrt(p2))
    return {"rq1_dz": {str(e): math.ceil(TTestPower().solve_power(effect_size=e, alpha=alpha, power=power)) for e in (.20, .15, .10)},
            "rq2_or": {str(o): rq2(o) for o in (1.5, 1.3, 1.2)},
            "rq3_d_per_group": {str(e): math.ceil(TTestIndPower().solve_power(effect_size=e, alpha=alpha, power=power)) for e in (.30, .20, .15)},
            "rq4_p2025_per_survey": {str(p2): math.ceil(2 * (z / h(p2)) ** 2) for p2 in (.48, .53, .55)}}


def main():
    primary = PRIMARY if (PROCESSED / f"cancer_episodes_{PRIMARY}.csv").exists() else "nss75"
    if primary != PRIMARY:
        print(f"cancer_episodes_{PRIMARY}.csv not found: analysing nss75 instead")
    df = _models.load_round(primary).reset_index(drop=True)
    y, preds = cv_predict(df, FEATS)
    res = {"round": primary, "n": len(df),
           "A_stronger_baseline": score(*cv_predict(df, FEATS, log_stay=True)),
           "B_no_stay": score(*cv_predict(df, [f for f in FEATS if f != "stay_days"])),
           "C_sex_audit_known": sex_audit(df, np.abs(y - preds["random_forest"])),
           "D_died_or_left": died_or_left(df),
           "E_rq3_adjusted": rq3_adjusted(df)}
    if {"psu", "stratum"} <= set(df):
        res["G_rq1_clustered"] = rq1_clustered(df, np.abs(y - preds["random_forest"]), np.abs(y - preds["linear"]))
        res["H_rq2_clustered"] = rq2_clustered(df)
    res["I_sample_size_sensitivity"] = sample_size_sensitivity()
    f = OUTPUTS / "design_based_check.json"
    res["F_rq4_design"] = design_effects(json.load(open(f))) if f.exists() else {"note": "run 08_design_based_check.py first"}
    json.dump(res, open(OUTPUTS / "robustness_checks.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
