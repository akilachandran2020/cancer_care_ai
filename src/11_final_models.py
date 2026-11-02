"""Step 11 - Final models for the final report (all on the primary 2025 round).

1. Hold-out split      80/20 split by first-stage unit (FSU, 'psu'): no FSU is in both parts, so a household cluster cannot
                       leak between training and test data. Seed 42.
2. Tuning              RandomizedSearchCV on the training part only, 5-fold GroupKFold by FSU, scored on MAE (log scale).
                       The final model is chosen on this cross-validation score, never on the test part.
3. Models              linear regression (baseline), random forest, LightGBM, XGBoost and a multilayer perceptron (a small
                       deep-learning baseline), each with the fixed interim settings ("default") and tuned settings.
4. Evaluation          training and test MAE / RMSE / R^2, MAE gain over linear with a cluster bootstrap interval, paired
                       test with FSU-clustered SE.
5. Intervals           LightGBM 10th/90th percentile models, with and without split-conformal calibration; coverage on the
                       test part, by subgroup, and over 10 repeated 5-fold runs.
6. Back-transformation Duan smearing factor for rupee estimates; share of episodes under-predicted by >20%.
7. Two-part model      P(OOPE > 0) x E[OOPE | OOPE > 0] as a sensitivity analysis for the 14% zero-cost episodes.
8. Group calibration   social-group offsets estimated on training residuals, error audit before and after.
9. CHE classifiers     logistic regression and gradient boosting for the CHE risk flag: accuracy, precision, recall, F1,
                       ROC AUC, confusion matrices, training vs test.
10. Linearity of logit Box-Tidwell test for age, stay and household size in the RQ2 model.

Usage: python 11_final_models.py [--n-iter 20] [--bootstrap 1000]
Writes outputs/final_models.json, outputs/report/table_final_*.csv, outputs/report/fig12-16, and models/*.joblib
(git-ignored). Only aggregates are published.
"""
import argparse, importlib, json, math, warnings
import numpy as np, pandas as pd
from scipy import stats
import joblib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from config import OUTPUTS, MODELS, SEED, PRIMARY
from db import load_episodes
import plot_style as ps

warnings.filterwarnings("ignore")
_m = importlib.import_module("03_models_and_tests")
FEATS, CAT_FEATS = _m.FEATS, _m.CAT_FEATS
CONT = ["age", "stay_days", "hh_size"]                     # scaled for linear / neural models
REPORT = OUTPUTS / "report"
GROUPS = {"sector": "sector_urban", "social_group": "social_group", "hospital": "facility_type", "quintile": "mpce_quintile"}


# ---------------------------------------------------------------------------------------------- preprocessing
def tree_prep():
    """Trees: median imputation only (codes are used directly)."""
    return SimpleImputer(strategy="median")


def linear_prep():
    """Linear / neural: median imputation, one-hot encoding of ward, social group and quintile, scaling of the
    continuous variables. Imputation statistics come from the training part only (inside the pipeline)."""
    num_other = [f for f in FEATS if f not in CAT_FEATS + CONT]
    return ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), CAT_FEATS),
        ("cont", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), CONT),
        ("bin", SimpleImputer(strategy="median"), num_other)])


def default_models():
    """Settings used in the interim report (script 03)."""
    return {
        "linear": Pipeline([("prep", linear_prep()), ("m", LinearRegression())]),
        "random_forest": Pipeline([("prep", tree_prep()), ("m", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, random_state=SEED, n_jobs=-1))]),
        "lightgbm": Pipeline([("prep", tree_prep()), ("m", lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, verbose=-1))]),
        "xgboost": Pipeline([("prep", tree_prep()), ("m", xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED))]),
        "mlp": Pipeline([("prep", linear_prep()), ("m", MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, early_stopping=True,
            max_iter=500, random_state=SEED))]),
    }


def search_spaces():
    from scipy.stats import loguniform, randint, uniform
    return {
        "random_forest": {"m__n_estimators": [300, 500, 800], "m__max_depth": [None, 6, 10, 16],
                          "m__min_samples_leaf": [2, 3, 5, 10, 20], "m__max_features": [0.5, 0.7, 1.0]},
        "lightgbm": {"m__n_estimators": randint(150, 900), "m__learning_rate": loguniform(0.01, 0.12),
                     "m__num_leaves": [7, 15, 31, 63], "m__min_child_samples": randint(8, 60),
                     "m__subsample": uniform(0.6, 0.4), "m__colsample_bytree": uniform(0.5, 0.5),
                     "m__reg_lambda": loguniform(1e-2, 20)},
        "xgboost": {"m__n_estimators": randint(150, 900), "m__learning_rate": loguniform(0.01, 0.12),
                    "m__max_depth": [2, 3, 4, 5, 6], "m__min_child_weight": randint(1, 20),
                    "m__subsample": uniform(0.6, 0.4), "m__colsample_bytree": uniform(0.5, 0.5),
                    "m__reg_lambda": loguniform(1e-2, 20)},
        "mlp": {"m__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (128, 64, 32)],
                "m__alpha": loguniform(1e-4, 1e-1), "m__learning_rate_init": [1e-3, 3e-3, 1e-2]},
    }


# ---------------------------------------------------------------------------------------------------- metrics
def reg_scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"mae_log": float(np.mean(np.abs(y - p))), "rmse_log": float(np.sqrt(np.mean((y - p) ** 2))),
            "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))}


def split(df, test_size=.2):
    """80/20 split by FSU; returns train and test index arrays and verifies there is no FSU overlap."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED).split(df, groups=df["psu"]))
    assert not set(df["psu"].iloc[tr]) & set(df["psu"].iloc[te]), "FSU appears in both training and test parts"
    return tr, te


def cluster_bootstrap_gain(y, p_base, p_ml, psu, reps=1000, seed=SEED):
    """Percentile CI for the MAE gain (%) of the ML model over the baseline, resampling whole FSUs."""
    rng = np.random.default_rng(seed)
    e_b, e_m = np.abs(y - p_base), np.abs(y - p_ml)
    codes, uniq = pd.factorize(psu)
    sums_b = np.bincount(codes, e_b, len(uniq)); sums_m = np.bincount(codes, e_m, len(uniq)); cnt = np.bincount(codes, minlength=len(uniq))
    gains = np.empty(reps)
    for i in range(reps):
        s = rng.integers(0, len(uniq), len(uniq))
        n = cnt[s].sum()
        gains[i] = 100 * (sums_b[s].sum() - sums_m[s].sum()) / sums_b[s].sum() if n else np.nan
    return [float(np.nanpercentile(gains, 2.5)), float(np.nanpercentile(gains, 97.5))]


def clustered_paired_test(e_ml, e_base, psu):
    """Paired mean difference in absolute error with an FSU-clustered SE. The 20% test part leaves many strata with a
    single FSU, so strata are ignored here (conservative); df = number of FSUs - 1."""
    d = pd.Series(np.asarray(e_ml) - np.asarray(e_base))
    tot = d.groupby(np.asarray(psu)).sum()
    n_c, n = len(tot), len(d)
    mean = d.mean()
    u = tot - mean * d.groupby(np.asarray(psu)).size()
    se = float(np.sqrt(n_c / (n_c - 1) * (u ** 2).sum()) / n)
    tv = mean / se
    return {"mean_diff": float(mean), "se_clustered": se, "t": float(tv), "df": int(n_c - 1),
            "p": float(2 * stats.t.sf(abs(tv), n_c - 1)), "dz": float(d.mean() / d.std(ddof=1))}


# ----------------------------------------------------------------------------------------- regression block
def regression_block(df, tr, te, n_iter, boot):
    X, y = df[FEATS], df["log_oop"].to_numpy()
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    gtr = df["psu"].iloc[tr].to_numpy()
    out = {"split": {"n_train": len(tr), "n_test": len(te), "psu_train": int(df["psu"].iloc[tr].nunique()),
                     "psu_test": int(df["psu"].iloc[te].nunique()), "test_size": .2, "seed": SEED, "split_by": "FSU (psu)",
                     "tuning_cv": "5-fold GroupKFold by FSU on the training part", "search_iterations": n_iter}}
    models, spaces = default_models(), search_spaces()
    fitted, preds_te, preds_tr, rows, tuned_params = {}, {}, {}, [], {}
    cv = GroupKFold(n_splits=5)
    for name, base in models.items():
        res = {}
        m0 = clone(base).fit(Xtr, ytr)                                            # default settings
        res["default"] = {"train": reg_scores(ytr, m0.predict(Xtr)), "test": reg_scores(yte, m0.predict(Xte))}
        fitted[(name, "default")] = m0
        if name in spaces:                                                          # tuned settings
            rs = RandomizedSearchCV(clone(base), spaces[name], n_iter=n_iter, cv=cv, scoring="neg_mean_absolute_error",
                                    random_state=SEED, n_jobs=-1, refit=True)
            rs.fit(Xtr, ytr, groups=gtr)
            m1 = rs.best_estimator_
            tuned_params[name] = {k.removeprefix("m__"): (v if isinstance(v, (str, tuple, type(None))) else
                                                         (float(v) if isinstance(v, (float, np.floating)) else int(v)))
                                  for k, v in rs.best_params_.items()}
            res["tuned"] = {"train": reg_scores(ytr, m1.predict(Xtr)), "test": reg_scores(yte, m1.predict(Xte)),
                            "cv_mae_log": float(-rs.best_score_)}
            fitted[(name, "tuned")] = m1
        else:
            res["tuned"] = res["default"] | {"cv_mae_log": None}
            fitted[(name, "tuned")] = m0
        out.setdefault("models", {})[name] = res
    out["tuned_params"] = tuned_params
    # CV score of every model on the training part with default settings (for choosing among models on training data only)
    for name in ("linear",):
        sc = []
        for a, b in cv.split(Xtr, ytr, groups=gtr):
            sc.append(np.mean(np.abs(ytr[b] - clone(models[name]).fit(Xtr.iloc[a], ytr[a]).predict(Xtr.iloc[b]))))
        out["models"][name]["tuned"]["cv_mae_log"] = float(np.mean(sc))
    # choose the best ML model by tuned CV MAE on the training part (not by test error)
    ml = [k for k in out["models"] if k not in ("linear", "mlp")]
    best = min(ml, key=lambda k: out["models"][k]["tuned"]["cv_mae_log"])
    out["best_ml_by_training_cv"] = best
    base_mae = out["models"]["linear"]["default"]["test"]["mae_log"]
    for name in out["models"]:
        for kind in ("default", "tuned"):
            r = out["models"][name][kind]
            r["test_gain_vs_linear_pct"] = float((base_mae - r["test"]["mae_log"]) / base_mae * 100)
    pb, pl = fitted[(best, "tuned")].predict(Xte), fitted[("linear", "default")].predict(Xte)
    out["best_test"] = {"model": best, "gain_pct": out["models"][best]["tuned"]["test_gain_vs_linear_pct"],
                        "gain_ci95_cluster_bootstrap": cluster_bootstrap_gain(yte, pl, pb, df["psu"].iloc[te].to_numpy(), boot),
                        "paired_test_clustered": clustered_paired_test(np.abs(yte - pb), np.abs(yte - pl),
                                                                       df["psu"].iloc[te]),
                        "paired_t_srs": {"t": float(stats.ttest_rel(np.abs(yte - pb), np.abs(yte - pl)).statistic),
                                         "p": float(stats.ttest_rel(np.abs(yte - pb), np.abs(yte - pl)).pvalue)}}
    for name in out["models"]:
        pt = fitted[(name, "tuned")].predict(Xte)
        preds_te[name] = pt
    return out, fitted, preds_te, best


# ------------------------------------------------------------------------------- intervals, smearing, calibration
def lgb_quantile(alpha, params=None):
    p = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, random_state=SEED, verbose=-1)
    p.update(params or {})
    return Pipeline([("prep", tree_prep()), ("m", lgb.LGBMRegressor(objective="quantile", alpha=alpha, **p))])


def conformal_q(lo, hi, y, level=.8):
    """Split-conformal (CQR) adjustment: widen (or narrow) both ends by the level-quantile of the conformity scores."""
    s = np.maximum(lo - y, y - hi)
    k = min(int(math.ceil((len(s) + 1) * level)), len(s))
    return float(np.sort(s)[k - 1])


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_block(df, tr, te):
    X, y = df[FEATS], df["log_oop"].to_numpy()
    # calibration set: 25% of the training FSUs
    sub = df.iloc[tr]
    t2, c2 = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(sub, groups=sub["psu"]))
    fit_idx, cal_idx = tr[t2], tr[c2]
    lo_m, hi_m = lgb_quantile(.10).fit(X.iloc[fit_idx], y[fit_idx]), lgb_quantile(.90).fit(X.iloc[fit_idx], y[fit_idx])
    q = conformal_q(lo_m.predict(X.iloc[cal_idx]), hi_m.predict(X.iloc[cal_idx]), y[cal_idx])
    lo, hi = lo_m.predict(X.iloc[te]), hi_m.predict(X.iloc[te])
    yte = y[te]
    res = {"raw_coverage_test": coverage(yte, lo, hi), "conformal_adjustment_log": q,
           "conformal_coverage_test": coverage(yte, lo - q, hi + q),
           "mean_width_log_raw": float(np.mean(hi - lo)), "mean_width_log_conformal": float(np.mean(hi - lo + 2 * q)),
           "n_fit": len(fit_idx), "n_calibration": len(cal_idx), "n_test": len(te)}
    d = df.iloc[te].reset_index(drop=True).assign(lo=lo - q, hi=hi + q, y=yte)
    by = {}
    for lab, col in GROUPS.items():
        for lvl, g in d.groupby(col):
            by[f"{lab}={int(lvl)}"] = {"n": int(len(g)), "coverage": coverage(g["y"], g["lo"], g["hi"])}
    sex_known = d["female"].notna()
    for lab, m in (("sex=male", sex_known & (d["female"] == 0)), ("sex=female", sex_known & (d["female"] == 1)), ("sex=unknown", ~sex_known)):
        by[lab] = {"n": int(m.sum()), "coverage": coverage(d.loc[m, "y"], d.loc[m, "lo"], d.loc[m, "hi"])}
    res["coverage_by_group_test_conformal"] = by
    # repeated 5-fold coverage of the raw quantile models (interim 'in progress' item)
    cov = []
    for seed in range(10):
        c = np.zeros(len(df), bool)
        for a, b in KFold(5, shuffle=True, random_state=seed).split(X):
            l = lgb_quantile(.10).fit(X.iloc[a], y[a]).predict(X.iloc[b]); h = lgb_quantile(.90).fit(X.iloc[a], y[a]).predict(X.iloc[b])
            c[b] = (y[b] >= l) & (y[b] <= h)
        cov.append(float(c.mean()))
    res["repeated_5fold_coverage"] = {"seeds": 10, "mean": float(np.mean(cov)), "min": float(np.min(cov)), "max": float(np.max(cov)),
                                      "sd": float(np.std(cov, ddof=1)), "all_within_75_85": bool(all(.75 <= c <= .85 for c in cov))}
    return res, (lo_m, hi_m, q), (lo - q, hi + q)


def oof_predictions(model, X, y, groups, n=5):
    """Out-of-fold training predictions (grouped), used for the smearing factor and group offsets."""
    p = np.zeros(len(y))
    for a, b in GroupKFold(n).split(X, y, groups):
        p[b] = clone(model).fit(X.iloc[a], y[a]).predict(X.iloc[b])
    return p


def smearing_block(df, tr, te, best_model, X, y):
    gtr = df["psu"].iloc[tr].to_numpy()
    oof = oof_predictions(best_model, X.iloc[tr], y[tr], gtr)
    resid = y[tr] - oof
    smear = float(np.mean(np.exp(resid)))                                         # Duan (1983) smearing factor
    fit = clone(best_model).fit(X.iloc[tr], y[tr])
    p = fit.predict(X.iloc[te]); actual = np.expm1(y[te])
    naive = np.expm1(p)
    smeared = np.maximum(smear * np.exp(p) - 1, 0)
    def pack(pred):
        return {"mae_rupees": float(np.mean(np.abs(actual - pred))), "median_abs_error": float(np.median(np.abs(actual - pred))),
                "mean_predicted": float(pred.mean()), "mean_actual": float(actual.mean()),
                "bias_pct": float(100 * (pred.mean() - actual.mean()) / actual.mean()),
                "share_underpredicted_20pct": float(np.mean(pred < .8 * actual))}
    return {"smearing_factor": smear, "naive_back_transform": pack(naive), "with_smearing": pack(smeared)}, smear, fit, p


def twopart_block(df, tr, te, best_name, smear_single):
    X, y = df[FEATS], df["log_oop"].to_numpy()
    pos = y > 0
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    clf = Pipeline([("prep", tree_prep()), ("m", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                                                    min_child_samples=20, random_state=SEED, verbose=-1))])
    clf.fit(Xtr, pos[tr])
    tr_pos = tr[pos[tr]]
    reg = clone(default_models()["lightgbm"] if best_name == "lightgbm" else default_models()[best_name]).fit(X.iloc[tr_pos], y[tr_pos])
    resid = y[tr_pos] - oof_predictions(clone(default_models()[best_name]), X.iloc[tr_pos], y[tr_pos], df["psu"].iloc[tr_pos].to_numpy())
    smear2 = float(np.mean(np.exp(resid)))
    p_pos = clf.predict_proba(Xte)[:, 1]
    cond = smear2 * np.exp(reg.predict(Xte))
    expected = p_pos * cond
    actual = np.expm1(y[te])
    single = smear_single
    return {"part1_auc_test_nonzero": float(roc_auc_score(pos[te], p_pos)), "n_zero_cost_train": int((~pos[tr]).sum()),
            "n_zero_cost_test": int((~pos[te]).sum()), "smearing_factor_positive": smear2,
            "mae_rupees_expected": float(np.mean(np.abs(actual - expected))),
            "median_abs_error_expected": float(np.median(np.abs(actual - expected))),
            "bias_pct": float(100 * (expected.mean() - actual.mean()) / actual.mean()),
            "share_underpredicted_20pct": float(np.mean(expected < .8 * actual)),
            "mae_log_conditional_on_positive": float(np.mean(np.abs(y[te][pos[te]] - reg.predict(Xte)[pos[te]]))),
            "single_part_log_mae_on_positive_only": None}, reg


def group_calibration_block(df, tr, te, best_model, X, y):
    """Offsets by social group = mean out-of-fold residual in the training part. Audit on the test part.
    Acceptance band (0.80-1.25) applies to the demographic groups: sector, social group and sex. Hospital type and
    consumption quintile are care-setting / income descriptors and are reported but not used for acceptance."""
    gtr = df["psu"].iloc[tr].to_numpy()
    oof = oof_predictions(best_model, X.iloc[tr], y[tr], gtr)
    resid = pd.Series(y[tr] - oof, index=tr)
    offsets = resid.groupby(df["social_group"].iloc[tr]).mean()
    fit = clone(best_model).fit(X.iloc[tr], y[tr])
    p = fit.predict(X.iloc[te])
    p_adj = p + df["social_group"].iloc[te].map(offsets).fillna(0).to_numpy()
    d = df.iloc[te].reset_index(drop=True)
    d["sex"] = np.where(d["female"].isna(), -1, d["female"]).astype(int)          # -1 = sex unknown (imputed)
    res = {"offsets_by_social_group": {int(k): float(v) for k, v in offsets.items()},
           "overall_mae_before": float(np.mean(np.abs(y[te] - p))), "overall_mae_after": float(np.mean(np.abs(y[te] - p_adj)))}
    cols = {**GROUPS, "sex": "sex"}
    for lab, pred in (("before", p), ("after", p_adj)):
        tab, ov = {}, np.mean(np.abs(y[te] - pred))
        for glab, col in cols.items():
            for lvl, idx in d.groupby(col).groups.items():
                idx = np.array(list(idx)); e = y[te][idx] - pred[idx]
                tab[f"{glab}={int(lvl)}"] = {"n": int(len(idx)), "mae": float(np.mean(np.abs(e))), "ratio": float(np.mean(np.abs(e)) / ov),
                                             "mean_signed_error": float(np.mean(e)), "acceptance_group": glab in ("sector", "social_group", "sex")}
        res[lab] = tab
    for lab in ("before", "after"):
        r = [v["ratio"] for v in res[lab].values() if v["acceptance_group"] and v["n"] >= 30]
        res[f"max_ratio_{lab}"], res[f"min_ratio_{lab}"] = float(max(r)), float(min(r))
    return res, offsets


# ----------------------------------------------------------------------------------------- CHE classification
def class_scores(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)), "specificity": float(tn / (tn + fp)), "f1": float(f1_score(y, pred)),
            "auc": float(roc_auc_score(y, prob)), "brier": float(brier_score_loss(y, prob)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def classifier_block(df, tr, te, n_iter):
    from scipy.stats import loguniform, randint, uniform
    X, y = df[FEATS], df["che"].to_numpy()
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    gtr = df["psu"].iloc[tr].to_numpy()
    cv = GroupKFold(5)
    models = {
        "logistic": (Pipeline([("prep", linear_prep()), ("m", LogisticRegression(max_iter=2000, C=1.0))]), {"m__C": loguniform(.01, 30)}),
        "lightgbm": (Pipeline([("prep", tree_prep()), ("m", lgb.LGBMClassifier(random_state=SEED, verbose=-1))]),
                     {"m__n_estimators": randint(100, 600), "m__learning_rate": loguniform(.01, .1), "m__num_leaves": [7, 15, 31],
                      "m__min_child_samples": randint(10, 50), "m__subsample": uniform(.6, .4), "m__subsample_freq": [1],
                      "m__colsample_bytree": uniform(.5, .5), "m__reg_lambda": loguniform(.01, 20)}),
        "random_forest": (Pipeline([("prep", tree_prep()), ("m", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
                          {"m__n_estimators": [300, 500], "m__max_depth": [None, 6, 10], "m__min_samples_leaf": [3, 5, 10, 20],
                           "m__max_features": ["sqrt", 0.5]}),
    }
    out, fitted, probs = {}, {}, {}
    for name, (est, space) in models.items():
        rs = RandomizedSearchCV(est, space, n_iter=n_iter, cv=cv, scoring="roc_auc", random_state=SEED, n_jobs=-1)
        rs.fit(Xtr, ytr, groups=gtr)
        m = rs.best_estimator_
        ptr, pte = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]
        # operating threshold: maximises Youden's J on out-of-fold training predictions (never on the test part)
        oof = np.zeros(len(ytr))
        for a, b in cv.split(Xtr, ytr, gtr):
            oof[b] = clone(m).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
        fpr, tpr, thr = roc_curve(ytr, oof)
        t_star = float(thr[np.argmax(tpr - fpr)])
        out[name] = {"cv_auc_train": float(rs.best_score_), "threshold_youden_oof": t_star,
                     "train_at_0.5": class_scores(ytr, ptr, .5), "test_at_0.5": class_scores(yte, pte, .5),
                     "train_at_youden": class_scores(ytr, ptr, t_star), "test_at_youden": class_scores(yte, pte, t_star),
                     "best_params": {k.removeprefix("m__"): (float(v) if isinstance(v, (float, np.floating)) else
                                                            (int(v) if isinstance(v, (int, np.integer)) else v)) for k, v in rs.best_params_.items()}}
        fitted[name], probs[name] = m, (ptr, pte)
    best = max(out, key=lambda k: out[k]["cv_auc_train"])
    out["best_by_training_cv_auc"] = best
    # RQ2 logistic (statsmodels) evaluated out of sample
    cc = df.dropna(subset=FEATS)
    cc_tr, cc_te = cc.loc[cc.index.isin(df.index[tr])], cc.loc[cc.index.isin(df.index[te])]
    f = "che ~ " + " + ".join(_m.RQ2_TERMS)
    mod = smf.logit(f, data=cc_tr).fit(disp=0)
    out["rq2_logistic_holdout"] = {"n_train_complete": int(len(cc_tr)), "n_test_complete": int(len(cc_te)),
                                   "auc_train": float(roc_auc_score(cc_tr["che"], mod.predict(cc_tr))),
                                   "auc_test": float(roc_auc_score(cc_te["che"], mod.predict(cc_te))),
                                   "pseudo_r2_train": float(mod.prsquared)}
    return out, fitted, probs, best, (ytr, yte)


def box_tidwell(df):
    """Linearity of the logit for the two continuous RQ2 predictors (age, length of stay): add x*ln(x) terms to the
    RQ2 model; a significant term signals non-linearity. Bonferroni threshold 0.05 / 2. Age is shifted by 1 (ages of 0)."""
    cc = df.dropna(subset=FEATS).copy()
    shifted = {"age": cc["age"] + 1, "stay_days": cc["stay_days"]}
    for k, v in shifted.items():
        cc[f"{k}_xlnx"] = v * np.log(v)
    base = " + ".join(_m.RQ2_TERMS)
    m = smf.logit("che ~ " + base + " + age_xlnx + stay_days_xlnx", data=cc).fit(disp=0)
    res = {k: {"coef": float(m.params[f"{k}_xlnx"]), "p": float(m.pvalues[f"{k}_xlnx"]),
               "nonlinear_at_bonferroni": bool(m.pvalues[f"{k}_xlnx"] < .05 / 2)} for k in shifted}
    # remedy check: use ln(stay) instead of stay and compare AIC
    base_m = smf.logit("che ~ " + base, data=cc).fit(disp=0)
    cc["log_stay"] = np.log(cc["stay_days"].clip(lower=1))
    alt = smf.logit("che ~ " + " + ".join("log_stay" if t == "stay_days" else t for t in _m.RQ2_TERMS), data=cc).fit(disp=0)
    return {"n": int(len(cc)), "terms": res, "any_nonlinear": any(v["nonlinear_at_bonferroni"] for v in res.values()),
            "aic_linear_stay": float(base_m.aic), "aic_log_stay": float(alt.aic),
            "or_per_log_stay_unit": float(np.exp(alt.params["log_stay"])), "p_log_stay": float(alt.pvalues["log_stay"])}



def leakage_check(df):
    """Random 5-fold vs FSU-grouped 5-fold cross-validation with the interim (fixed) settings on all episodes."""
    X, y, g = df[FEATS], df["log_oop"].to_numpy(), df["psu"].to_numpy()
    models = {k: v for k, v in default_models().items() if k != "mlp"}
    res = {"episodes_in_multi_episode_fsu": int(df["psu"].map(df["psu"].value_counts()).gt(1).sum()),
           "fsus": int(df["psu"].nunique()), "fsus_with_more_than_one_episode": int((df["psu"].value_counts() > 1).sum())}
    for lab, splitter in (("random_5fold", KFold(5, shuffle=True, random_state=SEED)), ("fsu_grouped_5fold", GroupKFold(5))):
        pred = {k: np.zeros(len(y)) for k in models}
        for a, b in splitter.split(X, y, g):
            for k, m in models.items():
                pred[k][b] = clone(m).fit(X.iloc[a], y[a]).predict(X.iloc[b])
        mae = {k: float(np.mean(np.abs(y - p))) for k, p in pred.items()}
        res[lab] = {"mae_log": mae, "gain_vs_linear_pct": {k: float((mae["linear"] - v) / mae["linear"] * 100) for k, v in mae.items() if k != "linear"}}
    return res


def tweedie_block(df, tr, te):
    """Direct rupee-scale model for expected cost: LightGBM with a Tweedie objective handles the zero-cost spike and the
    right skew without back-transformation. Compared with the back-transformed log model on the test part."""
    X, y = df[FEATS], df["oop_medical"].to_numpy(float)
    best, best_cv = None, 1e18
    for power in (1.2, 1.5, 1.8):
        for lr, leaves in ((0.03, 7), (0.03, 15)):
            sc = []
            for a, b in GroupKFold(5).split(X.iloc[tr], y[tr], df["psu"].iloc[tr]):
                m = Pipeline([("prep", tree_prep()), ("m", lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=power, n_estimators=300,
                                                                            learning_rate=lr, num_leaves=leaves, min_child_samples=30, random_state=SEED, verbose=-1))])
                sc.append(np.mean(np.abs(y[tr][b] - m.fit(X.iloc[tr].iloc[a], y[tr][a]).predict(X.iloc[tr].iloc[b]))))
            if np.mean(sc) < best_cv:
                best_cv, best = float(np.mean(sc)), (power, lr, leaves)
    power, lr, leaves = best
    m = Pipeline([("prep", tree_prep()), ("m", lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=power, n_estimators=300,
                                                                learning_rate=lr, num_leaves=leaves, min_child_samples=30, random_state=SEED, verbose=-1))])
    p = m.fit(X.iloc[tr], y[tr]).predict(X.iloc[te]); a = y[te]
    return {"variance_power": power, "learning_rate": lr, "num_leaves": leaves, "cv_mae_rupees_train": best_cv,
            "test_mae_rupees": float(np.mean(np.abs(a - p))), "test_median_abs_error": float(np.median(np.abs(a - p))),
            "mean_predicted": float(p.mean()), "mean_actual": float(a.mean()), "bias_pct": float(100 * (p.mean() - a.mean()) / a.mean()),
            "share_underpredicted_20pct": float(np.mean(p < .8 * a)), "pearson_r": float(np.corrcoef(p, a)[0, 1])}


def rq2_final(df, tr, te):
    """RQ2 logistic regression with ln(stay) in place of stay (Box-Tidwell found stay non-linear): odds ratios, FSU-clustered
    CIs, VIF, held-out AUC."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    cc = df.dropna(subset=FEATS).copy()
    cc["log_stay"] = np.log(cc["stay_days"].clip(lower=1))
    terms = [("log_stay" if t == "stay_days" else t) for t in _m.RQ2_TERMS]
    f = "che ~ " + " + ".join(terms)
    m = smf.logit(f, data=cc).fit(disp=0)
    mc = smf.logit(f, data=cc).fit(disp=0, cov_type="cluster", cov_kwds={"groups": cc["psu"]})
    ci, cic = np.exp(m.conf_int()), np.exp(mc.conf_int())
    vif = {n: float(variance_inflation_factor(m.model.exog, i)) for i, n in enumerate(m.model.exog_names) if n != "Intercept"}
    tr_idx = cc.index.isin(df.index[tr]); te_idx = cc.index.isin(df.index[te])
    mh = smf.logit(f, data=cc[tr_idx]).fit(disp=0)
    return {"formula": f, "n": int(m.nobs), "events": int(m.model.endog.sum()), "events_per_parameter": float(m.model.endog.sum() / (len(m.params) - 1)),
            "pseudo_r2": float(m.prsquared), "auc_in_sample": float(roc_auc_score(m.model.endog, m.predict())),
            "auc_train_part": float(roc_auc_score(cc[tr_idx]["che"], mh.predict(cc[tr_idx]))),
            "auc_test_part": float(roc_auc_score(cc[te_idx]["che"], mh.predict(cc[te_idx]))), "max_vif": max(vif.values()), "vif": vif,
            "terms": {k: {"or": float(np.exp(m.params[k])), "ci_low": float(ci.loc[k, 0]), "ci_high": float(ci.loc[k, 1]), "p": float(m.pvalues[k]),
                          "or_cluster_ci_low": float(cic.loc[k, 0]), "or_cluster_ci_high": float(cic.loc[k, 1]), "p_cluster": float(mc.pvalues[k])}
                      for k in m.params.index if k != "Intercept"},
            "llr_pvalue": float(m.llr_pvalue)}


# --------------------------------------------------------------------------------------------------- figures
def figures(out, df, te, y_te_reg, preds_te, best, probs, ycls, interval_te, fitted_cls, cls_best):
    REPORT.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    # Fig 12: ROC curves, training vs test (best classifier) and test curves of all classifiers
    ytr, yte = ycls
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (lab, which) in zip(axs, (("A. Best model: training vs test", None), ("B. All models: test part", "all"))):
        if which is None:
            for nm, idx, col in (("Training", 0, ps.NAVY), ("Test", 1, ps.ORANGE)):
                yy = ytr if idx == 0 else yte
                f, t, _ = roc_curve(yy, probs[cls_best][idx])
                ax.plot(f, t, color=col, lw=2, label=f"{nm} (AUC = {roc_auc_score(yy, probs[cls_best][idx]):.2f})")
        else:
            for nm, col in zip(probs, (ps.NAVY, ps.ORANGE, ps.MID)):
                f, t, _ = roc_curve(yte, probs[nm][1])
                ax.plot(f, t, color=col, lw=2, label=f"{nm.replace('_', ' ').title()} (AUC = {roc_auc_score(yte, probs[nm][1]):.2f})")
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="Chance")
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.set_title(lab, fontsize=10, loc="left")
        ax.legend(loc="lower right", frameon=False); ps.grid(ax, "both")
    ps.save(fig, REPORT / "fig12_roc_curves.png")
    # Fig 13: confusion matrix of the best classifier on the test part at the training-derived threshold
    cm = out["classifier"][cls_best]["test_at_youden"]["confusion"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(mat, cmap=ps.BLUE_ORANGE.reversed() if False else "Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}\n({100 * mat[i, j] / mat.sum():.1f}%)", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["No CHE", "CHE"]); ax.set_yticks([0, 1], ["No CHE", "CHE"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.spines[:].set_visible(False)
    ps.save(fig, REPORT / "fig13_confusion_matrix.png")
    # Fig 14: train vs test MAE for every model (tuned)
    names = list(out["regression"]["models"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = .38; xs = np.arange(len(names))
    tr_ = [out["regression"]["models"][n]["tuned"]["train"]["mae_log"] for n in names]
    te_ = [out["regression"]["models"][n]["tuned"]["test"]["mae_log"] for n in names]
    ax.bar(xs - w / 2, tr_, w, color=ps.NAVY, label="Training"); ax.bar(xs + w / 2, te_, w, color=ps.ORANGE, label="Test")
    for x, a, b in zip(xs, tr_, te_):
        ax.text(x - w / 2, a + .03, f"{a:.2f}", ha="center", fontsize=8); ax.text(x + w / 2, b + .03, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs, [n.replace("_", " ").replace("mlp", "neural network (MLP)").title().replace("Lightgbm", "LightGBM").replace("Xgboost", "XGBoost") for n in names], rotation=12)
    ax.set_ylabel("MAE on ln(OOPE + 1)"); ax.legend(frameon=False); ps.grid(ax)
    ps.nice_axis(ax, max(tr_ + te_), 0.5)
    ps.save(fig, REPORT / "fig14_train_test_mae.png")
    # Fig 15: predicted vs actual (best model, test part)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    p = preds_te[best]
    ax.scatter(p, y_te_reg, s=7, alpha=.35, color=ps.NAVY, edgecolors="none")
    lim = [0, max(p.max(), y_te_reg.max()) + .5]
    ax.plot(lim, lim, color=ps.ORANGE, lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Predicted ln(OOPE + 1)"); ax.set_ylabel("Actual ln(OOPE + 1)"); ax.legend(frameon=False, loc="upper left")
    ps.grid(ax, "both"); ax.set_xlim(lim); ax.set_ylim(lim)
    ps.save(fig, REPORT / "fig15_predicted_vs_actual.png")
    # Fig 16: coverage by group (conformal 80% interval, test part)
    cov = out["intervals"]["coverage_by_group_test_conformal"]
    keys = [k for k, v in cov.items() if v["n"] >= 30]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [100 * cov[k]["coverage"] for k in keys]
    ax.barh(range(len(keys)), vals, color=[ps.NAVY if 75 <= v <= 85 else ps.ORANGE for v in vals])
    ax.axvspan(75, 85, color=ps.LIGHT, alpha=.45, zorder=0, label="75%-85% target")
    ax.axvline(80, color="#555555", lw=1, ls="--")
    ax.set_yticks(range(len(keys)), [f"{k} (n = {cov[k]['n']})" for k in keys], fontsize=8)
    ax.set_xlabel("Actual costs inside the 80% range (%)"); ax.set_xlim(40, 100); ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis(); ps.grid(ax, "x")
    ps.save(fig, REPORT / "fig16_interval_coverage_by_group.png")


# ------------------------------------------------------------------------------------------------------ main
def main(n_iter=20, boot=1000):
    df = load_episodes(PRIMARY).dropna(subset=["log_oop", "annual_consumption"]).reset_index(drop=True)
    X, y = df[FEATS], df["log_oop"].to_numpy()
    tr, te = split(df)
    print(f"split: {len(tr)} train / {len(te)} test episodes; {df['psu'].iloc[tr].nunique()} / {df['psu'].iloc[te].nunique()} FSUs")
    out = {"round": PRIMARY, "n": len(df)}
    reg, fitted, preds_te, best = regression_block(df, tr, te, n_iter, boot)
    out["regression"] = reg
    print("regression done; best ML model by training CV:", best)
    best_model = clone(fitted[(best, "tuned")])
    out["intervals"], (lo_m, hi_m, q), (lo_te, hi_te) = interval_block(df, tr, te)
    print("intervals done")
    out["rupee_estimates"], smear, fit_tr, p_te = smearing_block(df, tr, te, best_model, X, y)
    out["two_part"], _ = twopart_block(df, tr, te, best, smear)
    # single-part log MAE on positive episodes for comparison with part 2
    pos_te = y[te] > 0
    out["two_part"]["single_part_log_mae_on_positive_only"] = float(np.mean(np.abs(y[te][pos_te] - p_te[pos_te])))
    out["group_calibration"], offsets = group_calibration_block(df, tr, te, best_model, X, y)
    out["classifier"], fitted_cls, probs, cls_best, ycls = classifier_block(df, tr, te, n_iter)
    out["linearity_of_logit"] = box_tidwell(df)
    out["rq2_final"] = rq2_final(df, tr, te)
    out["leakage_check"] = leakage_check(df)
    out["tweedie_expected_cost"] = tweedie_block(df, tr, te)
    print("classification and linearity done; best classifier:", cls_best)
    figures(out, df, te, y[te], preds_te, best, probs, ycls, (lo_te, hi_te), fitted_cls, cls_best)

    # ---- final fitted objects on ALL episodes for the companion prototype (after the test evaluation is finished)
    MODELS.mkdir(parents=True, exist_ok=True)
    final_cost = clone(best_model).fit(X, y)
    oof_all = oof_predictions(best_model, X, y, df["psu"].to_numpy())
    smear_all = float(np.mean(np.exp(y - oof_all)))
    lo_f, hi_f = lgb_quantile(.10).fit(X, y), lgb_quantile(.90).fit(X, y)
    final_cls = clone(fitted_cls[cls_best]).fit(X, df["che"].to_numpy())
    explainer = default_models()["xgboost"].fit(X, y)
    t_star = out["classifier"][cls_best]["threshold_youden_oof"]
    joblib.dump({"feats": FEATS, "cost_model": final_cost, "cost_model_name": best, "smearing": smear_all, "q_low": lo_f, "q_high": hi_f,
                 "conformal_q": q, "che_model": final_cls, "che_model_name": cls_best, "che_threshold": t_star,
                 "explainer": explainer, "social_group_offsets": {int(k): float(v) for k, v in offsets.items()},
                 "trained_on": f"NSS 80th round (2025), n = {len(df)}"}, MODELS / "companion_models.joblib")
    out["deployment_artifact"] = {"file": "models/companion_models.joblib", "cost_model": best, "che_classifier": cls_best,
                                  "che_threshold": t_star, "smearing_factor_all": smear_all, "git_ignored": True}
    json.dump(out, open(OUTPUTS / "final_models.json", "w"), indent=2, default=float)
    # flat tables for the report
    rows = []
    for n, v in reg["models"].items():
        for kind in ("default", "tuned"):
            r = v[kind]
            rows.append({"model": n, "settings": kind, "train_mae": r["train"]["mae_log"], "test_mae": r["test"]["mae_log"],
                         "train_rmse": r["train"]["rmse_log"], "test_rmse": r["test"]["rmse_log"], "train_r2": r["train"]["r2"],
                         "test_r2": r["test"]["r2"], "test_gain_pct": r["test_gain_vs_linear_pct"]})
    pd.DataFrame(rows).to_csv(REPORT / "table_final_regression.csv", index=False)
    crow = []
    for n in ("logistic", "lightgbm", "random_forest"):
        for part in ("train", "test"):
            r = out["classifier"][n][f"{part}_at_youden"]
            crow.append({"model": n, "part": part, **{k: r[k] for k in ("threshold", "accuracy", "precision", "recall", "specificity", "f1", "auc", "brier")}})
    pd.DataFrame(crow).to_csv(REPORT / "table_final_classification.csv", index=False)
    print(json.dumps({"best": best, "test_gain_pct": reg["best_test"]["gain_pct"], "ci": reg["best_test"]["gain_ci95_cluster_bootstrap"],
                      "conformal_cov": out["intervals"]["conformal_coverage_test"], "smearing": smear,
                      "best_cls": cls_best, "cls_test": out["classifier"][cls_best]["test_at_youden"]}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    main(a.n_iter, a.bootstrap)
