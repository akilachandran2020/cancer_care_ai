"""Step 13 - Design-based (survey-weighted) estimates for RQ2-RQ4 in Python: a cross-check for the SPSS Complex Samples run.

Same design as spss/rq2_rq4.sps: analysis weight = weight, strata = NSS region x sector (stratum), clusters = first-stage units
(psu), with-replacement variance, Taylor-series linearisation, t-based confidence intervals with
df = (number of FSUs) - (number of strata).

  descriptives  weighted mean OOPE and CHE by hospital type (domain estimation, full design kept)
  RQ2           weighted (pseudo-likelihood) logistic regression for CHE; ln(stay) version = final model, stay version = interim
  RQ3           weighted regression of ln(OOPE + 1) on hospital type; pairwise differences (private - public etc.)
  RQ4           weighted CHE by survey, difference, and adjusted logistic regression on the combined file

RQ2 is run twice: listwise on complete cases (what CSLOGISTIC does by default) and as a subpopulation of complete cases that
keeps the full design (what the SPSS SUBPOP option does). Writes outputs/design_based_models.json (aggregates only).
"""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import patsy
from config import OUTPUTS, PRIMARY
from db import load_episodes

warnings.filterwarnings("ignore")
FAC = {1: "Public", 2: "Charitable/NGO", 3: "Private"}


def _meat(S, stratum, psu):
    """Between-FSU variance of weighted score totals within strata (with replacement); returns B, n_fsu, n_strata."""
    T = pd.DataFrame(S).groupby([np.asarray(stratum), np.asarray(psu)]).sum()
    B = np.zeros((S.shape[1], S.shape[1])); n_psu = n_str = 0
    for _, g in T.groupby(level=0):
        n = len(g)
        if n < 2:
            continue
        dev = g.to_numpy() - g.to_numpy().mean(0)
        B += n / (n - 1) * dev.T @ dev
        n_psu += n; n_str += 1
    return B, n_psu, n_str


def _tab(names, beta, V, df, exp=False):
    se = np.sqrt(np.diag(V)); t = beta / se; tcrit = stats.t.ppf(.975, df)
    out = {}
    for n, b, s, tv in zip(names, beta, se, t):
        r = {"b": float(b), "se": float(s), "t": float(tv), "p": float(2 * stats.t.sf(abs(tv), df)),
             "ci_low": float(b - tcrit * s), "ci_high": float(b + tcrit * s)}
        if exp:
            r |= {"or": float(np.exp(b)), "or_ci_low": float(np.exp(b - tcrit * s)), "or_ci_high": float(np.exp(b + tcrit * s))}
        out[n] = r
    return out


def weighted_logit(df, formula, domain=None):
    """Pseudo-likelihood logistic regression with Taylor-series variance. `domain` (bool Series) keeps the full design but
    gives zero weight outside the domain (subpopulation analysis); otherwise rows with missing values are dropped."""
    d = df.copy()
    if domain is not None:
        d.loc[~domain, "weight"] = 0.0
        cols = [c.strip() for c in formula.replace("~", "+").replace("C(", " ").replace(")", " ").split("+")]
        for c in cols:
            if c in d and d[c].isna().any():
                d[c] = d[c].fillna(d[c].median())
    y, X = patsy.dmatrices(formula, d, return_type="dataframe", NA_action="drop")
    d = d.loc[y.index]
    w = d["weight"].to_numpy()
    fit = sm.GLM(y.iloc[:, 0], X, family=sm.families.Binomial(), freq_weights=w + 1e-12).fit()
    mu = fit.predict(X).to_numpy()
    A = X.to_numpy().T @ (X.to_numpy() * (w * mu * (1 - mu))[:, None])
    S = X.to_numpy() * (w * (y.iloc[:, 0].to_numpy() - mu))[:, None]
    B, n_psu, n_str = _meat(S, d["stratum"], d["psu"])
    Ai = np.linalg.inv(A); V = Ai @ B @ Ai
    df_ = n_psu - n_str
    res = {"n_used": int((w > 0).sum()), "n_fsu": int(n_psu), "n_strata": int(n_str), "df": int(df_),
           "params": _tab(list(X.columns), fit.params.to_numpy(), V, df_, exp=True)}
    # Wald F test for each factor (all its dummy columns jointly), SPSS-style F with (q, df - q + 1) degrees of freedom
    names = list(X.columns); res["factor_tests"] = {}
    for fct in sorted({n.split("[")[0] for n in names if "[" in n}):
        idx = [i for i, n in enumerate(names) if n.startswith(fct + "[")]
        b, Vq = fit.params.to_numpy()[idx], V[np.ix_(idx, idx)]
        q = len(idx); wald = float(b @ np.linalg.solve(Vq, b)); F = wald / q * (df_ - q + 1) / df_ if df_ > q else np.nan
        res["factor_tests"][fct] = {"q": q, "wald_chi2": wald, "F": float(F), "df1": q, "df2": int(df_ - q + 1),
                                    "p": float(stats.f.sf(F, q, df_ - q + 1))}
    return res


def weighted_ols(df, formula, contrasts=None):
    """Weighted least squares with Taylor-series variance; `contrasts` = {label: vector over the design columns}."""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe", NA_action="drop")
    d = df.loc[y.index]; w = d["weight"].to_numpy(); Xa = X.to_numpy(); ya = y.iloc[:, 0].to_numpy()
    A = Xa.T @ (Xa * w[:, None]); beta = np.linalg.solve(A, Xa.T @ (w * ya)); e = ya - Xa @ beta
    B, n_psu, n_str = _meat(Xa * (w * e)[:, None], d["stratum"], d["psu"])
    Ai = np.linalg.inv(A); V = Ai @ B @ Ai; df_ = n_psu - n_str
    res = {"n_used": int(len(d)), "n_fsu": int(n_psu), "n_strata": int(n_str), "df": int(df_), "params": _tab(list(X.columns), beta, V, df_)}
    if contrasts:
        res["contrasts"] = {}
        for lab, c in contrasts(list(X.columns)).items():
            c = np.asarray(c, float); est = float(c @ beta); se = float(np.sqrt(c @ V @ c)); t = est / se; tc = stats.t.ppf(.975, df_)
            res["contrasts"][lab] = {"diff": est, "se": se, "t": float(t), "df": int(df_), "p": float(2 * stats.t.sf(abs(t), df_)),
                                     "ci_low": est - tc * se, "ci_high": est + tc * se}
    return res


def domain_mean(df, y, mask):
    """Weighted mean of y within a domain with a linearised SE; full design retained (SPSS CSDESCRIPTIVES /SUBPOP)."""
    w = df["weight"].to_numpy() * mask.to_numpy(); yv = df[y].to_numpy(float)
    p = (w * yv).sum() / w.sum()
    z = pd.DataFrame({"z": w * (yv - p) / w.sum()}).groupby([df["stratum"].to_numpy(), df["psu"].to_numpy()]).sum()
    var, n_psu, n_str = 0.0, 0, 0
    for _, g in z.groupby(level=0):
        n = len(g)
        if n < 2:
            continue
        var += n / (n - 1) * ((g["z"] - g["z"].mean()) ** 2).sum(); n_psu += n; n_str += 1
    se = float(np.sqrt(var)); df_ = n_psu - n_str; tc = stats.t.ppf(.975, df_)
    return {"n_unweighted": int(mask.sum()), "weighted_count": float(w.sum()), "mean": float(p), "se": se, "df": int(df_),
            "ci_low": float(p - tc * se), "ci_high": float(p + tc * se)}


def main():
    d25 = load_episodes(PRIMARY).dropna(subset=["log_oop", "annual_consumption"]).reset_index(drop=True)
    d75 = load_episodes("nss75").dropna(subset=["log_oop", "annual_consumption"]).reset_index(drop=True)
    d25["log_stay"] = np.log(d25["stay_days"].clip(lower=1))
    out = {"design": "strata = region x sector; clusters = FSU; weight = survey multiplier; with-replacement Taylor-series variance"}

    out["descriptives_by_hospital"] = {
        FAC[c]: {"oop_medical": domain_mean(d25, "oop_medical", d25["facility_type"] == c), "che": domain_mean(d25, "che", d25["facility_type"] == c)}
        for c in (1, 2, 3)}
    out["descriptives_all_2025"] = {"oop_medical": domain_mean(d25, "oop_medical", d25["facility_type"] > 0),
                                    "che": domain_mean(d25, "che", d25["facility_type"] > 0)}

    base = "age + female + sector_urban + facility_public + {stay} + surgery_received + outside_state + insured + C(mpce_quintile) + C(social_group)"
    cc = d25[["age", "female", "sector_urban", "facility_public", "stay_days", "surgery_received", "outside_state", "insured", "mpce_quintile", "social_group"]].notna().all(axis=1)
    out["rq2"] = {}
    for lab, stay in (("final_log_stay", "log_stay"), ("interim_stay_days", "stay_days")):
        f = "che ~ " + base.format(stay=stay)
        out["rq2"][lab] = {"listwise_complete_cases": weighted_logit(d25[cc].reset_index(drop=True), f),
                           "subpopulation_complete_cases": weighted_logit(d25, f, domain=cc)}
        # unweighted reference for the report text
    out["rq2"]["n_complete_cases"] = int(cc.sum())

    contrasts = lambda cols: {
        "Private - Public": [1 if c == "C(facility_type)[T.3]" else 0 for c in cols],
        "Charitable/NGO - Public": [1 if c == "C(facility_type)[T.2]" else 0 for c in cols],
        "Private - Charitable/NGO": [(1 if c == "C(facility_type)[T.3]" else -1 if c == "C(facility_type)[T.2]" else 0) for c in cols]}
    out["rq3"] = weighted_ols(d25, "log_oop ~ C(facility_type)", contrasts)
    pub_priv = d25[d25["facility_type"].isin([1, 3])].reset_index(drop=True)
    out["rq3_public_vs_private_only"] = weighted_ols(pub_priv, "log_oop ~ C(facility_type)", lambda cols: {"Private - Public": [1 if c == "C(facility_type)[T.3]" else 0 for c in cols]})

    cols = ["che", "facility_public", "facility_type", "sector_urban", "mpce_quintile", "weight", "oop_medical", "psu", "stratum"]
    both = pd.concat([d75[cols].assign(survey_2025=0),
                      d25[cols].assign(survey_2025=1, stratum=d25["stratum"] + 100000, psu=d25["psu"] + 10000000)], ignore_index=True)
    out["rq4"] = {"che_by_survey": {lab: domain_mean(both, "che", both["survey_2025"] == s) for s, lab in ((0, "2017-18"), (1, "2025"))}}
    p1, p2 = out["rq4"]["che_by_survey"]["2017-18"], out["rq4"]["che_by_survey"]["2025"]
    diff = p2["mean"] - p1["mean"]
    # the two surveys are independent, so the SE of the difference combines the two domain SEs (the combined-file design
    # contains both, SPSS CSGLM gives the same estimate)
    glm = weighted_ols(both, "che ~ survey_2025", lambda c: {"2025 - 2017-18": [0, 1]})
    out["rq4"]["difference_csglm_equivalent"] = glm["contrasts"]["2025 - 2017-18"]
    out["rq4"]["adjusted_logistic"] = weighted_logit(both, "che ~ survey_2025 + C(mpce_quintile) + sector_urban + facility_public")
    json.dump(out, open(OUTPUTS / "design_based_models.json", "w"), indent=2, default=float)

    print("Weighted 2025 descriptives:")
    for k, v in out["descriptives_by_hospital"].items():
        print(f"  {k:15s} OOPE mean {v['oop_medical']['mean']:>10,.0f} (SE {v['oop_medical']['se']:,.0f})  CHE {100*v['che']['mean']:.1f}% (SE {100*v['che']['se']:.1f})")
    for lab in ("final_log_stay", "interim_stay_days"):
        r = out["rq2"][lab]["listwise_complete_cases"]
        print(f"RQ2 [{lab}] n={r['n_used']}, FSUs={r['n_fsu']}, strata={r['n_strata']}, df={r['df']}")
        for k, v in r["params"].items():
            if k != "Intercept":
                print(f"   {k:28s} OR {v['or']:.2f} [{v['or_ci_low']:.2f}, {v['or_ci_high']:.2f}] p={v['p']:.3f}")
    print("RQ3 contrasts:", {k: (round(v['diff'], 3), round(v['se'], 3), round(v['p'], 4)) for k, v in out["rq3"]["contrasts"].items()})
    print("RQ4 CHE:", {k: (round(v["mean"], 4), round(v["se"], 4)) for k, v in out["rq4"]["che_by_survey"].items()},
          "diff", {k: round(v, 4) for k, v in out["rq4"]["difference_csglm_equivalent"].items() if k in ("diff", "se", "t", "p")})
    a = out["rq4"]["adjusted_logistic"]["params"]["survey_2025"]; print("RQ4 adjusted OR:", round(a["or"], 3), [round(a["or_ci_low"], 3), round(a["or_ci_high"], 3)], round(a["p"], 4))


if __name__ == "__main__":
    main()
