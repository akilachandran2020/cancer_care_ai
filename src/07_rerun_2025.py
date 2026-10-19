"""Step 7 - Robustness check: rerun RQ1-RQ3 and the fairness audit on the 2025 (NSS 80th round) episode file.

The 2025 episode file has no ward type, length of stay or surgery fields (they are not mapped from hhscsL4),
so a like-for-like comparison refits BOTH rounds on the features they share. The full 14-feature 2017-18
results in results.json are not changed.

Run after 01 (both rounds) and 03:  python 07_rerun_2025.py
Writes outputs/results_2025.json, outputs/episode_predictions_2025.csv,
       outputs/report/table_rerun_2025.csv and outputs/report/table_rq2_or_by_round.csv
"""
import importlib, json
import numpy as np, pandas as pd
from config import PROCESSED, OUTPUTS

m = importlib.import_module("03_models_and_tests")   # reuse the exact RQ1-RQ3 and fairness code
REPORT = OUTPUTS / "report"

def load(round_):
    f = PROCESSED / f"cancer_episodes_{round_}.csv"
    if not f.exists():
        raise SystemExit(f"Missing {f.name}: set ROUND = '{round_}' in config.py and rerun 01_build_cancer_episodes.py")
    return pd.read_csv(f).dropna(subset=["log_oop", "annual_consumption"])

def run(df, feats):
    df, r1 = m.rq1(df, feats)
    out = {"n": len(df), "rq1": r1, "rq2": m.rq2(df, feats), "rq3": m.rq3(df), "fairness": m.fairness(df),
           "che_rate": float(df["che"].mean()),
           "weighted_che_rate": float(np.average(df["che"], weights=df["weight"]))}
    return df, out

def summary_rows(label, r):
    r1, r2, r3 = r["rq1"], r["rq2"], r["rq3"]
    best, s = r1["best_ml"], r1["scores"]
    o, p = r2["odds_ratios"], r2["p"]
    rows = {"Episodes (n)": r["n"], "Best ML model": best,
            "MAE log, linear": s["linear"]["mae_log"], "MAE log, best ML": s[best]["mae_log"],
            "MAE improvement (%)": r1["mae_improvement_pct"], "R2, linear": s["linear"]["r2"], "R2, best ML": s[best]["r2"],
            "MAE rupees, best ML": s[best]["mae_rupees"], "RQ1 paired t": r1["t"], "RQ1 p": r1["p"], "RQ1 dz": r1["dz"],
            "80% interval coverage": r1["pi80_coverage"], "Top SHAP feature": next(iter(r1["mean_abs_shap"])),
            "RQ2 n (complete cases)": r2["n"], "RQ2 pseudo-R2": r2["pseudo_r2"],
            "OR public hospital": o.get("facility_public"), "p public hospital": p.get("facility_public"),
            "OR scheme coverage": o.get("insured"), "p scheme coverage": p.get("insured"),
            "OR outside home state": o.get("outside_state"), "p outside home state": p.get("outside_state"),
            "RQ3 n public / private": f'{r3["n_public"]} / {r3["n_private"]}',
            "RQ3 median OOPE public": r3["median_oop_public"], "RQ3 median OOPE private": r3["median_oop_private"],
            "RQ3 Welch t": r3["welch_t"], "RQ3 Welch df": r3["welch_df"], "RQ3 p": r3["p"], "RQ3 Cohen d": r3["cohen_d"],
            "CHE rate (unweighted)": r["che_rate"], "CHE rate (weighted)": r["weighted_che_rate"]}
    return pd.Series(rows, name=label)

def main():
    d75, d80 = load("nss75"), load("nss80")
    common = [f for f in m.FEATS if f in d80.columns and d80[f].notna().any()]
    dropped = [f for f in m.FEATS if f not in common]
    print(f"Shared features ({len(common)}): {common}\nNot in the 2025 file: {dropped}")

    _, r75 = run(d75, common)
    d80, r80 = run(d80, common)
    results = {"common_features": common, "dropped_features": dropped, "nss75_common": r75, "nss80_common": r80}
    json.dump(results, open(OUTPUTS / "results_2025.json", "w"), indent=2, default=float)
    d80.to_csv(OUTPUTS / "episode_predictions_2025.csv", index=False)

    REPORT.mkdir(parents=True, exist_ok=True)
    table = pd.concat([summary_rows("2017-18 (shared features)", r75), summary_rows("2025 (shared features)", r80)], axis=1)
    table.to_csv(REPORT / "table_rerun_2025.csv")
    ors = pd.DataFrame({"OR 2017-18": r75["rq2"]["odds_ratios"], "p 2017-18": r75["rq2"]["p"],
                        "OR 2025": r80["rq2"]["odds_ratios"], "p 2025": r80["rq2"]["p"]}).drop(index="Intercept").round(4)
    ors.to_csv(REPORT / "table_rq2_or_by_round.csv")
    print(table.to_string())

if __name__ == "__main__":
    main()
