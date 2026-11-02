"""Step 12 - Price adjustment and CHE-threshold sensitivity for the 2017-18 vs 2025 comparison (RQ4).

A. Real costs: 2017-18 rupees are inflated to 2024-25 rupees with the CPI-Combined general index (135 -> 193, Economic
   Survey 2025-26, Table 4.3; factor 1.43). OOPE by hospital type is compared in constant prices, with design-based SEs.
   CHE is a ratio of two rupee amounts in the same year, so a uniform price level cancels; the price adjustment therefore
   changes the cost comparison, not the CHE comparison.
B. Threshold sensitivity: the 10% CHE threshold is the interim definition. The rate and the 2017-18 vs 2025 difference are
   recomputed at 10%, 15%, 20%, 25% and 40% of annual consumption (design-based SE and p-value).
C. Medical-price scenario: if medical prices rose faster than the general CPI, 2017-18 spending would be understated in
   2025 prices. The CHE difference is recomputed after inflating 2017-18 OOPE by an extra 0%, 10%, 20% and 30% relative to
   consumption (a scenario, not an estimate).

Writes outputs/price_threshold_sensitivity.json and outputs/report/table_price_threshold.csv (aggregates only).
"""
import importlib, json, math
import numpy as np, pandas as pd
from scipy import stats
from config import OUTPUTS, CPI, CPI_FACTOR
from db import load_episodes

chk = importlib.import_module("08_design_based_check")
REPORT = OUTPUTS / "report"
FAC = {1: "Public", 2: "Charitable/NGO", 3: "Private"}


def design_diff(a, b, y):
    """Difference (b - a) in the design-based weighted mean of column y between two independent surveys."""
    pa, sa, da = chk.design_mean(a, y)
    pb, sb, db_ = chk.design_mean(b, y)
    diff, se, df = pb - pa, float(np.hypot(sa, sb)), da + db_
    t = diff / se
    return {"mean_a": float(pa), "se_a": sa, "mean_b": float(pb), "se_b": sb, "diff": float(diff), "se_diff": se, "df": int(df),
            "t": float(t), "p": float(2 * stats.t.sf(abs(t), df))}


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def main():
    d75, d80 = load_episodes("nss75"), load_episodes("nss80")
    for d in (d75, d80):
        d.dropna(subset=["log_oop", "annual_consumption"], inplace=True)
    res = {"cpi": {"index": CPI, "factor_2017_18_to_2024_25": CPI_FACTOR,
                   "source": "Economic Survey 2025-26, Statistical Table 4.3 (CPI Combined, general, base 2012 = 100, annual averages)"}}

    # ---- A. real costs
    a75 = d75.assign(real_oop=d75["oop_medical"] * CPI_FACTOR)
    a75["log_real"] = np.log1p(a75["real_oop"])
    a80 = d80.assign(real_oop=d80["oop_medical"]); a80["log_real"] = d80["log_oop"]
    real = {"all": {
        "median_2017_18_nominal": float(d75["oop_medical"].median()), "median_2017_18_in_2025_prices": float(a75["real_oop"].median()),
        "median_2025": float(d80["oop_medical"].median()),
        "ln_oope_diff_design_based": design_diff(a75, a80, "log_real"),
        "weighted_mean_oope_2017_18_in_2025_prices": float(np.average(a75["real_oop"], weights=a75["weight"])),
        "weighted_mean_oope_2025": float(np.average(a80["real_oop"], weights=a80["weight"]))}}
    for code, lab in FAC.items():
        m75, m80 = a75[a75["facility_type"] == code], a80[a80["facility_type"] == code]
        real[lab] = {"n_2017_18": len(m75), "n_2025": len(m80), "median_2017_18_nominal": float(m75["oop_medical"].median()),
                     "median_2017_18_in_2025_prices": float(m75["real_oop"].median()), "median_2025": float(m80["real_oop"].median()),
                     "median_change_real_pct": float(100 * (m80["real_oop"].median() / m75["real_oop"].median() - 1)),
                     "weighted_mean_2017_18_in_2025_prices": float(np.average(m75["real_oop"], weights=m75["weight"])),
                     "weighted_mean_2025": float(np.average(m80["real_oop"], weights=m80["weight"]))}
    res["A_real_costs"] = real

    # ---- B. threshold sensitivity (survey-offset strata and FSUs so the two designs are independent)
    def che_at(d, thr):
        return (d["oop_total"] > thr * d["annual_consumption"]).astype(int)
    rows = []
    for thr in (.10, .15, .20, .25, .40):
        x75, x80 = d75.assign(c=che_at(d75, thr)), d80.assign(c=che_at(d80, thr))
        r = design_diff(x75, x80, "c")
        p1u, p2u = x75["c"].mean(), x80["c"].mean()
        rows.append({"threshold": thr, "che_2017_18_unweighted": float(p1u), "che_2025_unweighted": float(p2u),
                     "che_2017_18_weighted": r["mean_a"], "se_2017_18": r["se_a"], "che_2025_weighted": r["mean_b"], "se_2025": r["se_b"],
                     "fall_points_weighted": float(-100 * r["diff"]), "se_points": float(100 * r["se_diff"]), "t": float(-r["t"]),
                     "df": r["df"], "p_design_based": r["p"], "cohen_h_weighted": float(cohen_h(r["mean_a"], r["mean_b"]))})
    res["B_threshold_sensitivity"] = rows

    # ---- C. medical-price scenario
    scen = []
    for extra in (0.0, .10, .20, .30):
        x75 = d75.assign(c=((d75["oop_total"] * (1 + extra)) > .10 * d75["annual_consumption"]).astype(int))
        r = design_diff(x75, d80.assign(c=d80["che"]), "c")
        scen.append({"extra_medical_inflation": extra, "che_2017_18_weighted": r["mean_a"], "che_2025_weighted": r["mean_b"],
                     "fall_points": float(-100 * r["diff"]), "se_points": float(100 * r["se_diff"]), "p": r["p"]})
    res["C_medical_price_scenario"] = scen

    json.dump(res, open(OUTPUTS / "price_threshold_sensitivity.json", "w"), indent=2, default=float)
    REPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(REPORT / "table_price_threshold.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k in ("cpi",)}, indent=1))
    print("Real costs (2025 rupees):")
    for lab in ("Public", "Charitable/NGO", "Private"):
        r = real[lab]; print(f"  {lab}: 2017-18 {r['median_2017_18_nominal']:,.0f} nominal -> {r['median_2017_18_in_2025_prices']:,.0f} real; 2025 {r['median_2025']:,.0f}; change {r['median_change_real_pct']:+.0f}%")
    print("  ln(OOPE+1) difference (2025 - 2017-18 real):", {k: round(v, 3) for k, v in real["all"]["ln_oope_diff_design_based"].items() if k in ("diff", "se_diff", "t", "p")})
    print("Threshold sensitivity:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(scen).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
