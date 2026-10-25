"""Step 8 - Design-based CHE rates and the RQ4 difference (strata, clusters, weights); cross-check for SPSS."""
import json
import numpy as np, pandas as pd
from scipy import stats
from config import PROCESSED, OUTPUTS

def design_mean(df, y):
    """Weighted mean, design-based SE and degrees of freedom."""
    w = df["weight"]
    p = (w * df[y]).sum() / w.sum()
    z = w * (df[y] - p) / w.sum()
    totals = z.groupby([df["stratum"], df["psu"]]).sum().reset_index()
    var, n_psu, n_strata = 0.0, 0, 0
    for _, g in totals.groupby("stratum"):
        n = len(g)
        n_psu, n_strata = n_psu + n, n_strata + 1
        var += n / (n - 1) * ((g[0] - g[0].mean()) ** 2).sum()
    return p, float(np.sqrt(var)), n_psu - n_strata

def main():
    res = {}
    for r in ("nss75", "nss80"):
        d = pd.read_csv(PROCESSED / f"cancer_episodes_{r}.csv")
        p, se, df = design_mean(d, "che")
        t = stats.t.ppf(.975, df)
        res[r] = {"n": len(d), "weighted_che": p, "se": se, "df": df, "ci_low": p - t * se, "ci_high": p + t * se,
                  "n_strata": int(d["stratum"].nunique()), "n_psu": int(d["psu"].nunique())}
    diff = res["nss75"]["weighted_che"] - res["nss80"]["weighted_che"]
    se = float(np.hypot(res["nss75"]["se"], res["nss80"]["se"]))
    df = res["nss75"]["df"] + res["nss80"]["df"]
    res["difference"] = {"fall_points": 100 * diff, "se_points": 100 * se, "z": diff / se, "df": df,
                         "p": float(2 * stats.t.sf(abs(diff / se), df))}
    json.dump(res, open(OUTPUTS / "design_based_check.json", "w"), indent=2)
    for r in ("nss75", "nss80"):
        v = res[r]; print(f"{r}: weighted CHE {100*v['weighted_che']:.1f}% (SE {100*v['se']:.1f}, 95% CI {100*v['ci_low']:.1f}-{100*v['ci_high']:.1f}); {v['n_strata']} strata, {v['n_psu']} FSUs")
    d_ = res["difference"]; print(f"fall {d_['fall_points']:.1f} points, SE {d_['se_points']:.1f}, t({d_['df']}) = {d_['z']:.2f}, p = {d_['p']:.3f}")

if __name__ == "__main__":
    main()
