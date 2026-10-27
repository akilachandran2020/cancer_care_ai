"""Step 2 - Sample size for RQ1-RQ4 (alpha = .05, power = .80)."""
import math
from scipy import stats
from statsmodels.stats.power import TTestPower, TTestIndPower, NormalIndPower

A, PW = 0.05, 0.80
Z = stats.norm.ppf(1 - A / 2) + stats.norm.ppf(PW)

rq1 = math.ceil(TTestPower().solve_power(effect_size=0.20, alpha=A, power=PW))          # paired errors
b = math.log(1.5); P, R2 = 0.63, 0.30                                                   # Hsieh et al. (1998)
rq2_raw = Z ** 2 / (P * (1 - P) * b ** 2); rq2 = math.ceil(rq2_raw / (1 - R2))
rq3_g = math.ceil(TTestIndPower().solve_power(effect_size=0.30, alpha=A, power=PW))     # per group
h = 2 * math.asin(math.sqrt(0.63)) - 2 * math.asin(math.sqrt(0.48))                     # CHE 2017-18 vs 2025
rq4_g = math.ceil(NormalIndPower().solve_power(effect_size=h, alpha=A, power=PW))       # per survey

print(f"RQ1 paired t (dz=0.20):                    n = {rq1} episodes")
print(f"RQ2 logistic, Hsieh (OR=1.5/SD, P=.63, R2=.3): {rq2_raw:.1f} -> {rq2} episodes")
print(f"RQ3 two-sample t (d=0.30):                 {rq3_g} per group = {2 * rq3_g}")
print(f"RQ4 CHE 2017-18 vs 2025 (.63 vs .48, h={h:.4f}): {rq4_g} per survey = {2 * rq4_g}")
print(f"FINAL N = max = {max(rq1, rq2, 2 * rq3_g, 2 * rq4_g)} cancer hospitalisation episodes")

# Design effect (report: Sample Size Adequacy). The minimums above assume simple random sampling; the NSS is stratified
# and clustered. Once 08_design_based_check.py has run, show the effective sample size for the RQ4 CHE comparison.
import importlib, json
from config import OUTPUTS
f = OUTPUTS / "design_based_check.json"
if f.exists():
    d = importlib.import_module("09_robustness_checks").design_effects(json.load(open(f)))
    for k, lab in (("nss75", "2017-18"), ("nss80", "2025")):
        v = d[k]
        print(f"RQ4 {lab}: design effect {v['deff']:.1f}, effective n = {v['n_effective']:.0f} (required {rq4_g} per survey)")
    print(f"RQ4 design-based power for a 15-point fall = {d['design_power_15pt_fall']:.2f}; "
          f"minimum detectable fall = {d['minimum_detectable_fall_pts']:.1f} points")
else:
    print("Run 08_design_based_check.py to add design effects for RQ4.")
