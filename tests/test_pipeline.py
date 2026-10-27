"""Unit tests on synthetic data. Run: python -m pytest -q"""
import numpy as np, pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind

# ---------- 01: variable derivation ----------

def test_che_uses_strict_10pct_threshold(build):
    out = pd.DataFrame({"oop_medical": [0, 100, 100, 100, 100], "oop_total": [0, 1199, 1200, 1201, 50000],
                        "hh_monthly_consumption": [1000] * 5, "hh_size": [1, 2, 3, 4, 5],
                        "major_finance": [1, 2, 3, 4, 1]})
    r = build.finish(out)
    # exactly 10% is not catastrophic
    assert r["annual_consumption"].tolist() == [12000] * 5
    assert r["che"].tolist() == [0, 0, 0, 1, 1]
    assert np.allclose(r["log_oop"], np.log1p(r["oop_medical"]))
    assert r["distress_financing"].tolist() == [0, 1, 1, 0, 0]       # borrowing or asset sale

def test_mpce_quintiles_are_balanced(build):
    n = 100
    out = pd.DataFrame({"oop_medical": 0, "oop_total": 0, "hh_monthly_consumption": np.arange(1, n + 1) * 1000.0,
                        "hh_size": 1, "major_finance": 1})
    r = build.finish(out)
    assert sorted(r["mpce_quintile"].unique()) == [1, 2, 3, 4, 5]
    assert (r["mpce_quintile"].value_counts() == n // 5).all()
    assert r.loc[0, "mpce_quintile"] == 1 and r.loc[n - 1, "mpce_quintile"] == 5   # 1 = poorest

@pytest.fixture
def raw75(tmp_path, build, monkeypatch):
    """Synthetic NSS 75th round files."""
    key = {"Hamlet_Sub_block": 1, "Second_stage_stratum": 1, "Sample_hhld": 1}
    hh = pd.DataFrame([{**key, "FSU": f, "Household_size": 4, "Social_group": 2,
                        "Household_usual_consumer_expendi": 10000} for f in (101, 102)])
    person = pd.DataFrame([{**key, "FSU": 101, "Person_serial_no": 1, "Gender": 2, "covered_by_any_scheme_for_health": 1},
                           {**key, "FSU": 102, "Person_serial_no": 1, "Gender": 1, "covered_by_any_scheme_for_health": 5}])
    cases = [  # (FSU, case, member, ailment, medical, reimbursed, place, NSS, NSC)
        (101, 1, 1, 13, 50000, 0, 5, 1, 1),      # outside state, insured woman
        (101, 2, 1, 13, 20000, 25000, 1, 1, 2),  # reimbursed more than spent
        (102, 1, 91, 13, 8000, 0, 3, 1, 1),      # member died or left
        (102, 2, 1, 4, 9000, 0, 1, 1, 1)]        # not cancer
    hosp, e1, e2 = [], [], []
    for f, c, mem, ail, med, reimb, place, nss, nsc in cases:
        ck = {**key, "FSU": f, "Srl_no_of_hospitalisation_case": c}
        hosp.append({**ck, "Srl_no_of_member_hospitalised": mem, "Age": 50, "NSS_Region": 271, "Sector": 2,
                     "Nature_of_ailment": ail, "Type_of_medical_institution": 3, "Type_of_ward": 2,
                     "Duration_of_stay_in_hospital_day": 6, "Surgery": 2, "NSS": nss, "NSC": nsc, "MULT": 1000})
        e1.append({**ck, "Medical_expenditure_totalitem_Rs": med, "Transport_for_patient_Rs": 1000,
                   "Other_non_medical_expenses_Rs": None})
        e2.append({**ck, "Total_amount_reimbursed_by_medic": reimb, "Major_source_of_finance": 1,
                   "Place_of_hospitalisation_code": place, "Loss_of_household_income_due_to_": 0})
    tables = {"hh": hh, "person": person, "hosp": pd.DataFrame(hosp), "exp1": pd.DataFrame(e1), "exp2": pd.DataFrame(e2)}
    for k, d in tables.items():
        d.to_csv(tmp_path / f"{build.FILES[k]}.csv", index=False)
    monkeypatch.setattr(build, "RAW", tmp_path)
    monkeypatch.setattr(build, "ROUND", "nss75")
    return build.build_nss75()

def test_build_keeps_only_cancer_episodes(raw75):
    assert len(raw75) == 3

def test_oope_never_negative_and_total_includes_nonmedical(raw75):
    assert (raw75["oop_medical"] >= 0).all() and (raw75["oop_total"] >= 0).all()
    over = raw75[raw75["reimbursed"] > raw75["medical_exp"]].iloc[0]
    assert over["oop_medical"] == 0 and over["oop_total"] == 0      # clipped to 0
    first = raw75[raw75["medical_exp"] == 50000].iloc[0]
    assert first["oop_total"] == 51000                               # missing cost counts as 0

def test_deceased_member_has_missing_coverage_and_sex(raw75):
    gone = raw75[raw75["member_deceased"] == 1]
    assert len(gone) == 1
    assert gone["insured"].isna().all() and gone["female"].isna().all()
    alive = raw75[raw75["member_deceased"] == 0]
    assert alive["insured"].tolist() == [1.0, 1.0] and alive["female"].tolist() == [1.0, 1.0]

def test_place_of_treatment_flags(raw75):
    by_place = raw75.set_index("place_code")
    assert by_place.loc[5, "outside_state"] == 1 and by_place.loc[5, "outside_district"] == 1
    assert by_place.loc[3, "outside_state"] == 0 and by_place.loc[3, "outside_district"] == 1
    assert by_place.loc[1, "outside_state"] == 0 and by_place.loc[1, "outside_district"] == 0

def test_survey_weight_rule(raw75):
    w = raw75.set_index("medical_exp")["weight"]
    assert w[50000] == 10 and w[20000] == 5                          # MULT/100 if NSS == NSC, else MULT/200

# ---------- 03: models and tests ----------

COST_COLUMNS = {"oop_medical", "oop_total", "log_oop", "medical_exp", "total_exp", "reimbursed", "transport_exp",
                "other_nonmed_exp", "che", "annual_consumption", "hh_monthly_consumption", "mpce"}

def test_no_cost_leakage_into_features(models):
    assert len(models.FEATS) == 14
    assert not COST_COLUMNS & set(models.FEATS)

def test_rq1_outputs_are_consistent(models, episodes):
    df, r = models.rq1(episodes)
    s = r["scores"]
    assert set(s) == {"linear", "random_forest", "lightgbm", "xgboost"}
    assert r["best_ml"] == min(("random_forest", "lightgbm", "xgboost"), key=lambda k: s[k]["mae_log"])
    expected = (s["linear"]["mae_log"] - s[r["best_ml"]]["mae_log"]) / s["linear"]["mae_log"] * 100
    assert r["mae_improvement_pct"] == pytest.approx(expected)
    assert 0 <= r["pi80_coverage"] <= 1
    assert set(r["mean_abs_shap"]) == set(models.FEATS)
    assert {"pred_oop", "pi_low", "pi_high"} <= set(df.columns) and len(df) == len(episodes)

def test_rq1_and_rq2_accept_a_reduced_feature_set(models, episodes):
    shared = [f for f in models.FEATS if f not in ("ward_type", "stay_days", "surgery_received")]
    reduced = episodes.drop(columns=["ward_type", "stay_days", "surgery_received"])
    _, r = models.rq1(reduced, shared)
    assert set(r["mean_abs_shap"]) == set(shared)
    r2 = models.rq2(reduced, shared)
    assert not {"stay_days", "surgery_received"} & set(r2["odds_ratios"])

def test_rq2_default_matches_original_formula(models, episodes):
    original = smf.logit("che ~ age + female + sector_urban + facility_public + stay_days + surgery_received + "
                         "outside_state + insured + C(mpce_quintile) + C(social_group)",
                         data=episodes.dropna(subset=models.FEATS)).fit(disp=0)
    r = models.rq2(episodes)
    assert r["n"] == int(original.nobs) == episodes.dropna(subset=models.FEATS).shape[0]
    for term, coef in original.params.items():
        assert r["odds_ratios"][term] == pytest.approx(np.exp(coef))
    assert r["pseudo_r2"] == pytest.approx(original.prsquared)

def test_rq3_welch_t_and_df_match_statsmodels(models, episodes):
    r = models.rq3(episodes)
    pub = episodes.loc[episodes["facility_type"] == 1, "log_oop"]
    pvt = episodes.loc[episodes["facility_type"] == 3, "log_oop"]
    t, p, dof = sm_ttest_ind(pub, pvt, usevar="unequal")
    assert r["welch_t"] == pytest.approx(t)
    assert r["welch_df"] == pytest.approx(dof)
    assert r["p"] == pytest.approx(p)
    assert r["n_public"] + r["n_private"] == len(pub) + len(pvt)      # charitable excluded

def test_fairness_under_prediction_definition(models):
    df = pd.DataFrame({"pred_oop": [79.0, 80.0, 120.0], "oop_medical": [100.0, 100.0, 100.0],
                       "sector_urban": [0, 0, 1], "social_group": [1, 1, 2], "female": [0.0, 0.0, 1.0]})
    df["log_oop"] = np.log1p(df["oop_medical"])
    r = models.fairness(df)
    assert r["sector_urban"]["under"][0] == 0.5                       # 79 < 80 counts; exactly 80% does not
    assert r["sector_urban"]["under"][1] == 0.0

# ---------- 07: compare rounds ----------

def test_shared_features_drop_fields_missing_in_2025(rerun, models, episodes):
    d80 = episodes.drop(columns=["ward_type", "stay_days", "surgery_received"])
    d80["hh_size"] = np.nan                                           # empty column is dropped
    shared = rerun.shared_features(d80)
    assert shared == [f for f in models.FEATS if f not in ("ward_type", "stay_days", "surgery_received", "hh_size")]

def test_rq2_confidence_intervals_bracket_odds_ratios(models, episodes):
    r = models.rq2(episodes)
    for term, orv in r["odds_ratios"].items():
        assert r["ci_low"][term] < orv < r["ci_high"][term]

def test_rq2_reports_auc_events_and_vif(models, episodes):
    r = models.rq2(episodes)
    assert 0.5 <= r["auc"] <= 1
    assert r["events_per_parameter"] == r["events"] / (len(r["odds_ratios"]) - 1)
    assert r["max_vif"] >= 1


def test_strata_have_at_least_two_psus_for_complex_samples():
    import importlib
    build = importlib.import_module("01_build_cancer_episodes")
    df = pd.DataFrame({"psu": [1, 2, 3, 4, 5], "stratum": [11, 11, 21, 21, 31]})   # stratum 31 has one PSU
    out = build.collapse_strata(df.copy())
    assert (out.groupby("stratum")["psu"].nunique() >= 2).all()


def test_design_based_mean_matches_weighted_mean_and_has_positive_se():
    import importlib
    chk = importlib.import_module("08_design_based_check")
    df = pd.DataFrame({"weight": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0], "che": [1, 0, 1, 1, 0, 0],
                       "stratum": [1, 1, 1, 1, 2, 2], "psu": [1, 1, 2, 2, 3, 4]})
    p, se, dof = chk.design_mean(df, "che")
    assert p == pytest.approx((df.weight * df.che).sum() / df.weight.sum())
    assert se > 0 and dof == 2


def test_design_effects_match_report_formula():
    import importlib
    rob = importlib.import_module("09_robustness_checks")
    res = {"nss75": {"weighted_che": .63, "n": 1600, "se": .036},
           "nss80": {"weighted_che": .53, "n": 3300, "se": .029},
           "difference": {"se_points": 4.6}}
    d = rob.design_effects(res)
    srs_se = (.63 * .37 / 1600) ** .5
    assert d["nss75"]["deff"] == pytest.approx((.036 / srs_se) ** 2)
    assert d["nss75"]["n_effective"] == pytest.approx(1600 / d["nss75"]["deff"])
    assert d["n_required_per_survey_srs"] == pytest.approx(170.9, abs=0.1)     # Table 2, RQ4
    assert 0 < d["design_power_15pt_fall"] < 1


def test_robustness_checks_run_on_synthetic_episodes(episodes):
    import importlib
    rob = importlib.import_module("09_robustness_checks")
    ep = episodes.assign(psu=np.arange(len(episodes)) % 40)
    stronger = rob.linear_design(ep[rob.FEATS].fillna(0), rob.FEATS, log_stay=True)
    assert {"log_stay", "public_x_log_stay"} <= set(stronger.columns)
    assert "stay_days" not in [f for f in rob.FEATS if f != "stay_days"]
    gone = rob.died_or_left(ep)
    assert gone["died_or_left"]["n"] == int(ep["member_deceased"].sum())
    assert gone["died_or_left"]["n"] + gone["others"]["n"] == len(ep)
    adj = rob.rq3_adjusted(ep)
    assert adj["n"] == int(ep["facility_type"].isin([1, 3]).sum())
    assert adj["adjusted"]["ci_low"] < adj["adjusted"]["b"] < adj["adjusted"]["ci_high"]
    err = pd.Series(np.abs(np.random.default_rng(0).normal(size=len(ep))))
    audit = rob.sex_audit(ep, err.to_numpy())
    assert audit["n_known"] + audit["n_imputed"] == len(ep)
    assert audit["male"]["n"] + audit["female"]["n"] == audit["n_known"]


def test_sample_size_sensitivity_matches_table_2():
    import importlib
    rob = importlib.import_module("09_robustness_checks")
    s = rob.sample_size_sensitivity()
    assert s["rq1_dz"]["0.2"] == 199 and s["rq2_or"]["1.5"] == 293
    assert s["rq3_d_per_group"]["0.3"] == 176 and s["rq4_p2025_per_survey"]["0.48"] == 171
    assert s["rq3_d_per_group"]["0.2"] > s["rq3_d_per_group"]["0.3"]


def test_rq1_clustered_se_is_at_least_srs_for_clustered_errors():
    import importlib
    rob = importlib.import_module("09_robustness_checks")
    rng = np.random.default_rng(3)
    psu = np.repeat(np.arange(40), 10)
    shift = rng.normal(0, 1, 40)[psu]                      # errors share a cluster effect
    df = pd.DataFrame({"psu": psu, "stratum": psu % 4})
    r = rob.rq1_clustered(df, rng.normal(1, .5, 400) + shift, rng.normal(1.2, .5, 400))
    assert r["se_clustered"] > r["se_srs"] and r["deff"] > 1

