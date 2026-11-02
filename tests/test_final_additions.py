"""Unit tests for scripts 10-14 on synthetic tables (no survey microdata)."""
import importlib, sqlite3
import numpy as np, pandas as pd
import pytest
from conftest import make_episodes


def with_design(n=400, seed=3):
    df = make_episodes(n, seed)
    rng = np.random.default_rng(seed)
    df["psu"] = rng.integers(1, 120, n)
    df["stratum"] = df["psu"] % 12 + 1
    return df


@pytest.fixture(scope="session")
def final():
    return importlib.import_module("11_final_models")


def test_split_has_no_fsu_overlap(final):
    df = with_design()
    tr, te = final.split(df)
    assert set(df["psu"].iloc[tr]).isdisjoint(df["psu"].iloc[te])
    assert len(tr) + len(te) == len(df) and 0.1 < len(te) / len(df) < 0.35


def test_conformal_adjustment_gives_requested_coverage(final):
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    lo, hi = np.full(5000, -.3), np.full(5000, .3)            # intervals that are too narrow
    q = final.conformal_q(lo[:2500], hi[:2500], y[:2500], .8)
    assert q > 0
    assert abs(final.coverage(y[2500:], lo[2500:] - q, hi[2500:] + q) - .8) < .03


def test_cluster_bootstrap_gain_brackets_point_estimate(final):
    rng = np.random.default_rng(1)
    n = 600; psu = rng.integers(0, 150, n); y = rng.normal(0, 1, n)
    base, ml = y + rng.normal(0, 1.0, n), y + rng.normal(0, .6, n)
    lo, hi = final.cluster_bootstrap_gain(y, base, ml, psu, reps=300)
    point = 100 * (np.abs(y - base).mean() - np.abs(y - ml).mean()) / np.abs(y - base).mean()
    assert lo < point < hi


def test_clustered_paired_test_matches_hand_calculation(final):
    d_ml, d_base = np.array([1., 1, 2, 2]), np.array([2., 2, 2, 2])
    r = final.clustered_paired_test(d_ml, d_base, np.array([1, 1, 2, 2]))
    assert r["mean_diff"] == pytest.approx(-.5) and r["df"] == 1


def test_box_tidwell_flags_nonlinear_stay(final):
    df = with_design(1500, 5)
    rng = np.random.default_rng(5)
    df["stay_days"] = rng.integers(1, 60, len(df))
    eta = 1.2 * np.log(df["stay_days"]) - 3.5
    df["che"] = (rng.random(len(df)) < 1 / (1 + np.exp(-eta))).astype(int)
    res = final.box_tidwell(df)
    assert res["terms"]["stay_days"]["nonlinear_at_bonferroni"] and res["aic_log_stay"] < res["aic_linear_stay"]


def test_feature_set_has_no_cost_columns(final):
    assert not {"oop_medical", "oop_total", "log_oop", "che", "reimbursed", "medical_exp", "total_exp"} & set(final.FEATS)


def test_database_build_and_validate(tmp_path, monkeypatch):
    dbm = importlib.import_module("10_build_database")
    proc = tmp_path / "processed"; proc.mkdir()
    for r, seed in (("nss75", 1), ("nss80", 2)):
        with_design(60, seed).to_csv(proc / f"cancer_episodes_{r}.csv", index=False)
    monkeypatch.setattr(dbm, "PROCESSED", proc); monkeypatch.setattr(dbm, "DB_PATH", tmp_path / "db" / "t.db")
    monkeypatch.setattr(dbm, "OUTPUTS", tmp_path); monkeypatch.setattr(dbm, "ROOT", tmp_path)
    out = dbm.validate(dbm.build())
    assert all(v["all_match"] for v in out["integrity_checks"].values())
    with sqlite3.connect(tmp_path / "db" / "t.db") as con:
        assert con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 120
        assert con.execute("SELECT COUNT(*) FROM v_che_by_survey").fetchone()[0] == 2


def test_price_factor_and_design_diff():
    cfg = importlib.import_module("config")
    assert cfg.CPI_FACTOR == pytest.approx(193 / 135)
    sens = importlib.import_module("12_price_and_threshold_sensitivity")
    a, b = with_design(300, 1), with_design(300, 2)
    r = sens.design_diff(a, b, "che")
    assert r["diff"] == pytest.approx(r["mean_b"] - r["mean_a"]) and 0 <= r["p"] <= 1


def test_design_based_logit_matches_weighted_point_estimates():
    mod = importlib.import_module("13_design_based_models")
    df = with_design(800, 7).dropna(subset=["female", "insured"]).reset_index(drop=True)
    r = mod.weighted_logit(df, "che ~ age + facility_public")
    from sklearn.linear_model import LogisticRegression
    X = df[["age", "facility_public"]].to_numpy(float)
    m = LogisticRegression(penalty=None, max_iter=5000).fit(X, df["che"], sample_weight=df["weight"] / df["weight"].mean())
    assert r["params"]["facility_public"]["b"] == pytest.approx(m.coef_[0][1], abs=1e-3)
    assert r["df"] > 0 and r["params"]["age"]["se"] > 0


def test_companion_estimate_checklist_and_validation():
    comp = importlib.import_module("14_companion")
    assert comp.validate(comp.DEMO) == []
    assert comp.validate({**comp.DEMO, "age": 200, "insured": 3})
    est = {"che_risk": "High"}
    items = comp.checklist({**comp.DEMO, "insured": 0, "outside_state": 1}, est)
    text = " ".join(items)
    assert "PM-JAY" in text and "HMCPF" in text and "another state" in text
    with pytest.raises(ValueError):
        comp.estimate({"age": 5})
