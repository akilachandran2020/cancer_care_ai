"""Shared fixtures. Every test uses small SYNTHETIC tables built here; no survey microdata is read."""
import importlib, pathlib, sys
import numpy as np, pandas as pd
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

@pytest.fixture(scope="session")
def build():
    return importlib.import_module("01_build_cancer_episodes")

@pytest.fixture(scope="session")
def models():
    return importlib.import_module("03_models_and_tests")

@pytest.fixture(scope="session")
def rerun():
    return importlib.import_module("07_rerun_2025")

def make_episodes(n=400, seed=1):
    """Synthetic episode file with every column the models use (not survey data)."""
    rng = np.random.default_rng(seed)
    fac = rng.choice([1, 2, 3], n, p=[.45, .05, .5])
    stay = rng.integers(1, 40, n)
    ward = rng.choice([1, 2, 3], n)
    surgery = rng.integers(0, 2, n)
    oop = np.where(rng.random(n) < .08, 0,
                   np.exp(9 + 1.1 * (fac == 3) + .03 * stay + .4 * (ward == 3) + .3 * surgery + rng.normal(0, 1, n))).round()
    cons = np.exp(rng.normal(11.8, .5, n)).round()
    df = pd.DataFrame({
        "age": rng.integers(1, 90, n), "female": rng.integers(0, 2, n).astype(float),
        "sector_urban": rng.integers(0, 2, n), "facility_public": (fac == 1).astype(int), "facility_type": fac,
        "ward_type": ward, "stay_days": stay, "surgery_received": surgery,
        "outside_district": rng.integers(0, 2, n), "outside_state": rng.integers(0, 2, n),
        "insured": rng.integers(0, 2, n).astype(float), "member_deceased": 0,
        "mpce_quintile": rng.integers(1, 6, n), "social_group": rng.choice([1, 2, 3, 9], n),
        "hh_size": rng.integers(1, 10, n), "oop_medical": oop, "oop_total": oop + rng.integers(0, 5000, n),
        "annual_consumption": cons, "weight": rng.uniform(50, 5000, n)})
    df.loc[:30, ["female", "insured"]] = np.nan          # patients who died or left the household
    df.loc[:30, "member_deceased"] = 1
    df["log_oop"] = np.log1p(df["oop_medical"])
    df["che"] = (df["oop_total"] > .10 * df["annual_consumption"]).astype(int)
    return df

@pytest.fixture
def episodes():
    return make_episodes()
