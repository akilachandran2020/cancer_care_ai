"""Step 14 - CAC (Cancer Care AI Companion) prototype: core logic and command-line demo.

Input   what is known when a family meets the hospital social worker or patient navigator (age, sex, urban/rural, hospital type,
        ward, expected length of stay, surgery planned, treatment outside the home district/state, scheme coverage, household
        consumption quintile, social group, household size).
Output  1. a cost range for the hospitalization: typical OOPE (median-type estimate) with an 80% range from conformal-calibrated
           10th/90th percentile models;
        2. a CHE risk flag and band from a classifier whose threshold was chosen on training data;
        3. the three factors that push the estimate up and down most (tree-SHAP contributions);
        4. a financial-aid checklist built from rules, not from the model.
The companion estimates; it does not diagnose, set prices or decide eligibility for any scheme.

Run:  python 14_companion.py            (demo patient)      streamlit run ../app/companion_app.py   (web form)
Needs models/companion_models.joblib written by 11_final_models.py.
"""
import json, sys
import numpy as np, pandas as pd
import joblib
import xgboost as xgb
from config import MODELS

LABELS = {"age": "Age", "female": "Sex", "sector_urban": "Urban residence", "facility_public": "Public hospital",
          "ward_type": "Type of ward", "stay_days": "Length of stay", "surgery_received": "Surgery", "outside_district": "Treated outside home district",
          "outside_state": "Treated outside home state", "insured": "Scheme coverage status", "member_deceased": "Missing person record",
          "mpce_quintile": "Household consumption quintile", "social_group": "Social group", "hh_size": "Household size"}
REQUIRED = ["age", "female", "sector_urban", "facility_public", "ward_type", "stay_days", "surgery_received", "outside_district",
            "outside_state", "insured", "mpce_quintile", "social_group", "hh_size"]
DISCLAIMER = ("This is a statistical estimate from the 2025 national household survey (NSS 80th round). It is not medical, legal or "
              "financial advice, it does not set hospital prices and it does not decide eligibility for any scheme.")


def load_bundle(path=None):
    path = path or (MODELS / "companion_models.joblib")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run python src/11_final_models.py first")
    return joblib.load(path)


def validate(p):
    """Return a list of problems with the input (empty when valid)."""
    bad = [f"missing: {k}" for k in REQUIRED if k not in p or p[k] is None]
    rng = {"age": (0, 110), "stay_days": (1, 365), "hh_size": (1, 30), "mpce_quintile": (1, 5), "ward_type": (1, 3)}
    for k, (lo, hi) in rng.items():
        if k in p and p[k] is not None and not lo <= p[k] <= hi:
            bad.append(f"{k} must be between {lo} and {hi}")
    for k in ("female", "sector_urban", "facility_public", "surgery_received", "outside_district", "outside_state", "insured"):
        if k in p and p[k] is not None and p[k] not in (0, 1):
            bad.append(f"{k} must be 0 or 1")
    if "social_group" in p and p["social_group"] not in (1, 2, 3, 9, None):
        bad.append("social_group must be 1 (ST), 2 (SC), 3 (OBC) or 9 (other)")
    return bad


HIDDEN = {"member_deceased"}   # model feature the app always sets to 0 (living patient); never entered, so never shown as a driver
VALUE_LABELS = {"female": ("Male", "Female"), "sector_urban": ("Rural residence", "Urban residence"),
                "facility_public": ("Private or charitable hospital", "Public hospital"), "surgery_received": ("No surgery", "Surgery planned"),
                "outside_district": ("Treated in home district", "Treated outside home district"),
                "outside_state": ("Treated in home state", "Treated outside home state"),
                "insured": ("No scheme coverage", "Covered by a health scheme")}


def label(k, row):
    """Name a driver by the patient's own value (e.g. 'No scheme coverage'), not just the variable."""
    v = row.iloc[0][k]
    return VALUE_LABELS[k][int(v)] if k in VALUE_LABELS and v in (0, 1) else LABELS[k]


def drivers(bundle, row, top=3):
    """Top factors raising and lowering the cost estimate (tree-SHAP contributions of the XGBoost explainer, log scale).
    Only the inputs the user entered are ranked; HIDDEN features are excluded."""
    ex = bundle["explainer"]
    X = ex.named_steps["prep"].transform(row)
    contrib = ex.named_steps["m"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    items = sorted(((k, v) for k, v in zip(bundle["feats"], contrib) if k not in HIDDEN), key=lambda kv: kv[1])
    up = [(label(k, row), float(v)) for k, v in items[::-1][:top] if v > 0.02]
    down = [(label(k, row), float(v)) for k, v in items[:top] if v < -0.02]
    return {"raises_cost": up, "lowers_cost": down}


def checklist(p, est):
    """Rule-based aid checklist. Every item is a question to ask or a document to gather, never an eligibility decision."""
    items = []
    if not p.get("insured"):
        items.append("Ask the hospital help desk whether the patient is covered by PM-JAY (Ayushman Bharat) or a state scheme; PM-JAY "
                     "lists chemotherapy, radiotherapy and surgical oncology among its covered packages (Press Information Bureau, n.d.).")
    else:
        items.append("Confirm that the scheme card is valid at this hospital and ask which packages and pre-authorization steps apply before treatment starts.")
    if p.get("mpce_quintile", 5) <= 2 or est["che_risk"] == "High":
        items.append("Ask the social worker about the Health Minister's Cancer Patient Fund (HMCPF) and state or district relief funds for low-income patients; "
                     "applications usually need the treatment estimate and income documents.")
    if not p.get("facility_public"):
        items.append("Ask the hospital for a written package-rate estimate and ask whether a public or charitable cancer centre can provide the same treatment; "
                     "in the 2025 survey, private hospitalizations had far higher out-of-pocket cost than public ones.")
    if p.get("outside_state"):
        items.append("Check whether the scheme card is accepted in another state, and plan for travel and lodging; treatment outside the home state "
                     "had the largest association with catastrophic spending.")
    elif p.get("outside_district"):
        items.append("Budget for travel and attendant costs; treatment outside the home district adds non-medical spending.")
    if p.get("ward_type", 1) >= 2:
        items.append("Ask whether a free or general ward is available; ward type was the strongest driver of cost in the model.")
    if est["che_risk"] in ("High", "Moderate"):
        items.append("Plan for borrowing risk early: ask about instalment plans, hospital concessions and charitable trusts before taking loans or selling assets.")
    items.append("Gather documents: identity card, ration or income certificate, scheme card, diagnosis and treatment plan, and the hospital's cost estimate.")
    return items


def estimate(p, bundle=None):
    """Estimate for one patient (dict with the keys in REQUIRED, plus optional annual_consumption in rupees)."""
    problems = validate(p)
    if problems:
        raise ValueError("; ".join(problems))
    bundle = bundle or load_bundle()
    row = pd.DataFrame([{**{k: p[k] for k in REQUIRED}, "member_deceased": 0}])[bundle["feats"]]
    pred = float(bundle["cost_model"].predict(row)[0])
    lo = float(bundle["q_low"].predict(row)[0]) - bundle["conformal_q"]
    hi = float(bundle["q_high"].predict(row)[0]) + bundle["conformal_q"]
    lo, mid, hi = (max(float(np.expm1(v)), 0.0) for v in sorted((lo, pred, hi)))
    prob = float(bundle["che_model"].predict_proba(row)[0, 1])
    thr = float(bundle["che_threshold"])
    band = "High" if prob >= thr else "Moderate" if prob >= min(.30, thr / 2) else "Low"
    est = {"cost_typical_rupees": round(mid, -2), "cost_low_rupees": round(lo, -2), "cost_high_rupees": round(hi, -2),
           "che_probability": round(prob, 3), "che_risk": band, "che_flag": bool(prob >= thr), "che_threshold": round(thr, 3),
           "model": {"cost": bundle["cost_model_name"], "che": bundle["che_model_name"], "trained_on": bundle["trained_on"]}}
    if p.get("annual_consumption"):
        c = float(p["annual_consumption"])
        est["share_of_annual_consumption"] = {"typical": round(mid / c, 3), "high": round(hi / c, 3), "che_threshold": 0.10,
                                              "note": "medical cost only; CHE also counts transport and other non-medical costs"}
    est["drivers"] = drivers(bundle, row)
    est["checklist"] = checklist(p, est)
    est["disclaimer"] = DISCLAIMER
    return est


DEMO = {"age": 54, "female": 1, "sector_urban": 0, "facility_public": 0, "ward_type": 2, "stay_days": 10, "surgery_received": 1,
        "outside_district": 1, "outside_state": 0, "insured": 0, "mpce_quintile": 2, "social_group": 3, "hh_size": 5, "annual_consumption": 180000}

if __name__ == "__main__":
    patient = json.loads(sys.argv[1]) if len(sys.argv) > 1 else DEMO
    print(json.dumps(estimate(patient), indent=2, ensure_ascii=False))
