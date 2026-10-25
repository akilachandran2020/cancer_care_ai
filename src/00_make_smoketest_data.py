"""Smoke test: writes synthetic level files with the real NSS 75th round column names. Never analyse or report these values."""
import numpy as np, pandas as pd
from config import RAW, FILES, SEED
if any((RAW / f"{f}.csv").exists() for f in FILES.values()):
    raise SystemExit("Real data found in data/raw/nss75 - smoke test not written (it would overwrite them).")
rng = np.random.default_rng(SEED); n_hh = 3000; rows = {k: [] for k in FILES}
for i in range(n_hh):
    key = {"FSU": 10000 + i // 8, "Hamlet_Sub_block": 1, "Second_stage_stratum": 1, "Sample_hhld": i % 8 + 1}
    region = rng.choice([11, 21, 33, 91, 192, 271]); sector = rng.choice([1, 2], p=[.65, .35])
    size = rng.integers(2, 9); cons = int(rng.lognormal(9.3, .5)); insured = rng.random() < .2
    rows["hh"].append({**key, "Household_size": size, "Social_group": rng.choice([1, 2, 3, 9]),
                       "Household_usual_consumer_expendi": cons, "Amount_of_medical_insurance_prem": 0})
    rows["person"].append({**key, "Person_serial_no": 1, "Gender": rng.choice([1, 2]), "General_education": 5,
                           "covered_by_any_scheme_for_health": 1 if insured else 5})
    if rng.random() < .6:
        ailment = 13 if rng.random() < .6 else 4
        fac = rng.choice([1, 2, 3], p=[.45, .05, .5]); stay = int(rng.integers(1, 30))
        med = int(np.exp(9.5 + (1.2 if fac == 3 else 0) + .03 * stay + rng.normal(0, .9)))
        reimb = int(med * .3) if insured and rng.random() < .4 else 0
        case = {**key, "Srl_no_of_hospitalisation_case": 1, "Srl_no_of_member_hospitalised": 1, "Age": int(rng.integers(20, 80))}
        rows["hosp"].append({**case, "NSS_Region": region, "Sector": sector, "Nature_of_ailment": ailment,
                             "Type_of_medical_institution": fac, "Type_of_ward": rng.choice([1, 2, 3]),
                             "Duration_of_stay_in_hospital_day": stay, "Surgery": rng.choice([1, 2, 4])})
        rows["exp1"].append({**case, "Medical_expenditure_totalitem_Rs": med, "Transport_for_patient_Rs": int(rng.integers(100, 5000)),
                             "Other_non_medical_expenses_Rs": int(rng.integers(200, 8000)), "NSS": 1, "NSC": 2, "MULT": 120000})
        distress = rng.random() < (.2 if insured else .35)
        rows["exp2"].append({**case, "Total_amount_reimbursed_by_medic": reimb, "Major_source_of_finance": 2 if distress else 1,
                             "Place_of_hospitalisation_code": rng.choice([1, 2, 3, 4, 5]), "Loss_of_household_income_due_to_": 0,
                             "NSS": 1, "NSC": 2, "MULT": 120000})
for k, f in FILES.items():
    d = pd.DataFrame(rows[k])
    if k == "hosp":
        d = d.merge(pd.DataFrame(rows["exp1"])[["FSU", "Sample_hhld", "NSS", "NSC", "MULT"]], on=["FSU", "Sample_hhld"])
    d.to_csv(RAW / f"{f}.csv", index=False)
print("SYNTHETIC smoke-test files written - delete before real analysis")
