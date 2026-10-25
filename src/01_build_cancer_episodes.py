"""Step 1 - Build one row per cancer hospitalisation episode.
python src/01_build_cancer_episodes.py         both rounds
python src/01_build_cancer_episodes.py nss80   2025 only
python src/01_build_cancer_episodes.py nss75   2017-18 only"""
import sys
import numpy as np, pandas as pd
from config import ROOT, ROUND, RAW, PROCESSED, FILES, FILES80, HH_KEY, HH_KEY80, CASE_KEY, CANCER_CODE, CHE_THRESHOLD

def load(key):
    f = RAW / f"{(FILES80 if ROUND == 'nss80' else FILES)[key]}.csv"
    if not f.exists():
        raise SystemExit(f"Missing {f.name} - see README step 1.")
    return pd.read_csv(f, low_memory=False)

def num(s):
    return pd.to_numeric(s, errors="coerce")

def build_nss80():
    """2025 fields (hhscsL4): b6i5 ailment (13 = cancer), b6i7 institution, b6i9 ward, b6i12 stay (days), b6i13 surgery (2, 3, 4 = surgery),
    b6i3 age, b7i12 medical, b7i15 total, b7i16 reimbursed, b7i17 finance source, b7i18 place, b6i2 member serial (91/92 = died or left).
    hhscsL1: umce, hhsz. hhscsL2: b3c1 member, b3c4 sex, b3c17 scheme (19 = not covered)."""
    hosp, hh, person = load("hosp"), load("hh"), load("person")
    for d in (hosp, hh, person):
        for c in HH_KEY80:
            d[c] = d[c].astype(str).str.strip()
    hosp = hosp[num(hosp["b6i5"]) == CANCER_CODE].copy()
    hosp["member"], person["member"] = num(hosp["b6i2"]), num(person["b3c1"])
    ep = (hosp.merge(hh[HH_KEY80 + ["hhsz", "b5i3", "umce"]], on=HH_KEY80, how="left")
              .merge(person[HH_KEY80 + ["member", "b3c4", "b3c17"]], on=HH_KEY80 + ["member"], how="left"))
    assert len(ep) == len(hosp), "member merge duplicated episodes - check keys"
    out = pd.DataFrame({
        "sector_urban": (num(ep["sec"]) == 2).astype(int),
        "age": num(ep["b6i3"]),
        "female": np.where(num(ep["b3c4"]).isna(), np.nan, (num(ep["b3c4"]) == 2).astype(float)),
        # died or left: coverage unknown
        "insured": np.where(num(ep["b3c17"]).isna(), np.nan, (num(ep["b3c17"]) != 19).astype(float)),
        "member_deceased": ep["member"].isin([91, 92]).astype(int),
        "social_group": num(ep["b5i3"]),
        "hh_size": num(ep["hhsz"]),
        "hh_monthly_consumption": num(ep["umce"]),
        "facility_public": (num(ep["b6i7"]) == 1).astype(int),
        "facility_type": num(ep["b6i7"]),
        "ward_type": num(ep["b6i9"]),
        "stay_days": num(ep["b6i12"]),
        "surgery_received": num(ep["b6i13"]).isin([2, 3, 4]).astype(int),
        "place_code": num(ep["b7i18"]),
        "medical_exp": num(ep["b7i12"]).fillna(0),
        "total_exp": num(ep["b7i15"]).fillna(0),
        "reimbursed": num(ep["b7i16"]).fillna(0),
        "major_finance": num(ep["b7i17"]),
        "weight": num(ep["mult"]),
        "psu": num(ep["fsu"]),                                   # first-stage unit
        "stratum": num(ep["nssreg"]) * 10 + num(ep["sec"]),      # NSS region x sector
    })
    out["outside_state"] = (out["place_code"] == 5).astype(int)
    out["outside_district"] = out["place_code"].isin([3, 4, 5]).astype(int)
    out["oop_medical"] = (out["medical_exp"] - out["reimbursed"]).clip(lower=0)
    out["oop_total"] = (out["total_exp"] - out["reimbursed"]).clip(lower=0)
    return finish(out)


def collapse_strata(out):
    """Merge strata with a single PSU into the nearest stratum of the same sector."""
    if "psu" not in out or "stratum" not in out:
        return out
    stratum = out["stratum"].copy()
    for _ in range(5):
        n = out.assign(stratum=stratum).groupby("stratum")["psu"].nunique()
        lone = n[n < 2].index
        if len(lone) == 0:
            break
        for k in lone:
            same = [j for j in n.index if j != k and j % 10 == k % 10]
            if same:
                stratum[stratum == k] = min(same, key=lambda j: abs(j - k))
    out["stratum"] = stratum.astype(int)
    out["psu"] = out["psu"].astype(int)
    return out

def finish(out):
    out = collapse_strata(out)
    out["log_oop"] = np.log1p(out["oop_medical"])
    out["annual_consumption"] = out["hh_monthly_consumption"] * 12
    out["mpce"] = out["hh_monthly_consumption"] / out["hh_size"]
    out["mpce_quintile"] = pd.qcut(out["mpce"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["che"] = (out["oop_total"] > CHE_THRESHOLD * out["annual_consumption"]).astype(int)
    out["distress_financing"] = out["major_finance"].isin([2, 3]).astype(int)   # borrowing or asset sale
    return out

def build_nss75():
    hosp, e1, e2 = load("hosp"), load("exp1"), load("exp2")
    hh, person = load("hh"), load("person")
    for d in (hosp, e1, e2, hh, person):
        for c in HH_KEY + ["Srl_no_of_hospitalisation_case"]:
            if c in d:
                d[c] = d[c].astype(str).str.strip()
    hosp = hosp[num(hosp["Nature_of_ailment"]) == CANCER_CODE]
    keep1 = CASE_KEY + ["Medical_expenditure_totalitem_Rs", "Transport_for_patient_Rs", "Other_non_medical_expenses_Rs"]
    keep2 = CASE_KEY + ["Total_amount_reimbursed_by_medic", "Major_source_of_finance", "Place_of_hospitalisation_code",
                        "Loss_of_household_income_due_to_"]
    ep = (hosp.merge(e1[keep1], on=CASE_KEY, how="left")
              .merge(e2[keep2], on=CASE_KEY, how="left")
              .merge(hh[HH_KEY + ["Household_size", "Social_group", "Household_usual_consumer_expendi"]], on=HH_KEY, how="left"))
    person = person.rename(columns={"Person_serial_no": "Srl_no_of_member_hospitalised"})
    for d in (person, ep):
        d["Srl_no_of_member_hospitalised"] = d["Srl_no_of_member_hospitalised"].astype(str).str.strip()
    ep = ep.merge(person[HH_KEY + ["Srl_no_of_member_hospitalised", "Gender", "covered_by_any_scheme_for_health"]],
                  on=HH_KEY + ["Srl_no_of_member_hospitalised"], how="left")
    out = pd.DataFrame({
        "state_region": ep["NSS_Region"].astype(str).str[:2],
        "sector_urban": (num(ep["Sector"]) == 2).astype(int),
        "age": num(ep["Age"]),
        "female": np.where(num(ep["Gender"]).isna(), np.nan, (num(ep["Gender"]) == 2).astype(float)),
        "social_group": num(ep["Social_group"]),
        "hh_size": num(ep["Household_size"]),
        "hh_monthly_consumption": num(ep["Household_usual_consumer_expendi"]),
        "facility_public": (num(ep["Type_of_medical_institution"]) == 1).astype(int),
        "facility_type": num(ep["Type_of_medical_institution"]),
        "ward_type": num(ep["Type_of_ward"]),
        "stay_days": num(ep["Duration_of_stay_in_hospital_day"]),
        "surgery_received": num(ep["Surgery"]).isin([2, 3, 4]).astype(int),
        "place_code": num(ep["Place_of_hospitalisation_code"]),
        # died or left: coverage and sex unknown
        "insured": np.where(num(ep["covered_by_any_scheme_for_health"]).isna(), np.nan,
                            num(ep["covered_by_any_scheme_for_health"]).isin([1, 2, 3, 4, 9]).astype(float)),
        "member_deceased": num(ep["Srl_no_of_member_hospitalised"]).isin([91, 92]).astype(int),
        "medical_exp": num(ep["Medical_expenditure_totalitem_Rs"]).fillna(0),
        "transport_exp": num(ep["Transport_for_patient_Rs"]),
        "other_nonmed_exp": num(ep["Other_non_medical_expenses_Rs"]),
        "reimbursed": num(ep["Total_amount_reimbursed_by_medic"]).fillna(0),
        "major_finance": num(ep["Major_source_of_finance"]),
        "income_loss": num(ep["Loss_of_household_income_due_to_"]),
        "weight": np.where(num(ep["NSS"]) == num(ep["NSC"]), num(ep["MULT"]) / 100, num(ep["MULT"]) / 200),
        "psu": num(ep["FSU"]),
        "stratum": num(ep["NSS_Region"]) * 10 + num(ep["Sector"]),
    })
    out["outside_state"] = (out["place_code"] == 5).astype(int)
    out["outside_district"] = out["place_code"].isin([3, 4, 5]).astype(int)
    out["oop_medical"] = (out["medical_exp"] - out["reimbursed"]).clip(lower=0)
    out["oop_total"] = (out["medical_exp"] + out["transport_exp"].fillna(0) + out["other_nonmed_exp"].fillna(0)
                        - out["reimbursed"]).clip(lower=0)
    return finish(out)

def main(rounds=None):
    global ROUND, RAW
    rounds = rounds or (sys.argv[1:] or ["nss80", "nss75"])
    for r in rounds:
        if r not in ("nss75", "nss80"):
            raise SystemExit("Usage: python src/01_build_cancer_episodes.py [nss80] [nss75]   (no argument builds both)")
        ROUND, RAW = r, ROOT / "data/raw" / r
        out = build_nss80() if r == "nss80" else build_nss75()
        out.to_csv(PROCESSED / f"cancer_episodes_{r}.csv", index=False)
        print(f"{r}: {len(out)} cancer hospitalisation episodes written")

if __name__ == "__main__":
    main()
