"""Step 4 - Export episode files to SPSS (.sav)."""
import pandas as pd, pyreadstat
from config import PROCESSED, OUTPUTS, PRIMARY

KEEP = ["oop_medical", "log_oop", "oop_total", "che", "distress_financing", "facility_public", "facility_type",
        "ward_type", "stay_days", "surgery_received", "outside_district", "outside_state", "insured", "age", "female",
        "sector_urban", "social_group", "hh_size", "mpce_quintile", "annual_consumption", "weight", "psu", "stratum"]
LABELS = {"oop_medical": "OOPE per cancer hospitalization (Rs)", "log_oop": "ln(OOPE + 1)",
          "oop_total": "OOPE incl. transport and other non-medical (Rs)", "che": "Catastrophic health expenditure (CHE)",
          "distress_financing": "Borrowing or asset sale as major source", "facility_public": "Public hospital",
          "facility_type": "Type of medical institution", "survey_2025": "Survey round", "psu": "First-stage unit (FSU), cluster for complex samples",
          "stratum": "Stratum: NSS region x sector (collapsed to >= 2 FSUs)",
          "mpce_quintile": "MPCE quintile (within survey)", "weight": "Survey weight"}
VALUES = {"che": {0: "No", 1: "Yes"}, "facility_type": {1: "Public", 2: "Charitable/NGO", 3: "Private"},
          "survey_2025": {0: "2017-18 (NSS 75th)", 1: "2025 (NSO 80th)"}, "female": {0: "Male", 1: "Female"},
          "sector_urban": {0: "Rural", 1: "Urban"}}

def save(df, name):
    out = OUTPUTS / "spss"; out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in df.columns]
    pyreadstat.write_sav(df, out / name, column_labels={c: LABELS.get(c, c) for c in cols},
                         variable_value_labels={c: v for c, v in VALUES.items() if c in cols})
    print(f"{name}: {len(df)} episodes")

def main():
    year = {"nss75": "2017", "nss80": "2025"}
    primary = PRIMARY if (PROCESSED / f"cancer_episodes_{PRIMARY}.csv").exists() else "nss75"
    dp = pd.read_csv(PROCESSED / f"cancer_episodes_{primary}.csv")
    save(dp[KEEP], f"cancer_{year[primary]}.sav")
    f75, f80 = PROCESSED / "cancer_episodes_nss75.csv", PROCESSED / "cancer_episodes_nss80.csv"
    if f75.exists() and f80.exists():
        d17, d25 = pd.read_csv(f75), pd.read_csv(f80)
        cols = ["che", "facility_public", "facility_type", "sector_urban", "mpce_quintile", "weight", "oop_medical", "psu", "stratum"]
        # offset codes so rounds share no strata or FSUs
        both = pd.concat([d17[cols].assign(survey_2025=0), d25[cols].assign(survey_2025=1, stratum=d25["stratum"] + 100000,
                         psu=d25["psu"] + 10000000)], ignore_index=True)
        save(both, "cancer_2017_2025.sav")

if __name__ == "__main__":
    main()
