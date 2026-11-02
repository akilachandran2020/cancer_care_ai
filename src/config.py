"""Configuration (NSS 80th round, 2025, primary; NSS 75th round, 2017-18, for RQ4)."""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
ROUND = "nss75"            # smoke-test folder
PRIMARY = "nss80"          # primary survey (2025)
RAW = ROOT / "data/raw" / ROUND
PROCESSED = ROOT / "data/processed"
OUTPUTS = ROOT / "outputs"
DB_PATH = ROOT / "data/db/cancer_care.db"      # SQLite database built by 10_build_database.py (git-ignored)
MODELS = ROOT / "models"                       # fitted models written by 11_final_models.py (git-ignored)
SEED = 42

# MoSPI CSV files
# 2017-18 (catalog 152): data/raw/nss75/
FILES = {"hh": "Block_3_Level_2_R75250L02", "person": "Block_4_Level_3_R75250L03",
         "hosp": "Block_6_Level_5_R75250L05", "exp1": "Block_7_Level_6_R75250L06",
         "exp2": "Block_7_Level_7_R75250L07"}
# 2025 (catalog 290): data/raw/nss80/
FILES80 = {"hh": "hhscsL1", "person": "hhscsL2", "hosp": "hhscsL4"}
HH_KEY80 = ["fsu", "sd", "sss", "hhd"]

HH_KEY = ["FSU", "Hamlet_Sub_block", "Second_stage_stratum", "Sample_hhld"]
CASE_KEY = HH_KEY + ["Srl_no_of_hospitalisation_case"]
CANCER_CODE = 13            # cancer
CHE_THRESHOLD = 0.10        # 10% of annual consumption

# Consumer Price Index, Combined (all-India), General, base 2012 = 100, annual averages.
# Source: Government of India, Economic Survey 2025-26, Statistical Table 4.3.
CPI = {"2017-18": 135, "2024-25": 193}
CPI_FACTOR = CPI["2024-25"] / CPI["2017-18"]   # 2017-18 rupees -> 2024-25 rupees (approximates 2025 prices)
