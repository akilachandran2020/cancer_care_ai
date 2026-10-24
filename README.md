# AI Cancer Care Companion: predicting cancer treatment cost and financial risk (India)

Predicts out-of-pocket expenditure (OOPE) and catastrophic health expenditure (CHE) risk for cancer
hospitalisation episodes using the NSO 80th round (January-December 2025) health survey: 3,283 cancer
hospitalisations (ailment code 13) among 121,584 hospitalisation records (RQ1-RQ3). RQ4 compares the CHE rate
with the NSS 75th round (2017-18) survey: 1,606 cancer hospitalisations among 93,925 records. The primary round
is set by `PRIMARY` in `src/config.py`.

## 1. Get the data (register as research student at https://microdata.gov.in)
1. **2017-18, catalog 152** (https://microdata.gov.in/NADA/index.php/catalog/152): download the CSV zip
   (`CSV_HSCH_75`) and copy these five files, unchanged, into `data/raw/nss75/`:
   `Block_3_Level_2_R75250L02.csv`, `Block_4_Level_3_R75250L03.csv`, `Block_6_Level_5_R75250L05.csv`,
   `Block_7_Level_6_R75250L06.csv`, `Block_7_Level_7_R75250L07.csv`.
2. **2025, catalog 290** (https://microdata.gov.in/NADA/index.php/catalog/290): download the CSV zip
   (`CSV_data_household_social_consumption_heaith_Jan_Dec25`) and copy `hhscsL1.csv`, `hhscsL2.csv` and
   `hhscsL4.csv` into `data/raw/nss80/`. Field mapping used: hhscsL4 `b6i5` ailment (13 = cancer, same list as
   2017-18), `b6i7` institution (1 public, 2 charitable, 3 private), `b6i9` type of ward (1 free, 2 paying
   general, 3 paying special), `b6i12` duration of stay (days), `b6i13` surgery (2, 3, 4 = surgery, as in
   2017-18), `b7i12` medical total,
   `b7i15` total incl. non-medical, `b7i16` reimbursed, `b7i17` source of finance, `b7i18` place, `mult`
   weight, `b6i2` serial of the member hospitalised (91/92 = died or left); hhscsL1 `umce` monthly consumption,
   `hhsz`; hhscsL2 `b3c1` member serial, `b3c4` sex, `b3c17` health-expenditure scheme (19 = not covered);
   keys `fsu, sd, sss, hhd`.
3. Follow MoSPI's terms of use: no attempt to identify households, cite the datasets. 

## 2. Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd src
python 02_sample_size.py              # reproduces the synopsis sample-size table
python 01_build_cancer_episodes.py    # builds both rounds -> data/processed/cancer_episodes_nss80.csv and _nss75.csv (add nss80 or nss75 to build one)
python 03_models_and_tests.py         # -> outputs/results.json: RQ1-RQ3 on PRIMARY (2025), RQ4 on both rounds
python 04_export_for_spss.py          # -> outputs/spss/cancer_2025.sav, cancer_2017_2025.sav
python 05_figure1_coverage.py         # -> outputs/figure1_coverage.csv/.png (Figure 1)
python 07_compare_rounds.py           # same models on 2017-18 and 2025 with the same 14 features
# -> outputs/results_by_round.json, outputs/report/table_by_round.csv, table_rq2_or_by_round.csv
python 06_report_outputs.py           # -> outputs/report/ (cleaning log, EDA tables, Figures 2-11)
```

## 3. SPSS (descriptives and RQ2-RQ4) - work in progress!
Open `spss/rq2_rq4.sps` in IBM SPSS Statistics (27 or later; Complex Samples module for the weighted
estimates), set the FILE HANDLE path to your `outputs/spss` folder, and run all. Welch's test is the
"equal variances not assumed" row; the RQ4 two-proportion test is the Pearson chi-square (chi-square = z^2).
`03_models_and_tests.py` reruns the same tests in Python as a cross-check. RQ1 (machine learning, SHAP,
prediction intervals) runs in Python only.

## 4. Unit tests
`python -m pytest -q` (from the repository root) runs 16 tests in `tests/` on small synthetic tables: the CHE
threshold, OOPE clipping, missing coverage for members who died or left, place-of-treatment flags, survey weights,
no cost columns among the model features, the RQ2 model and its AUC, the Welch degrees of freedom and the
shared-feature selection used by the cross-round comparison. They never read the survey microdata.

## 5. Smoke test
`python 00_make_smoketest_data.py` writes clearly labelled SYNTHETIC files with the real column names so
the pipeline can be tested before the download. Delete them before real analysis; never report them.

Seed 42 throughout. Variable definitions: `data_dictionary.csv`.
