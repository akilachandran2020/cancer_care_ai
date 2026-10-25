# Outputs
Files written by the scripts in `src/`. They are git-ignored because they are derived from the
MoSPI microdata; run the scripts to regenerate them.

| File | Written by | Contents |
|---|---|---|
| `results.json` | `03_models_and_tests.py` | Model scores and RQ1–RQ3 results on the primary round (2025), RQ4 on both rounds |
| `episode_predictions.csv` | `03_models_and_tests.py` | Predicted OOPE and 80% range for each episode |
| `spss/cancer_2025.sav` | `04_export_for_spss.py` | 2025 episodes for SPSS (descriptives, RQ2, RQ3) |
| `spss/cancer_2017_2025.sav` | `04_export_for_spss.py` | Both surveys combined for SPSS (RQ4) |
| `figure1_coverage.csv`, `figure1_coverage.png` | `05_figure1_coverage.py` | Synopsis Figure 1: cancer patients' scheme coverage by sector, 2017–18 vs 2025 |
| `results_by_round.json`, `episode_predictions_nss75.csv` | `07_compare_rounds.py` | RQ1–RQ3 and fairness on 2017–18 and 2025 with the same features |
| `report/table_by_round.csv`, `report/table_rq2_or_by_round.csv` | `07_compare_rounds.py` | Side-by-side 2017–18 vs 2025 summary and RQ2 odds ratios |
| `design_based_check.json` | `08_design_based_check.py` | Weighted CHE in each round with design-based SEs (strata + clusters) and the RQ4 difference; cross-check for SPSS |
| `report/` (other files) | `06_report_outputs.py` | Cleaning log, EDA tables and Figures 2–11 for the primary round |
