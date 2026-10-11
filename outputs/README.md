# Outputs
Files written by the scripts in `src/`. They are git-ignored because they are derived from the
MoSPI microdata; run the scripts to regenerate them.

| File | Written by | Contents |
|---|---|---|
| `results.json` | `03_models_and_tests.py` | Model scores and RQ1–RQ4 test results |
| `episode_predictions.csv` | `03_models_and_tests.py` | Predicted OOPE and 80% range for each episode |
| `spss/cancer_2017.sav` | `04_export_for_spss.py` | 2017–18 episodes for SPSS (descriptives, RQ2, RQ3) |
| `spss/cancer_2017_2025.sav` | `04_export_for_spss.py` | Both surveys combined for SPSS (RQ4) |
| `figure1_coverage.csv`, `figure1_coverage.png` | `05_figure1_coverage.py` | Synopsis Figure 1: cancer patients' scheme coverage by sector, 2017–18 vs 2025 |
