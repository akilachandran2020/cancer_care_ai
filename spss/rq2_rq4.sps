* AI Cancer Care Companion - SPSS syntax for descriptives and RQ2-RQ4.
* Run src/04_export_for_spss.py first. Set the folder below to your copy of outputs/spss.
* IBM SPSS Statistics 27 or later; the Complex Samples module is needed for the weighted estimates.
FILE HANDLE spssdir /NAME='C:\cancer_repo\outputs\spss'.

* ---------- 2025 cancer episodes (primary analysis file) ----------.
GET FILE='spssdir/cancer_2025.sav'.
DATASET NAME c25.

* Weighted national descriptives (Complex Samples; weight only, with-replacement estimator).
CSPLAN ANALYSIS
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /PLANVARS ANALYSISWEIGHT=weight
  /DESIGN
  /ESTIMATOR TYPE=WR.
CSDESCRIPTIVES
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /SUMMARY VARIABLES=oop_medical che
  /SUBPOP TABLE=facility_type
  /MEAN
  /STATISTICS SE CIN(95).

* RQ2: multiple logistic regression for CHE (unweighted, as in the synopsis); add /PRINT=GOODFIT for Hosmer-Lemeshow.
LOGISTIC REGRESSION VARIABLES che
  /METHOD=ENTER age female sector_urban facility_public stay_days surgery_received outside_state insured
    mpce_quintile social_group
  /CATEGORICAL=mpce_quintile social_group
  /PRINT=CI(95) GOODFIT
  /CRITERIA=PIN(.05) POUT(.10) ITERATE(20) CUT(.5).

* RQ3: Welch t-test on ln(OOPE), public (1) vs private (3); read the "Equal variances not assumed" row.
T-TEST GROUPS=facility_type(1 3)
  /VARIABLES=log_oop
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).
* RQ3 check: Mann-Whitney U.
NPAR TESTS /M-W=log_oop BY facility_type(1 3).

* ---------- RQ4: 2017-18 vs 2025 ----------.
GET FILE='spssdir/cancer_2017_2025.sav'.
DATASET NAME c1725.
* Two-proportion test: for a 2 x 2 table the Pearson chi-square equals z squared.
CROSSTABS /TABLES=che BY survey_2025
  /CELLS=COUNT COLUMN
  /STATISTICS=CHISQ.
* RQ4 adjusted: logistic regression of CHE on survey round.
LOGISTIC REGRESSION VARIABLES che
  /METHOD=ENTER survey_2025 mpce_quintile sector_urban facility_public
  /CATEGORICAL=mpce_quintile
  /PRINT=CI(95).
