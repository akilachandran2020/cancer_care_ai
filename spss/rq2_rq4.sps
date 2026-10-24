* AI Cancer Care Companion - SPSS syntax for design-based (weighted) RQ2-RQ4.
* Run src/04_export_for_spss.py first. Set the folder below to your copy of outputs/spss.
* IBM SPSS Statistics 27 or later; the Complex Samples module is required.
* Design: strata = NSS region x sector (stratum), clusters = first-stage units (psu), analysis weight = weight,
* with-replacement estimator. The same plan is used for every Complex Samples procedure below.
FILE HANDLE spssdir /NAME='C:\cancer_repo\outputs\spss'.

* ---------- 2025 cancer episodes (primary analysis file) ----------.
GET FILE='spssdir/cancer_2025.sav'.
DATASET NAME c25.

CSPLAN ANALYSIS
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /PLANVARS ANALYSISWEIGHT=weight
  /DESIGN STRATA=stratum CLUSTER=psu
  /ESTIMATOR TYPE=WR.

* Weighted national descriptives by hospital type, with design-based standard errors.
CSDESCRIPTIVES
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /SUMMARY VARIABLES=oop_medical che
  /SUBPOP TABLE=facility_type DISPLAY=LAYERED
  /MEAN
  /STATISTICS SE CIN(95) COUNT.

* RQ2: design-based logistic regression for CHE (odds ratios and 95% CIs).
CSLOGISTIC che (LOW) BY mpce_quintile social_group WITH age female sector_urban facility_public stay_days
    surgery_received outside_state insured
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /MODEL age female sector_urban facility_public stay_days surgery_received outside_state insured
    mpce_quintile social_group
  /INTERCEPT INCLUDE=YES SHOW=YES
  /STATISTICS PARAMETER EXP SE CINTERVAL
  /TEST TYPE=F PADJUST=LSD
  /ODDSRATIOS FACTOR=[mpce_quintile social_group]
  /MISSING CLASSMISSING=EXCLUDE
  /CRITERIA MXITER=100 MXSTEP=5 PCONVERGE=[1E-006 RELATIVE] LCONVERGE=[0] CHKSEP=20 CILEVEL=95.

* RQ3: design-based comparison of mean ln(OOPE), private vs public (pairwise contrast between hospital types).
CSGLM log_oop BY facility_type
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /MODEL facility_type
  /INTERCEPT INCLUDE=YES SHOW=YES
  /STATISTICS PARAMETER SE CINTERVAL
  /EMMEANS TABLES=facility_type COMPARE=facility_type CONTRAST=PAIRWISE
  /CRITERIA CILEVEL=95.

* ---------- RQ4: 2017-18 vs 2025 ----------.
GET FILE='spssdir/cancer_2017_2025.sav'.
DATASET NAME c1725.

CSPLAN ANALYSIS
  /PLAN FILE='spssdir/plan_1725.csaplan'
  /PLANVARS ANALYSISWEIGHT=weight
  /DESIGN STRATA=stratum CLUSTER=psu
  /ESTIMATOR TYPE=WR.

* Weighted CHE rate in each survey.
CSDESCRIPTIVES
  /PLAN FILE='spssdir/plan_1725.csaplan'
  /SUMMARY VARIABLES=che
  /SUBPOP TABLE=survey_2025 DISPLAY=LAYERED
  /MEAN
  /STATISTICS SE CIN(95) COUNT.

* Difference in weighted CHE rates (2025 minus 2017-18) with its design-based SE and p-value.
CSGLM che BY survey_2025
  /PLAN FILE='spssdir/plan_1725.csaplan'
  /MODEL survey_2025
  /INTERCEPT INCLUDE=YES SHOW=YES
  /STATISTICS PARAMETER SE CINTERVAL
  /EMMEANS TABLES=survey_2025 COMPARE=survey_2025 CONTRAST=PAIRWISE
  /CRITERIA CILEVEL=95.

* RQ4 adjusted: design-based logistic regression of CHE on survey round, adjusting for the same covariates as the report.
CSLOGISTIC che (LOW) BY survey_2025 mpce_quintile WITH sector_urban facility_public
  /PLAN FILE='spssdir/plan_1725.csaplan'
  /MODEL survey_2025 mpce_quintile sector_urban facility_public
  /INTERCEPT INCLUDE=YES SHOW=YES
  /STATISTICS PARAMETER EXP SE CINTERVAL
  /ODDSRATIOS FACTOR=[survey_2025]
  /CRITERIA MXITER=100 CILEVEL=95.

* ---------- Unweighted checks used in earlier drafts (not design-based) ----------.
LOGISTIC REGRESSION VARIABLES che
  /METHOD=ENTER survey_2025 mpce_quintile sector_urban facility_public
  /CATEGORICAL=mpce_quintile
  /PRINT=CI(95).
