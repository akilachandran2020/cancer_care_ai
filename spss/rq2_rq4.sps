* AI Cancer Care Companion - SPSS syntax for design-based (survey-weighted) RQ2-RQ4.
* 1. Run src/04_export_for_spss.py (writes outputs/spss/*.sav).  2. Set the folder in FILE HANDLE below.
* 3. Run > All.  Needs IBM SPSS Statistics 27 or later with the Complex Samples module.
* Design: strata = NSS region x sector (stratum), clusters = first-stage units (psu), analysis weight = weight,
* with-replacement estimator. Results are also written to spss_results.txt in the same folder (OMS).
* Reference categories: Complex Samples uses the HIGHEST code of a factor as reference, so quintile and social group are
* recoded below to make Q1 (poorest) and Scheduled Tribe the reference, as in the report and the Python models.
FILE HANDLE spssdir /NAME='C:\Users\akila\OneDrive\Documents\GitHub\cancer_care_ai\outputs\spss'.

OMS /SELECT TABLES /IF COMMANDS=['CSDescriptives' 'CSLogistic' 'CSGLM' 'Logistic Regression']
  /DESTINATION FORMAT=TEXT OUTFILE='spssdir/spss_results.txt' /TAG='cancer'.

* ---------- 2025 cancer episodes (primary analysis file) ----------.
GET FILE='spssdir/cancer_2025.sav'.
DATASET NAME c25 WINDOW=FRONT.

COMPUTE log_stay = LN(stay_days).
RECODE mpce_quintile (1=5) (2=4) (3=3) (4=2) (5=1) INTO q_ref.
RECODE social_group (1=10) (2=2) (3=3) (9=9) INTO social_ref.
VARIABLE LABELS log_stay 'ln(length of stay in days)' q_ref 'MPCE quintile (reference = Q1 poorest)'
  social_ref 'Social group (reference = Scheduled Tribe)'.
VALUE LABELS q_ref 5 'Q1 poorest (ref)' 4 'Q2' 3 'Q3' 2 'Q4' 1 'Q5 richest' /
  social_ref 10 'ST (ref)' 2 'SC' 3 'OBC' 9 'Other'.
EXECUTE.

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

* RQ2 (final model): design-based logistic regression for CHE with ln(stay); Exp(B) = odds ratios.
CSLOGISTIC che (LOW) BY q_ref social_ref WITH age female sector_urban facility_public log_stay
    surgery_received outside_state insured
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /MODEL age female sector_urban facility_public log_stay surgery_received outside_state insured q_ref social_ref
  /INTERCEPT INCLUDE=YES SHOW=YES
  /STATISTICS PARAMETER EXP SE CINTERVAL
  /TEST TYPE=F PADJUST=LSD
  /MISSING CLASSMISSING=EXCLUDE
  /CRITERIA MXITER=100 MXSTEP=5 PCONVERGE=[1E-006 RELATIVE] LCONVERGE=[0] CHKSEP=20 CILEVEL=95.

* RQ2 (interim model with stay in days, for comparison with the interim report).
CSLOGISTIC che (LOW) BY q_ref social_ref WITH age female sector_urban facility_public stay_days
    surgery_received outside_state insured
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /MODEL age female sector_urban facility_public stay_days surgery_received outside_state insured q_ref social_ref
  /INTERCEPT INCLUDE=YES SHOW=YES
  /STATISTICS PARAMETER EXP SE CINTERVAL
  /TEST TYPE=F PADJUST=LSD
  /MISSING CLASSMISSING=EXCLUDE
  /CRITERIA MXITER=100 MXSTEP=5 PCONVERGE=[1E-006 RELATIVE] LCONVERGE=[0] CHKSEP=20 CILEVEL=95.

* RQ3: design-based comparison of mean ln(OOPE + 1) between hospital types (pairwise contrasts).
CSGLM log_oop BY facility_type
  /PLAN FILE='spssdir/plan_2025.csaplan'
  /MODEL facility_type
  /INTERCEPT INCLUDE=YES SHOW=YES
  /STATISTICS PARAMETER SE CINTERVAL
  /EMMEANS TABLES=facility_type COMPARE=facility_type CONTRAST=PAIRWISE
  /CRITERIA CILEVEL=95.

* ---------- RQ4: 2017-18 vs 2025 ----------.
GET FILE='spssdir/cancer_2017_2025.sav'.
DATASET NAME c1725 WINDOW=FRONT.
RECODE mpce_quintile (1=5) (2=4) (3=3) (4=2) (5=1) INTO q_ref.
VALUE LABELS q_ref 5 'Q1 poorest (ref)' 4 'Q2' 3 'Q3' 2 'Q4' 1 'Q5 richest'.
EXECUTE.

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

* RQ4 adjusted: design-based logistic regression of CHE on survey round, adjusting for quintile, sector and hospital type.
CSLOGISTIC che (LOW) BY survey_2025 q_ref WITH sector_urban facility_public
  /PLAN FILE='spssdir/plan_1725.csaplan'
  /MODEL survey_2025 q_ref sector_urban facility_public
  /INTERCEPT INCLUDE=YES SHOW=YES
  /STATISTICS PARAMETER EXP SE CINTERVAL
  /CRITERIA MXITER=100 CILEVEL=95.

* ---------- Unweighted check (not design-based) ----------.
LOGISTIC REGRESSION VARIABLES che
  /METHOD=ENTER survey_2025 q_ref sector_urban facility_public
  /CATEGORICAL=q_ref
  /PRINT=CI(95).

OMSEND TAG='cancer'.
