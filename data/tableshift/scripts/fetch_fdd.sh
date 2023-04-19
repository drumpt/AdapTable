
# A script to compute FDD for all of the TableShift tasks.
# FDD measures are printed to STDOUT.

python scripts/fdd.py --experiment acsfoodstamps --uid acsfoodstampsdomain_split_varname_DIVISIONdomain_split_ood_value_06
python scripts/fdd.py --experiment acsincome --uid acsincomedomain_split_varname_DIVISIONdomain_split_ood_value_01
python scripts/fdd.py --experiment acspubcov --uid acspubcovdomain_split_varname_DISdomain_split_ood_value_1.0
python scripts/fdd.py --experiment acsunemployment --uid acsunemploymentdomain_split_varname_SCHLdomain_split_ood_value_010203040506070809101112131415
python scripts/fdd.py --experiment anes --uid anesdomain_split_varname_VCF0112domain_split_ood_value_3.0
python scripts/fdd.py --experiment brfss_diabetes --uid brfss_diabetesdomain_split_varname_PRACE1domain_split_ood_value_23456domain_split_id_values_1
python scripts/fdd.py --experiment brfss_blood_pressure --uid brfss_blood_pressuredomain_split_varname_BMI5CATdomain_split_ood_value_3.04.0
python scripts/fdd.py --experiment diabetes --uid diabetes_readmissiondomain_split_varname_admission_source_iddomain_split_ood_value_7
python scripts/fdd.py --experiment heloc --uid helocdomain_split_varname_ExternalRiskEstimateLowdomain_split_ood_value_0
python scripts/fdd.py --experiment mimic_extract_los_3 --uid mimic_extract_los_3domain_split_varname_insurancedomain_split_ood_value_Medicare
python scripts/fdd.py --experiment mimic_extract_hosp_mort --uid mimic_extract_mort_hospdomain_split_varname_insurancedomain_split_ood_value_MedicareMedicaid
python scripts/fdd.py --experiment nhanes_lead --uid nhanes_leaddomain_split_varname_INDFMPIRBelowCutoffdomain_split_ood_value_1.0
python scripts/fdd.py --experiment physionet_los47 --uid physionetICULOSgt47.0
