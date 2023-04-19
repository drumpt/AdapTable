import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

PHYSIONET_FEATURES = FeatureList(features=[
    Feature("HR", float, "Heart rate (beats per minute)"),
    Feature("O2Sat", float, "Pulse oximetry (%)"),
    Feature("Temp", float, "Temperature (deg C)"),
    Feature("SBP", float, "Systolic BP (mm Hg)"),
    Feature("MAP", float, "Mean arterial pressure (mm Hg)"),
    Feature("DBP", float, "Diastolic BP (mm Hg)"),
    Feature("Resp", float, "Respiration rate (breaths per minute)"),
    Feature("EtCO2", float, "End tidal carbon dioxide (mm Hg)"),
    Feature("BaseExcess", float, "Excess bicarbonate (mmol/L)"),
    Feature("HCO3", float, "Bicarbonate (mmol/L)"),
    Feature("FiO2", float, "Fraction of inspired oxygen (%)"),
    Feature("pH", float, "pH"),
    Feature("PaCO2", float, "Partial pressure of carbon dioxide from arterial "
                            "blood (mm Hg)"),
    Feature("SaO2", float, "Oxygen saturation from arterial blood (%)"),
    Feature("AST", float, "Aspartate transaminase (IU/L)"),
    Feature("BUN", float, "Blood urea nitrogen (mg/dL)"),
    Feature("Alkalinephos", float, "Alkaline phosphatase (IU/L)"),
    Feature("Calcium", float, "Calcium (mg/dL)"),
    Feature("Chloride", float, "Chloride (mmol/L)"),
    Feature("Creatinine", float, "Creatinine (mg/dL)"),
    Feature("Bilirubin_direct", float, "Direct bilirubin (mg/dL)"),
    Feature("Glucose", float, "Serum glucose (mg/dL)"),
    Feature("Lactate", float, "Lactic acid (mg/dL)"),
    Feature("Magnesium", float, "Magnesium (mmol/dL)"),
    Feature("Phosphate", float, "Phosphate (mg/dL)"),
    Feature("Potassium", float, "Potassiam (mmol/L)"),
    Feature("Bilirubin_total", float, "Total bilirubin (mg/dL)"),
    Feature("TroponinI", float, "Troponin I (ng/mL)"),
    Feature("Hct", float, "Hematocrit (%)"),
    Feature("Hgb", float, "Hemoglobin (g/dL)"),
    Feature("PTT", float, "Partial thromboplastin time (seconds)"),
    Feature("WBC", float, "Leukocyte count (count/L)"),
    Feature("Fibrinogen", float, "Fibrinogen concentration (mg/dL)"),
    Feature("Platelets", float, "Platelet count (count/mL)"),
    Feature("Age", int, "Age (years)"),
    Feature("Gender", int, "Female (0) or male (1)"),
    Feature("Unit1", int, "Administrative identifier for ICU unit (MICU); "
                          "false (0) or true (1)"),
    Feature("Unit2", int, "Administrative identifier for ICU unit (SICU); "
                          "false (0) or true (1)"),
    Feature("HospAdmTime", float, "Time between hospital and ICU admission ("
                                  "hours since ICU admission)"),
    Feature("ICULOS", float, "ICU length of stay (hours since ICU admission)"),
    Feature("SepsisLabel", int, "For septic patients, SepsisLabel is 1 if t ≥ "
                                "t_sepsis − 6 and 0 if t < t_sepsis − 6. For "
                                "non-septic patients, SepsisLabel is 0.",
            is_target=True),
    Feature("set", cat_dtype,
            "The training set (i..e hospital) from which an example is drawn "
            "unique (values: 'a', 'b').")
], documentation="https://physionet.org/content/challenge-2019/1.0.0"
                 "/physionet_challenge_2019_ccm_manuscript.pdf")


def preprocess_physionet(df: pd.DataFrame) -> pd.DataFrame:
    return df