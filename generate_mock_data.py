import pandas as pd
import os
from datetime import datetime

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

SUBJECT_ID = 249
HADM_ID = 116935
PREFIX = f"{DATA_DIR}/Subject_ID_{SUBJECT_ID}_"

# 1. PATIENTS
pd.DataFrame(
    {"SUBJECT_ID": [SUBJECT_ID], "GENDER": ["F"], "DOB": ["2075-05-05"]}
).to_csv(f"{PREFIX}PATIENTS.csv", index=False)

# 2. ADMISSIONS
pd.DataFrame(
    {
        "SUBJECT_ID": [SUBJECT_ID],
        "HADM_ID": [HADM_ID],
        "ADMITTIME": ["2149-12-15 14:00:00"],
        "ADMISSION_TYPE": ["URGENT"],
    }
).to_csv(f"{PREFIX}ADMISSIONS.csv", index=False)

# 3. D_ICD_DIAGNOSES (Dictionary)
d_diag = pd.DataFrame(
    {
        "ICD9_CODE": ["428", "410", "250", "493", "51881"],
        "LONG_TITLE": [
            "Congestive heart failure",
            "Acute myocardial infarction",
            "Diabetes mellitus",
            "Asthma",
            "Acute respiratory failure",
        ],
    }
)
d_diag.to_csv(f"{PREFIX}D_ICD_DIAGNOSES.csv", index=False)

# 4. DIAGNOSES_ICD
pd.DataFrame(
    {
        "SUBJECT_ID": [SUBJECT_ID, SUBJECT_ID],
        "HADM_ID": [HADM_ID, HADM_ID],
        "ICD9_CODE": ["428", "250"],  # HF and Diabetes
    }
).to_csv(f"{PREFIX}DIAGNOSES_ICD.csv", index=False)

# 5. D_ICD_PROCEDURES (Dictionary)
d_proc = pd.DataFrame(
    {
        "ICD9_CODE": ["361", "3961"],
        "LONG_TITLE": ["Bypass coronary artery", "Extracorporeal circulation"],
    }
)
d_proc.to_csv(f"{PREFIX}D_ICD_PROCEDURES.csv", index=False)

# 6. PROCEDURES_ICD
pd.DataFrame(
    {"SUBJECT_ID": [SUBJECT_ID], "HADM_ID": [HADM_ID], "ICD9_CODE": ["361"]}  # CABG
).to_csv(f"{PREFIX}PROCEDURES_ICD.csv", index=False)

# 7. D_LABITEMS (Dictionary)
d_lab = pd.DataFrame(
    {
        "ITEMID": [50912, 50910, 51002, 51221, 50889],
        "LABEL": [
            "Creatinine",
            "BNP",
            "Troponin I",
            "Hemoglobin",
            "C-Reactive Protein",
        ],
    }
)
d_lab.to_csv(f"{PREFIX}D_LABITEMS.csv", index=False)

# 8. LABEVENTS (The actual lab values)
pd.DataFrame(
    {
        "SUBJECT_ID": [SUBJECT_ID, SUBJECT_ID, SUBJECT_ID, SUBJECT_ID],
        "HADM_ID": [HADM_ID, HADM_ID, HADM_ID, HADM_ID],
        "ITEMID": [50912, 50910, 51221, 50889],
        "CHARTTIME": [
            "2149-12-16 08:00:00",  # Creatinine
            "2149-12-16 08:00:00",  # BNP
            "2149-12-16 08:00:00",  # Hgb
            "2149-12-16 09:00:00",  # CRP
        ],
        "VALUENUM": [
            2.1,  # High Creatinine
            450.0,  # High BNP
            11.5,  # Low Hgb
            12.0,  # High CRP
        ],
        "VALUEUOM": ["mg/dL", "pg/mL", "g/dL", "mg/L"],
    }
).to_csv(f"{PREFIX}LABEVENTS.csv", index=False)

# 9. D_ITEMS (For Vitals)
d_items = pd.DataFrame(
    {
        "ITEMID": [220050, 220179, 646],
        "LABEL": ["Heart Rate", "Non Invasive Blood Pressure Systolic", "SpO2"],
    }
)
d_items.to_csv(f"{PREFIX}D_ITEMS.csv", index=False)

# 10. CHARTEVENTS (Vitals)
pd.DataFrame(
    {
        "SUBJECT_ID": [SUBJECT_ID, SUBJECT_ID, SUBJECT_ID],
        "HADM_ID": [HADM_ID, HADM_ID, HADM_ID],
        "ITEMID": [220050, 220179, 646],
        "CHARTTIME": [
            "2149-12-16 10:00:00",
            "2149-12-16 10:00:00",
            "2149-12-16 10:00:00",
        ],
        "VALUENUM": [95, 130, 96],
    }
).to_csv(f"{PREFIX}CHARTEVENTS.csv", index=False)

print(f"Mock data created in {DATA_DIR} for Subject {SUBJECT_ID}")
