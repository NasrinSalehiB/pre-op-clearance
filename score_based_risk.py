"""
Score-Based Cardiac Risk Extractor for MIMIC-III

This module computes several adapted cardiac risk scores and lab-based trends
using ONLY fields available in the MIMIC-III CSV extract you have:

A. FORMAL / SEMI-FORMAL RISK SCORES (ADAPTED)
   - RCRI:
       * Diagnoses: ischemic heart disease (MI/CAD), heart failure, stroke/TIA, diabetes
       * Labs: creatinine > 2.0 mg/dL
       * High-risk surgery: thoracic / major abdominal / major vascular (from procedures)
   - AUB-HAS-2:
       * Age (from PATIENTS + ADMISSIONS)
       * Surgery risk (inferred from procedure category)
       * Cardiac history (prior MI/HF/arrhythmia)
       * Anemia (Hgb from LABEVENTS)
   - NSQIP MACE (simplified):
       * Predictors: age, creatinine, surgery risk
       * ASA and detailed comorbids are not fully available; we approximate and
         mark missing factors explicitly.
   - Gupta MICA (simplified):
       * Predictors: procedure type, age, emergency status.
       * Functional status is not directly available; we mark it as unknown and
         treat this as a partial score.

B. LAB INTELLIGENCE & TRENDS (LABEVENTS)
   - BNP / Troponin:
       * Last available pre-admission value.
       * Delta = last - previous if multiple values exist.
   - Electrolytes:
       * Last K+, last Mg2+.
   - Renal trend:
       * All creatinine values within 72h before admission; if >=2 values, compute
         simple slope (delta creatinine / delta hours).
   - Anemia:
       * Last Hgb; last RDW if available.
   - Inflammatory:
       * Last CRP, last WBC.

C. DATA GAPS
   - If a required lab is missing, record a note such as:
       "No pre-admission BNP available".
   - If a formal score is partially computed, return:
       * score
       * components
       * missing_components list

The main entry point is:
    ScoreBasedRiskExtractor.build_score_based_risk_report(subject_id, hadm_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import pandas as pd


@dataclass
class ScoreComponentResult:
    score: Optional[float]
    components: Dict[str, Any] = field(default_factory=dict)
    missing_components: List[str] = field(default_factory=list)


class ScoreBasedRiskExtractor:
    """
    Compute adapted RCRI, AUB-HAS-2, NSQIP-MACE, and Gupta MICA scores
    plus lab trends from MIMIC-III CSVs.
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir

        # Core tables
        self.patients_df: Optional[pd.DataFrame] = None
        self.admissions_df: Optional[pd.DataFrame] = None
        self.procedures_icd_df: Optional[pd.DataFrame] = None
        self.d_icd_procedures_df: Optional[pd.DataFrame] = None
        self.diagnoses_icd_df: Optional[pd.DataFrame] = None
        self.d_icd_diagnoses_df: Optional[pd.DataFrame] = None
        self.labevents_df: Optional[pd.DataFrame] = None
        self.d_labitems_df: Optional[pd.DataFrame] = None

        # Detected configuration
        self.icd_version: Optional[str] = None  # "ICD9" or "ICD10"
        self.icd_code_column: Optional[str] = None

        # Lab item ID cache by semantic name
        self.item_ids_cache: Dict[str, List[int]] = {}

    # -------------------------------------------------------------------------
    # Setup / helpers
    # -------------------------------------------------------------------------

    def _match_hadm_id(self, df: pd.DataFrame, hadm_id: int) -> pd.Series:
        """Handle HADM_ID as int/float consistently."""
        return df["HADM_ID"].astype(float) == float(hadm_id)

    def _detect_icd_version(self) -> Optional[str]:
        """Detect ICD version from diagnoses table."""
        if self.diagnoses_icd_df is None or self.d_icd_diagnoses_df is None:
            return None

        if "ICD9_CODE" in self.diagnoses_icd_df.columns:
            self.icd_code_column = "ICD9_CODE"
            return "ICD9"
        if "ICD10_CODE" in self.diagnoses_icd_df.columns:
            self.icd_code_column = "ICD10_CODE"
            return "ICD10"
        if "ICD_CODE" in self.diagnoses_icd_df.columns:
            sample = self.diagnoses_icd_df["ICD_CODE"].dropna().head(100).astype(str)
            icd10_count = sample.str.match(r"^[A-Z]").sum()
            icd9_count = sample.str.match(r"^\d").sum()
            self.icd_code_column = "ICD_CODE"
            return "ICD10" if icd10_count > icd9_count else "ICD9"
        return None

    def _check_diag_patterns(
        self,
        df: pd.DataFrame,
        code_col: str,
        codes: List[str],
        patterns: List[str],
        exact_match: bool = False,
    ) -> pd.DataFrame:
        """Filter merged diagnoses by ICD code and/or LONG_TITLE patterns."""
        if df.empty:
            return df

        codes = [c for c in codes if c] or [""]
        if exact_match:
            code_match = df[code_col].astype(str).isin(codes)
        else:
            code_match = df[code_col].astype(str).str.startswith(tuple(codes))

        if "LONG_TITLE" in df.columns and patterns:
            pattern_match = df["LONG_TITLE"].astype(str).str.lower().str.contains(
                "|".join(patterns), na=False, regex=False
            )
            return df[code_match | pattern_match]
        return df[code_match]

    def _find_lab_item_ids(self, terms: List[str]) -> List[int]:
        """Search D_LABITEMS for labels matching given terms."""
        if self.d_labitems_df is None:
            return []
        df = self.d_labitems_df
        label = df["LABEL"].astype(str).str.lower()
        itemids: List[int] = []
        for term in terms:
            m = df[label.str.contains(term.lower(), na=False, regex=False)]
            if not m.empty:
                itemids.extend(m["ITEMID"].tolist())
        return list(set(itemids))

    def _init_lab_item_ids(self):
        """Initialize lab item IDs for needed markers."""
        # BNP
        self.item_ids_cache["bnp"] = self._find_lab_item_ids(
            ["bnp", "brain natriuretic", "b-type natriuretic", "ntprobnp", "nt-probnp"]
        ) or [50910, 50963]

        # Troponin
        self.item_ids_cache["troponin"] = self._find_lab_item_ids(
            ["troponin", "trop i", "trop t", "tni", "tnt"]
        ) or [51002, 51003]

        # Creatinine
        self.item_ids_cache["creatinine"] = self._find_lab_item_ids(
            ["creatinine", "creat"]
        ) or [50912, 50902]

        # Hgb
        self.item_ids_cache["hgb"] = self._find_lab_item_ids(
            ["hemoglobin", "hgb"]
        ) or [51221, 50811]

        # RDW
        self.item_ids_cache["rdw"] = self._find_lab_item_ids(["rdw", "red cell dist"])

        # WBC
        self.item_ids_cache["wbc"] = self._find_lab_item_ids(
            ["wbc", "white blood"]
        ) or [51300, 51301]

        # CRP
        self.item_ids_cache["crp"] = self._find_lab_item_ids(
            ["c-reactive", "crp"]
        ) or [50889]

        # Potassium
        self.item_ids_cache["k"] = self._find_lab_item_ids(
            ["potassium", "k "]
        ) or [50971, 50822]

        # Magnesium
        self.item_ids_cache["mg"] = self._find_lab_item_ids(
            ["magnesium", "mg "]
        ) or [50960]

        # Sodium (for electrolyte stability)
        self.item_ids_cache["na"] = self._find_lab_item_ids(
            ["sodium", "na "]
        ) or [50983, 50824]

        # Sodium
        self.item_ids_cache["na"] = self._find_lab_item_ids(
            ["sodium", "na "]
        ) or [50983, 50824]

    def _get_admit_time(self, hadm_id: int) -> Optional[datetime]:
        if self.admissions_df is None:
            return None
        adm = self.admissions_df[self.admissions_df["HADM_ID"] == hadm_id]
        if adm.empty:
            return None
        return pd.to_datetime(adm["ADMITTIME"].iloc[0])

    def _get_lab_series(
        self,
        subject_id: int,
        hadm_id: int,
        lab_name: str,
        hours_before: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Return all lab rows for a given lab semantic name and admission,
        optionally restricted to a time window before admission.

        NOTE: This helper is admission-based (filters by HADM_ID). For
        cross-admission pre-op windows relative to a specific time (e.g.
        surgery_time), use `_get_lab_series_by_time_window`, which only
        uses SUBJECT_ID + CHARTTIME.
        """
        if self.labevents_df is None:
            return None
        itemids = self.item_ids_cache.get(lab_name, [])
        if not itemids:
            return None

        labs = self.labevents_df[
            (self.labevents_df["SUBJECT_ID"] == subject_id)
            & (self.labevents_df["ITEMID"].isin(itemids))
            & (self.labevents_df["HADM_ID"].astype(float) == float(hadm_id))
        ].copy()
        if labs.empty:
            return None

        labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])
        admit_time = self._get_admit_time(hadm_id)
        if hours_before is not None and admit_time is not None:
            start = admit_time - timedelta(hours=hours_before)
            labs = labs[(labs["CHARTTIME"] >= start) & (labs["CHARTTIME"] <= admit_time)]
        if labs.empty:
            return None
        labs = labs.sort_values("CHARTTIME")
        return labs

    def _get_lab_series_by_time_window(
        self,
        subject_id: int,
        lab_name: str,
        anchor_time: datetime,
        hours_before: int,
    ) -> Optional[pd.DataFrame]:
        """
        Return all lab rows for a given lab semantic name for the subject
        within a time window [anchor_time - hours_before, anchor_time),
        regardless of HADM_ID.

        This is closer to a true pre-operative window and is what you want
        for trends that may span multiple admissions but are temporally close
        to the planned surgery. Values at or after anchor_time are excluded
        to avoid using intra-/post-operative labs when surgery_time is known.
        """
        if self.labevents_df is None:
            return None
        itemids = self.item_ids_cache.get(lab_name, [])
        if not itemids or anchor_time is None:
            return None

        labs = self.labevents_df[
            (self.labevents_df["SUBJECT_ID"] == subject_id)
            & (self.labevents_df["ITEMID"].isin(itemids))
        ].copy()
        if labs.empty:
            return None

        labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])
        start = anchor_time - timedelta(hours=hours_before)
        labs = labs[(labs["CHARTTIME"] >= start) & (labs["CHARTTIME"] < anchor_time)]
        if labs.empty:
            return None
        return labs.sort_values("CHARTTIME")

    def _get_last_and_delta(
        self, series_df: Optional[pd.DataFrame]
    ) -> Dict[str, Optional[float]]:
        """
        Given a lab series dataframe, return last value and delta
        (last - previous) if at least 2 data points exist.
        """
        if series_df is None or series_df.empty or "VALUENUM" not in series_df.columns:
            return {"last": None, "delta": None}

        vals = series_df["VALUENUM"].dropna().astype(float)
        if vals.empty:
            return {"last": None, "delta": None}

        last = float(vals.iloc[-1])
        if len(vals) >= 2:
            delta = float(vals.iloc[-1] - vals.iloc[-2])
        else:
            delta = None
        return {"last": last, "delta": delta}

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def load_data(
        self,
        subject_id: str = "249",
        patients_file: str = "PATIENTS.csv",
        admissions_file: str = "ADMISSIONS.csv",
        procedures_icd_file: str = "PROCEDURES_ICD.csv",
        diagnoses_icd_file: str = "DIAGNOSES_ICD.csv",
        d_icd_diagnoses_file: str = "D_ICD_DIAGNOSES.csv",
        d_icd_procedures_file: str = "D_ICD_PROCEDURES.csv",
        labevents_file: str = "LABEVENTS.csv",
        d_labitems_file: str = "D_LABITEMS.csv",
    ):
        """
        Load all required MIMIC-III CSV files for a given subject slice.
        """
        prefix = f"{self.data_dir}/Subject_ID_{subject_id}_"
        print("Loading MIMIC-III data files for score-based risk extraction...")

        self.patients_df = pd.read_csv(f"{prefix}{patients_file}")
        self.admissions_df = pd.read_csv(f"{prefix}{admissions_file}")
        self.procedures_icd_df = pd.read_csv(f"{prefix}{procedures_icd_file}")
        self.diagnoses_icd_df = pd.read_csv(f"{prefix}{diagnoses_icd_file}")
        self.d_icd_diagnoses_df = pd.read_csv(f"{prefix}{d_icd_diagnoses_file}")
        self.d_icd_procedures_df = pd.read_csv(f"{prefix}{d_icd_procedures_file}")
        self.labevents_df = pd.read_csv(f"{prefix}{labevents_file}")
        self.d_labitems_df = pd.read_csv(f"{prefix}{d_labitems_file}")

        print("Data files loaded successfully.")

        self.icd_version = self._detect_icd_version() or "ICD9"
        print(f"Detected ICD version: {self.icd_version}")

        self._init_lab_item_ids()

    # -------------------------------------------------------------------------
    # A. Formal / adapted risk scores
    # -------------------------------------------------------------------------

    def _get_age(self, subject_id: int, hadm_id: int) -> Optional[int]:
        if self.patients_df is None or self.admissions_df is None:
            return None
        patient = self.patients_df[self.patients_df["SUBJECT_ID"] == subject_id]
        if patient.empty:
            return None
        dob = pd.to_datetime(patient["DOB"].iloc[0])
        adm = self.admissions_df[self.admissions_df["HADM_ID"] == hadm_id]
        if adm.empty:
            return None
        admittime = pd.to_datetime(adm["ADMITTIME"].iloc[0])
        age = (admittime - dob).days / 365.25
        return min(int(age), 90) if age >= 0 else None

    def _get_diagnoses_for_admission(
        self, subject_id: int, hadm_id: int
    ) -> Optional[pd.DataFrame]:
        if self.diagnoses_icd_df is None or self.d_icd_diagnoses_df is None:
            return None
        diag = self.diagnoses_icd_df[
            (self.diagnoses_icd_df["SUBJECT_ID"] == subject_id)
            & (self.diagnoses_icd_df["HADM_ID"] == hadm_id)
        ].copy()
        if diag.empty:
            return None

        if self.icd_version == "ICD9":
            merge_col = "ICD9_CODE"
            code_col = "ICD9_CODE"
        else:
            merge_col = (
                "ICD10_CODE"
                if "ICD10_CODE" in self.d_icd_diagnoses_df.columns
                else "ICD_CODE"
            )
            code_col = self.icd_code_column or "ICD_CODE"

        if code_col not in diag.columns:
            return None

        merged = diag.merge(
            self.d_icd_diagnoses_df,
            left_on=code_col,
            right_on=merge_col,
            how="left",
        )
        merged["CODE_COL"] = code_col if code_col in merged.columns else merge_col
        return merged

    def _classify_surgery_risk(self, subject_id: int, hadm_id: int) -> str:
        """
        Roughly classify surgery as high/intermediate/low risk
        based on ICD procedure descriptions.
        """
        if self.procedures_icd_df is None or self.d_icd_procedures_df is None:
            return "unknown"

        procs = self.procedures_icd_df[
            (self.procedures_icd_df["SUBJECT_ID"] == subject_id)
            & (self.procedures_icd_df["HADM_ID"] == hadm_id)
        ].copy()
        if procs.empty:
            return "unknown"

        if self.icd_version == "ICD9":
            merge_col = "ICD9_CODE"
            code_col = "ICD9_CODE"
        else:
            merge_col = (
                "ICD10_CODE"
                if "ICD10_CODE" in self.d_icd_procedures_df.columns
                else "ICD_CODE"
            )
            code_col = self.icd_code_column or "ICD_CODE"

        if code_col not in procs.columns:
            return "unknown"

        merged = procs.merge(
            self.d_icd_procedures_df,
            left_on=code_col,
            right_on=merge_col,
            how="left",
        )
        text = " ".join(merged.get("LONG_TITLE", pd.Series([], dtype=str)).astype(str)).lower()

        # High-risk: According to RCRI, high-risk surgeries include:
        # - Intraperitoneal procedures
        # - Intrathoracic procedures
        # - Suprainguinal vascular procedures
        high_keywords = [
            # Intrathoracic procedures
            "thoracotomy",
            "lobectomy",
            "pneumonectomy",
            "lung resection",
            "wedge resection",
            "esophagectomy",
            "mediastinal",
            "tracheostomy",
            "tracheotomy",
            "endotracheal",
            "airway",
            # Intraperitoneal procedures (major abdominal)
            "laparotomy",
            "colectomy",
            "gastrectomy",
            "hysterectomy",
            "cholecystectomy",
            "splenectomy",
            "hepatectomy",
            "pancreatectomy",
            "nephrectomy",
            # Suprainguinal vascular procedures
            "aortic",
            "aneurysm",
            "carotid endarterectomy",
            "endarterectomy",
            "coronary artery bypass",
            "cabg",
            "vascular",
        ]
        if any(k in text for k in high_keywords):
            return "high"

        # Intermediate: other procedures (head/neck, peripheral)
        intermediate_keywords = [
            "thyroid",
            "laryngectomy",
            "peripheral",
        ]
        if any(k in text for k in intermediate_keywords):
            return "intermediate"

        # Otherwise consider low risk
        return "low"

    # ---- RCRI ----

    def _compute_rcri(
        self,
        subject_id: int,
        hadm_id: int,
        temporal: Optional[Dict[str, Any]] = None,
        surgery_time: Optional[datetime] = None,
    ) -> ScoreComponentResult:
        """
        Compute an adapted RCRI score using available MIMIC-III fields.
        """
        components: Dict[str, bool] = {
            "ischemic_heart_disease": False,
            "heart_failure": False,
            "stroke_tia": False,
            "diabetes": False,
            "creatinine_gt_2": False,
            "high_risk_surgery": False,
        }
        missing: List[str] = []

        diag = self._get_diagnoses_for_admission(subject_id, hadm_id)
        if diag is None:
            missing.extend(
                [
                    "ischemic_heart_disease",
                    "heart_failure",
                    "stroke_tia",
                    "diabetes",
                ]
            )
        else:
            code_col = "CODE_COL"

            # Ischemic heart disease
            if self.icd_version == "ICD9":
                ihd_codes = [str(c) for c in range(410, 415)]
            else:
                ihd_codes = ["I20", "I21", "I22", "I23", "I24", "I25"]
            ihd_patterns = ["myocardial infarction", "ischemic heart", "coronary artery", "angina"]
            ihd_df = self._check_diag_patterns(diag, code_col, ihd_codes, ihd_patterns)
            components["ischemic_heart_disease"] = not ihd_df.empty

            # Heart failure
            if self.icd_version == "ICD9":
                hf_codes = ["428"]
            else:
                hf_codes = ["I50"]
            hf_patterns = ["heart failure", "cardiac failure", "congestive heart failure"]
            hf_df = self._check_diag_patterns(diag, code_col, hf_codes, hf_patterns)
            components["heart_failure"] = not hf_df.empty

            # Stroke/TIA
            if self.icd_version == "ICD9":
                stroke_codes = [str(c) for c in range(430, 439)]
            else:
                stroke_codes = ["I63", "I64", "G45"]
            stroke_patterns = ["stroke", "cerebrovascular", "tia", "transient ischemic"]
            stroke_df = self._check_diag_patterns(
                diag, code_col, stroke_codes, stroke_patterns
            )
            components["stroke_tia"] = not stroke_df.empty

            # Diabetes (any)
            if self.icd_version == "ICD9":
                diab_codes = ["250"]
            else:
                diab_codes = ["E10", "E11"]
            diab_patterns = ["diabetes"]
            diab_df = self._check_diag_patterns(diag, code_col, diab_codes, diab_patterns)
            components["diabetes"] = not diab_df.empty

        # Creatinine > 2 (pre-op window, across admissions)
        # Use surgery_time if provided, otherwise fall back to admit_time
        anchor_time = surgery_time or self._get_admit_time(hadm_id)
        creat_series = self._get_lab_series_by_time_window(
            subject_id, "creatinine", anchor_time, 72
        ) if anchor_time is not None else None
        creat_info = self._get_last_and_delta(creat_series)
        if creat_info["last"] is None:
            missing.append("creatinine_gt_2")
        else:
            components["creatinine_gt_2"] = creat_info["last"] > 2.0

        # High-risk surgery
        surg_risk = self._classify_surgery_risk(subject_id, hadm_id)
        components["high_risk_surgery"] = surg_risk == "high"

        # Integrate temporal flags (BNP/troponin/electrolytes) as contextual info,
        # but DO NOT change the canonical RCRI numeric score.
        if temporal is not None:
            components["worsening_bnp"] = temporal.get("bnp", {}).get("worsening_bnp", False)
            components["rising_troponin"] = temporal.get("troponin", {}).get("rising_troponin", False)
            components["unstable_electrolytes"] = temporal.get("electrolytes", {}).get(
                "unstable_electrolytes", False
            )

        # RCRI score = count of the classical components only
        classical_keys = [
            "ischemic_heart_disease",
            "heart_failure",
            "stroke_tia",
            "diabetes",
            "creatinine_gt_2",
            "high_risk_surgery",
        ]
        score = float(sum(1 for k in classical_keys if components.get(k)))

        # Derive a simple categorical risk level and temporal-adjusted category.
        # Common interpretation:
        #   0 → low, 1–2 → intermediate, ≥3 → high
        if score == 0:
            base_cat = "low"
            idx = 0
        elif score <= 2:
            base_cat = "intermediate"
            idx = 1
        else:
            base_cat = "high"
            idx = 2

        adjusted_idx = idx
        temporal_explanations: List[str] = []

        if temporal is not None:
            if components.get("worsening_bnp"):
                temporal_explanations.append(
                    "BNP rising over 48h pre-op suggests possible heart failure exacerbation."
                )
                if adjusted_idx < 2:
                    adjusted_idx += 1
            if components.get("rising_troponin"):
                temporal_explanations.append(
                    "Troponin rising over 6h pre-op suggests active myocardial injury."
                )
                if adjusted_idx < 2:
                    adjusted_idx += 1
            if components.get("unstable_electrolytes"):
                temporal_explanations.append(
                    "Electrolyte shifts over 24h (K/Na) increase arrhythmia and anesthesia risk."
                )
                # We leave category unchanged here; used as contextual risk.

        categories = ["low", "intermediate", "high"]
        components["risk_category"] = base_cat
        components["temporal_adjusted_category"] = categories[adjusted_idx]
        if temporal_explanations:
            components["temporal_explanations"] = temporal_explanations

        return ScoreComponentResult(score=score, components=components, missing_components=missing)

    # ---- AUB-HAS-2 (simplified) ----

    def _compute_aub_has2(
        self,
        subject_id: int,
        hadm_id: int,
        temporal: Optional[Dict[str, Any]] = None,
        surgery_time: Optional[datetime] = None,
    ) -> ScoreComponentResult:
        """
        Compute a simplified AUB-HAS-2 style score based on:
        - Age
        - Surgery risk (high/intermediate/low)
        - Cardiac history (any of: MI, HF, arrhythmia)
        - Anemia (Hgb < 13 for men, < 12 for women)
        """
        components: Dict[str, Any] = {
            "age": None,
            "surgery_risk": "unknown",
            "cardiac_history": False,
            "anemia": False,
        }
        missing: List[str] = []

        age = self._get_age(subject_id, hadm_id)
        components["age"] = age
        if age is None:
            missing.append("age")

        surg_risk = self._classify_surgery_risk(subject_id, hadm_id)
        components["surgery_risk"] = surg_risk
        if surg_risk == "unknown":
            missing.append("surgery_risk")

        # Cardiac history from diagnoses: MI/HF/arrhythmia
        diag = self._get_diagnoses_for_admission(subject_id, hadm_id)
        if diag is None:
            missing.append("cardiac_history")
        else:
            code_col = "CODE_COL"
            cardiac_history = False

            # MI/IHD
            if self.icd_version == "ICD9":
                ihd_codes = [str(c) for c in range(410, 415)]
            else:
                ihd_codes = ["I20", "I21", "I22", "I23", "I24", "I25"]
            ihd_df = self._check_diag_patterns(
                diag, code_col, ihd_codes, ["myocardial infarction", "ischemic heart"]
            )
            if not ihd_df.empty:
                cardiac_history = True

            # HF
            if self.icd_version == "ICD9":
                hf_codes = ["428"]
            else:
                hf_codes = ["I50"]
            hf_df = self._check_diag_patterns(
                diag, code_col, hf_codes, ["heart failure"]
            )
            if not hf_df.empty:
                cardiac_history = True

            # Arrhythmia
            if self.icd_version == "ICD9":
                arr_codes = ["427"]
            else:
                arr_codes = ["I47", "I48", "I49"]
            arr_df = self._check_diag_patterns(
                diag,
                code_col,
                arr_codes,
                ["arrhythmia", "atrial fibrillation", "ventricular tachycardia", "afib"],
            )
            if not arr_df.empty:
                cardiac_history = True

            components["cardiac_history"] = cardiac_history

        # Anemia from Hgb (pre-op window, across admissions)
        # Use surgery_time if provided, otherwise fall back to admit_time
        anchor_time = surgery_time or self._get_admit_time(hadm_id)
        hgb_series = self._get_lab_series_by_time_window(
            subject_id, "hgb", anchor_time, 72
        ) if anchor_time is not None else None
        hgb_info = self._get_last_and_delta(hgb_series)
        if hgb_info["last"] is None:
            missing.append("anemia")
        else:
            # Gender-specific thresholds if possible
            gender = None
            if self.patients_df is not None:
                p = self.patients_df[self.patients_df["SUBJECT_ID"] == subject_id]
                if not p.empty and "GENDER" in p.columns:
                    gender = p["GENDER"].iloc[0]
            if gender == "M":
                components["anemia"] = hgb_info["last"] < 13.0
            elif gender == "F":
                components["anemia"] = hgb_info["last"] < 12.0
            else:
                components["anemia"] = hgb_info["last"] < 12.5

        # Simple point system (not validated AUB-HAS-2):
        #   +1 age >= 65
        #   +1 high-risk surgery
        #   +1 cardiac history
        #   +1 anemia
        score = 0.0
        if age is not None and age >= 65:
            score += 1
        if surg_risk == "high":
            score += 1
        if components["cardiac_history"]:
            score += 1
        if components["anemia"]:
            score += 1

        # Attach temporal patterns context without changing the numeric score
        if temporal is not None:
            components["worsening_bnp"] = temporal.get("bnp", {}).get("worsening_bnp", False)
            components["rising_troponin"] = temporal.get("troponin", {}).get("rising_troponin", False)
            components["unstable_electrolytes"] = temporal.get("electrolytes", {}).get(
                "unstable_electrolytes", False
            )

        # Simple category + temporal-adjusted category for narrative context
        #   0–1 → low, 2–3 → intermediate, 4 → high (very rough buckets)
        if score <= 1:
            base_cat = "low"
            idx = 0
        elif score <= 3:
            base_cat = "intermediate"
            idx = 1
        else:
            base_cat = "high"
            idx = 2

        adjusted_idx = idx
        temporal_explanations: List[str] = []
        if temporal is not None:
            if components.get("worsening_bnp"):
                temporal_explanations.append(
                    "BNP rising pre-op supports history of or evolving heart failure."
                )
                if adjusted_idx < 2:
                    adjusted_idx += 1
            if components.get("rising_troponin"):
                temporal_explanations.append(
                    "Rising troponin pre-op suggests active cardiac injury."
                )
                if adjusted_idx < 2:
                    adjusted_idx += 1
            if components.get("unstable_electrolytes"):
                temporal_explanations.append(
                    "Unstable K+/Na+ over 24h increases perioperative cardiac risk."
                )

        categories = ["low", "intermediate", "high"]
        components["risk_category"] = base_cat
        components["temporal_adjusted_category"] = categories[adjusted_idx]
        if temporal_explanations:
            components["temporal_explanations"] = temporal_explanations

        return ScoreComponentResult(score=score, components=components, missing_components=missing)

    # ---- NSQIP MACE (very simplified) ----

    def _compute_nsqip_mace(
        self, 
        subject_id: int, 
        hadm_id: int,
        surgery_time: Optional[datetime] = None,
    ) -> ScoreComponentResult:
        """
        Very simplified NSQIP MACE-like risk:
        - Age
        - Creatinine
        - Surgery risk

        We do NOT attempt to replicate the original logistic model; instead we
        provide a rough points-based surrogate and highlight missing pieces.
        """
        components: Dict[str, Any] = {
            "age": None,
            "creatinine": None,
            "surgery_risk": "unknown",
        }
        missing: List[str] = []

        age = self._get_age(subject_id, hadm_id)
        components["age"] = age
        if age is None:
            missing.append("age")

        # Use surgery_time if provided, otherwise fall back to admit_time
        anchor_time = surgery_time or self._get_admit_time(hadm_id)
        creat_series = self._get_lab_series_by_time_window(
            subject_id, "creatinine", anchor_time, 72
        ) if anchor_time is not None else None
        creat_info = self._get_last_and_delta(creat_series)
        components["creatinine"] = creat_info["last"]
        if creat_info["last"] is None:
            missing.append("creatinine")

        surg_risk = self._classify_surgery_risk(subject_id, hadm_id)
        components["surgery_risk"] = surg_risk
        if surg_risk == "unknown":
            missing.append("surgery_risk")

        # Simple points:
        #  age >= 70 → +2, 60–69 → +1
        #  creatinine >= 1.5 → +2
        #  high-risk surgery → +2, intermediate → +1
        score = 0.0
        if age is not None:
            if age >= 70:
                score += 2
            elif age >= 60:
                score += 1
        if creat_info["last"] is not None and creat_info["last"] >= 1.5:
            score += 2
        if surg_risk == "high":
            score += 2
        elif surg_risk == "intermediate":
            score += 1

        # Derive risk category (0-2 → low, 3-4 → intermediate, 5-6 → high)
        if score <= 2:
            risk_category = "low"
        elif score <= 4:
            risk_category = "intermediate"
        else:
            risk_category = "high"
        
        components["risk_category"] = risk_category

        return ScoreComponentResult(score=score, components=components, missing_components=missing)

    # ---- Gupta MICA (simplified) ----

    def _get_emergency_status(self, hadm_id: int) -> Optional[str]:
        if self.admissions_df is None:
            return None
        adm = self.admissions_df[self.admissions_df["HADM_ID"] == hadm_id]
        if adm.empty:
            return None
        adm_type = str(adm["ADMISSION_TYPE"].iloc[0]).lower()
        if "emergency" in adm_type:
            return "emergent"
        if "urgent" in adm_type:
            return "urgent"
        return "elective"

    def _classify_procedure_type(self, subject_id: int, hadm_id: int) -> Optional[str]:
        """
        Rough procedure type for Gupta MICA (cardiac / vascular / thoracic / abdominal / other).
        """
        if self.procedures_icd_df is None or self.d_icd_procedures_df is None:
            return None
        procs = self.procedures_icd_df[
            (self.procedures_icd_df["SUBJECT_ID"] == subject_id)
            & (self.procedures_icd_df["HADM_ID"] == hadm_id)
        ].copy()
        if procs.empty:
            return None

        if self.icd_version == "ICD9":
            merge_col = "ICD9_CODE"
            code_col = "ICD9_CODE"
        else:
            merge_col = (
                "ICD10_CODE"
                if "ICD10_CODE" in self.d_icd_procedures_df.columns
                else "ICD_CODE"
            )
            code_col = self.icd_code_column or "ICD_CODE"

        if code_col not in procs.columns:
            return None

        merged = procs.merge(
            self.d_icd_procedures_df,
            left_on=code_col,
            right_on=merge_col,
            how="left",
        )
        text = " ".join(merged.get("LONG_TITLE", pd.Series([], dtype=str)).astype(str)).lower()

        # Cardiac
        cardiac_keywords = ["coronary artery bypass", "cabg", "valve", "pacemaker", "defibrillator"]
        if any(k in text for k in cardiac_keywords):
            return "Cardiac"

        # Vascular
        vascular_keywords = ["aortic", "carotid endarterectomy", "peripheral bypass", "aneurysm"]
        if any(k in text for k in vascular_keywords):
            return "vascular"

        # Thoracic
        thoracic_keywords = ["lobectomy", "pneumonectomy", "lung resection", "thoracotomy"]
        if any(k in text for k in thoracic_keywords):
            return "thoracic"

        # Abdominal
        abdominal_keywords = ["laparotomy", "colectomy", "gastrectomy", "hysterectomy"]
        if any(k in text for k in abdominal_keywords):
            return "abdominal"

        return "other"

    def _compute_gupta_mica(
        self, subject_id: int, hadm_id: int
    ) -> ScoreComponentResult:
        """
        Very simplified Gupta MICA-style risk:
        - Procedure type
        - Age
        - Emergency status
        Functional status is not available and is marked as missing.
        """
        components: Dict[str, Any] = {
            "procedure_type": None,
            "age": None,
            "emergency_status": None,
            "functional_status": "unknown",  # cannot be inferred reliably
        }
        missing: List[str] = []

        age = self._get_age(subject_id, hadm_id)
        components["age"] = age
        if age is None:
            missing.append("age")

        proc_type = self._classify_procedure_type(subject_id, hadm_id)
        components["procedure_type"] = proc_type
        if proc_type is None:
            missing.append("procedure_type")

        emerg = self._get_emergency_status(hadm_id)
        components["emergency_status"] = emerg
        if emerg is None:
            missing.append("emergency_status")

        missing.append("functional_status")  # explicit data gap

        # Simple points:
        #   cardiac / vascular / thoracic → +2
        #   emergency/urgent → +2
        #   age >= 70 → +2, 60–69 → +1
        score = 0.0
        if proc_type in ["Cardiac", "cardiac", "vascular", "thoracic"]:
            score += 2
        if emerg in ["emergent", "urgent"]:
            score += 2
        if age is not None:
            if age >= 70:
                score += 2
            elif age >= 60:
                score += 1

        # Derive risk category (0-2 → low, 3-4 → intermediate, 5-6 → high)
        if score <= 2:
            risk_category = "low"
        elif score <= 4:
            risk_category = "intermediate"
        else:
            risk_category = "high"
        
        components["risk_category"] = risk_category

        return ScoreComponentResult(score=score, components=components, missing_components=missing)

    # -------------------------------------------------------------------------
    # B. Lab intelligence & trends
    # -------------------------------------------------------------------------

    def _compute_lab_intelligence(
        self,
        subject_id: int,
        hadm_id: int,
        notes: List[str],
        anchor_time: Optional[datetime] = None,
        hours_before: int = 72,
    ) -> Dict[str, Any]:
        labs: Dict[str, Any] = {}

        # Determine anchor time: if not provided, use ADMITTIME as proxy
        if anchor_time is None:
            anchor_time = self._get_admit_time(hadm_id)

        # BNP & Troponin: last pre-op within window, with delta (SUBJECT_ID + time window).
        # Document if values span multiple admissions (cross-admission trend).
        bnp_series = self._get_lab_series_by_time_window(
            subject_id, "bnp", anchor_time, hours_before
        ) if anchor_time is not None else None
        troponin_series = self._get_lab_series_by_time_window(
            subject_id, "troponin", anchor_time, hours_before
        ) if anchor_time is not None else None
        bnp_info = self._get_last_and_delta(bnp_series)
        trop_info = self._get_last_and_delta(troponin_series)
        if bnp_info["last"] is None:
            notes.append("No pre-admission BNP available.")
        if trop_info["last"] is None:
            notes.append("No pre-admission troponin available.")
        # Cross-admission documentation for BNP/troponin
        if bnp_series is not None and "HADM_ID" in bnp_series.columns:
            hadms = sorted(set(bnp_series["HADM_ID"].dropna().astype(int)))
            if len(hadms) > 1:
                notes.append(f"BNP window includes values from multiple admissions: {hadms}.")
        if troponin_series is not None and "HADM_ID" in troponin_series.columns:
            hadms = sorted(set(troponin_series["HADM_ID"].dropna().astype(int)))
            if len(hadms) > 1:
                notes.append(f"Troponin window includes values from multiple admissions: {hadms}.")

        labs["bnp"] = bnp_info
        labs["troponin"] = trop_info

        # Electrolytes: last K+, Mg2+ within window
        k_series = self._get_lab_series_by_time_window(
            subject_id, "k", anchor_time, hours_before
        ) if anchor_time is not None else None
        mg_series = self._get_lab_series_by_time_window(
            subject_id, "mg", anchor_time, hours_before
        ) if anchor_time is not None else None
        k_info = self._get_last_and_delta(k_series)
        mg_info = self._get_last_and_delta(mg_series)
        if k_info["last"] is None:
            notes.append("No pre-admission potassium (K+) available.")
        if mg_info["last"] is None:
            notes.append("No pre-admission magnesium (Mg2+) available.")
        # Also capture Na for stability checks (delta handled in cardiac temporal patterns)
        na_series = self._get_lab_series_by_time_window(
            subject_id, "na", anchor_time, hours_before
        ) if anchor_time is not None else None
        na_info = self._get_last_and_delta(na_series)
        labs["electrolytes"] = {
            "k": k_info["last"],
            "mg": mg_info["last"],
            "na": na_info["last"],
        }

        # Renal trend: creatinine slope over window if >=2 values
        creat_series = self._get_lab_series_by_time_window(
            subject_id, "creatinine", anchor_time, hours_before
        ) if anchor_time is not None else None
        renal_trend = {"slope_per_hour": None, "n_points": 0}
        if creat_series is not None and not creat_series.empty:
            vals = creat_series["VALUENUM"].dropna().astype(float)
            times = creat_series["CHARTTIME"]
            if len(vals) >= 2:
                delta_creat = float(vals.iloc[-1] - vals.iloc[0])
                delta_hours = max(
                    (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600.0, 0.1
                )
                renal_trend["slope_per_hour"] = delta_creat / delta_hours
                renal_trend["n_points"] = len(vals)
            else:
                renal_trend["n_points"] = len(vals)
        else:
            notes.append("No pre-admission creatinine trend available (<=1 value).")
        labs["renal_trend"] = renal_trend

        # Anemia: last Hgb + RDW if available in window
        hgb_series = self._get_lab_series_by_time_window(
            subject_id, "hgb", anchor_time, hours_before
        ) if anchor_time is not None else None
        rdw_series = self._get_lab_series_by_time_window(
            subject_id, "rdw", anchor_time, hours_before
        ) if anchor_time is not None else None
        hgb_info = self._get_last_and_delta(hgb_series)
        rdw_info = self._get_last_and_delta(rdw_series)
        if hgb_info["last"] is None:
            notes.append("No pre-admission hemoglobin (Hgb) available.")
        anemia_block = {
            "hgb_last": hgb_info["last"],
            "rdw_last": rdw_info["last"],
        }
        labs["anemia"] = anemia_block

        # Inflammatory: last CRP, WBC in window
        crp_series = self._get_lab_series_by_time_window(
            subject_id, "crp", anchor_time, hours_before
        ) if anchor_time is not None else None
        wbc_series = self._get_lab_series_by_time_window(
            subject_id, "wbc", anchor_time, hours_before
        ) if anchor_time is not None else None
        crp_info = self._get_last_and_delta(crp_series)
        wbc_info = self._get_last_and_delta(wbc_series)
        if crp_info["last"] is None:
            notes.append("No pre-admission CRP available.")
        if wbc_info["last"] is None:
            notes.append("No pre-admission WBC available.")
        labs["inflammatory"] = {"crp_last": crp_info["last"], "wbc_last": wbc_info["last"]}

        return labs

    # generic temporal feature aggregation function is unused and removed for simplicity

    # -------------------------------------------------------------------------
    # Cardiac temporal patterns (BNP, troponin, electrolytes)
    # -------------------------------------------------------------------------

    def analyze_cardiac_temporal_patterns(
        self,
        subject_id: int,
        anchor_time: datetime,
    ) -> Dict[str, Any]:
        """
        Analyze cardiac temporal patterns prior to surgery / index event:

        - BNP 48h delta:
            ΔBNP_48h = BNP(t0) - BNP(t-48h) (approximated as last - first in window)
            Threshold: > 100 pg/mL increase → worsening_bnp flag

        - Troponin 6h delta:
            ΔTrop_6h = Trop(t0) - Trop(t-6h)
            Threshold: any increase above ULN (~0.04) → rising_troponin flag

        - Electrolyte stability (K, Na) over 24h:
            ΔK_24h, ΔNa_24h = last - first in window
            Threshold: |ΔK| > 0.5 or |ΔNa| > 5 → unstable_electrolytes flag
        """
        patterns: Dict[str, Any] = {
            "bnp": {
                "delta_48h": None,
                "worsening_bnp": False,
            },
            "troponin": {
                "delta_6h": None,
                "rising_troponin": False,
            },
            "electrolytes": {
                "delta_k_24h": None,
                "delta_na_24h": None,
                "unstable_electrolytes": False,
            },
        }

        # BNP 48h delta
        bnp_series = self._get_lab_series_by_time_window(
            subject_id, "bnp", anchor_time, 48
        )
        if bnp_series is not None and not bnp_series.empty:
            vals = bnp_series["VALUENUM"].dropna().astype(float)
            if len(vals) >= 2:
                first_val = float(vals.iloc[0])
                last_val = float(vals.iloc[-1])
                delta_48h = last_val - first_val
                patterns["bnp"]["delta_48h"] = delta_48h
                if delta_48h > 100.0:
                    patterns["bnp"]["worsening_bnp"] = True

        # Troponin 6h delta
        trop_series = self._get_lab_series_by_time_window(
            subject_id, "troponin", anchor_time, 6
        )
        if trop_series is not None and not trop_series.empty:
            vals = trop_series["VALUENUM"].dropna().astype(float)
            if len(vals) >= 2:
                first_val = float(vals.iloc[0])
                last_val = float(vals.iloc[-1])
                delta_6h = last_val - first_val
                patterns["troponin"]["delta_6h"] = delta_6h

                # ULN assumption for troponin (depends on assay; 0.04 is a common cutoff)
                ULN_TROP = 0.04
                if delta_6h > 0 and last_val > ULN_TROP:
                    patterns["troponin"]["rising_troponin"] = True

        # Electrolyte stability over 24h (K, Na)
        k_series = self._get_lab_series_by_time_window(
            subject_id, "k", anchor_time, 24
        )
        na_series = self._get_lab_series_by_time_window(
            subject_id, "na", anchor_time, 24
        )
        delta_k = None
        delta_na = None

        if k_series is not None and not k_series.empty:
            kv = k_series["VALUENUM"].dropna().astype(float)
            if len(kv) >= 2:
                delta_k = float(kv.iloc[-1] - kv.iloc[0])
        if na_series is not None and not na_series.empty:
            nav = na_series["VALUENUM"].dropna().astype(float)
            if len(nav) >= 2:
                delta_na = float(nav.iloc[-1] - nav.iloc[0])

        patterns["electrolytes"]["delta_k_24h"] = delta_k
        patterns["electrolytes"]["delta_na_24h"] = delta_na
        if (delta_k is not None and abs(delta_k) > 0.5) or (
            delta_na is not None and abs(delta_na) > 5.0
        ):
            patterns["electrolytes"]["unstable_electrolytes"] = True

        return patterns

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build_score_based_risk_report(
        self,
        subject_id: int,
        hadm_id: int,
        surgery_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Build a combined report:
        - rcri
        - aub_has2
        - nsqip_mace
        - gupta_mica
        - labs (trends)
        - data_quality (notes)
        """
        data_quality_notes: List[str] = []

        # Anchor time for pre-op window: prefer explicit surgery_time, else ADMITTIME
        anchor_time = surgery_time or self._get_admit_time(hadm_id)

        # Cardiac temporal patterns (BNP, troponin, electrolytes)
        cardiac_temporal = (
            self.analyze_cardiac_temporal_patterns(subject_id, anchor_time)
            if anchor_time is not None
            else {}
        )

        rcri = self._compute_rcri(subject_id, hadm_id, temporal=cardiac_temporal, surgery_time=anchor_time)
        aub_has2 = self._compute_aub_has2(subject_id, hadm_id, temporal=cardiac_temporal, surgery_time=anchor_time)
        nsqip = self._compute_nsqip_mace(subject_id, hadm_id, surgery_time=anchor_time)
        gupta = self._compute_gupta_mica(subject_id, hadm_id)

        labs = self._compute_lab_intelligence(
            subject_id,
            hadm_id,
            data_quality_notes,
            anchor_time=anchor_time,
            hours_before=72,
        )

        # Record missing components as data-quality gaps
        scores = {
            "rcri": rcri,
            "aub_has2": aub_has2,
            "nsqip_mace": nsqip,
            "gupta_mica": gupta,
        }
        missing_scores: Dict[str, List[str]] = {}
        for name, res in scores.items():
            if res.missing_components:
                missing_scores[name] = res.missing_components

        if missing_scores:
            data_quality_notes.append(
                f"Some scores have missing components: {missing_scores}"
            )

        return {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "scores": {
                "rcri": {
                    "score": rcri.score,
                    "components": rcri.components,
                    "missing_components": rcri.missing_components,
                },
                "aub_has2": {
                    "score": aub_has2.score,
                    "components": aub_has2.components,
                    "missing_components": aub_has2.missing_components,
                },
                "nsqip_mace": {
                    "score": nsqip.score,
                    "components": nsqip.components,
                    "missing_components": nsqip.missing_components,
                },
                "gupta_mica": {
                    "score": gupta.score,
                    "components": gupta.components,
                    "missing_components": gupta.missing_components,
                },
            },
            "labs": labs,
            "cardiac_temporal_patterns": cardiac_temporal,
            "data_quality": {
                "notes": data_quality_notes,
            },
        }


if __name__ == "__main__":
    # Example usage / quick test for subject 249
    extractor = ScoreBasedRiskExtractor(data_dir="./data")
    extractor.load_data(subject_id="249")

    subject_id = 249
    hadm_id = 116935

    report = extractor.build_score_based_risk_report(subject_id, hadm_id)
    print(report)


