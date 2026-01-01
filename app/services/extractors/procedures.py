"""
Procedure Event Risk Extractor for MIMIC-III

This module builds a structured report of:

A. PRIOR RELATED SURGERIES (that impact cardiac risk):
   - Cardiac surgeries: CABG, valve repair/replacement, PCI (stent), pacemaker/ICD implant
   - Vascular surgeries: aortic repair, carotid endarterectomy, peripheral bypass
   - Thoracic surgeries: lung resection, esophagectomy
   - Any surgery within last 90 days (systemic stress)
   For each: date, type, and if possible, complications (MI, stroke, shock).

B. PRIOR ADVERSE CARDIAC EVENTS:
   - Myocardial infarction (with date)
   - Stroke/TIA (with date)
   - Heart failure admission (with date + BNP trend)
   - Arrhythmia requiring treatment (AFib, VT)
   - Cardiogenic shock or cardiac arrest
   For each: date, severity proxy, and whether likely post-operative.

C. CURRENT SURGICAL CARDIAC RISK CONTEXT:
   - Planned surgery type & expected duration
   - Emergency status (emergent/urgent/elective)
   - Expected hemodynamic stress (keywords)
   - Positioning risks (prone, Trendelenburg, lateral decubitus)

Returns:
   Structured dictionary with sections:
   - prior_surgeries
   - prior_events
   - surgical_context
Each section includes a simple risk flag: "High" | "Medium" | "Low".
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


class ProcedureEventRiskExtractor:
    """
    Extracts procedure- and event-based cardiac risk context from MIMIC-III.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the extractor.

        Args:
            data_dir: Directory containing MIMIC-III CSV files (default: "./data")
        """
        self.data_dir = data_dir

        self.patients_df: Optional[pd.DataFrame] = None
        self.admissions_df: Optional[pd.DataFrame] = None
        self.procedures_icd_df: Optional[pd.DataFrame] = None
        self.d_icd_procedures_df: Optional[pd.DataFrame] = None
        self.diagnoses_icd_df: Optional[pd.DataFrame] = None
        self.d_icd_diagnoses_df: Optional[pd.DataFrame] = None
        self.labevents_df: Optional[pd.DataFrame] = None
        self.d_labitems_df: Optional[pd.DataFrame] = None

        # Detected configuration
        self.icd_version: Optional[str] = None
        self.icd_code_column: Optional[str] = None

        # Lab item ID cache (BNP, troponin)
        self.item_ids_cache: Dict[str, List[int]] = {}

    # -------------------------
    # Helper / setup functions
    # -------------------------

    def _match_hadm_id(self, df: pd.DataFrame, hadm_id: int) -> pd.Series:
        """Helper to match HADM_ID handling both int and float types."""
        return df["HADM_ID"].astype(float) == float(hadm_id)

    def detect_icd_version(self) -> Optional[str]:
        """Detect whether ICD-9 or ICD-10 codes are used."""
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

    def _check_diagnosis_patterns(
        self,
        df: pd.DataFrame,
        code_col: str,
        codes: List[str],
        patterns: List[str],
        exact_match: bool = False,
    ) -> pd.DataFrame:
        """Filter diagnoses by code and/or LONG_TITLE patterns."""
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

    def _init_lab_item_ids(self):
        """Initialize lab item IDs (BNP, troponin) from D_LABITEMS."""
        if self.d_labitems_df is None:
            return

        def find_item_ids(terms: List[str]) -> List[int]:
            df = self.d_labitems_df
            label = df["LABEL"].astype(str).str.lower()
            itemids: List[int] = []
            for term in terms:
                m = df[label.str.contains(term.lower(), na=False, regex=False)]
                if not m.empty:
                    itemids.extend(m["ITEMID"].tolist())
            return list(set(itemids))

        # BNP
        self.item_ids_cache["bnp"] = find_item_ids(
            ["bnp", "brain natriuretic", "b-type natriuretic", "ntprobnp", "nt-probnp"]
        )
        if not self.item_ids_cache["bnp"]:
            self.item_ids_cache["bnp"] = [50910, 50963]

        # Troponin (for MI support)
        self.item_ids_cache["troponin"] = find_item_ids(
            ["troponin", "trop i", "trop t", "tni", "tnt"]
        )
        if not self.item_ids_cache["troponin"]:
            # Common MIMIC-III troponin item IDs (may vary slightly by extract)
            self.item_ids_cache["troponin"] = [51002, 51003]

    def _get_lab_series_for_admission(
        self, subject_id: int, hadm_id: int, lab_name: str
    ) -> Optional[pd.DataFrame]:
        """Return all lab rows for a given lab_name and admission (for trends)."""
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
        labs = labs.sort_values("CHARTTIME")
        return labs

    # -------------------------
    # Data loading
    # -------------------------

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
        """Load required MIMIC-III CSV files."""
        prefix = f"{self.data_dir}/Subject_ID_{subject_id}_"
        print("Loading MIMIC-III data files for procedure/event risk...")

        self.patients_df = pd.read_csv(f"{prefix}{patients_file}")
        self.admissions_df = pd.read_csv(f"{prefix}{admissions_file}")
        self.procedures_icd_df = pd.read_csv(f"{prefix}{procedures_icd_file}")
        self.diagnoses_icd_df = pd.read_csv(f"{prefix}{diagnoses_icd_file}")
        self.d_icd_diagnoses_df = pd.read_csv(f"{prefix}{d_icd_diagnoses_file}")
        self.d_icd_procedures_df = pd.read_csv(f"{prefix}{d_icd_procedures_file}")
        self.labevents_df = pd.read_csv(f"{prefix}{labevents_file}")
        self.d_labitems_df = pd.read_csv(f"{prefix}{d_labitems_file}")

        print("Data files loaded successfully.")

        self.icd_version = self.detect_icd_version() or "ICD9"
        print(f"Detected ICD version: {self.icd_version}")

        self._init_lab_item_ids()

    # -------------------------
    # Core extraction functions
    # -------------------------

    def _get_current_admission_time(self, hadm_id: int) -> Optional[datetime]:
        if self.admissions_df is None:
            return None
        adm = self.admissions_df[self.admissions_df["HADM_ID"] == hadm_id]
        if adm.empty:
            return None
        return pd.to_datetime(adm["ADMITTIME"].iloc[0])

    def _get_prior_admissions(self, subject_id: int, current_hadm_id: int) -> pd.DataFrame:
        """Return all admissions before current admission."""
        if self.admissions_df is None:
            return pd.DataFrame()
        current_time = self._get_current_admission_time(current_hadm_id)
        if current_time is None:
            return pd.DataFrame()
        prior = self.admissions_df[
            (self.admissions_df["SUBJECT_ID"] == subject_id)
            & (pd.to_datetime(self.admissions_df["ADMITTIME"]) < current_time)
        ].copy()
        return prior

    # ---- A. Prior related surgeries ----

    def _classify_procedure_type(self, text: str, code: str) -> Optional[str]:
        t = text.lower()

        # Cardiac surgeries
        cardiac_keywords = [
            "coronary artery bypass",
            "cabg",
            "bypass graft",
            "valve replacement",
            "valve repair",
            "valvuloplasty",
            "valvotomy",
            "percutaneous coronary",
            "angioplasty",
            "stent",
            "pacemaker",
            "defibrillator",
            "icd implantation",
        ]
        if any(k in t for k in cardiac_keywords):
            return "Cardiac"

        # Vascular surgeries
        vascular_keywords = [
            "aortic",
            "aorta",
            "carotid endarterectomy",
            "endarterectomy",
            "femoral-popliteal bypass",
            "peripheral bypass",
            "aneurysm repair",
        ]
        if any(k in t for k in vascular_keywords):
            return "vascular"

        # Thoracic surgeries
        thoracic_keywords = [
            "lobectomy",
            "pneumonectomy",
            "lung resection",
            "wedge resection",
            "thoracotomy",
            "esophagectomy",
            "mediastinal",
            "tracheostomy",
            "tracheotomy",
            "endotracheal",
            "airway",
        ]
        if any(k in t for k in thoracic_keywords):
            return "thoracic"

        return None

    def _get_prior_related_surgeries(
        self, subject_id: int, current_hadm_id: int
    ) -> Dict[str, Any]:
        """
        Get prior related surgeries (cardiac, vascular, thoracic, any within 90 days).
        """
        result: Dict[str, Any] = {
            "cardiac": [],
            "vascular": [],
            "thoracic": [],
            "recent_any": [],
            "risk_flag": "Low",
            "duration_inferred": False,  # no direct duration in MIMIC-III
        }

        if (
            self.procedures_icd_df is None
            or self.d_icd_procedures_df is None
            or self.admissions_df is None
        ):
            return result

        current_time = self._get_current_admission_time(current_hadm_id)
        if current_time is None:
            return result

        prior_adm = self._get_prior_admissions(subject_id, current_hadm_id)
        if prior_adm.empty:
            return result

        prior_hadm_ids = prior_adm["HADM_ID"].tolist()

        prior_procs = self.procedures_icd_df[
            (self.procedures_icd_df["SUBJECT_ID"] == subject_id)
            & (self.procedures_icd_df["HADM_ID"].isin(prior_hadm_ids))
        ].copy()
        if prior_procs.empty:
            return result

        # Merge with D_ICD_PROCEDURES
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

        if code_col not in prior_procs.columns:
            return result

        merged = prior_procs.merge(
            self.d_icd_procedures_df,
            left_on=code_col,
            right_on=merge_col,
            how="left",
        )

        merged["ADMITTIME"] = pd.to_datetime(
            merged["HADM_ID"].map(
                prior_adm.set_index("HADM_ID")["ADMITTIME"].to_dict()
            )
        )

        # Determine if within last 90 days
        ninety_days_ago = current_time - timedelta(days=90)

        for _, row in merged.iterrows():
            long_title = str(row.get("LONG_TITLE", "")).lower()
            code = str(row.get(code_col, ""))
            hadm = row.get("HADM_ID")
            adm_time = row.get("ADMITTIME")

            proc_type = self._classify_procedure_type(long_title, code)
            if proc_type:
                entry = {
                    "hadm_id": hadm,
                    "date": str(adm_time) if pd.notna(adm_time) else None,
                    "procedure_code": code,
                    "description": long_title if long_title else None,
                    "suspected_complications": [],  # complications inferred via prior_events
                }
                result[proc_type].append(entry)

            # Any surgery within last 90 days
            if pd.notna(adm_time) and adm_time >= ninety_days_ago:
                result["recent_any"].append(
                    {
                        "hadm_id": hadm,
                        "date": str(adm_time),
                        "procedure_code": code,
                        "description": long_title if long_title else None,
                    }
                )

        # Simple risk flag: High if any cardiac or vascular surgery, or any surgery <90d
        if result["cardiac"] or result["vascular"] or result["recent_any"]:
            result["risk_flag"] = "High"
        elif result["thoracic"]:
            result["risk_flag"] = "Medium"
        else:
            result["risk_flag"] = "Low"

        return result

    # ---- B. Prior adverse cardiac events ----

    def _get_prior_adverse_events(
        self, subject_id: int, current_hadm_id: int
    ) -> Dict[str, Any]:
        """
        Extract prior adverse cardiac events:
        MI, stroke/TIA, HF admissions, arrhythmia admissions, cardiogenic shock/arrest.
        """
        result: Dict[str, Any] = {
            "mi": [],
            "stroke_tia": [],
            "heart_failure": [],
            "arrhythmia": [],
            "shock_or_arrest": [],
            "risk_flag": "Low",
            "post_op_heuristic_used": False,
        }

        if (
            self.diagnoses_icd_df is None
            or self.d_icd_diagnoses_df is None
            or self.admissions_df is None
        ):
            return result

        current_time = self._get_current_admission_time(current_hadm_id)
        if current_time is None:
            return result

        prior_adm = self._get_prior_admissions(subject_id, current_hadm_id)
        if prior_adm.empty:
            return result

        prior_hadm_ids = prior_adm["HADM_ID"].tolist()

        diag = self.diagnoses_icd_df[
            (self.diagnoses_icd_df["SUBJECT_ID"] == subject_id)
            & (self.diagnoses_icd_df["HADM_ID"].isin(prior_hadm_ids))
        ].copy()
        if diag.empty:
            return result

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
            return result

        merged = diag.merge(
            self.d_icd_diagnoses_df,
            left_on=code_col,
            right_on=merge_col,
            how="left",
        )

        # Attach admission times
        adm_times = prior_adm.set_index("HADM_ID")["ADMITTIME"].to_dict()
        merged["ADMITTIME"] = pd.to_datetime(merged["HADM_ID"].map(adm_times))

        code_col_merged = code_col if code_col in merged.columns else merge_col

        # Map HADM_ID → whether that admission had any procedures (for post-op heuristic)
        hadm_with_procs = set(
            self.procedures_icd_df["HADM_ID"].unique()
        ) if self.procedures_icd_df is not None else set()

        def add_event(mask: pd.Series, bucket: str, severity: str, lab_support: Optional[Dict] = None):
            """
            Add events to a bucket. lab_support can be used to attach lab-based
            evidence (e.g., troponin or BNP).
            """
            nonlocal result
            used_heuristic = False
            for _, row in merged[mask].iterrows():
                hadm = row["HADM_ID"]
                adm_time = row["ADMITTIME"]
                # Heuristic: event is possibly post-op if same admission has any procedures
                possibly_post_op = hadm in hadm_with_procs
                if possibly_post_op:
                    used_heuristic = True

                entry = {
                    "hadm_id": hadm,
                    "date": str(adm_time) if pd.notna(adm_time) else None,
                    "code": str(row.get(code_col_merged, "")),
                    "description": str(row.get("LONG_TITLE", "")).lower()
                    if "LONG_TITLE" in row
                    else None,
                    "severity": severity,
                    # MIMIC-III lacks exact event timestamps at diagnosis level,
                    # so we can only provide a heuristic flag:
                    "post_operative_heuristic": possibly_post_op,
                    "post_operative_exact": False,
                }
                if lab_support:
                    entry.update(lab_support)
                result[bucket].append(entry)

            if used_heuristic:
                result["post_op_heuristic_used"] = True

        # MI
        if self.icd_version == "ICD9":
            mi_codes = [str(c) for c in range(410, 415)]
        else:
            mi_codes = ["I21", "I22"]
        mi_patterns = ["myocardial infarction", "acute mi", "stemi", "nstemi"]
        mi_df = self._check_diagnosis_patterns(
            merged, code_col_merged, mi_codes, mi_patterns
        )
        # For MI, try to capture a troponin peak if available
        if not mi_df.empty:
            for _, row in mi_df.iterrows():
                hadm = row["HADM_ID"]
                trop_series = self._get_lab_series_for_admission(subject_id, hadm, "troponin")
                trop_support = None
                if trop_series is not None and not trop_series["VALUENUM"].isna().all():
                    trop_support = {
                        "troponin_peak": float(trop_series["VALUENUM"].max()),
                        "troponin_min": float(trop_series["VALUENUM"].min()),
                    }
                # add_event expects a mask; we handle MI rows individually here
                add_event(
                    merged.index.to_series() == row.name,
                    "mi",
                    "major",
                    lab_support=trop_support,
                )

        # Stroke/TIA
        if self.icd_version == "ICD9":
            stroke_codes = [str(c) for c in range(430, 439)]
        else:
            stroke_codes = ["I63", "I64", "G45"]
        stroke_patterns = ["stroke", "cerebrovascular", "tia", "transient ischemic"]
        stroke_df = self._check_diagnosis_patterns(
            merged, code_col_merged, stroke_codes, stroke_patterns
        )
        if not stroke_df.empty:
            for _, row in stroke_df.iterrows():
                add_event(
                    merged.index.to_series() == row.name,
                    "stroke_tia",
                    "major",
                )

        # Heart failure
        if self.icd_version == "ICD9":
            hf_codes = ["428"]
        else:
            hf_codes = ["I50"]
        hf_patterns = ["heart failure", "cardiac failure", "congestive heart failure"]
        hf_df = self._check_diagnosis_patterns(
            merged, code_col_merged, hf_codes, hf_patterns
        )
        if not hf_df.empty:
            for _, row in hf_df.iterrows():
                hadm = row["HADM_ID"]
                # BNP trend during that admission (if available)
                bnp_series = self._get_lab_series_for_admission(subject_id, hadm, "bnp")
                bnp_trend = (
                    {
                        "bnp_min": float(bnp_series["VALUENUM"].min()),
                        "bnp_max": float(bnp_series["VALUENUM"].max()),
                    }
                    if bnp_series is not None and not bnp_series["VALUENUM"].isna().all()
                    else None
                )
                add_event(
                    merged.index.to_series() == row.name,
                    "heart_failure",
                    "major",
                    lab_support={"bnp_trend": bnp_trend} if bnp_trend else None,
                )

        # Arrhythmia requiring treatment (AFib, VT, etc.)
        if self.icd_version == "ICD9":
            arr_codes = ["427"]
        else:
            arr_codes = ["I47", "I48", "I49"]
        arr_patterns = [
            "atrial fibrillation",
            "afib",
            "atrial flutter",
            "ventricular tachycardia",
            "vt",
            "supraventricular tachycardia",
        ]
        arr_df = self._check_diagnosis_patterns(
            merged, code_col_merged, arr_codes, arr_patterns
        )
        if not arr_df.empty:
            for _, row in arr_df.iterrows():
                add_event(
                    merged.index.to_series() == row.name,
                    "arrhythmia",
                    "moderate",
                )

        # Cardiogenic shock or cardiac arrest
        if self.icd_version == "ICD9":
            shock_codes = ["78551", "78550"]
            arrest_codes = ["4275"]
        else:
            shock_codes = ["R57.0"]
            arrest_codes = ["I46"]
        shock_patterns = ["cardiogenic shock"]
        arrest_patterns = ["cardiac arrest"]

        shock_df = self._check_diagnosis_patterns(
            merged, code_col_merged, shock_codes, shock_patterns
        )
        arrest_df = self._check_diagnosis_patterns(
            merged, code_col_merged, arrest_codes, arrest_patterns
        )
        if not shock_df.empty:
            for _, row in shock_df.iterrows():
                add_event(
                    merged.index.to_series() == row.name,
                    "shock_or_arrest",
                    "critical",
                )
        if not arrest_df.empty:
            for _, row in arrest_df.iterrows():
                add_event(
                    merged.index.to_series() == row.name,
                    "shock_or_arrest",
                    "critical",
                )

        # Simple risk flag: High if any major/critical events, Medium if only arrhythmias
        if result["mi"] or result["stroke_tia"] or result["heart_failure"] or result["shock_or_arrest"]:
            result["risk_flag"] = "High"
        elif result["arrhythmia"]:
            result["risk_flag"] = "Medium"
        else:
            result["risk_flag"] = "Low"

        return result

    # ---- C. Current surgical cardiac risk context ----

    def _get_surgical_context(
        self, subject_id: int, hadm_id: int
    ) -> Dict[str, Any]:
        """
        Extract current surgical cardiac risk context.
        """
        context: Dict[str, Any] = {
            "planned_surgery_type": None,
            "expected_duration_minutes": None,
            "duration_inferred": False,
            "primary_procedure_name": None,
            "emergency_status": None,
            "hemodynamic_stress": None,
            "hemodynamic_stress_inferred": False,
            "positioning_risk": None,
            "risk_flag": "Low",
        }

        if (
            self.procedures_icd_df is None
            or self.d_icd_procedures_df is None
            or self.admissions_df is None
        ):
            return context

        # Procedures for this admission
        procs = self.procedures_icd_df[
            (self.procedures_icd_df["SUBJECT_ID"] == subject_id)
            & (self.procedures_icd_df["HADM_ID"] == hadm_id)
        ].copy()

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

        if not procs.empty and code_col in procs.columns:
            merged = procs.merge(
                self.d_icd_procedures_df,
                left_on=code_col,
                right_on=merge_col,
                how="left",
            )
            long_title_series = merged.get("LONG_TITLE", pd.Series([], dtype=str)).astype(str)
            long_titles = " ".join(long_title_series).lower()

            # Choose a primary procedure name for reporting (e.g., \"Replacement of tracheostomy tube\")
            primary_name = None
            if not long_title_series.empty:
                # Prefer any procedure containing 'trach' / 'tracheostomy'
                trach_matches = long_title_series[
                    long_title_series.str.contains("trach|tracheostomy", case=False, na=False)
                ]
                if not trach_matches.empty:
                    primary_name = trach_matches.iloc[0]
                else:
                    primary_name = long_title_series.iloc[0]
            context["primary_procedure_name"] = primary_name

            # Surgery type (cardiac, vascular, thoracic, abdominal, etc.)
            proc_type = self._classify_procedure_type(long_titles, "")
            if proc_type:
                context["planned_surgery_type"] = proc_type
            else:
                if any(k in long_titles for k in ["laparotomy", "colectomy", "gastrectomy"]):
                    context["planned_surgery_type"] = "abdominal"

            # Hemodynamic stress: infer from surgery category / keywords
            stress_keywords = [
                "aortic",
                "aneurysm",
                "major resection",
                "bypass",
                "thoracotomy",
                "lobectomy",
                "pneumonectomy",
            ]
            if any(k in long_titles for k in stress_keywords):
                context["hemodynamic_stress"] = "high"
                context["hemodynamic_stress_inferred"] = True
            elif any(k in long_titles for k in ["laparoscopic", "arthroscopy"]):
                context["hemodynamic_stress"] = "low"
                context["hemodynamic_stress_inferred"] = True
            else:
                context["hemodynamic_stress"] = "unknown"
                context["hemodynamic_stress_inferred"] = False

            # Positioning risks: usually not explicit in codes; infer from keywords
            if any(k in long_titles for k in ["spine", "laminectomy", "posterior fusion"]):
                # Likely prone
                context["positioning_risk"] = "prone"
            elif any(k in long_titles for k in ["robotic prostatectomy", "pelvic", "gynecologic"]):
                context["positioning_risk"] = "Trendelenburg"
            else:
                context["positioning_risk"] = "standard"

            # Duration: MIMIC-III does not store OR time; infer from surgery name/type (rule-based)
            # More specific keyword-based mapping overrides generic type-based mapping.
            duration_minutes = None
            if "trach" in long_titles or "tracheostomy" in long_titles:
                # Replacement of tracheostomy tube and similar airway procedures are typically shorter.
                duration_minutes = 60
            else:
                inferred_durations = {
                    "cardiac": 240,   # ~4h
                    "vascular": 210,  # ~3.5h
                    "thoracic": 180,  # ~3h
                    "abdominal": 150, # ~2.5h
                }
                if context["planned_surgery_type"] in inferred_durations:
                    duration_minutes = inferred_durations[context["planned_surgery_type"]]

            if duration_minutes is not None:
                context["expected_duration_minutes"] = duration_minutes
                context["duration_inferred"] = True

        # Emergency status and admission type
        adm = self.admissions_df[self.admissions_df["HADM_ID"] == hadm_id]
        if not adm.empty:
            adm_type = str(adm["ADMISSION_TYPE"].iloc[0]).lower()
            if "emergency" in adm_type:
                context["emergency_status"] = "emergent"
            elif "urgent" in adm_type:
                context["emergency_status"] = "urgent"
            else:
                context["emergency_status"] = "elective"

        # Simple risk flag logic based on inferred surgical stressors
        risk_score = 0
        if context["planned_surgery_type"] in ["cardiac", "vascular", "thoracic", "abdominal"]:
            risk_score += 2
        if context["emergency_status"] in ["emergent", "urgent"]:
            risk_score += 2
        if context["hemodynamic_stress"] == "high":
            risk_score += 2

        if risk_score >= 4:
            context["risk_flag"] = "High"
        elif risk_score >= 2:
            context["risk_flag"] = "Medium"
        else:
            context["risk_flag"] = "Low"

        return context

    # -------------------------
    # Public API
    # -------------------------

    def build_procedure_event_risk_report(
        self, subject_id: int, hadm_id: int
    ) -> Dict[str, Any]:
        """
        Build full structured report:
        - prior_surgeries
        - prior_events
        - surgical_context
        """
        prior_surgeries = self._get_prior_related_surgeries(subject_id, hadm_id)
        prior_events = self._get_prior_adverse_events(subject_id, hadm_id)
        surgical_context = self._get_surgical_context(subject_id, hadm_id)

        return {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "prior_surgeries": prior_surgeries,
            "prior_events": prior_events,
            "surgical_context": surgical_context,
        }


if __name__ == "__main__":
    # Simple manual test harness
    extractor = ProcedureEventRiskExtractor(data_dir="./data")
    extractor.load_data(subject_id="249")

    subject_id = 249
    hadm_id = 116935

    report = extractor.build_procedure_event_risk_report(subject_id, hadm_id)
    print(report)


