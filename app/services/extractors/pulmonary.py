"""
Pulmonary Risk Extractor for MIMIC-III Database

This module extracts pulmonary risk factors from MIMIC-III CSV files
for ARISCAT (Assess Respiratory Risk in Surgical Patients in Catalonia) calculation.

ARISCAT Risk Factors:
- Age
- Preoperative SpO₂
- Respiratory infection in last month
- Preoperative anemia (Hgb)
- Surgical incision site
- Duration of surgery
- Emergency surgery
- COPD/Asthma/OSA diagnosis
- Smoking status
- Prior respiratory complications
- Lab markers (WBC, CRP, Albumin, Lactate, BNP)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class PulmonaryRiskExtractor:
    """
    Extracts pulmonary risk factors from MIMIC-III data for ARISCAT calculation.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the extractor with path to MIMIC-III CSV files.
        
        Args:
            data_dir: Directory containing MIMIC-III CSV files (default: "./data")
        """
        self.data_dir = data_dir
        self.patients_df = None
        self.chartevents_df = None
        self.labevents_df = None
        self.diagnoses_icd_df = None
        self.noteevents_df = None
        self.admissions_df = None
        self.procedures_icd_df = None
        self.d_icd_diagnoses_df = None
        self.d_icd_procedures_df = None
        self.d_labitems_df = None
        self.d_items_df = None
        
        # Detected configurations
        self.icd_version = None  # 'ICD9' or 'ICD10'
        self.item_ids_cache = {}  # Cache for detected item IDs
        self.icd_code_column = None  # Column name for ICD codes

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _match_hadm_id(self, df: pd.DataFrame, hadm_id: int) -> pd.Series:
        """
        Helper function to match HADM_ID handling both int and float types.
        """
        return df['HADM_ID'].astype(float) == float(hadm_id)

    def _get_lab_series_by_time_window(
        self,
        subject_id: int,
        lab_name: str,
        anchor_time: pd.Timestamp,
        hours_before: int,
    ) -> Optional[pd.DataFrame]:
        """
        Return all lab rows for a given lab semantic name for the subject
        within a time window [anchor_time - hours_before, anchor_time),
        regardless of HADM_ID.

        This is used for temporal pre-op patterns (e.g. lactate 3h trend,
        CRP 48h slope, albumin 7d slope).
        """
        if self.labevents_df is None or anchor_time is None:
            return None

        itemids = self.item_ids_cache.get(lab_name, [])
        if not itemids:
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
        
    def _match_hadm_id(self, df: pd.DataFrame, hadm_id: int) -> pd.Series:
        """
        Helper function to match HADM_ID handling both int and float types.
        
        Args:
            df: DataFrame with HADM_ID column
            hadm_id: Hospital admission ID to match
            
        Returns:
            Boolean series for filtering
        """
        # Handle both int and float HADM_ID values
        return df['HADM_ID'].astype(float) == float(hadm_id)
    
    def _check_diagnosis_patterns(self, df: pd.DataFrame, code_col: str, 
                                  codes: List[str], patterns: List[str],
                                  exact_match: bool = False) -> pd.DataFrame:
        """
        Helper function to check diagnosis patterns by code or text description.
        
        Args:
            df: Merged diagnosis dataframe
            code_col: Column name for ICD codes
            codes: List of ICD codes to match
            patterns: List of text patterns to match in LONG_TITLE
            exact_match: If True, use exact match for codes; if False, use startswith
            
        Returns:
            Filtered dataframe with matching diagnoses
        """
        if df.empty:
            return df
        
        # Check by code
        if exact_match:
            code_match = df[code_col].astype(str).isin(codes)
        else:
            code_match = df[code_col].astype(str).str.startswith(tuple(codes))
        
        # Check by text pattern if LONG_TITLE exists
        if 'LONG_TITLE' in df.columns:
            pattern_match = df['LONG_TITLE'].astype(str).str.lower().str.contains(
                '|'.join(patterns), na=False, regex=False
            )
            return df[code_match | pattern_match]
        else:
            return df[code_match]
    
    def detect_icd_version(self) -> str:
        """
        Detect whether the database uses ICD-9 or ICD-10 codes.
        
        Returns:
            'ICD9' or 'ICD10'
        """
        if self.diagnoses_icd_df is None or self.d_icd_diagnoses_df is None:
            return None
        
        # Check column names to determine ICD version
        if 'ICD9_CODE' in self.diagnoses_icd_df.columns:
            self.icd_code_column = 'ICD9_CODE'
            return 'ICD9'
        elif 'ICD10_CODE' in self.diagnoses_icd_df.columns:
            self.icd_code_column = 'ICD10_CODE'
            return 'ICD10'
        elif 'ICD_CODE' in self.diagnoses_icd_df.columns:
            # Need to inspect actual codes to determine version
            sample_codes = self.diagnoses_icd_df['ICD_CODE'].dropna().head(100).astype(str)
            # ICD-9 codes are numeric (3-5 digits), ICD-10 codes start with letter
            icd10_count = sample_codes.str.match(r'^[A-Z]').sum()
            icd9_count = sample_codes.str.match(r'^\d').sum()
            
            if icd10_count > icd9_count:
                self.icd_code_column = 'ICD_CODE'
                return 'ICD10'
            else:
                self.icd_code_column = 'ICD_CODE'
                return 'ICD9'
        
        return None
    
    def find_item_ids_by_label(self, search_terms: List[str], 
                               search_in: str = 'chartevents') -> List[int]:
        """
        Dynamically find item IDs by searching D_ITEMS or D_LABITEMS for matching labels.
        
        Args:
            search_terms: List of search terms (e.g., ['spo2', 'oxygen saturation'])
            search_in: 'chartevents' or 'labevents'
            
        Returns:
            List of matching item IDs
        """
        if search_in == 'chartevents':
            if self.d_items_df is None:
                return []
            df = self.d_items_df
            label_column = 'LABEL' if 'LABEL' in df.columns else 'DBSOURCE'
        else:  # labevents
            if self.d_labitems_df is None:
                return []
            df = self.d_labitems_df
            label_column = 'LABEL' if 'LABEL' in df.columns else 'FLUID'
        
        item_ids = []
        
        # Search in label column
        if label_column in df.columns:
            df_lower = df[label_column].astype(str).str.lower()
            for term in search_terms:
                term_lower = term.lower()
                matches = df[df_lower.str.contains(term_lower, na=False, regex=False)]
                if not matches.empty:
                    item_ids.extend(matches['ITEMID'].tolist())
        
        # Also search in other text columns if available
        text_columns = [col for col in df.columns if col in ['DBSOURCE', 'LINKSTO', 'FLUID', 'CATEGORY']]
        for col in text_columns:
            df_lower = df[col].astype(str).str.lower()
            for term in search_terms:
                term_lower = term.lower()
                matches = df[df_lower.str.contains(term_lower, na=False, regex=False)]
                if not matches.empty:
                    item_ids.extend(matches['ITEMID'].tolist())
        
        # Remove duplicates and return
        return list(set(item_ids))
    
    def initialize_item_ids(self):
        """
        Initialize and cache item IDs by searching metadata tables.
        This should be called after load_data().
        """
        print("Detecting item IDs from metadata tables...")
        
        # SpO2 item IDs
        spo2_terms = ['spo2', 'oxygen saturation', 'o2 sat', 'pulse oximetry']
        self.item_ids_cache['spo2'] = self.find_item_ids_by_label(spo2_terms, 'chartevents')
        if not self.item_ids_cache['spo2']:
            # Fallback to common IDs
            self.item_ids_cache['spo2'] = [646, 220277]
        print(f"  SpO2 item IDs: {self.item_ids_cache['spo2']}")
        
        # Hemoglobin item IDs
        hgb_terms = ['hemoglobin', 'hgb', 'hgb (calc)', 'hemoglobin, whole blood']
        self.item_ids_cache['hgb'] = self.find_item_ids_by_label(hgb_terms, 'labevents')
        if not self.item_ids_cache['hgb']:
            # Fallback to common IDs
            self.item_ids_cache['hgb'] = [51221, 50811]
        print(f"  Hgb item IDs: {self.item_ids_cache['hgb']}")
        
        # WBC item IDs
        wbc_terms = ['white blood cell', 'wbc', 'leukocyte', 'white cell count']
        self.item_ids_cache['wbc'] = self.find_item_ids_by_label(wbc_terms, 'labevents')
        if not self.item_ids_cache['wbc']:
            self.item_ids_cache['wbc'] = [51300, 51301]
        print(f"  WBC item IDs: {self.item_ids_cache['wbc']}")
        
        # CRP item IDs
        crp_terms = ['c-reactive protein', 'crp', 'c reactive protein']
        self.item_ids_cache['crp'] = self.find_item_ids_by_label(crp_terms, 'labevents')
        if not self.item_ids_cache['crp']:
            self.item_ids_cache['crp'] = [50889]
        print(f"  CRP item IDs: {self.item_ids_cache['crp']}")
        
        # Albumin item IDs
        albumin_terms = ['albumin', 'alb']
        self.item_ids_cache['albumin'] = self.find_item_ids_by_label(albumin_terms, 'labevents')
        if not self.item_ids_cache['albumin']:
            self.item_ids_cache['albumin'] = [50862]
        print(f"  Albumin item IDs: {self.item_ids_cache['albumin']}")
        
        # Lactate item IDs
        lactate_terms = ['lactate', 'lactic acid', 'lact']
        self.item_ids_cache['lactate'] = self.find_item_ids_by_label(lactate_terms, 'labevents')
        if not self.item_ids_cache['lactate']:
            self.item_ids_cache['lactate'] = [50813]
        print(f"  Lactate item IDs: {self.item_ids_cache['lactate']}")
        
        # BNP item IDs (include both BNP and NTproBNP)
        bnp_terms = ['bnp', 'b-type natriuretic peptide', 'brain natriuretic peptide', 'nt-probnp', 'ntprobnp']
        self.item_ids_cache['bnp'] = self.find_item_ids_by_label(bnp_terms, 'labevents')
        if not self.item_ids_cache['bnp']:
            # Fallback to common BNP item IDs (50910 for BNP, 50963 for NTproBNP)
            self.item_ids_cache['bnp'] = [50910, 50963]
        print(f"  BNP item IDs: {self.item_ids_cache['bnp']}")
        
        # Smoking status item IDs
        smoking_terms = ['smoking', 'tobacco', 'smoker', 'cigarette']
        self.item_ids_cache['smoking'] = self.find_item_ids_by_label(smoking_terms, 'chartevents')
        if not self.item_ids_cache['smoking']:
            self.item_ids_cache['smoking'] = [229792, 229793, 229794]
        print(f"  Smoking item IDs: {self.item_ids_cache['smoking']}")
        
    def load_data(self, subject_id : str = "249",
                  patients_file: str = "PATIENTS.csv",
                  chartevents_file: str = "CHARTEVENTS.csv",
                  labevents_file: str = "LABEVENTS.csv",
                  diagnoses_icd_file: str = "DIAGNOSES_ICD.csv",
                  noteevents_file: str = "NOTES.csv",
                  admissions_file: str = "ADMISSIONS.csv",
                  procedures_icd_file: str = "PROCEDURES_ICD.csv",
                  d_icd_diagnoses_file: str = "D_ICD_DIAGNOSES.csv",
                  d_icd_procedures_file: str = "D_ICD_PROCEDURES.csv",
                  d_labitems_file: str = "D_LABITEMS.csv",
                  d_items_file: str = "D_ITEMS.csv"):
        """
        Load MIMIC-III CSV files into memory.
        
        Args:
            All parameters are filenames for respective MIMIC-III tables
        """
        try:
            print("Loading MIMIC-III data files...")
            # Construct file paths: ./data/Subject_ID_<subject_id>_<filename>
            file_prefix = f"{self.data_dir}/Subject_ID_{subject_id}_"

            self.patients_df = pd.read_csv(f"{file_prefix}{patients_file}")
            self.chartevents_df = pd.read_csv(f"{file_prefix}{chartevents_file}")
            self.labevents_df = pd.read_csv(f"{file_prefix}{labevents_file}")
            self.diagnoses_icd_df = pd.read_csv(f"{file_prefix}{diagnoses_icd_file}")
            
            # NOTEEVENTS is optional - may not exist in all datasets
            try:
                self.noteevents_df = pd.read_csv(f"{file_prefix}{noteevents_file}")
                print(f"  Loaded {noteevents_file}")
            except FileNotFoundError:
                print(f"  Warning: {noteevents_file} not found. Skipping noteevents.")
                self.noteevents_df = None
            
            self.admissions_df = pd.read_csv(f"{file_prefix}{admissions_file}")
            self.procedures_icd_df = pd.read_csv(f"{file_prefix}{procedures_icd_file}")
            self.d_icd_diagnoses_df = pd.read_csv(f"{file_prefix}{d_icd_diagnoses_file}")
            self.d_icd_procedures_df = pd.read_csv(f"{file_prefix}{d_icd_procedures_file}")
            self.d_labitems_df = pd.read_csv(f"{file_prefix}{d_labitems_file}")
            self.d_items_df = pd.read_csv(f"{file_prefix}{d_items_file}")
            print("Data files loaded successfully.")
            
            # Detect ICD version
            self.icd_version = self.detect_icd_version()
            if self.icd_version:
                print(f"Detected ICD version: {self.icd_version}")
            else:
                print("Warning: Could not detect ICD version. Defaulting to ICD-9.")
                self.icd_version = 'ICD9'
                self.icd_code_column = 'ICD9_CODE'
            
            # Initialize item IDs
            self.initialize_item_ids()
            
        except FileNotFoundError as e:
            print(f"Error loading data files: {e}")
            raise
    
    def get_patient_age_at_surgery(self, subject_id: int, hadm_id: int) -> Optional[int]:
        """
        Calculate patient age at time of surgery.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            
        Returns:
            Age in years, or None if not available
        """
        if self.patients_df is None or self.admissions_df is None:
            return None
        
        # Get patient DOB
        patient = self.patients_df[self.patients_df['SUBJECT_ID'] == subject_id]
        if patient.empty:
            return None
        
        dob = pd.to_datetime(patient['DOB'].iloc[0])
        
        # Get admission date
        admission = self.admissions_df[self.admissions_df['HADM_ID'] == hadm_id]
        if admission.empty:
            return None
        
        admittime = pd.to_datetime(admission['ADMITTIME'].iloc[0])
        
        # Calculate age
        age = (admittime - dob).days / 365.25
        
        # MIMIC-III de-identifies ages >89, so cap at 90
        return min(int(age), 90) if age >= 0 else None
    
    def get_last_preop_spo2(self, subject_id: int, hadm_id: int, 
                            surgery_time: pd.Timestamp) -> Optional[float]:
        """
        Get last pre-operative SpO₂ from chartevents.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            surgery_time: Timestamp of surgery
            
        Returns:
            Last SpO₂ value before surgery, or None if not available
        """
        if self.chartevents_df is None or self.d_items_df is None:
            return None
        
        # Use detected SpO2 item IDs
        spo2_itemids = self.item_ids_cache.get('spo2', [646, 220277])
        
        # Filter chartevents for this patient and admission
        chart_events = self.chartevents_df[
            (self.chartevents_df['SUBJECT_ID'] == subject_id) &
            (self._match_hadm_id(self.chartevents_df, hadm_id)) &
            (self.chartevents_df['ITEMID'].isin(spo2_itemids))
        ].copy()
        
        if chart_events.empty:
            return None
        
        # Convert CHARTTIME to datetime
        chart_events['CHARTTIME'] = pd.to_datetime(chart_events['CHARTTIME'])
        
        # Filter for pre-operative values
        preop_events = chart_events[chart_events['CHARTTIME'] < surgery_time]
        
        if preop_events.empty:
            return None
        
        # Get the most recent value
        preop_events = preop_events.sort_values('CHARTTIME', ascending=False)
        last_spo2 = preop_events['VALUENUM'].iloc[0]
        
        return float(last_spo2) if pd.notna(last_spo2) else None
    
    def get_preop_hgb(self, subject_id: int, hadm_id: int, 
                     surgery_time: pd.Timestamp, hours_before: int = 48) -> Optional[float]:
        """
        Get hemoglobin within specified hours before surgery.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            surgery_time: Timestamp of surgery
            hours_before: Hours before surgery to look for Hgb (default 48)
            
        Returns:
            Hgb value in g/dL, or None if not available
        """
        if self.labevents_df is None or self.d_labitems_df is None:
            return None
        
        # Use detected Hgb item IDs and a pure time window (SUBJECT_ID + CHARTTIME);
        # do not constrain by HADM_ID so that values from nearby admissions can
        # still be used if they fall in the pre-op window.
        hgb_itemids = self.item_ids_cache.get('hgb', [51221, 50811])

        labs = self.labevents_df[
            (self.labevents_df['SUBJECT_ID'] == subject_id) &
            (self.labevents_df['ITEMID'].isin(hgb_itemids))
        ].copy()
        if labs.empty:
            return None

        labs['CHARTTIME'] = pd.to_datetime(labs['CHARTTIME'])
        time_threshold = surgery_time - timedelta(hours=hours_before)
        preop_labs = labs[
            (labs['CHARTTIME'] >= time_threshold) &
            (labs['CHARTTIME'] < surgery_time)
        ]
        if preop_labs.empty:
            return None

        preop_labs = preop_labs.sort_values('CHARTTIME', ascending=False)
        hgb_value = preop_labs['VALUENUM'].iloc[0]
        return float(hgb_value) if pd.notna(hgb_value) else None

    def analyze_pulmonary_temporal_patterns(
        self,
        subject_id: int,
        surgery_time: pd.Timestamp,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Analyze temporal pulmonary lab patterns around surgery:

        - Lactate 3h rolling trend:
            LacRise_3h = lactate(t0) - lactate(t-3h)
            Window: 3 hours pre-op
            Threshold: > 0.5 mmol/L increase → rising_lactate flag

        - CRP 48h slope:
            Window: 48 hours pre-op
            slope_48h = (CRP_last - CRP_first) / delta_hours
            increasing_crp flag if slope_48h > 0

        - Albumin 7d slope:
            Window: 7 days (168h) pre-op
            slope_7d = (Alb_last - Alb_first) / delta_hours
            falling_albumin flag if slope_7d < 0
        """
        patterns: Dict[str, Dict[str, Optional[float]]] = {
            "lactate": {
                "lac_rise_3h": None,
                "rising_lactate": False,
            },
            "crp": {
                "slope_48h": None,
                "increasing_crp": False,
            },
            "albumin": {
                "slope_7d": None,
                "falling_albumin": False,
            },
        }

        # --- Lactate 3h rolling trend ---
        lactate_series = self._get_lab_series_by_time_window(
            subject_id=subject_id,
            lab_name="lactate",
            anchor_time=surgery_time,
            hours_before=3,
        )
        if lactate_series is not None and not lactate_series.empty:
            vals = lactate_series["VALUENUM"].dropna().astype(float)
            if not vals.empty:
                first_val = float(vals.iloc[0])
                last_val = float(vals.iloc[-1])
                lac_rise = last_val - first_val
                patterns["lactate"]["lac_rise_3h"] = lac_rise
                if lac_rise > 0.5:
                    patterns["lactate"]["rising_lactate"] = True

        # --- CRP 48h slope ---
        crp_series = self._get_lab_series_by_time_window(
            subject_id=subject_id,
            lab_name="crp",
            anchor_time=surgery_time,
            hours_before=48,
        )
        if crp_series is not None and not crp_series.empty:
            crp_vals = crp_series["VALUENUM"].dropna().astype(float)
            times = crp_series["CHARTTIME"]
            if len(crp_vals) >= 2:
                first_val = float(crp_vals.iloc[0])
                last_val = float(crp_vals.iloc[-1])
                delta_val = last_val - first_val
                delta_hours = max(
                    (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600.0, 0.1
                )
                slope_48h = delta_val / delta_hours
                patterns["crp"]["slope_48h"] = slope_48h
                if slope_48h > 0:
                    patterns["crp"]["increasing_crp"] = True

        # --- Albumin 7d slope ---
        albumin_series = self._get_lab_series_by_time_window(
            subject_id=subject_id,
            lab_name="albumin",
            anchor_time=surgery_time,
            hours_before=7 * 24,
        )
        if albumin_series is not None and not albumin_series.empty:
            alb_vals = albumin_series["VALUENUM"].dropna().astype(float)
            times = albumin_series["CHARTTIME"]
            if len(alb_vals) >= 2:
                first_val = float(alb_vals.iloc[0])
                last_val = float(alb_vals.iloc[-1])
                delta_val = last_val - first_val
                delta_hours = max(
                    (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600.0, 0.1
                )
                slope_7d = delta_val / delta_hours
                patterns["albumin"]["slope_7d"] = slope_7d
                if slope_7d < 0:
                    patterns["albumin"]["falling_albumin"] = True

        return patterns
    
    def get_respiratory_diagnoses(self, subject_id: int, hadm_id: int) -> Dict[str, bool]:
        """
        Check for COPD, asthma, and OSA diagnoses.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            
        Returns:
            Dictionary with diagnosis flags
        """
        result = {
            'copd': False,
            'asthma': False,
            'osa': False
        }
        
        if self.diagnoses_icd_df is None or self.d_icd_diagnoses_df is None:
            return result
        
        # Get diagnoses for this admission
        diagnoses = self.diagnoses_icd_df[
            (self.diagnoses_icd_df['SUBJECT_ID'] == subject_id) &
            (self.diagnoses_icd_df['HADM_ID'] == hadm_id)
        ]
        
        if diagnoses.empty:
            return result
        
        # Determine merge column based on ICD version
        if self.icd_version == 'ICD9':
            merge_col = 'ICD9_CODE'
            code_col = 'ICD9_CODE'
        else:
            merge_col = 'ICD10_CODE' if 'ICD10_CODE' in self.d_icd_diagnoses_df.columns else 'ICD_CODE'
            code_col = self.icd_code_column
        
        # Merge with diagnosis descriptions
        diagnoses_merged = diagnoses.merge(
            self.d_icd_diagnoses_df,
            left_on=code_col,
            right_on=merge_col,
            how='left'
        )
        
        # Get code column name in merged dataframe
        merged_code_col = code_col if code_col in diagnoses_merged.columns else merge_col
        
        # Check for COPD
        copd_patterns = ['copd', 'chronic obstructive', 'emphysema', 'chronic bronchitis']
        if self.icd_version == 'ICD9':
            copd_codes = [str(code) for code in range(490, 497)]
        else:  # ICD-10
            copd_codes = ['J44', 'J43', 'J41', 'J42']
        copd_diag = self._check_diagnosis_patterns(diagnoses_merged, merged_code_col, copd_codes, copd_patterns)
        result['copd'] = not copd_diag.empty
        
        # Check for Asthma
        asthma_patterns = ['asthma']
        if self.icd_version == 'ICD9':
            asthma_codes = ['493']
        else:  # ICD-10
            asthma_codes = ['J45', 'J46']
        asthma_diag = self._check_diagnosis_patterns(diagnoses_merged, merged_code_col, asthma_codes, asthma_patterns)
        result['asthma'] = not asthma_diag.empty
        
        # Check for OSA
        osa_patterns = ['obstructive sleep apnea', 'osa', 'sleep apnea']
        if self.icd_version == 'ICD9':
            # ICD-9: specific OSA-related codes
            osa_codes = ['32723', '78051', '78053', '78057']
            osa_diag = self._check_diagnosis_patterns(
                diagnoses_merged, merged_code_col, osa_codes, osa_patterns, exact_match=True
            )
        else:  # ICD-10
            # ICD-10: G47.33 Obstructive sleep apnea
            osa_codes = ['G47.33', 'G4733']
            osa_diag = self._check_diagnosis_patterns(
                diagnoses_merged, merged_code_col, osa_codes, osa_patterns, exact_match=False
            )
        result['osa'] = not osa_diag.empty
        
        return result
    
    def get_smoking_status(self, subject_id: int, hadm_id: int) -> Optional[str]:
        """
        Extract smoking status from chartevents or noteevents.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            
        Returns:
            Smoking status: 'current', 'former', 'never', or None
        """
        if self.chartevents_df is None and self.noteevents_df is None:
            return None
        
        smoking_status = None
        
        # Check chartevents for smoking status
        if self.chartevents_df is not None and self.d_items_df is not None:
            # Use detected smoking item IDs
            smoking_itemids = self.item_ids_cache.get('smoking', [229792, 229793, 229794])
            
            chart_smoking = self.chartevents_df[
                (self.chartevents_df['SUBJECT_ID'] == subject_id) &
                (self._match_hadm_id(self.chartevents_df, hadm_id)) &
                (self.chartevents_df['ITEMID'].isin(smoking_itemids))
            ]
            
            if not chart_smoking.empty:
                # Extract smoking status from VALUE
                value = str(chart_smoking['VALUE'].iloc[0]).lower()
                if 'current' in value or 'smoker' in value:
                    smoking_status = 'current'
                elif 'former' in value or 'ex-smoker' in value:
                    smoking_status = 'former'
                elif 'never' in value or 'non-smoker' in value:
                    smoking_status = 'never'
        
        # Check noteevents if not found in chartevents
        if smoking_status is None and self.noteevents_df is not None:
            notes = self.noteevents_df[
                (self.noteevents_df['SUBJECT_ID'] == subject_id) &
                (self._match_hadm_id(self.noteevents_df, hadm_id))
            ]
            
            if not notes.empty:
                # Search in note text
                note_text = ' '.join(notes['TEXT'].astype(str)).lower()
                
                smoking_patterns = {
                    'current': ['current smoker', 'smoking', 'smokes', 'tobacco use'],
                    'former': ['former smoker', 'ex-smoker', 'quit smoking', 'past smoker'],
                    'never': ['never smoked', 'non-smoker', 'no smoking history']
                }
                
                for status, patterns in smoking_patterns.items():
                    if any(pattern in note_text for pattern in patterns):
                        smoking_status = status
                        break
        
        return smoking_status
    
    def get_prior_respiratory_admissions(self, subject_id: int, 
                                        current_hadm_id: int) -> Dict[str, bool]:
        """
        Check for prior admissions for pneumonia or respiratory failure.
        
        Args:
            subject_id: Patient subject ID
            current_hadm_id: Current hospital admission ID
            
        Returns:
            Dictionary with flags for prior pneumonia/respiratory failure
        """
        result = {
            'prior_pneumonia': False,
            'prior_respiratory_failure': False
        }
        
        if self.admissions_df is None or self.diagnoses_icd_df is None:
            return result
        
        # Get current admission date
        current_adm = self.admissions_df[self.admissions_df['HADM_ID'] == current_hadm_id]
        if current_adm.empty:
            return result
        
        current_admittime = pd.to_datetime(current_adm['ADMITTIME'].iloc[0])
        
        # Get all prior admissions for this patient
        prior_admissions = self.admissions_df[
            (self.admissions_df['SUBJECT_ID'] == subject_id) &
            (pd.to_datetime(self.admissions_df['ADMITTIME']) < current_admittime)
        ]
        
        if prior_admissions.empty:
            return result
        
        prior_hadm_ids = prior_admissions['HADM_ID'].tolist()
        
        # Check diagnoses in prior admissions
        prior_diagnoses = self.diagnoses_icd_df[
            (self.diagnoses_icd_df['SUBJECT_ID'] == subject_id) &
            (self.diagnoses_icd_df['HADM_ID'].isin(prior_hadm_ids))
        ]
        
        if prior_diagnoses.empty:
            return result
        
        # Merge with diagnosis descriptions
        if self.d_icd_diagnoses_df is not None:
            # Determine merge column based on ICD version
            if self.icd_version == 'ICD9':
                merge_col = 'ICD9_CODE'
                code_col = 'ICD9_CODE'
            else:
                merge_col = 'ICD10_CODE' if 'ICD10_CODE' in self.d_icd_diagnoses_df.columns else 'ICD_CODE'
                code_col = self.icd_code_column
            
            prior_diagnoses_merged = prior_diagnoses.merge(
                self.d_icd_diagnoses_df,
                left_on=code_col,
                right_on=merge_col,
                how='left'
            )
            
            # Get code column name in merged dataframe
            merged_code_col = code_col if code_col in prior_diagnoses_merged.columns else merge_col
            
            # Check for pneumonia
            pneumonia_patterns = ['pneumonia']
            if self.icd_version == 'ICD9':
                pneumonia_codes = [str(code) for code in range(480, 487)]
            else:  # ICD-10
                pneumonia_codes = ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18']
            pneumonia_diag = self._check_diagnosis_patterns(
                prior_diagnoses_merged, merged_code_col, pneumonia_codes, pneumonia_patterns
            )
            result['prior_pneumonia'] = not pneumonia_diag.empty
            
            # Check for respiratory failure
            resp_failure_patterns = ['respiratory failure']
            if self.icd_version == 'ICD9':
                resp_failure_codes = ['51881', '51882', '51884']
                resp_failure_diag = prior_diagnoses_merged[
                    prior_diagnoses_merged[merged_code_col].astype(str).isin(resp_failure_codes)
                ]
                if 'LONG_TITLE' in prior_diagnoses_merged.columns:
                    resp_pattern_match = prior_diagnoses_merged['LONG_TITLE'].astype(str).str.lower().str.contains(
                        '|'.join(resp_failure_patterns), na=False, regex=False
                    )
                    resp_failure_diag = prior_diagnoses_merged[resp_failure_diag.index | resp_pattern_match]
            else:  # ICD-10
                resp_failure_codes = ['J96']
                resp_failure_diag = self._check_diagnosis_patterns(
                    prior_diagnoses_merged, merged_code_col, resp_failure_codes, resp_failure_patterns
                )
            result['prior_respiratory_failure'] = not resp_failure_diag.empty
        
        return result
    
    def get_surgery_details(self, subject_id: int, hadm_id: int) -> Dict[str, any]:
        """
        Extract surgery details: incision site, duration, emergency status.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            
        Returns:
            Dictionary with surgery details
        """
        result = {
            'incision_site': None,
            'surgery_duration_minutes': None,
            'emergency_status': None,
            'surgery_time': None
        }
        
        if self.procedures_icd_df is None or self.admissions_df is None:
            return result
        
        # Get procedures for this admission
        procedures = self.procedures_icd_df[
            (self.procedures_icd_df['SUBJECT_ID'] == subject_id) &
            (self.procedures_icd_df['HADM_ID'] == hadm_id)
        ]
        
        if procedures.empty:
            return result
        
        # Merge with procedure descriptions
        if self.d_icd_procedures_df is not None:
            # Determine merge column based on ICD version
            if self.icd_version == 'ICD9':
                merge_col = 'ICD9_CODE'
                # Find the code column in procedures dataframe
                if 'ICD9_CODE' in procedures.columns:
                    code_col = 'ICD9_CODE'
                else:
                    code_col = 'ICD_CODE' if 'ICD_CODE' in procedures.columns else None
            else:
                merge_col = 'ICD10_CODE' if 'ICD10_CODE' in self.d_icd_procedures_df.columns else 'ICD_CODE'
                # Find the code column in procedures dataframe
                if 'ICD10_CODE' in procedures.columns:
                    code_col = 'ICD10_CODE'
                elif 'ICD_CODE' in procedures.columns:
                    code_col = 'ICD_CODE'
                else:
                    code_col = None
            
            if code_col:
                procedures_merged = procedures.merge(
                    self.d_icd_procedures_df,
                    left_on=code_col,
                    right_on=merge_col,
                    how='left'
                )
            else:
                # Fallback: try to merge on any matching column
                procedures_merged = procedures.copy()
                if 'LONG_TITLE' not in procedures_merged.columns:
                    procedures_merged['LONG_TITLE'] = ''
            
            # Determine incision site based on procedure codes
            # Major surgical sites for ARISCAT
            incision_sites = {
                'upper_abdominal': ['upper abdominal', 'laparotomy', 'cholecystectomy', 'gastrectomy'],
                'lower_abdominal': ['lower abdominal', 'appendectomy', 'colectomy', 'hysterectomy'],
                'thoracic': ['thoracic', 'thoracotomy', 'lung', 'cardiac', 'esophagectomy'],
                'head_neck': ['head', 'neck', 'thyroid', 'laryngectomy'],
                'peripheral': ['peripheral', 'extremity', 'orthopedic']
            }
            
            # Get procedure text from LONG_TITLE if available, otherwise use code
            if 'LONG_TITLE' in procedures_merged.columns:
                procedure_text = ' '.join(procedures_merged['LONG_TITLE'].astype(str)).lower()
            elif code_col and code_col in procedures_merged.columns:
                procedure_text = ' '.join(procedures_merged[code_col].astype(str)).lower()
            else:
                procedure_text = ''
            
            for site, keywords in incision_sites.items():
                if any(keyword in procedure_text for keyword in keywords):
                    result['incision_site'] = site
                    break
        
        # Get admission details for emergency status
        admission = self.admissions_df[self.admissions_df['HADM_ID'] == hadm_id]
        if not admission.empty:
            # Emergency status from ADMISSION_TYPE
            adm_type = str(admission['ADMISSION_TYPE'].iloc[0]).lower()
            result['emergency_status'] = 'emergency' in adm_type or 'urgent' in adm_type
            
            # Use admission time as proxy for surgery time
            result['surgery_time'] = pd.to_datetime(admission['ADMITTIME'].iloc[0])
        
        # Surgery duration is typically not directly available in MIMIC-III
        # Would need to calculate from procedure start/end times if available
        
        return result
    
    def get_lab_markers(self, subject_id: int, hadm_id: int, 
                       surgery_time: pd.Timestamp, hours_before: int = 48) -> Dict[str, Optional[float]]:
        """
        Get lab markers within specified hours before surgery.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            surgery_time: Timestamp of surgery
            hours_before: Hours before surgery to look for labs (default 48)
            
        Returns:
            Dictionary with lab marker values
        """
        result = {
            'wbc': None,
            'crp': None,
            'albumin': None,
            'lactate': None,
            'bnp': None
        }
        
        if self.labevents_df is None or self.d_labitems_df is None:
            return result
        
        # Use detected lab item IDs
        lab_itemids = {
            'wbc': self.item_ids_cache.get('wbc', [51300, 51301]),
            'crp': self.item_ids_cache.get('crp', [50889]),
            'albumin': self.item_ids_cache.get('albumin', [50862]),
            'lactate': self.item_ids_cache.get('lactate', [50813]),
            'bnp': self.item_ids_cache.get('bnp', [50910])
        }
        
        # Pure time-window search across admissions: SUBJECT_ID + CHARTTIME
        all_lab_itemids = [item for items in lab_itemids.values() for item in items]
        labs = self.labevents_df[
            (self.labevents_df['SUBJECT_ID'] == subject_id) &
            (self.labevents_df['ITEMID'].isin(all_lab_itemids))
        ].copy()
        if labs.empty:
            return result

        labs['CHARTTIME'] = pd.to_datetime(labs['CHARTTIME'])
        time_threshold = surgery_time - timedelta(hours=hours_before)
        preop_labs = labs[
            (labs['CHARTTIME'] >= time_threshold) &
            (labs['CHARTTIME'] < surgery_time)
        ]
        if preop_labs.empty:
            return result

        # Track provenance: which HADM_IDs contributed data
        hadms_included = sorted(
            set(preop_labs['HADM_ID'].dropna().astype(float))
        ) if 'HADM_ID' in preop_labs.columns else []

        # Get most recent value for each lab marker
        preop_labs = preop_labs.sort_values('CHARTTIME', ascending=False)

        for marker, itemids in lab_itemids.items():
            marker_labs = preop_labs[preop_labs['ITEMID'].isin(itemids)]
            if not marker_labs.empty:
                value = marker_labs['VALUENUM'].iloc[0]
                result[marker] = float(value) if pd.notna(value) else None

        # Attach provenance metadata under a special key
        result["_metadata"] = {
            "hadm_ids_included": hadms_included,
            "window_start": str(time_threshold),
            "window_end": str(surgery_time),
        }

        return result
    
    def extract_pulmonary_risk_factors(self, subject_id: int, hadm_id: int) -> Dict[str, any]:
        """
        Extract all pulmonary risk factors for a patient admission.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            
        Returns:
            Structured dictionary with all risk factors for ARISCAT calculation
        """
        # Get surgery details first (needed for time-based queries)
        surgery_details = self.get_surgery_details(subject_id, hadm_id)
        surgery_time = surgery_details.get('surgery_time')
        
        if surgery_time is None:
            # Fallback to admission time if surgery time not available
            if self.admissions_df is not None:
                admission = self.admissions_df[self.admissions_df['HADM_ID'] == hadm_id]
                if not admission.empty:
                    surgery_time = pd.to_datetime(admission['ADMITTIME'].iloc[0])
        
        # Extract all risk factors
        risk_factors = {
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'age_at_surgery': self.get_patient_age_at_surgery(subject_id, hadm_id),
            'preop_spo2': self.get_last_preop_spo2(subject_id, hadm_id, surgery_time) if surgery_time else None,
            'preop_hgb': self.get_preop_hgb(subject_id, hadm_id, surgery_time) if surgery_time else None,
            'respiratory_diagnoses': self.get_respiratory_diagnoses(subject_id, hadm_id),
            'smoking_status': self.get_smoking_status(subject_id, hadm_id),
            'prior_respiratory_admissions': self.get_prior_respiratory_admissions(subject_id, hadm_id),
            'surgery_details': surgery_details,
            'lab_markers': self.get_lab_markers(subject_id, hadm_id, surgery_time) if surgery_time else {}
        }
        
        return risk_factors
    
    def calculate_ariscat_score(self, risk_factors: Dict[str, any]) -> Dict[str, any]:
        """
        Calculate ARISCAT risk score based on extracted risk factors.
        
        Args:
            risk_factors: Dictionary of extracted risk factors
            
        Returns:
            Dictionary with ARISCAT score and risk stratification
        """
        score = 0
        score_details = []
        
        # Age scoring
        age = risk_factors.get('age_at_surgery')
        if age is not None:
            if age <= 50:
                score += 0
            elif age <= 59:
                score += 3
                score_details.append(f"Age {age}: +3 points")
            elif age <= 69:
                score += 16
                score_details.append(f"Age {age}: +16 points")
            else:
                score += 13
                score_details.append(f"Age {age}: +13 points")
        
        # SpO₂ scoring
        spo2 = risk_factors.get('preop_spo2')
        if spo2 is not None:
            if spo2 >= 96:
                score += 0
            elif spo2 >= 91:
                score += 8
                score_details.append(f"SpO₂ {spo2}%: +8 points")
            else:
                score += 24
                score_details.append(f"SpO₂ {spo2}%: +24 points")
        
        # Respiratory infection in last month (using prior admissions as proxy)
        prior_admissions = risk_factors.get('prior_respiratory_admissions', {})
        if prior_admissions.get('prior_pneumonia', False):
            score += 17
            score_details.append("Prior pneumonia: +17 points")
        
        # Preoperative anemia (Hgb < 10 g/dL)
        hgb = risk_factors.get('preop_hgb')
        if hgb is not None and hgb < 10:
            score += 11
            score_details.append(f"Hgb {hgb} g/dL: +11 points")
        
        # Surgical incision site
        incision_site = risk_factors.get('surgery_details', {}).get('incision_site')
        if incision_site == 'upper_abdominal':
            score += 15
            score_details.append("Upper abdominal surgery: +15 points")
        elif incision_site == 'thoracic':
            score += 14
            score_details.append("Thoracic surgery: +14 points")
        elif incision_site == 'head_neck':
            score += 8
            score_details.append("Head/neck surgery: +8 points")
        elif incision_site == 'lower_abdominal':
            score += 0
        
        # Duration of surgery (>2 hours)
        duration = risk_factors.get('surgery_details', {}).get('surgery_duration_minutes')
        if duration is not None and duration > 120:
            score += 16
            score_details.append(f"Surgery duration {duration} min: +16 points")
        
        # Emergency surgery
        emergency = risk_factors.get('surgery_details', {}).get('emergency_status')
        if emergency:
            score += 8
            score_details.append("Emergency surgery: +8 points")
        
        # Risk stratification
        if score <= 25:
            risk_level = "Low"
        elif score <= 44:
            risk_level = "Intermediate"
        else:
            risk_level = "High"
        
        return {
            'ariscat_score': score,
            'risk_level': risk_level,
            'score_details': score_details,
            'risk_factors': risk_factors
        }

    def calculate_ariscat(self, risk_factors: Dict[str, any]) -> Dict[str, any]:
        """
        Calculate ARISCAT score (0–123) and risk category using the extracted
        pulmonary risk dictionary.

        This implementation follows the validated ARISCAT weighting:

        Inputs (7 total):
            1) Age (years)
            2) Pre-op SpO2 on room air (%)
            3) Respiratory infection in last month (boolean)
            4) Pre-op anemia (lowest Hgb in last 30 days, g/dL)
            5) Surgical incision site: peripheral / upper_abdominal / intrathoracic
            6) Duration of surgery (hours)
            7) Emergency surgery (boolean)

        Point allocation (Canet et al. 2010):
            Age:
                - ≤50 years → 0 points
                - 51–80 years → 3 points
                - >80 years → 16 points

            SpO2 (room air):
                - ≥96% → 0 points
                - 91–95% → 8 points
                - ≤90% → 24 points

            Respiratory infection (last month):
                - No → 0 points
                - Yes → 17 points

            Pre-op anemia (lowest Hgb in last 30 days):
                - ≥10 g/dL → 0 points
                - <10 g/dL → 11 points

            Surgical incision site:
                - Peripheral → 0 points
                - Upper abdominal → 15 points
                - Intrathoracic → 24 points

            Duration of surgery:
                - ≤2 hours → 0 points
                - 2–3 hours → 16 points
                - >3 hours → 23 points

            Emergency surgery:
                - No → 0 points
                - Yes → 8 points

        Risk stratification:
            - Low risk: ≤26 points
            - Intermediate risk: 27–44 points
            - High risk: ≥45 points

        Args:
            risk_factors: Output of extract_pulmonary_risk_factors(...)

        Returns:
            dict with:
                - score: numeric ARISCAT score
                - risk_category: 'low' | 'intermediate' | 'high'
                - contributors: list of strings describing contributing factors
        """
        score = 0
        contributors: List[str] = []

        # --- Age ---
        age = risk_factors.get('age_at_surgery')
        if age is not None:
            if age <= 50:
                age_points = 0
            elif 51 <= age <= 80:
                age_points = 3
            else:  # > 80
                age_points = 16
            score += age_points
            contributors.append(f"Age {age} years: {age_points} points")

        # --- SpO2 (room air) ---
        spo2 = risk_factors.get('preop_spo2')
        if spo2 is not None:
            if spo2 >= 96:
                spo2_points = 0
            elif 91 <= spo2 <= 95:
                spo2_points = 8
            else:  # ≤ 90
                spo2_points = 24
            score += spo2_points
            contributors.append(f"SpO2 {spo2}%: {spo2_points} points")

        # --- Respiratory infection in last month ---
        # Use prior respiratory admissions and current diagnoses as proxy.
        resp_diagnoses = risk_factors.get('respiratory_diagnoses', {}) or {}
        prior_resp = risk_factors.get('prior_respiratory_admissions', {}) or {}

        has_resp_infection = (
            prior_resp.get('prior_pneumonia', False)
            or prior_resp.get('prior_respiratory_failure', False)
        )

        # Also check current admission diagnoses for bronchitis/pneumonia/respiratory failure
        if not has_resp_infection and self.diagnoses_icd_df is not None:
            subject_id = risk_factors.get('subject_id')
            hadm_id = risk_factors.get('hadm_id')
            if subject_id and hadm_id:
                current_diag = self.diagnoses_icd_df[
                    (self.diagnoses_icd_df['SUBJECT_ID'] == subject_id)
                    & (self.diagnoses_icd_df['HADM_ID'] == hadm_id)
                ]
                if not current_diag.empty and self.d_icd_diagnoses_df is not None:
                    code_col = 'ICD9_CODE' if self.icd_version == 'ICD9' else self.icd_code_column
                    merge_col = (
                        'ICD9_CODE'
                        if self.icd_version == 'ICD9'
                        else ('ICD10_CODE' if 'ICD10_CODE' in self.d_icd_diagnoses_df.columns else 'ICD_CODE')
                    )
                    if code_col in current_diag.columns:
                        merged = current_diag.merge(
                            self.d_icd_diagnoses_df,
                            left_on=code_col,
                            right_on=merge_col,
                            how='left',
                        )
                        if 'LONG_TITLE' in merged.columns:
                            titles_lower = merged['LONG_TITLE'].astype(str).str.lower()
                            has_resp_infection = titles_lower.str.contains(
                                'bronchitis|pneumonia|respiratory failure',
                                case=False,
                                na=False,
                            ).any()
                        if not has_resp_infection and code_col in merged.columns:
                            codes_str = merged[code_col].astype(str)
                            if self.icd_version == 'ICD9':
                                resp_codes = [
                                    '466',
                                    '480',
                                    '481',
                                    '482',
                                    '483',
                                    '484',
                                    '485',
                                    '486',
                                    '51881',
                                    '51882',
                                    '51884',
                                ]
                                has_resp_infection = codes_str.str.startswith(tuple(resp_codes)).any()
                            else:
                                resp_codes = ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J96']
                                has_resp_infection = codes_str.str.startswith(tuple(resp_codes)).any()

        if has_resp_infection:
            score += 17
            contributors.append("Respiratory infection in last month: 17 points")

        # --- Pre-op anemia (Hgb < 10 g/dL, lowest in last 30 days) ---
        hgb = risk_factors.get('preop_hgb')
        if hgb is not None:
            if hgb < 10.0:
                anemia_points = 11
            else:
                anemia_points = 0
            score += anemia_points
            contributors.append(f"Pre-op hemoglobin {hgb} g/dL: {anemia_points} points")

        # --- Surgical incision site ---
        incision_site = (risk_factors.get('surgery_details') or {}).get('incision_site')
        site_points = 0
        site_label = incision_site or "unknown"
        if incision_site is not None:
            if incision_site == 'upper_abdominal':
                site_points = 15
            elif incision_site in {'thoracic', 'intrathoracic'}:
                site_points = 24
            else:
                # Treat all other sites as peripheral for ARISCAT purposes
                site_points = 0
        score += site_points
        contributors.append(f"Surgical incision site ({site_label}): {site_points} points")

        # --- Duration of surgery (hours) ---
        duration_min = (risk_factors.get('surgery_details') or {}).get('surgery_duration_minutes')
        duration_points = 0
        if duration_min is not None:
            duration_hours = duration_min / 60.0
            if duration_hours <= 2.0:
                duration_points = 0
            elif 2.0 < duration_hours <= 3.0:
                duration_points = 16
            else:  # > 3 hours
                duration_points = 23
            score += duration_points
            contributors.append(f"Surgery duration {duration_hours:.1f} hours: {duration_points} points")

        # --- Emergency surgery ---
        emergency = (risk_factors.get('surgery_details') or {}).get('emergency_status')
        emergency_points = 8 if emergency else 0
        score += emergency_points
        contributors.append(f"Emergency surgery: {emergency_points} points")

        # --- Risk category using validated thresholds ---
        if score <= 26:
            category = "low"
        elif 27 <= score <= 44:
            category = "intermediate"
        else:
            category = "high"

        return {
            "score": score,
            "risk_category": category,
            "contributors": contributors,
        }

    def enhance_pulmonary_risk_with_labs(self, risk_factors: Dict[str, any]) -> Dict[str, any]:
        """
        Enhance pulmonary risk assessment by combining ARISCAT score with
        lab-derived risk flags (WBC, CRP, Albumin, Lactate, BNP).

        Lab ontology / TRD assumptions:
        - WBC, CRP, Albumin, Lactate, BNP have already been mapped via D_LABITEMS
          in get_lab_markers(), so here we only interpret values.
        - Thresholds (per TRD-style clinical conventions):
            * CRP > 10 mg/L → active inflammation
            * Albumin < 3.5 g/dL → malnutrition → increased respiratory risk
            * Lactate > 2.0 mmol/L → perfusion issue → respiratory failure risk
            * BNP > 300 pg/mL (or IU/L equivalent) → heart failure / pulmonary edema risk

        Args:
            risk_factors: Output from extract_pulmonary_risk_factors(...)

        Returns:
            Consolidated pulmonary risk report:
                {
                  "overall_risk_category": "low|moderate|high",
                  "ariscat": { ... from calculate_ariscat(...) ... },
                  "lab_risk": {
                      "wbc": float | None,
                      "crp": float | None,
                      "albumin": float | None,
                      "lactate": float | None,
                      "bnp": float | None,
                      "flags": {
                          "active_inflammation": bool,
                          "malnutrition": bool,
                          "perfusion_issue": bool,
                          "heart_failure": bool,
                          "wbc_abnormal": bool,
                      },
                      "flag_details": [ ... ]
                  },
                  "summary": str
                }
        """
        # Base ARISCAT calculation
        ariscat = self.calculate_ariscat(risk_factors)

        labs = risk_factors.get("lab_markers") or {}
        wbc = labs.get("wbc")
        crp = labs.get("crp")
        albumin = labs.get("albumin")
        lactate = labs.get("lactate")
        bnp = labs.get("bnp")

        lab_flags = {
            "active_inflammation": False,
            "malnutrition": False,
            "perfusion_issue": False,
            "heart_failure": False,
            "wbc_abnormal": False,
        }
        flag_details: List[str] = []

        # WBC: generic abnormal range (approx. 4–12 x10^9/L)
        if wbc is not None:
            if wbc < 4 or wbc > 12:
                lab_flags["wbc_abnormal"] = True
                flag_details.append(f"WBC abnormal ({wbc})")

        # CRP: inflammation
        if crp is not None and crp > 10:
            lab_flags["active_inflammation"] = True
            flag_details.append(f"CRP {crp} > 10 → active inflammation")

        # Albumin: malnutrition
        if albumin is not None and albumin < 3.5:
            lab_flags["malnutrition"] = True
            flag_details.append(f"Albumin {albumin} < 3.5 → malnutrition / higher pulmonary risk")

        # Lactate: perfusion issue / impending organ failure
        if lactate is not None and lactate > 2.0:
            lab_flags["perfusion_issue"] = True
            flag_details.append(f"Lactate {lactate} > 2.0 → perfusion issue / respiratory failure risk")

        # BNP: heart failure / pulmonary edema risk
        if bnp is not None and bnp > 300:
            lab_flags["heart_failure"] = True
            flag_details.append(f"BNP {bnp} > 300 → heart failure / pulmonary edema risk")

        lab_risk = {
            "wbc": wbc,
            "crp": crp,
            "albumin": albumin,
            "lactate": lactate,
            "bnp": bnp,
            "flags": lab_flags,
            "flag_details": flag_details,
        }

        # ------------------------------------------------------------------
        # Temporal pulmonary patterns (subject_id + time window)
        # ------------------------------------------------------------------
        temporal_patterns = {}
        surgery_details = risk_factors.get("surgery_details") or {}
        surgery_time = surgery_details.get("surgery_time")
        subject_id = risk_factors.get("subject_id")

        if surgery_time is not None and subject_id is not None:
            temporal_patterns = self.analyze_pulmonary_temporal_patterns(
                subject_id=subject_id,
                surgery_time=pd.to_datetime(surgery_time),
            )

            # Add temporal flags into summary context
            tp = temporal_patterns
            if tp.get("lactate", {}).get("rising_lactate"):
                flag_details.append(
                    "Rising lactate over 3 hours pre-op → possible perfusion decline / sepsis risk"
                )
            if tp.get("crp", {}).get("increasing_crp"):
                flag_details.append(
                    "CRP increasing over 48 hours pre-op → escalating inflammatory burden"
                )
            if tp.get("albumin", {}).get("falling_albumin"):
                flag_details.append(
                    "Albumin falling over 7 days pre-op → worsening nutritional status"
                )

        # Combine ARISCAT category with lab flags to derive an overall category.
        # Start from ARISCAT risk, then escalate if severe lab abnormalities exist.
        category_order = ["low", "moderate", "high"]
        base_cat = (ariscat.get("risk_category") or "low").lower()
        try:
            idx = category_order.index(base_cat)
        except ValueError:
            idx = 0

        # Escalation rules:
        # - Any of: perfusion_issue, heart_failure → at least moderate
        # - Multiple lab flags or malnutrition + inflammation → bump one level
        severe_flags = lab_flags["perfusion_issue"] or lab_flags["heart_failure"]
        num_flags = sum(1 for v in lab_flags.values() if v)

        new_idx = idx
        if severe_flags and new_idx < 2:  # ensure at least "high" if already moderate
            new_idx = max(new_idx + 1, 1)
        if num_flags >= 2 and new_idx < 2:
            new_idx = min(new_idx + 1, 2)

        overall_category = category_order[new_idx]

        # Build summary text
        summary_parts = [f"ARISCAT risk: {base_cat} (score {ariscat.get('score')})"]
        if flag_details:
            summary_parts.append("Lab-derived risk modifiers: " + "; ".join(flag_details))
        summary_parts.append(f"Overall pulmonary risk: {overall_category}")

        return {
            "overall_risk_category": overall_category,
            "ariscat": ariscat,
            "lab_risk": lab_risk,
            "temporal_patterns": temporal_patterns,
            "summary": " | ".join(summary_parts),
        }


def extract_for_multiple_patients(extractor: PulmonaryRiskExtractor, 
                                  subject_ids: List[int], 
                                  hadm_ids: List[int]) -> List[Dict[str, any]]:
    """
    Extract pulmonary risk factors for multiple patients.
    
    Args:
        extractor: PulmonaryRiskExtractor instance
        subject_ids: List of subject IDs
        hadm_ids: List of corresponding HADM IDs
        
    Returns:
        List of risk factor dictionaries
    """
    results = []
    for subject_id, hadm_id in zip(subject_ids, hadm_ids):
        try:
            risk_factors = extractor.extract_pulmonary_risk_factors(subject_id, hadm_id)
            ariscat_result = extractor.calculate_ariscat_score(risk_factors)
            results.append(ariscat_result)
        except Exception as e:
            print(f"Error processing subject_id={subject_id}, hadm_id={hadm_id}: {e}")
            results.append({
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'error': str(e)
            })
    
    return results


# Example usage
if __name__ == "__main__":
    # Initialize extractor with data directory
    extractor = PulmonaryRiskExtractor(data_dir="./data")
    
    # Load data files for subject ID 249
    extractor.load_data(subject_id="249")
    
    # Example: Extract risk factors for a patient
    # Based on available data: subject_id=249, hadm_id=116935
    subject_id = 249
    hadm_id = 158975
    
    print(f"\nExtracting pulmonary risk factors for Subject ID: {subject_id}, HADM ID: {hadm_id}")
    print("=" * 70)
    
    try:
        risk_factors = extractor.extract_pulmonary_risk_factors(subject_id, hadm_id)
        ariscat_result = extractor.calculate_ariscat_score(risk_factors)
        
        print(f"\nARISCAT Score: {ariscat_result['ariscat_score']}")
        print(f"Risk Level: {ariscat_result['risk_level']}")
        
        if ariscat_result['score_details']:
            print("\nScore Breakdown:")
            for detail in ariscat_result['score_details']:
                print(f"  - {detail}")
        
        print("\nExtracted Risk Factors:")
        print(f"  Age at surgery: {risk_factors.get('age_at_surgery')}")
        print(f"  Pre-op SpO2: {risk_factors.get('preop_spo2')}")
        print(f"  Pre-op Hgb: {risk_factors.get('preop_hgb')}")
        print(f"  Smoking status: {risk_factors.get('smoking_status')}")
        print(f"  Respiratory diagnoses:")
        resp_diag = risk_factors.get('respiratory_diagnoses', {})
        print(f"    - COPD: {resp_diag.get('copd')}")
        print(f"    - Asthma: {resp_diag.get('asthma')}")
        print(f"    - OSA: {resp_diag.get('osa')}")
        print(f"  Surgery details:")
        surgery = risk_factors.get('surgery_details', {})
        print(f"    - Incision site: {surgery.get('incision_site')}")
        print(f"    - Emergency: {surgery.get('emergency_status')}")
        print(f"  Lab markers:")
        labs = risk_factors.get('lab_markers', {})
        for lab_name, lab_value in labs.items():
            if lab_value is not None:
                print(f"    - {lab_name.upper()}: {lab_value}")
        
    except Exception as e:
        print(f"\nError extracting risk factors: {e}")
        import traceback
        traceback.print_exc()

