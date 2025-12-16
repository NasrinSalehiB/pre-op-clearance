"""
Cardiac & Pulmonary Risk Integration

This module runs three main risk modules:
  1) Cardiac score-based risk (RCRI, AUB-HAS-2-like, NSQIP-like, Gupta-like)
  2) Cardiac risk from events/labs/conditions (procedure & event context + temporal trends)
  3) Pulmonary risk (ARISCAT + labs + temporal pulmonary patterns)

It logs assumptions/heuristics and returns a final JSON-like dict with:
  - risk_summary (scores + consensus)
  - event_based_context (narrative)
  - lab_flags (abnormalities and trend-based flags)
  - data_quality (missing/inferred fields + assumptions)
  - recommendations (prioritized)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from pulmonary_risk_extractor import PulmonaryRiskExtractor
from score_based_risk import ScoreBasedRiskExtractor
from procedure_event_risk import ProcedureEventRiskExtractor
from trend_risk_flags import generate_trend_risk_flags
from temporal_narrative import generate_temporal_narrative


def _build_cardiac_consensus(scores: Dict[str, Any]) -> str:
    rcri = scores.get("rcri")
    aub = scores.get("aub_has2")
    if not rcri or not aub:
        return "Partial"
    r_cat = rcri["components"].get("temporal_adjusted_category")
    a_cat = aub["components"].get("temporal_adjusted_category")
    if r_cat == a_cat:
        return "Convergent"
    return "Divergent"


def _normalize_risk_tier(category: Optional[str]) -> str:
    """
    Normalize risk categories to tier-based system: [critical, high, moderate, low]
    
    Mapping:
    - critical: None (no data) or extreme high-risk scenarios
    - high: "high" from calculators
    - moderate: "intermediate" from calculators
    - low: "low" from calculators
    """
    if category is None:
        return "critical"  # No data = critical uncertainty
    
    cat_lower = category.lower()
    if cat_lower in ["high"]:
        return "high"
    elif cat_lower in ["intermediate", "moderate"]:
        return "moderate"
    elif cat_lower in ["low"]:
        return "low"
    else:
        return "critical"  # Unknown category = critical


def _calculate_copd_risk_tier(
    copd_diagnosis: Optional[bool],
    spo2: Optional[float],
    prior_respiratory_admissions: Optional[Dict[str, Any]],
    smoking_status: Optional[str]
) -> Optional[str]:
    """
    Calculate COPD risk tier based on available clinical data.
    
    Clinical logic:
    - Critical: COPD + SpO2 < 90% OR COPD + prior respiratory failure
    - High: COPD + SpO2 90-92% OR COPD + current/former smoker OR COPD + prior pneumonia
    - Moderate: COPD diagnosis present with normal SpO2 (>92%) and no high-risk factors
    - Low: No COPD diagnosis
    - None: Missing diagnosis data (cannot determine COPD status)
    
    Args:
        copd_diagnosis: Boolean indicating if COPD diagnosis is present, or None if data unavailable
        spo2: Pre-operative SpO2 value (percentage)
        prior_respiratory_admissions: Dictionary with prior respiratory admission flags
        smoking_status: Smoking status ('current', 'former', 'never', or None)
    
    Returns:
        Risk tier: "critical", "high", "moderate", "low", or None if data insufficient
    """
    # If COPD diagnosis status cannot be determined (None), return None
    if copd_diagnosis is None:
        return None
    
    # If COPD diagnosis is present
    if copd_diagnosis:
        # Critical: Severe hypoxemia (< 90%) or prior respiratory failure
        if spo2 is not None and spo2 < 90:
            return "critical"
        
        # Check for prior respiratory complications (proxies for COPD severity)
        if prior_respiratory_admissions:
            prior_resp_failure = prior_respiratory_admissions.get('prior_respiratory_failure', False)
            if prior_resp_failure:
                return "critical"
        
        # High: Moderate hypoxemia (90-92%) or smoking history or prior pneumonia
        if spo2 is not None and 90 <= spo2 <= 92:
            return "high"
        
        if smoking_status in ['current', 'former']:
            return "high"
        
        if prior_respiratory_admissions:
            prior_pneumonia = prior_respiratory_admissions.get('prior_pneumonia', False)
            if prior_pneumonia:
                return "high"
        
        # Moderate: COPD present but SpO2 normal (>92%) and no other high-risk factors
        return "moderate"
    
    # No COPD diagnosis = low risk
    return "low"


def _calculate_score_percentage(calculator_name: str, score: Optional[float]) -> Optional[float]:
    """
    Convert risk scores to clinically validated percentage values.
    
    Based on published literature:
    - RCRI: Score 0 → ~0.4%, 1 → ~0.9%, 2 → ~6.6%, ≥3 → ~11.0% (MACE risk)
    - AUB-HAS-2: Score 0 → ~2%, 1 → ~5%, 2 → ~10%, 3 → ~15%, 4 → ~25% (MACE risk)
    - NSQIP MACE: Score 0-2 → ~2-5%, 3-4 → ~8-12%, 5-6 → ~15-20% (estimated)
    - Gupta MICA: Score 0-1 → ~1-3%, 2-3 → ~5-8%, 4-5 → ~12-18% (estimated)
    - ARISCAT: Score <26 → 1.6%, 26-44 → 13.3%, ≥45 → 42.1% (pulmonary complications, Canet et al. 2010)
    """
    if score is None:
        return None
    
    calculator_lower = calculator_name.lower()
    
    if calculator_lower == "rcri":
        # RCRI MACE risk percentages (from Lee et al. 1999)
        if score == 0:
            return 0.4
        elif score == 1:
            return 0.9
        elif score == 2:
            return 6.6
        else:  # score >= 3
            return 11.0
    
    elif calculator_lower == "aub-has-2" or calculator_lower == "aub_has2":
        # AUB-HAS-2 MACE risk percentages (estimated from literature)
        if score == 0:
            return 2.0
        elif score == 1:
            return 5.0
        elif score == 2:
            return 10.0
        elif score == 3:
            return 15.0
        else:  # score >= 4
            return 25.0
    
    elif calculator_lower == "nsqip_mace" or calculator_lower == "nsqip mace":
        # NSQIP MACE risk percentages (estimated)
        if score <= 2:
            return 3.0 + (score * 1.0)  # 3-5%
        elif score <= 4:
            return 8.0 + ((score - 2) * 2.0)  # 8-12%
        else:  # score >= 5
            return 15.0 + ((score - 4) * 2.5)  # 15-20%
    
    elif calculator_lower == "gupta_mica" or calculator_lower == "gupta mica":
        # Gupta MICA risk percentages (estimated)
        if score <= 1:
            return 1.0 + (score * 2.0)  # 1-3%
        elif score <= 3:
            return 5.0 + ((score - 1) * 1.5)  # 5-8%
        else:  # score >= 4
            return 12.0 + ((score - 3) * 3.0)  # 12-18%
    
    elif calculator_lower == "ariscat":
        # ARISCAT pulmonary complication risk percentages (from Canet et al. 2010)
        # Low risk (<26 points): 1.6% PPC rate
        # Intermediate risk (26-44 points): 13.3% PPC rate
        # High risk (≥45 points): 42.1% PPC rate
        if score <= 25:
            return 1.6  # Low risk category
        elif score <= 44:
            return 13.3  # Intermediate risk category
        else:  # score >= 45
            return 42.1  # High risk category
    
    return None  # Unknown calculator


def _extract_recent_vital_signs(
    pulm_extractor: PulmonaryRiskExtractor,
    subject_id: int,
    hadm_id: int,
    surgery_time: Optional[datetime],
) -> Dict[str, Any]:
    """
    Extract recent vital signs from chartevents before surgery_time.
    
    Collects the most recent pre-operative vital sign values:
    - Heart Rate (HR)
    - Blood Pressure (Systolic/Diastolic)
    - Temperature
    - Respiratory Rate
    - SpO2 (already extracted, but included for completeness)
    
    Args:
        pulm_extractor: PulmonaryRiskExtractor instance with loaded data
        subject_id: Patient subject ID
        hadm_id: Hospital admission ID
        surgery_time: Surgery time (all vitals must be before this time)
    
    Returns:
        Dictionary with recent vital sign values (None if not available)
    """
    vital_signs: Dict[str, Any] = {
        "heart_rate": None,
        "systolic_bp": None,
        "diastolic_bp": None,
        "mean_arterial_pressure": None,
        "temperature": None,
        "respiratory_rate": None,
        "spo2": None,
    }
    
    if pulm_extractor.chartevents_df is None or pulm_extractor.d_items_df is None:
        return vital_signs
    
    if surgery_time is None:
        return vital_signs
    
    # Convert surgery_time to pandas Timestamp if needed
    if isinstance(surgery_time, datetime):
        surgery_time_ts = pd.Timestamp(surgery_time)
    else:
        surgery_time_ts = pd.to_datetime(surgery_time)
    
    # Common MIMIC-III item IDs for vital signs (with fallbacks)
    # These are common IDs, but we should try to detect them dynamically
    vital_item_ids = {
        "heart_rate": [220050, 211],  # Heart Rate
        "systolic_bp": [51, 442, 455, 6701, 220179],  # Systolic BP
        "diastolic_bp": [8368, 8441, 8555, 220180],  # Diastolic BP
        "mean_arterial_pressure": [456, 52, 6702, 220181],  # MAP
        "temperature": [223761, 223762, 676, 677, 678, 679],  # Temperature (F/C)
        "respiratory_rate": [220210, 618, 615, 224690, 224689],  # Respiratory Rate
        "spo2": pulm_extractor.item_ids_cache.get('spo2', [646, 220277]),  # SpO2
    }
    
    # Try to detect item IDs dynamically if possible
    if pulm_extractor.d_items_df is not None:
        d_items_lower = pulm_extractor.d_items_df['LABEL'].astype(str).str.lower()
        
        # Heart Rate
        hr_matches = pulm_extractor.d_items_df[d_items_lower.str.contains('heart rate|hr|pulse rate', na=False, regex=True)]
        if not hr_matches.empty:
            vital_item_ids["heart_rate"] = hr_matches['ITEMID'].tolist()[:5]
        
        # Blood Pressure
        sbp_matches = pulm_extractor.d_items_df[d_items_lower.str.contains('systolic|sbp', na=False, regex=True)]
        if not sbp_matches.empty:
            vital_item_ids["systolic_bp"] = sbp_matches['ITEMID'].tolist()[:5]
        
        dbp_matches = pulm_extractor.d_items_df[d_items_lower.str.contains('diastolic|dbp', na=False, regex=True)]
        if not dbp_matches.empty:
            vital_item_ids["diastolic_bp"] = dbp_matches['ITEMID'].tolist()[:5]
        
        map_matches = pulm_extractor.d_items_df[d_items_lower.str.contains('mean arterial|map', na=False, regex=True)]
        if not map_matches.empty:
            vital_item_ids["mean_arterial_pressure"] = map_matches['ITEMID'].tolist()[:5]
        
        # Temperature
        temp_matches = pulm_extractor.d_items_df[d_items_lower.str.contains('temperature|temp', na=False, regex=True)]
        if not temp_matches.empty:
            vital_item_ids["temperature"] = temp_matches['ITEMID'].tolist()[:5]
        
        # Respiratory Rate
        rr_matches = pulm_extractor.d_items_df[d_items_lower.str.contains('respiratory rate|rr|fio2', na=False, regex=True)]
        if not rr_matches.empty:
            vital_item_ids["respiratory_rate"] = rr_matches['ITEMID'].tolist()[:5]
    
    # Extract all vital signs from chartevents
    # Use SUBJECT_ID + time window approach (like labs) to allow cross-admission data
    # but prioritize current admission
    all_vital_itemids = [itemid for itemids in vital_item_ids.values() for itemid in itemids]
    
    chart_events = pulm_extractor.chartevents_df[
        (pulm_extractor.chartevents_df['SUBJECT_ID'] == subject_id) &
        (pulm_extractor.chartevents_df['ITEMID'].isin(all_vital_itemids))
    ].copy()
    
    if chart_events.empty:
        return vital_signs
    
    # Convert CHARTTIME to datetime and filter for pre-operative values
    chart_events['CHARTTIME'] = pd.to_datetime(chart_events['CHARTTIME'])
    preop_events = chart_events[chart_events['CHARTTIME'] < surgery_time_ts]
    
    if preop_events.empty:
        return vital_signs
    
    # Sort by time (most recent first)
    preop_events = preop_events.sort_values('CHARTTIME', ascending=False)
    
    # Extract most recent value for each vital sign
    for vital_name, itemids in vital_item_ids.items():
        vital_events = preop_events[preop_events['ITEMID'].isin(itemids)]
        if not vital_events.empty:
            # Get most recent value
            value = vital_events['VALUENUM'].iloc[0]
            if pd.notna(value):
                vital_signs[vital_name] = float(value)
    
    return vital_signs


def _format_relative_time(capture_time: datetime, surgery_time: datetime) -> str:
    """
    Format relative time between capture_time and surgery_time.
    
    Examples:
        - "1 hour ago"
        - "2 days ago"
        - "1 week ago"
        - "2 weeks ago"
    """
    delta = surgery_time - capture_time
    
    if delta.total_seconds() < 3600:  # Less than 1 hour
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif delta.total_seconds() < 86400:  # Less than 1 day
        hours = int(delta.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.days < 7:  # Less than 1 week
        days = delta.days
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif delta.days < 30:  # Less than 1 month
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    else:
        months = delta.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"


def _get_lab_normal_ranges(lab_name: str, gender: Optional[str] = None) -> Dict[str, Optional[float]]:
    """
    Get normal range for a lab value based on lab name and patient gender.
    
    Args:
        lab_name: Name of the lab (e.g., 'hgb', 'creatinine', 'bnp')
        gender: Patient gender ('M' or 'F')
    
    Returns:
        Dictionary with 'min' and 'max' normal values, or None if not applicable
    """
    # Normal ranges (clinically standard values)
    ranges: Dict[str, Dict[str, Dict[str, float]]] = {
        'hgb': {
            'M': {'min': 13.0, 'max': 17.0},  # g/dL
            'F': {'min': 12.0, 'max': 15.0},  # g/dL
        },
        'creatinine': {
            'M': {'min': 0.7, 'max': 1.3},  # mg/dL
            'F': {'min': 0.6, 'max': 1.1},  # mg/dL
        },
        'bnp': {
            'M': {'min': 0, 'max': 100},  # pg/mL (age-dependent, but 100 is common cutoff)
            'F': {'min': 0, 'max': 100},  # pg/mL
        },
        'troponin': {
            'M': {'min': 0, 'max': 0.04},  # ng/mL (ULN varies by assay, 0.04 is common)
            'F': {'min': 0, 'max': 0.04},  # ng/mL
        },
        'wbc': {
            'M': {'min': 4.0, 'max': 11.0},  # K/uL
            'F': {'min': 4.0, 'max': 11.0},  # K/uL
        },
        'crp': {
            'M': {'min': 0, 'max': 3.0},  # mg/L (normal < 3, elevated > 10)
            'F': {'min': 0, 'max': 3.0},  # mg/L
        },
        'albumin': {
            'M': {'min': 3.5, 'max': 5.0},  # g/dL
            'F': {'min': 3.5, 'max': 5.0},  # g/dL
        },
        'lactate': {
            'M': {'min': 0.5, 'max': 2.2},  # mmol/L
            'F': {'min': 0.5, 'max': 2.2},  # mmol/L
        },
        'k': {
            'M': {'min': 3.5, 'max': 5.0},  # mEq/L
            'F': {'min': 3.5, 'max': 5.0},  # mEq/L
        },
        'na': {
            'M': {'min': 136, 'max': 145},  # mEq/L
            'F': {'min': 136, 'max': 145},  # mEq/L
        },
        'mg': {
            'M': {'min': 1.7, 'max': 2.2},  # mg/dL
            'F': {'min': 1.7, 'max': 2.2},  # mg/dL
        },
        'rdw': {
            'M': {'min': 11.5, 'max': 14.5},  # %
            'F': {'min': 11.5, 'max': 14.5},  # %
        },
    }
    
    if lab_name.lower() not in ranges:
        return {'min': None, 'max': None}
    
    lab_ranges = ranges[lab_name.lower()]
    
    # Use gender-specific range if available, otherwise use 'M' as default
    if gender and gender.upper() in lab_ranges:
        return lab_ranges[gender.upper()]
    elif 'M' in lab_ranges:
        return lab_ranges['M']
    else:
        return {'min': None, 'max': None}


def _extract_lab_summary(
    pulm_extractor: PulmonaryRiskExtractor,
    cardiac_extractor: ScoreBasedRiskExtractor,
    subject_id: int,
    hadm_id: int,
    surgery_time: Optional[datetime],
) -> Dict[str, Any]:
    """
    Extract lab summary with actual values, normal ranges (sex-based), and capture times.
    
    Args:
        pulm_extractor: PulmonaryRiskExtractor instance with loaded data
        cardiac_extractor: ScoreBasedRiskExtractor instance with loaded data
        subject_id: Patient subject ID
        hadm_id: Hospital admission ID
        surgery_time: Surgery time (all labs must be before this time)
    
    Returns:
        Dictionary with lab summaries including value, normal_range, and captured_ago
    """
    lab_summary: Dict[str, Any] = {}
    
    if surgery_time is None:
        return lab_summary
    
    # Convert surgery_time to datetime if needed
    if isinstance(surgery_time, datetime):
        surgery_time_dt = surgery_time
    else:
        surgery_time_dt = pd.to_datetime(surgery_time).to_pydatetime()
    
    # Get patient gender
    gender = None
    if pulm_extractor.patients_df is not None:
        patient_row = pulm_extractor.patients_df[pulm_extractor.patients_df['SUBJECT_ID'] == subject_id]
        if not patient_row.empty and 'GENDER' in patient_row.columns:
            gender = patient_row['GENDER'].iloc[0]
    
    # Lab names and their extraction methods
    labs_to_extract = [
        ('hgb', 'Hemoglobin', 'hgb'),
        ('creatinine', 'Creatinine', 'creatinine'),
        ('bnp', 'BNP', 'bnp'),
        ('troponin', 'Troponin', 'troponin'),
        ('wbc', 'WBC', 'wbc'),
        ('crp', 'CRP', 'crp'),
        ('albumin', 'Albumin', 'albumin'),
        ('lactate', 'Lactate', 'lactate'),
        ('k', 'Potassium', 'k'),
        ('na', 'Sodium', 'na'),
        ('mg', 'Magnesium', 'mg'),
        ('rdw', 'RDW', 'rdw'),
    ]
    
    # Extract labs from both extractors
    for lab_key, lab_display_name, lab_name in labs_to_extract:
        value = None
        capture_time = None
        
        # Try to get from cardiac extractor first (for cardiac labs)
        if lab_name in ['bnp', 'troponin', 'creatinine', 'hgb', 'k', 'na', 'mg', 'rdw', 'wbc']:
            if cardiac_extractor.labevents_df is not None:
                itemids = cardiac_extractor.item_ids_cache.get(lab_name, [])
                if itemids:
                    labs = cardiac_extractor.labevents_df[
                        (cardiac_extractor.labevents_df['SUBJECT_ID'] == subject_id) &
                        (cardiac_extractor.labevents_df['ITEMID'].isin(itemids))
                    ].copy()
                    if not labs.empty:
                        labs['CHARTTIME'] = pd.to_datetime(labs['CHARTTIME'])
                        preop_labs = labs[labs['CHARTTIME'] < surgery_time_dt]
                        if not preop_labs.empty:
                            preop_labs = preop_labs.sort_values('CHARTTIME', ascending=False)
                            value = float(preop_labs['VALUENUM'].iloc[0]) if pd.notna(preop_labs['VALUENUM'].iloc[0]) else None
                            capture_time = preop_labs['CHARTTIME'].iloc[0].to_pydatetime() if value is not None else None
        
        # Try pulmonary extractor for pulmonary labs
        if value is None and lab_name in ['wbc', 'crp', 'albumin', 'lactate', 'bnp']:
            if pulm_extractor.labevents_df is not None:
                itemids = pulm_extractor.item_ids_cache.get(lab_name, [])
                if itemids:
                    labs = pulm_extractor.labevents_df[
                        (pulm_extractor.labevents_df['SUBJECT_ID'] == subject_id) &
                        (pulm_extractor.labevents_df['ITEMID'].isin(itemids))
                    ].copy()
                    if not labs.empty:
                        labs['CHARTTIME'] = pd.to_datetime(labs['CHARTTIME'])
                        preop_labs = labs[labs['CHARTTIME'] < surgery_time_dt]
                        if not preop_labs.empty:
                            preop_labs = preop_labs.sort_values('CHARTTIME', ascending=False)
                            value = float(preop_labs['VALUENUM'].iloc[0]) if pd.notna(preop_labs['VALUENUM'].iloc[0]) else None
                            capture_time = preop_labs['CHARTTIME'].iloc[0].to_pydatetime() if value is not None else None
        
        # Build lab summary entry
        if value is not None:
            normal_range = _get_lab_normal_ranges(lab_name, gender)
            captured_ago = _format_relative_time(capture_time, surgery_time_dt) if capture_time else None
            
            lab_summary[lab_key] = {
                'name': lab_display_name,
                'value': value,
                'normal_range': {
                    'min': normal_range['min'],
                    'max': normal_range['max'],
                },
                'captured_ago': captured_ago,
                'capture_time': capture_time.isoformat() if capture_time else None,
            }
    
    return lab_summary


def _calculate_rcri_fraction(components: Dict[str, Any], insulin_therapy: Optional[bool] = None) -> Optional[str]:
    """
    Calculate RCRI component fraction (e.g., "2/6" or "2/5" if insulin therapy is None).
    
    Counts how many of the RCRI components are True.
    If insulin_therapy is None (cannot be determined), it is excluded from the count.
    
    Args:
        components: Dictionary with RCRI component flags
        insulin_therapy: Insulin therapy status (True/False/None). If None, excluded from count.
    
    Returns:
        Fraction string like "2/6" or "2/5" (if insulin_therapy is None)
    """
    rcri_keys = [
        "ischemic_heart_disease",
        "heart_failure",
        "stroke_tia",
        "creatinine_gt_2",
        "high_risk_surgery",
    ]
    
    # Count standard components
    count = sum(1 for k in rcri_keys if components.get(k, False))
    
    # Add insulin therapy if it's not None (i.e., if we can determine it)
    if insulin_therapy is not None:
        if insulin_therapy:
            count += 1
        total_components = 6
    else:
        # Insulin therapy is None (unknown), exclude from count
        total_components = 5
    
    return f"{count}/{total_components}"


def _collect_assumptions() -> List[str]:
    """
    Static list of key assumptions/heuristics used across risk modules.
    """
    return [
        "ICD version (ICD-9 vs ICD-10) is inferred from diagnoses tables and used to choose code ranges.",
        "Surgical risk and type (cardiac/vascular/thoracic/abdominal) are inferred from ICD procedure LONG_TITLE keyword matching.",
        "Emergency/urgent vs elective status is inferred from ADMISSION_TYPE.",
        "Pre-operative lab trends (BNP, troponin, lactate, CRP, albumin, creatinine, electrolytes) are computed using SUBJECT_ID + time windows, not limited by HADM_ID.",
        "Troponin upper limit of normal (ULN) is approximated as 0.04 for flagging active myocardial injury.",
        "Electrolyte instability thresholds: |ΔK_24h| > 0.5 mEq/L or |ΔNa_24h| > 5 mEq/L.",
        "Albumin < 3.5 g/dL and falling trend over 7 days is used as a proxy for poor nutritional reserve.",
        "NSQIP MACE and Gupta MICA models are simplified surrogates using only age, creatinine, surgery risk/type, and emergency status; ASA and detailed functional status are not available in this extract.",
        "Functional status is treated as 'unknown' because no direct mobility/ADL data are present in the current MIMIC-III slice.",
        "Prior cardiac events are marked as 'possibly post-operative' when they occur in admissions that also have surgical procedures, using a heuristic linkage.",
    ]


def integrate_cardiac_and_pulmonary_risk(
    subject_id: int,
    hadm_id: int,
    data_dir: str = "./data",
    surgery_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Run cardiac score-based, cardiac event/lab-based, and pulmonary
    risk modules, and return an integrated assessment JSON.
    """
    # ---------------- Pulmonary module ----------------
    pulm_extractor = PulmonaryRiskExtractor(data_dir=data_dir)
    pulm_extractor.load_data(subject_id=str(subject_id))

    pulmonary_risk_factors = pulm_extractor.extract_pulmonary_risk_factors(
        subject_id, hadm_id
    )
    surgery_details = pulmonary_risk_factors.get("surgery_details") or {}
    if surgery_time is None:
        surgery_time = surgery_details.get("surgery_time")
        if isinstance(surgery_time, str):
            surgery_time = datetime.fromisoformat(surgery_time)

    enhanced_pulm = pulm_extractor.enhance_pulmonary_risk_with_labs(
        pulmonary_risk_factors
    )

    # ---------------- Cardiac score-based module ----------------
    cardiac_extractor = ScoreBasedRiskExtractor(data_dir=data_dir)
    cardiac_extractor.load_data(subject_id=str(subject_id))

    cardiac_report = cardiac_extractor.build_score_based_risk_report(
        subject_id=subject_id,
        hadm_id=hadm_id,
        surgery_time=surgery_time,
    )
    scores = cardiac_report.get("scores", {})

    # ---------------- Procedure/event-based cardiac module ----------------
    proc_extractor = ProcedureEventRiskExtractor(data_dir=data_dir)
    proc_extractor.load_data(subject_id=str(subject_id))
    proc_report = proc_extractor.build_procedure_event_risk_report(subject_id, hadm_id)

    # ---------------- Trend-based flags ----------------
    trend_flags = generate_trend_risk_flags(
        pulmonary_temporal=enhanced_pulm.get("temporal_patterns"),
        pulmonary_lab_snapshot=enhanced_pulm.get("lab_risk"),
        cardiac_temporal=cardiac_report.get("cardiac_temporal_patterns"),
        cardiac_lab_snapshot=cardiac_report.get("labs"),
    )

    # ---------------- Temporal narratives ----------------
    temporal_narrative = generate_temporal_narrative(
        pulmonary_temporal=enhanced_pulm.get("temporal_patterns"),
        pulmonary_lab_snapshot=enhanced_pulm.get("lab_risk"),
        cardiac_temporal=cardiac_report.get("cardiac_temporal_patterns"),
        cardiac_lab_snapshot=cardiac_report.get("labs"),
        trend_flags=trend_flags,
    )

    # ---------------- calculated_risk_factors ----------------
    # Extract ARISCAT information
    ariscat = enhanced_pulm.get("ariscat", {})
    ariscat_score = ariscat.get("score") or ariscat.get("ariscat_score")
    ariscat_cat = ariscat.get("risk_category") or ariscat.get("risk_level")
    pulm_overall_cat = enhanced_pulm.get("overall_risk_category")
    
    # Extract ARISCAT component flags from risk_factors
    ariscat_flags = {}
    if pulmonary_risk_factors:
        # Respiratory infection
        prior_resp = pulmonary_risk_factors.get('prior_respiratory_admissions', {}) or {}
        resp_diagnoses = pulmonary_risk_factors.get('respiratory_diagnoses', {}) or {}
        ariscat_flags["respiratory_infection"] = (
            prior_resp.get('prior_pneumonia', False) or
            prior_resp.get('prior_respiratory_failure', False) or
            resp_diagnoses.get('bronchitis', False) or
            resp_diagnoses.get('pneumonia', False)
        )
        # Anemia (Hgb < 10)
        hgb = pulmonary_risk_factors.get('preop_hgb')
        ariscat_flags["anemia"] = hgb is not None and hgb < 10
        # Surgical incision site (thoracic/abdominal)
        incision_site = (pulmonary_risk_factors.get('surgery_details') or {}).get('incision_site')
        ariscat_flags["thoracic_or_abdominal_surgery"] = incision_site in {'thoracic', 'upper_abdominal', 'lower_abdominal'}
        # Surgery duration > 2h
        duration_min = (pulmonary_risk_factors.get('surgery_details') or {}).get('surgery_duration_minutes')
        ariscat_flags["surgery_duration_gt_2h"] = duration_min is not None and duration_min > 120
        # Emergency surgery
        emergency = (pulmonary_risk_factors.get('surgery_details') or {}).get('emergency_status')
        ariscat_flags["emergency_surgery"] = bool(emergency)
        # Asthma
        ariscat_flags["asthma"] = resp_diagnoses.get('asthma', False)
    
    # Calculate COPD risk tier
    copd_risk_tier = None
    if pulmonary_risk_factors:
        resp_diagnoses = pulmonary_risk_factors.get('respiratory_diagnoses')
        
        # Check if diagnosis data is available
        # If resp_diagnoses is None or empty dict, we can't determine COPD status
        if resp_diagnoses is not None:
            copd_diagnosis = resp_diagnoses.get('copd')  # Can be True, False, or None
            spo2 = pulmonary_risk_factors.get('preop_spo2')
            prior_resp = pulmonary_risk_factors.get('prior_respiratory_admissions', {}) or {}
            smoking_status = pulmonary_risk_factors.get('smoking_status')
            copd_risk_tier = _calculate_copd_risk_tier(
                copd_diagnosis=copd_diagnosis,
                spo2=spo2,
                prior_respiratory_admissions=prior_resp,
                smoking_status=smoking_status
            )
        # If resp_diagnoses is None, copd_risk_tier remains None (cannot determine)
    
    # Pulmonary lab_risk
    pulm_lab_risk = enhanced_pulm.get("lab_risk", {}) or {}
    pulm_lab_values = {
        "wbc": pulm_lab_risk.get("wbc"),
        "bnp": pulm_lab_risk.get("bnp"),
    }
    
    # Cardiac calculators - build structure for each
    cardiac_calculators = {}
    
    # RCRI
    rcri_res = scores.get("rcri")
    if rcri_res:
        rcri_comps = rcri_res.get("components", {})
        rcri_cardiac_labs = cardiac_report.get("labs", {}) or {}
        rcri_lab_values = {
            "wbc": rcri_cardiac_labs.get("inflammatory", {}).get("wbc_last"),
            "bnp": rcri_cardiac_labs.get("bnp", {}).get("last"),
        }
        rcri_flags = {
            "Ischemic heart disease": rcri_comps.get("ischemic_heart_disease", False),
            "heart_failure": rcri_comps.get("heart_failure", False),
            "Stroke/TIA": rcri_comps.get("stroke_tia", False),
            "Diabetes": rcri_comps.get("diabetes", False),
            "Creatinine > 2.0": rcri_comps.get("creatinine_gt_2", False),
            "High-risk surgery": rcri_comps.get("high_risk_surgery", False),
        }
        rcri_score = rcri_res.get("score")
        rcri_cat = rcri_comps.get("temporal_adjusted_category") or rcri_comps.get("risk_category")
        
        # Build RCRI factors list with TRUE/FALSE values
        # Note: According to RCRI, the component is "insulin therapy for diabetes", not just "diabetes"
        # Insulin therapy cannot be directly detected from available MIMIC-III data
        # (medication/prescription data not available in current extract)
        # If diabetes is False, insulin therapy is False (no diabetes = no insulin therapy)
        # If diabetes is True, insulin therapy is None (unknown - cannot determine without medication data)
        insulin_therapy = None
        if rcri_comps.get("diabetes", False):
            # Diabetes present but we cannot determine if on insulin therapy without medication data
            # Attempt to detect from chartevents if available (future enhancement)
            # For now, set to None to indicate unknown status
            insulin_therapy = None
        else:
            # No diabetes = no insulin therapy
            insulin_therapy = False
        
        rcri_factors = {
            "High_risk_surgery": rcri_comps.get("high_risk_surgery", False),
            "Ischemic_heart_disease": rcri_comps.get("ischemic_heart_disease", False),
            "CHF_history": rcri_comps.get("heart_failure", False),
            "Insuline_therapy_for_DM": insulin_therapy,  # None if diabetes present but insulin status unknown
            "Cerebrovascular_disease": rcri_comps.get("stroke_tia", False),
            "Preop_creatine>2_mg/dl": rcri_comps.get("creatinine_gt_2", False),
        }
        
        cardiac_calculators["RCRI"] = {
            "score": rcri_score,
            "score_percentage": round(_calculate_score_percentage("RCRI", rcri_score), 1) if rcri_score is not None else None,
            "component_fraction": _calculate_rcri_fraction(rcri_comps, insulin_therapy),
            "risk_tier": _normalize_risk_tier(rcri_cat),
            "factors": rcri_factors,
        }
    
    # AUB-HAS-2
    aub_res = scores.get("aub_has2")
    if aub_res:
        aub_comps = aub_res.get("components", {})
        aub_score = aub_res.get("score")
        aub_cat = aub_comps.get("temporal_adjusted_category") or aub_comps.get("risk_category")
        cardiac_calculators["AUB-HAS-2"] = {
            "score": aub_score,
            "score_percentage": round(_calculate_score_percentage("AUB-HAS-2", aub_score), 1) if aub_score is not None else None,
            "risk_tier": _normalize_risk_tier(aub_cat),
        }
    
    # NSQIP MACE
    nsqip_res = scores.get("nsqip_mace")
    if nsqip_res:
        nsqip_comps = nsqip_res.get("components", {})
        nsqip_score = nsqip_res.get("score")
        nsqip_cat = nsqip_comps.get("risk_category")
        cardiac_calculators["NSQIP_MACE"] = {
            "score": nsqip_score,
            "score_percentage": round(_calculate_score_percentage("NSQIP_MACE", nsqip_score), 1) if nsqip_score is not None else None,
            "risk_tier": _normalize_risk_tier(nsqip_cat),
        }
    
    # Gupta MICA
    gupta_res = scores.get("gupta_mica")
    if gupta_res:
        gupta_comps = gupta_res.get("components", {})
        gupta_score = gupta_res.get("score")
        gupta_cat = gupta_comps.get("risk_category")
        cardiac_calculators["Gupta_MICA"] = {
            "score": gupta_score,
            "score_percentage": round(_calculate_score_percentage("Gupta_MICA", gupta_score), 1) if gupta_score is not None else None,
            "risk_tier": _normalize_risk_tier(gupta_cat),
        }
    
    # Determine overall cardiac risk category (use RCRI if available, else AUB-HAS-2)
    cardiac_overall_cat = None
    if rcri_res:
        rcri_comps = rcri_res.get("components", {})
        cardiac_overall_cat = rcri_comps.get("temporal_adjusted_category") or rcri_comps.get("risk_category")
    elif aub_res:
        aub_comps = aub_res.get("components", {})
        cardiac_overall_cat = aub_comps.get("temporal_adjusted_category") or aub_comps.get("risk_category")
    
    # Normalize overall cardiac risk tier
    cardiac_overall_tier = _normalize_risk_tier(cardiac_overall_cat)
    
    # Build cardiac consensus for confidence calculation (not included in output)
    cardiac_consensus = _build_cardiac_consensus(scores)
    
    # Build shared cardiac lab_risk (using RCRI flags as primary, or general cardiac labs)
    cardiac_labs_shared = cardiac_report.get("labs", {}) or {}
    cardiac_lab_values_shared = {
        "wbc": cardiac_labs_shared.get("inflammatory", {}).get("wbc_last"),
        "bnp": cardiac_labs_shared.get("bnp", {}).get("last"),
    }
    # Use RCRI component flags as primary flags, plus lab-based flags
    cardiac_wbc = cardiac_lab_values_shared.get("wbc")
    cardiac_bnp = cardiac_lab_values_shared.get("bnp")
    cardiac_flags_shared = {}
    if rcri_res:
        rcri_comps = rcri_res.get("components", {})
        cardiac_flags_shared = {
            "Ischemic heart disease": rcri_comps.get("ischemic_heart_disease", False),
            "heart_failure": rcri_comps.get("heart_failure", False),
            "Stroke/TIA": rcri_comps.get("stroke_tia", False),
            "Diabetes": rcri_comps.get("diabetes", False),
            "Creatinine > 2.0": rcri_comps.get("creatinine_gt_2", False),
            "High-risk surgery": rcri_comps.get("high_risk_surgery", False),
        }
    # Add lab-based flags (wbc_abnormal and elevated BNP indicating heart failure)
    cardiac_flags_shared["wbc_abnormal"] = cardiac_wbc is not None and (cardiac_wbc < 4 or cardiac_wbc > 12)
    # Note: heart_failure above is from RCRI components (diagnoses), 
    # BNP > 300 also suggests heart failure but we keep the RCRI component flag as primary
    if cardiac_bnp is not None and cardiac_bnp > 300:
        # If BNP is elevated, ensure heart_failure flag is True (combines diagnosis + lab evidence)
        cardiac_flags_shared["heart_failure"] = True
    
    # Get temporal patterns
    cardiac_temporal = cardiac_report.get("cardiac_temporal_patterns", {})
    
    # Build calculated_risk_factors structure
    # Cardiac: shared lab_risk, overall_risk_category, and temporal_patterns at cardiac level
    # Pulmonary: ARISCAT info at pulmonary level, lab_risk also at pulmonary level
    calculated_risk_factors = {
        "cardiac": {
            "overall_risk_category": cardiac_overall_cat,
            "overall_risk_tier": cardiac_overall_tier,
            "lab_risk": {
                **cardiac_lab_values_shared,
                "flags": cardiac_flags_shared,
            },
            "temporal_patterns": cardiac_temporal,
            **cardiac_calculators,  # Individual calculators (RCRI, AUB-HAS-2, etc.)
        },
        "pulmonary": {
            "ARISCAT": {
                "score": ariscat_score,
                "score_percentage": round(_calculate_score_percentage("ARISCAT", ariscat_score), 1) if ariscat_score is not None else None,
                "risk_tier": _normalize_risk_tier(ariscat_cat),
            },
            "COPD_risk_tier": copd_risk_tier,
            "lab_risk": {
                **pulm_lab_values,
                "flags": ariscat_flags,
            },
        },
    }

    # ---------------- event_based_context (narrative) ----------------
    prior_surg = proc_report.get("prior_surgeries", {}) or {}
    prior_events = proc_report.get("prior_events", {}) or {}
    surg_ctx = proc_report.get("surgical_context", {}) or {}

    proc_lines: List[str] = []

    high_surg_entries: List[str] = []
    for group_name in ["cardiac", "vascular", "thoracic"]:
        group_list = prior_surg.get(group_name, []) or []
        for proc in group_list:
            desc = proc.get("description") or group_name
            date = proc.get("date")
            high_surg_entries.append(f"{desc} (HADM {proc.get('hadm_id')}, {date})")
    if high_surg_entries:
        proc_lines.append(
            "High-risk prior surgeries: " + "; ".join(high_surg_entries) + "."
        )
    else:
        proc_lines.append(
            "High-risk prior surgeries: none documented in available data."
        )

    event_entries: List[str] = []
    for bucket, label in [
        ("mi", "myocardial infarction"),
        ("stroke_tia", "stroke/TIA"),
        ("heart_failure", "heart failure admission"),
        ("arrhythmia", "arrhythmia admission"),
        ("shock_or_arrest", "cardiogenic shock/cardiac arrest"),
    ]:
        ev_list = prior_events.get(bucket, []) or []
        for ev in ev_list:
            postfix = (
                " (possibly post-operative)" if ev.get("post_operative_heuristic") else ""
            )
            event_entries.append(
                f"{label} (HADM {ev.get('hadm_id')}, {ev.get('date')}){postfix}"
            )
    if event_entries:
        proc_lines.append("Prior cardiac events: " + "; ".join(event_entries) + ".")
    else:
        proc_lines.append(
            "Prior cardiac events: none documented in available diagnoses."
        )

    surg_risk_flag = surg_ctx.get("risk_flag", "unknown")
    surg_type = surg_ctx.get("planned_surgery_type") or "unknown type"
    emerg_status = surg_ctx.get("emergency_status") or "unknown urgency"
    stress = surg_ctx.get("hemodynamic_stress") or "unknown hemodynamic stress"
    proc_lines.append(
        f"Surgical context: {surg_risk_flag} risk {surg_type} procedure, {emerg_status}, "
        f"with {stress} hemodynamic stress (inferred from procedure codes and ADMISSION_TYPE)."
    )

    event_based_context = " ".join(proc_lines)

    # ---------------- lab_flags (abnormalities) ----------------
    cardiac_labs = cardiac_report.get("labs", {}) or {}
    lab_flags: List[str] = []

    bnp_last = cardiac_labs.get("bnp", {}).get("last")
    if bnp_last is not None and bnp_last > 300:
        lab_flags.append(f"BNP {bnp_last:.1f} pg/mL (supports HF risk).")

    trop_last = cardiac_labs.get("troponin", {}).get("last")
    if trop_last is not None and trop_last > 0.04:
        lab_flags.append(f"Troponin {trop_last:.3f} (above typical ULN).")

    k_last = cardiac_labs.get("electrolytes", {}).get("k")
    if k_last is not None and (k_last < 3.5 or k_last > 5.2):
        lab_flags.append(f"K+ {k_last:.2f} mEq/L (may increase arrhythmia risk).")

    crp_last = cardiac_labs.get("inflammatory", {}).get("crp_last")
    if crp_last is not None and crp_last > 10:
        lab_flags.append(f"CRP {crp_last:.1f} mg/L (inflammatory burden).")

    # Also include codes from trend-based flags for more structured lab flags
    for flag in trend_flags.get("cardiac", []):
        lab_flags.append(f"{flag['code']}: {flag['condition']}")

    # ---------------- risk confidence score (cardiac + pulmonary) ----------------
    def _map_cat_to_score(cat: Optional[str], mapping: Dict[str, float]) -> float:
        if not cat:
            return 0.5
        return mapping.get(str(cat).lower(), 0.5)

    # Cardiac category confidence
    rcri_res = scores.get("rcri")
    aub_res = scores.get("aub_has2")
    if rcri_res:
        comps = rcri_res.get("components", {})
        cardiac_cat = comps.get("temporal_adjusted_category") or comps.get("risk_category")
    elif aub_res:
        comps = aub_res.get("components", {})
        cardiac_cat = comps.get("temporal_adjusted_category") or comps.get("risk_category")
    else:
        cardiac_cat = None

    cardiac_cat_score = _map_cat_to_score(
        cardiac_cat, {"low": 0.5, "intermediate": 0.7, "high": 0.9}
    )

    # Pulmonary category confidence
    pulm_cat_score = _map_cat_to_score(
        pulm_overall_cat, {"low": 0.5, "moderate": 0.7, "high": 0.9}
    )

    # Data completeness: presence of key labs and scores
    cardiac_labs = cardiac_report.get("labs", {}) or {}
    key_labs_present = 0
    key_labs_total = 0
    for block, key in [
        ("bnp", "last"),
        ("troponin", "last"),
        ("electrolytes", "k"),
        ("electrolytes", "na"),
        ("anemia", "hgb_last"),
        ("inflammatory", "crp_last"),
        ("inflammatory", "wbc_last"),
    ]:
        key_labs_total += 1
        val = cardiac_labs.get(block, {}).get(key)
        if val is not None:
            key_labs_present += 1
    labs_completeness = key_labs_present / key_labs_total if key_labs_total else 0.0

    score_presence = 0
    score_total = 0
    for k in ["rcri", "aub_has2"]:
        score_total += 1
        if scores.get(k):
            score_presence += 1
    score_completeness = score_presence / score_total if score_total else 0.0

    data_completeness = 0.5 * labs_completeness + 0.5 * score_completeness

    # Consensus factor
    consensus_factor = {"convergent": 1.0, "partial": 0.7, "divergent": 0.4}.get(
        cardiac_consensus.lower(), 0.6
    )

    cardiac_confidence = round(
        0.4 * cardiac_cat_score + 0.3 * data_completeness + 0.3 * consensus_factor, 3
    )
    pulmonary_confidence = round(
        0.6 * pulm_cat_score + 0.4 * data_completeness, 3
    )
    overall_confidence = round(
        0.5 * cardiac_confidence + 0.5 * pulmonary_confidence, 3
    )

    # ---------------- recommendations (prioritized) ----------------
    recommendations: List[str] = []

    # Basic prioritization based on consensus and categories
    rcri_res = scores.get("rcri")
    if rcri_res:
        comps = rcri_res.get("components", {})
        cat = comps.get("temporal_adjusted_category") or comps.get("risk_category")
        if cardiac_consensus == "Convergent":
            if cat == "low":
                recommendations.append(
                    "Cardiac risk appears low and calculators are convergent; "
                    "standard perioperative cardiac monitoring is appropriate."
                )
            elif cat == "intermediate":
                recommendations.append(
                    "Cardiac risk is intermediate with convergent calculators; "
                    "optimize comorbidities and consider telemetry/step-down monitoring."
                )
            else:
                recommendations.append(
                    "Cardiac risk is high and calculators are convergent; "
                    "obtain cardiology consultation, consider ICU-level monitoring, "
                    "and defer elective surgery if clinically feasible."
                )
        else:
            recommendations.append(
                "Cardiac calculators provide partial or divergent signals; integrate scores with "
                "clinical judgment, imaging, and functional capacity assessment."
            )
    else:
        recommendations.append(
            "RCRI score is not available; cardiac risk should be assessed primarily from history, "
            "examination, ECG, echocardiography, and clinical judgment."
        )

    # Augment recommendations based on lab flags
    for flag in trend_flags.get("cardiac", []):
        if flag["code"] == "CARD_RISK_ACTIVE_MI":
            recommendations.insert(
                0,
                "Possible active myocardial injury: defer non-emergent surgery and obtain urgent cardiology consultation.",
            )
        elif flag["code"] == "CARD_RISK_WORSENING_HF":
            recommendations.insert(
                0,
                "Evidence of worsening heart failure: intensify HF management and consider postponing surgery until optimized.",
            )
        elif flag["code"] == "CARD_RISK_ARRHYTHMIA_RISK":
            recommendations.append(
                "Correct K+/Na+ derangements pre-operatively and monitor rhythm closely during anesthesia."
            )

    # Fallback if no recommendations generated
    if not recommendations:
        recommendations.append(
            "No specific high-risk cardiac red flags identified; proceed per standard institutional protocol."
        )

    # ---------------- Recent Vital Signs (pre-operative) ---------------- 
    recent_vital_signs = _extract_recent_vital_signs(
        pulm_extractor=pulm_extractor,
        subject_id=subject_id,
        hadm_id=hadm_id,
        surgery_time=surgery_time,
    )

    # ---------------- Lab Summary (with normal ranges and capture times) ---------------- 
    lab_summary = _extract_lab_summary(
        pulm_extractor=pulm_extractor,
        cardiac_extractor=cardiac_extractor,
        subject_id=subject_id,
        hadm_id=hadm_id,
        surgery_time=surgery_time,
    )

    # ---------------- Final JSON-like dict ---------------- 
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "surgery_time": surgery_time.isoformat() if isinstance(surgery_time, datetime) else None,
        "last_updated": datetime.now().isoformat(),
        "calculated_risk_factors": calculated_risk_factors,
        "event_based_context": event_based_context,
        "cardiac_lab_flags": lab_flags,
        "recommendations": recommendations,
        "trend_analysis": temporal_narrative,
        "recent_vital_signs": recent_vital_signs,
        "lab_summary": lab_summary,
        "surgery": {
            "name": surg_ctx.get("primary_procedure_name"),
            "type": surg_ctx.get("planned_surgery_type"),
            "expected_duration_minutes": surg_ctx.get("expected_duration_minutes"),
            "duration_inferred": surg_ctx.get("duration_inferred"),
        },
        "risk_confidence": {
            "overall": overall_confidence,
            "cardiac": cardiac_confidence,
            "pulmonary": pulmonary_confidence,
            "data_completeness": round(data_completeness, 3),
            "cardiac_consensus_factor": consensus_factor,
        },
    }


def predict_preop_risk(
    subject_id: int,
    hadm_id: int,
    planned_surgery_time: datetime,
    data_dir: str = "./data",
) -> Dict[str, Any]:
    """
    Pre-operative risk prediction function.
    
    This function is specifically designed for PRE-OPERATIVE RISK ASSESSMENT,
    treating the planned surgery as a FUTURE event. All lab trends and temporal
    features are calculated using ONLY data BEFORE the planned_surgery_time.
    
    Key differences from integrate_cardiac_and_pulmonary_risk():
    - planned_surgery_time is REQUIRED (not optional)
    - All temporal windows are anchored to planned_surgery_time
    - Labs during or after planned_surgery_time are EXCLUDED
    - Report is labeled as "pre-operative prediction"
    
    Args:
        subject_id: Patient SUBJECT_ID
        hadm_id: Hospital admission ID for the planned surgery
        planned_surgery_time: Planned/expected surgery time (datetime)
                             This should be a FUTURE time relative to admission
        data_dir: Directory containing MIMIC-III CSV files
        
    Returns:
        Integrated risk assessment JSON with:
        - assessment_type: "pre_operative_prediction"
        - planned_surgery_time: ISO-8601 string
        - All risk scores, lab trends, and recommendations based on pre-op data only
        - risk_confidence: Overall confidence score (0-1)
        
    Example:
        from datetime import datetime
        
        planned_surgery = datetime(2149, 12, 20, 10, 0, 0)  # Future surgery time
        report = predict_preop_risk(
            subject_id=249,
            hadm_id=116935,
            planned_surgery_time=planned_surgery,
            data_dir="./data"
        )
    """
    # Validate that planned_surgery_time is provided
    if planned_surgery_time is None:
        raise ValueError(
            "planned_surgery_time is required for pre-operative prediction. "
            "This should be the expected/planned surgery time."
        )
    
    # Get admission time for validation (optional check)
    try:
        admissions_df = pd.read_csv(
            f"{data_dir}/Subject_ID_{subject_id}_ADMISSIONS.csv"
        )
        adm = admissions_df[admissions_df["HADM_ID"] == hadm_id]
        if not adm.empty:
            admittime = pd.to_datetime(adm["ADMITTIME"].iloc[0])
            if planned_surgery_time < admittime:
                # Warning: surgery time is before admission (might be data issue)
                import warnings
                warnings.warn(
                    f"planned_surgery_time ({planned_surgery_time}) is before "
                    f"admission time ({admittime}). This may indicate a data issue. "
                    f"Proceeding with assessment using planned_surgery_time as anchor.",
                    UserWarning
                )
    except Exception:
        # If we can't validate, proceed anyway
        pass
    
    # Call the main integration function with the planned surgery time
    report = integrate_cardiac_and_pulmonary_risk(
        subject_id=subject_id,
        hadm_id=hadm_id,
        data_dir=data_dir,
        surgery_time=planned_surgery_time,  # Always use planned_surgery_time as anchor
    )
    
    # Add pre-operative prediction metadata
    report["assessment_type"] = "pre_operative_prediction"
    report["planned_surgery_time"] = (
        planned_surgery_time.isoformat()
        if isinstance(planned_surgery_time, datetime)
        else str(planned_surgery_time)
    )
    
    return report


if __name__ == "__main__":
    from pprint import pprint

    # Pre-operative prediction example
    # This treats surgery_time as a FUTURE event and excludes intra/post-op labs
    planned_surgery = datetime(2149, 12, 20, 10, 0, 0)  # Example future surgery time
    preop_report = predict_preop_risk(
        subject_id=249,
        hadm_id=116935,
        planned_surgery_time=planned_surgery,
        data_dir="./data",
    )
    print("=" * 80)
    print("PRE-OPERATIVE RISK PREDICTION")
    print("=" * 80)
    pprint(preop_report)


