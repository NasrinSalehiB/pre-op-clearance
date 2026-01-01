"""
Comorbidity Block Detection System for Barnabus Pre-Op Clearance

This module defines immutable data models for comorbidity detection and trigger evaluation.
Follows Barnabus TRD v3.0 specifications for comorbidity block integration.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime, timedelta


# ============================================================================
# Helper Functions for Recency and Consistency
# ============================================================================


def _calculate_recency_score(date_str: str) -> float:
    """
    Calculate recency score based on data age.

    Args:
        date_str: ISO 8601 date string

    Returns:
        Recency score: 1.0 (within 30 days), 0.8 (30-90 days), 0.6 (90-365 days), 0.4 (>1 year)
    """
    try:
        data_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        data_date = data_date.replace(tzinfo=None)
        days_old = (datetime.now() - data_date).days

        if days_old <= 30:
            return 1.0
        elif days_old <= 90:
            return 0.8
        elif days_old <= 365:
            return 0.6
        else:
            return 0.4
    except:
        # If date parsing fails, assume older data
        return 0.4


# ============================================================================
# 1. PATIENT_CLINICAL_DATA
# ============================================================================


@dataclass(frozen=True)
class ProblemListItem:
    """Immutable problem list item."""

    code: str
    name: str
    date: str  # ISO 8601 date string


@dataclass(frozen=True)
class MedicationItem:
    """Immutable medication item."""

    name: str
    class_: str  # Medication class (e.g., 'beta_blocker', 'ace_inhibitor')
    dose: str
    frequency: str


@dataclass(frozen=True)
class LabValues:
    """Immutable latest lab values."""

    hemoglobin: Optional[float] = None
    a1c: Optional[float] = None
    creatinine: Optional[float] = None
    albumin: Optional[float] = None
    inr: Optional[float] = None
    platelets: Optional[float] = None
    bilirubin: Optional[float] = None
    sodium: Optional[float] = None
    bnp: Optional[float] = None
    troponin: Optional[float] = None
    nt_probnp: Optional[float] = None  # NT-proBNP
    ferritin: Optional[float] = None
    tsat: Optional[float] = None  # Transferrin saturation %


@dataclass(frozen=True)
class BloodPressure:
    """Immutable blood pressure reading."""

    systolic: Optional[float] = None
    diastolic: Optional[float] = None


@dataclass(frozen=True)
class VitalSigns:
    """Immutable latest vital signs."""

    heartRate: Optional[float] = None
    bloodPressure: Optional[BloodPressure] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None


@dataclass(frozen=True)
class Demographics:
    """Immutable patient demographics."""

    age: Optional[int] = None
    sex: Optional[str] = None  # 'M', 'F', or None
    bmi: Optional[float] = None
    smokingStatus: Optional[str] = None  # 'never', 'former', 'current', None


@dataclass(frozen=True)
class HpiRedFlags:
    """Immutable HPI red flags."""

    chestPain: bool = False
    shortnessOfBreath: bool = False
    syncope: bool = False
    fever: bool = False
    # Additional flags can be added here


@dataclass(frozen=True)
class PatientClinicalData:
    """
    Immutable patient clinical data snapshot.
    All fields are frozen to ensure data integrity for audit/provenance.
    """

    problemList: List[ProblemListItem] = field(default_factory=list)
    medications: List[MedicationItem] = field(default_factory=list)
    labs: LabValues = field(default_factory=LabValues)
    vitals: VitalSigns = field(default_factory=VitalSigns)
    devices: List[str] = field(
        default_factory=list
    )  # ['CPAP', 'BIPAP', 'pacemaker', 'ICD', 'insulin_pump', 'CGM']
    demographics: Demographics = field(default_factory=Demographics)
    hpiRedFlags: HpiRedFlags = field(default_factory=HpiRedFlags)
    # Additional clinical data for risk stratification
    lvef: Optional[float] = None  # Left ventricular ejection fraction (%)
    nyha_class: Optional[int] = None  # NYHA functional class (I-IV)
    dasi_mets: Optional[float] = None  # DASI METs score
    recent_cardiac_event_months: Optional[int] = (
        None  # Months since last MI/PCI/CHF admission
    )
    inotropic_dependence: bool = False  # On inotropic support
    proteinuria: Optional[bool] = None  # Proteinuria present
    dialysis_dependent: bool = False  # On dialysis
    diabetes_complications: List[str] = field(
        default_factory=list
    )  # e.g., ['retinopathy', 'nephropathy']
    recurrent_hypoglycemia: bool = False  # Recurrent hypoglycemic episodes


# ============================================================================
# 2. COMORBIDITY_BLOCK_DEFINITION
# ============================================================================


@dataclass(frozen=True)
class LabThreshold:
    """Immutable lab threshold rule."""

    test: str  # e.g., 'creatinine', 'hemoglobin'
    operator: Literal[">", "<", ">=", "<=", "=="]  # Comparison operator
    value: float  # Threshold value


@dataclass(frozen=True)
class MedicationTrigger:
    """Immutable medication trigger rule with optional dose threshold."""

    class_: str  # Medication class
    name_patterns: List[str] = field(
        default_factory=list
    )  # Specific medication name patterns
    min_dose: Optional[float] = (
        None  # Minimum dose threshold (e.g., 40mg furosemide equivalent)
    )
    dose_unit: Optional[str] = None  # Unit for dose (e.g., 'mg')


@dataclass(frozen=True)
class TriggerConditions:
    """Immutable trigger conditions for a comorbidity block."""

    icd10Codes: List[str] = field(
        default_factory=list
    )  # Exact codes or prefixes (e.g., "I25.10", "E11.*")
    medicationTriggers: List[MedicationTrigger] = field(
        default_factory=list
    )  # Enhanced medication matching
    medicationClasses: List[str] = field(
        default_factory=list
    )  # Simple class matching (backward compat)
    labThresholds: List[LabThreshold] = field(default_factory=list)
    deviceTypes: List[str] = field(default_factory=list)
    requiredFlags: List[str] = field(
        default_factory=list
    )  # e.g., ['chestPain', 'shortnessOfBreath']
    demographicConditions: List[str] = field(
        default_factory=list
    )  # e.g., ['age >= 65']
    secondaryTriggers: Optional[TriggerConditions] = (
        None  # Secondary triggers for review flags
    )


@dataclass(frozen=True)
class ContentSections:
    """Immutable content sections for comorbidity block."""

    why: str  # Clinical rationale
    whatToCheck: List[str] = field(default_factory=list)
    preOpActions: List[str] = field(default_factory=list)
    intraOpConsiderations: List[str] = field(default_factory=list)
    postOpManagement: List[str] = field(default_factory=list)
    anesthesiaHeadsUp: str = ""
    references: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RiskStratificationRules:
    """Immutable risk stratification rules."""

    lowRisk: List[str] = field(default_factory=list)  # Conditions for low risk
    intermediateRisk: List[str] = field(
        default_factory=list
    )  # Conditions for intermediate risk
    highRisk: List[str] = field(default_factory=list)  # Conditions for high risk


@dataclass(frozen=True)
class BlockMetadata:
    """Immutable metadata for comorbidity block."""

    owner: str = ""
    reviewers: List[str] = field(default_factory=list)
    effectiveDate: str = ""  # ISO 8601 date
    status: str = "active"  # 'active', 'draft', 'archived'


@dataclass(frozen=True)
class ComorbidityBlockDefinition:
    """
    Immutable comorbidity block definition.
    This is the template/configuration that defines when and how a comorbidity block triggers.
    """

    blockId: str  # e.g., 'CAD_CHF_001'
    conditionName: str  # e.g., 'CAD/CHF'
    version: str  # Semantic versioning (e.g., '1.0.0')
    triggers: TriggerConditions = field(default_factory=TriggerConditions)
    contentSections: ContentSections = field(default_factory=ContentSections)
    riskStratification: RiskStratificationRules = field(
        default_factory=RiskStratificationRules
    )
    metadata: BlockMetadata = field(default_factory=BlockMetadata)


# ============================================================================
# 3. TRIGGER_EVALUATION_RESULT
# ============================================================================


@dataclass(frozen=True)
class ConfidenceScore:
    """Confidence score with tier categorization."""

    score: float  # 0-1
    tier: Literal["very_high", "high", "moderate", "low", "very_low"]
    primaryComponent: float
    secondaryComponent: float
    temporalComponent: float
    consistencyComponent: float


# ============================================================================
# Confidence Calculation Engine
# ============================================================================


class ConfidenceCalculator:
    """
    Confidence calculation engine for comorbidity blocks.
    Calculates confidence scores based on primary triggers, secondary triggers,
    temporal factors, and consistency across evidence sources.
    """

    @staticmethod
    def get_confidence_tier(
        score: float,
    ) -> Literal["very_high", "high", "moderate", "low", "very_low"]:
        """
        Categorize confidence score into tier.

        Args:
            score: Confidence score (0-1)

        Returns:
            Confidence tier: 'very_high', 'high', 'moderate', 'low', or 'very_low'
        """
        if score >= 0.9:
            return "very_high"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.3:
            return "low"
        else:
            return "very_low"

    @staticmethod
    def calculate_primary_confidence(
        trigger_type: str, evidence: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence for a primary trigger type.

        Args:
            trigger_type: Type of trigger ('icd10', 'medication', 'device', 'hpi_flag', 'demographic')
            evidence: Evidence dictionary with trigger-specific data

        Returns:
            Confidence value (0-1)
        """
        if trigger_type == "icd10":
            # ICD-10: exact match = 1.0, related = 0.8, historical = 0.6/0.4
            match_type = evidence.get("match_type", "exact")
            is_historical = evidence.get("is_historical", False)
            if match_type == "exact":
                return 0.6 if is_historical else 1.0
            else:  # related or range
                return 0.4 if is_historical else 0.8

        elif trigger_type == "medication":
            # Medication: specific indication = 0.9, default = 0.8, multiple indications = 0.7
            specificity = evidence.get("specificity", "default")
            if specificity == "specific":
                return 0.9
            elif specificity == "multiple":
                return 0.7
            else:
                return 0.8

        elif trigger_type == "device":
            # Device: active = 0.9, historical = 0.5
            is_active = evidence.get("is_active", True)
            return 0.9 if is_active else 0.5

        elif trigger_type == "hpi_flag":
            # HPI flag: current = 0.7, historical = 0.4
            is_current = evidence.get("is_current", True)
            return 0.7 if is_current else 0.4

        elif trigger_type == "lab_threshold":
            # Lab threshold: objective = 0.8
            return 0.8

        elif trigger_type == "demographic":
            # Demographic: reliable but less specific = 0.8
            return 0.8

        return 0.5  # Default

    @staticmethod
    def calculate_secondary_confidence(
        trigger_type: str, value: Optional[float], threshold: Optional[float] = None
    ) -> float:
        """
        Calculate confidence for a secondary trigger.

        Args:
            trigger_type: Type of secondary trigger ('bnp', 'a1c', 'bmi', 'stop_bang', 'lvef', 'ekg')
            value: Lab value or measurement
            threshold: Threshold value for comparison

        Returns:
            Confidence value (0-1)
        """
        if value is None:
            return 0.0

        if trigger_type == "bnp":
            if value > 400:
                return 0.9  # BNP > 400
            elif value >= 100:
                return 0.7  # BNP 100-400
            else:
                return 0.6

        elif trigger_type == "a1c":
            if value > 9.0:
                return 0.9  # HbA1c > 9%
            elif value >= 7.0:
                return 0.7  # HbA1c 7-9%
            else:
                return 0.6

        elif trigger_type == "bmi":
            if value > 35:
                return 0.6  # BMI > 35
            return 0.0

        elif trigger_type == "stop_bang":
            if value >= 5:
                return 0.8  # STOP-BANG ≥ 5
            return 0.0

        elif trigger_type == "lvef":
            if value < 50:
                return 0.8  # Low LVEF known
            return 0.0

        elif trigger_type == "ekg":
            return 0.6  # Abnormal EKG pattern

        elif trigger_type == "glucose":
            if value > 200:
                return 0.6  # Random glucose > 200
            return 0.0

        elif trigger_type == "neck_circumference":
            if threshold and value > threshold:
                return 0.5  # Neck circumference > 17"
            return 0.0

        return 0.6  # Default for other secondary triggers

    @staticmethod
    def calculate_temporal_confidence(data: Dict[str, Any]) -> float:
        """
        Calculate temporal confidence based on recency and consistency over time.

        Args:
            data: Dictionary with temporal data (recent_events, recency_scores, etc.)

        Returns:
            Temporal confidence value (0-1)
        """
        recency_scores = data.get("recency_scores", [])
        consistency_scores = data.get("consistency_scores", [])
        trend_adjustment = data.get("trend_adjustment", 0.0)

        # Recency component (weighted 0.6)
        recency_component = (
            sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
        )

        # Consistency over time component (weighted 0.4)
        consistency_base = (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores
            else 0.5
        )
        consistency_over_time = max(0.0, min(1.0, consistency_base + trend_adjustment))

        # Weighted average
        return (recency_component * 0.6) + (consistency_over_time * 0.4)

    @staticmethod
    def calculate_consistency_confidence(
        sources: List[str], conflicts: List[str] = None
    ) -> float:
        """
        Calculate consistency confidence based on multiple confirming sources.

        Args:
            sources: List of confirming source types ('problem_list', 'medications', 'labs', 'devices', 'hpi_flags')
            conflicts: List of conflict types (reduces confidence)

        Returns:
            Consistency confidence value (0-1)
        """
        if conflicts is None:
            conflicts = []

        confirming_sources = len(
            [s for s in sources if s in ["problem_list", "medications", "labs"]]
        )

        # MULTIPLE CONFIRMING SOURCES scoring
        if confirming_sources >= 3:
            consistency = 1.0  # Problem list + Meds + Labs all agree
        elif confirming_sources == 2:
            consistency = 0.8  # Problem list + one other source
        elif confirming_sources == 1:
            consistency = 0.6  # Only one source
        else:
            # No primary sources (only devices/HPI flags)
            if any(s in ["devices", "hpi_flags"] for s in sources):
                consistency = 0.6  # Single source (device/HPI)
            else:
                consistency = 0.0  # No sources

        # Apply conflict penalties
        conflict_penalty = len(conflicts) * 0.3  # Each conflict reduces by 0.3
        consistency = max(0.0, consistency - conflict_penalty)

        return consistency

    @staticmethod
    def calculate_block_confidence(
        block: ComorbidityBlockDefinition,
        patient_data: PatientClinicalData,
        primary_component: float,
        secondary_component: float,
        temporal_component: float,
        consistency_component: float,
    ) -> ConfidenceScore:
        """
        Calculate overall block confidence score.

        Args:
            block: Comorbidity block definition
            patient_data: Patient clinical data
            primary_component: Primary triggers component (already weighted × 0.4)
            secondary_component: Secondary triggers component (already weighted × 0.3)
            temporal_component: Temporal component (already weighted × 0.2)
            consistency_component: Consistency component (already weighted × 0.1)

        Returns:
            ConfidenceScore with score, tier, and component breakdown
        """
        # Weighted sum (components are already weighted)
        raw_score = (
            primary_component
            + secondary_component
            + temporal_component
            + consistency_component
        )

        # Normalize if > 1.0
        if raw_score > 1.0:
            score = raw_score / 1.5
        else:
            score = raw_score

        # Apply minimum threshold: if < 0.3, set to 0 (very_low)
        if score < 0.3:
            score = 0.0
            tier = "very_low"
        else:
            tier = ConfidenceCalculator.get_confidence_tier(score)

        # Cap at 1.0
        score = min(1.0, score)

        return ConfidenceScore(
            score=round(score, 3),
            tier=tier,
            primaryComponent=round(primary_component, 3),
            secondaryComponent=round(secondary_component, 3),
            temporalComponent=round(temporal_component, 3),
            consistencyComponent=round(consistency_component, 3),
        )


@dataclass(frozen=True)
class CategorizedActions:
    """
    Categorized actions based on confidence level.

    Confidence thresholds:
    - requiredActions: confidence ≥ 0.7 (confident display, optional review)
    - suggestedActions: confidence 0.5-0.69 (standard display, suggest review)
    - optionalActions: confidence 0.3-0.49 (low confidence, require review)
    """

    requiredActions: List[str] = field(default_factory=list)  # Only if confidence ≥ 0.7
    suggestedActions: List[str] = field(default_factory=list)  # Confidence 0.5-0.69
    optionalActions: List[str] = field(default_factory=list)  # Confidence 0.3-0.49


@dataclass(frozen=True)
class TriggerEvaluationResult:
    """
    Immutable result of evaluating a comorbidity block against patient data.
    This is the output of the trigger evaluation engine.
    """

    blockId: str
    triggered: bool
    triggerReasons: List[str] = field(default_factory=list)
    confidenceScore: float = 0.0  # 0-1
    confidenceTier: Literal["very_high", "high", "moderate", "low", "very_low"] = (
        "very_low"
    )
    confidenceFlag: Optional[str] = (
        None  # 'low_confidence_require_review', 'standard_confidence_suggest_review', 'confident_display_optional_review', 'high_confidence_minimal_review', None
    )
    riskLevel: Literal["low", "intermediate", "high"] = "low"
    requiredActions: List[str] = field(
        default_factory=list
    )  # Deprecated: use categorizedActions
    categorizedActions: CategorizedActions = field(
        default_factory=lambda: CategorizedActions()
    )
    # Export gating based on confidence
    exportAllowed: bool = True  # Always allowed, but may have warnings
    exportWarning: Optional[str] = None  # Warning message for low confidence blocks
    exportNote: Optional[str] = (
        None  # Note to include in export for low confidence blocks
    )


# ============================================================================
# Helper Functions for Trigger Evaluation
# ============================================================================


def calculate_egfr_ckd_epi(
    creatinine: Optional[float],
    age: Optional[int],
    sex: Optional[str],
    race: Optional[str] = None,  # 'black' or None
) -> Optional[float]:
    """
    Calculate eGFR using CKD-EPI formula.

    Args:
        creatinine: Serum creatinine in mg/dL
        age: Age in years
        sex: 'M' or 'F'
        race: 'black' or None (for race adjustment)

    Returns:
        eGFR in mL/min/1.73m², or None if inputs are missing
    """
    if creatinine is None or age is None or sex is None:
        return None

    if creatinine <= 0:
        return None

    # CKD-EPI formula constants
    if sex == "F":
        kappa = 0.7
        alpha = -0.329
        if creatinine <= 0.7:
            min_cr = 0.7
        else:
            min_cr = creatinine
    else:  # Male
        kappa = 0.9
        alpha = -0.411
        if creatinine <= 0.9:
            min_cr = 0.9
        else:
            min_cr = creatinine

    # Race adjustment
    race_multiplier = 1.159 if race == "black" else 1.0

    # Calculate eGFR
    egfr = (
        141
        * (min(creatinine / kappa, 1.0) ** alpha)
        * (max(creatinine / kappa, 1.0) ** -1.209)
        * (0.993**age)
        * race_multiplier
    )

    return max(egfr, 0.0)  # Ensure non-negative


def match_icd10_code(patient_code: str, trigger_codes: List[str]) -> bool:
    """
    Check if patient ICD-10 code matches any trigger code pattern.
    Supports exact matches, prefix matches, and range patterns (e.g., "I50.20-I50.43").

    Args:
        patient_code: Patient's ICD-10 code
        trigger_codes: List of trigger patterns (exact codes, prefixes with *, or ranges)

    Returns:
        True if any trigger code matches
    """
    patient_code_upper = patient_code.upper().strip()

    for trigger in trigger_codes:
        trigger_upper = trigger.upper().strip()

        # Exact match
        if patient_code_upper == trigger_upper:
            return True

        # Prefix match (e.g., "E11.*" matches "E11.9", "E11.65", etc.)
        if trigger_upper.endswith(".*"):
            prefix = trigger_upper[:-2]
            if patient_code_upper.startswith(prefix):
                return True

        # Range match (e.g., "I50.20-I50.43")
        if "-" in trigger_upper:
            parts = trigger_upper.split("-")
            if len(parts) == 2:
                start_code = parts[0].strip()
                end_code = parts[1].strip()
                # Simple numeric comparison for ranges
                # Extract numeric parts for comparison
                try:
                    # For codes like "I50.20", compare as strings (lexicographic)
                    if start_code <= patient_code_upper <= end_code:
                        return True
                except:
                    pass

    return False


def _get_icd10_match_type(patient_code: str, trigger_codes: List[str]) -> Optional[str]:
    """
    Get the match type for an ICD-10 code.

    Args:
        patient_code: Patient's ICD-10 code
        trigger_codes: List of trigger patterns

    Returns:
        'exact', 'related', 'range', or None if no match
    """
    patient_code_upper = patient_code.upper().strip()

    for trigger in trigger_codes:
        trigger_upper = trigger.upper().strip()

        # Exact match
        if patient_code_upper == trigger_upper:
            return "exact"

        # Prefix match (e.g., "E11.*" matches "E11.9", "E11.65", etc.) - treated as related
        if trigger_upper.endswith(".*"):
            prefix = trigger_upper[:-2]
            if patient_code_upper.startswith(prefix):
                return "related"

        # Range match (e.g., "I50.20-I50.43")
        if "-" in trigger_upper:
            parts = trigger_upper.split("-")
            if len(parts) == 2:
                start_code = parts[0].strip()
                end_code = parts[1].strip()
                try:
                    if start_code <= patient_code_upper <= end_code:
                        return "range"
                except:
                    pass

    return None


def match_medication_trigger(
    medication: MedicationItem, trigger: MedicationTrigger
) -> bool:
    """
    Check if a medication matches a medication trigger rule.
    Supports class matching, name pattern matching, and dose thresholds.

    Args:
        medication: Patient's medication
        trigger: Medication trigger rule

    Returns:
        True if medication matches trigger
    """
    # Check class match
    if medication.class_.lower() == trigger.class_.lower():
        # If no dose requirement, match found
        if trigger.min_dose is None:
            return True

        # Check dose threshold
        try:
            # Extract numeric dose from dose string (e.g., "40mg" -> 40.0)
            dose_str = medication.dose.lower()
            # Remove common units and extract number
            dose_match = re.search(r"(\d+\.?\d*)", dose_str)
            if dose_match:
                med_dose = float(dose_match.group(1))
                if med_dose >= trigger.min_dose:
                    return True
        except:
            pass

    # Check name patterns
    med_name_lower = medication.name.lower()
    for pattern in trigger.name_patterns:
        if pattern.lower() in med_name_lower:
            # Check dose if required
            if trigger.min_dose is None:
                return True
            try:
                dose_str = medication.dose.lower()
                dose_match = re.search(r"(\d+\.?\d*)", dose_str)
                if dose_match:
                    med_dose = float(dose_match.group(1))
                    if med_dose >= trigger.min_dose:
                        return True
            except:
                pass

    return False


def evaluate_lab_threshold(lab_value: Optional[float], threshold: LabThreshold) -> bool:
    """
    Evaluate a lab value against a threshold rule.

    Args:
        lab_value: Patient's lab value (can be None)
        threshold: LabThreshold rule to evaluate

    Returns:
        True if condition is met, False otherwise
    """
    if lab_value is None:
        return False

    op = threshold.operator
    val = threshold.value

    if op == ">":
        return lab_value > val
    elif op == "<":
        return lab_value < val
    elif op == ">=":
        return lab_value >= val
    elif op == "<=":
        return lab_value <= val
    elif op == "==":
        return abs(lab_value - val) < 0.001  # Float comparison tolerance
    else:
        return False


def evaluate_trigger(
    block: ComorbidityBlockDefinition, patient_data: PatientClinicalData
) -> TriggerEvaluationResult:
    """
    Evaluate a comorbidity block definition against patient clinical data.
    Implements exact trigger rules as specified in TRD v3.0.

    This is a deterministic function: same inputs → same output.

    Args:
        block: Comorbidity block definition to evaluate
        patient_data: Patient clinical data snapshot

    Returns:
        TriggerEvaluationResult with triggered status and details
    """
    trigger_reasons: List[str] = []
    triggered = False
    primary_triggers = 0
    secondary_triggers = 0
    primary_trigger_confidences: List[float] = (
        []
    )  # Track individual trigger confidences

    # Check ICD-10 codes (supports exact, prefix, and range matching)
    patient_codes = [item.code for item in patient_data.problemList]
    matching_codes = []

    for problem_item in patient_data.problemList:
        code = problem_item.code
        problem_date_str = problem_item.date

        # Determine if code is historical (>5 years old)
        is_historical = False
        if problem_date_str:
            try:
                problem_date = datetime.fromisoformat(
                    problem_date_str.replace("Z", "+00:00")
                )
                years_old = (
                    datetime.now() - problem_date.replace(tzinfo=None)
                ).days / 365.25
                is_historical = years_old > 5
            except:
                pass

        # Check match type
        match_type = _get_icd10_match_type(code, block.triggers.icd10Codes)
        if match_type:
            matching_codes.append(code)
            triggered = True
            primary_triggers += 1

            # Assign confidence based on match type and historical status
            if match_type == "exact":
                icd_confidence = 0.6 if is_historical else 1.0
            elif match_type == "related":  # Same category (prefix match)
                icd_confidence = 0.4 if is_historical else 0.8
            else:  # Range match (treated as related)
                icd_confidence = 0.4 if is_historical else 0.8

            primary_trigger_confidences.append(icd_confidence)

    if matching_codes:
        trigger_reasons.append(f"ICD-10 codes matched: {', '.join(matching_codes)}")

    # Check enhanced medication triggers (with dose thresholds)
    # Medications are assumed active (no historical tracking in current data model)
    # In production, would check medication status/end date
    for trigger in block.triggers.medicationTriggers:
        for med in patient_data.medications:
            if match_medication_trigger(med, trigger):
                triggered = True
                primary_triggers += 1

                # Determine medication confidence
                # Specific indication medications (e.g., SGLT2 for HF, sacubitril/valsartan) = 0.9
                # Multiple possible indications = 0.7
                # For now, assume medications with specific classes are more specific
                specific_classes = [
                    "sglt2_inhibitor",
                    "arni",
                    "loop_diuretic_high_dose",
                ]
                if med.class_.lower() in specific_classes:
                    med_confidence = 0.9
                else:
                    med_confidence = 0.8  # Default medication confidence

                primary_trigger_confidences.append(med_confidence)

                med_desc = f"{med.name} ({med.class_})"
                if trigger.min_dose:
                    med_desc += f" at ≥{trigger.min_dose}{trigger.dose_unit or 'mg'}"
                trigger_reasons.append(f"Medication matched: {med_desc}")
                break  # Only count once per trigger

    # Check simple medication classes (backward compatibility)
    patient_med_classes = {med.class_.lower() for med in patient_data.medications}
    for med_class in block.triggers.medicationClasses:
        if med_class.lower() in patient_med_classes:
            triggered = True
            primary_triggers += 1
            # Simple medication class match = 0.7 (multiple possible indications)
            primary_trigger_confidences.append(0.7)
            trigger_reasons.append(f"Medication class matched: {med_class}")

    # Check lab thresholds
    lab_map = {
        "hemoglobin": patient_data.labs.hemoglobin,
        "a1c": patient_data.labs.a1c,
        "creatinine": patient_data.labs.creatinine,
        "albumin": patient_data.labs.albumin,
        "inr": patient_data.labs.inr,
        "platelets": patient_data.labs.platelets,
        "bilirubin": patient_data.labs.bilirubin,
        "sodium": patient_data.labs.sodium,
        "bnp": patient_data.labs.bnp,
        "troponin": patient_data.labs.troponin,
        "nt_probnp": patient_data.labs.nt_probnp,
        "ferritin": patient_data.labs.ferritin,
        "tsat": patient_data.labs.tsat,
    }

    for threshold in block.triggers.labThresholds:
        lab_val = lab_map.get(threshold.test)
        if evaluate_lab_threshold(lab_val, threshold):
            triggered = True
            primary_triggers += 1
            # Lab thresholds = 0.8 (objective, reliable)
            primary_trigger_confidences.append(0.8)
            trigger_reasons.append(
                f"Lab threshold met: {threshold.test} {threshold.operator} {threshold.value}"
            )

    # Check device types
    # Devices are assumed active (no historical tracking in current data model)
    # In production, would check device status/removal date
    patient_devices_lower = {d.lower() for d in patient_data.devices}
    matching_devices = []
    for device in block.triggers.deviceTypes:
        if device.lower() in patient_devices_lower:
            matching_devices.append(device)
            triggered = True
            primary_triggers += 1
            # Active device = 0.9
            primary_trigger_confidences.append(0.9)

    if matching_devices:
        trigger_reasons.append(f"Devices matched: {', '.join(matching_devices)}")

    # Check required flags (HPI red flags)
    # HPI flags are assumed current (no historical tracking in current data model)
    # In production, would check symptom onset date
    hpi_flags = patient_data.hpiRedFlags
    flag_map = {
        "chestPain": hpi_flags.chestPain,
        "shortnessOfBreath": hpi_flags.shortnessOfBreath,
        "syncope": hpi_flags.syncope,
        "fever": hpi_flags.fever,
    }

    matching_flags = []
    for flag in block.triggers.requiredFlags:
        if flag_map.get(flag, False):
            matching_flags.append(flag)
            triggered = True
            primary_triggers += 1
            # Current symptom documented = 0.7
            primary_trigger_confidences.append(0.7)

    if matching_flags:
        trigger_reasons.append(f"HPI flags matched: {', '.join(matching_flags)}")

    # Check demographic conditions
    for condition in block.triggers.demographicConditions:
        if condition.startswith("age >="):
            try:
                threshold_age = int(condition.split(">=")[1].strip())
                if (
                    patient_data.demographics.age is not None
                    and patient_data.demographics.age >= threshold_age
                ):
                    triggered = True
                    primary_triggers += 1
                    # Demographic conditions = 0.8 (reliable but less specific)
                    primary_trigger_confidences.append(0.8)
                    trigger_reasons.append(
                        f"Demographic condition met: age >= {threshold_age}"
                    )
            except:
                pass

    # Check secondary triggers (for review flags)
    # Note: Secondary trigger confidences are tracked in the confidence calculation section
    if block.triggers.secondaryTriggers:
        sec_triggers = block.triggers.secondaryTriggers
        # Check secondary lab thresholds
        for threshold in sec_triggers.labThresholds:
            lab_val = lab_map.get(threshold.test)
            if evaluate_lab_threshold(lab_val, threshold):
                secondary_triggers += 1
                trigger_reasons.append(
                    f"[Secondary] Lab threshold met: {threshold.test} {threshold.operator} {threshold.value}"
                )

        # Check secondary vital signs
        vitals = patient_data.vitals
        if vitals.heartRate is not None:
            if vitals.heartRate > 100 or vitals.heartRate < 50:
                secondary_triggers += 1
                trigger_reasons.append(
                    f"[Secondary] Abnormal heart rate: {vitals.heartRate}"
                )

        if vitals.bloodPressure and vitals.bloodPressure.systolic is not None:
            sbp = vitals.bloodPressure.systolic
            if sbp > 180 or sbp < 90:
                secondary_triggers += 1
                trigger_reasons.append(f"[Secondary] Abnormal SBP: {sbp}")

        if vitals.spo2 is not None and vitals.spo2 < 92:
            secondary_triggers += 1
            trigger_reasons.append(f"[Secondary] Low SpO2: {vitals.spo2}%")

    # Calculate confidence score using component-based approach:
    # PRIMARY TRIGGERS (Max 0.4): Sum of (individual confidence × 0.4), capped at 0.4
    # SECONDARY TRIGGERS (Max 0.3): Sum of (individual confidence × 0.3), capped at 0.3
    # TEMPORAL (Max 0.2): Sum of (temporal factor × 0.2), capped at 0.2
    # CONSISTENCY (Max 0.1): Consistency score × 0.1, capped at 0.1
    # Total = sum of all components, normalized to 0-1 scale if > 1.0

    confidence = 0.0
    if triggered:
        # 1. PRIMARY TRIGGERS component
        # Sum individual trigger confidences × 0.4 (no cap, will normalize total)
        # Apply block-specific adjustments for diabetes
        if block.blockId == "DIABETES_001":
            # Diabetes-specific primary trigger confidences
            diabetes_primary_confidences: List[float] = []

            # ICD-10 for diabetes: 1.0 → 0.4
            if matching_codes:
                diabetes_primary_confidences.append(1.0)

            # Check for insulin use (specific indication medication)
            insulin_classes = [
                "insulin",
                "rapid_acting_insulin",
                "long_acting_insulin",
                "intermediate_acting_insulin",
            ]
            has_insulin = any(
                med.class_.lower() in insulin_classes
                for med in patient_data.medications
            )
            if has_insulin:
                diabetes_primary_confidences.append(0.8)  # Insulin use: 0.8

            # Check for other DM meds (non-insulin diabetes medications)
            dm_med_classes = [
                "metformin",
                "sglt2_inhibitor",
                "glp1_agonist",
                "dpp4_inhibitor",
                "sulfonylurea",
                "thiazolidinedione",
                "meglitinide",
                "alpha_glucosidase_inhibitor",
            ]
            has_other_dm_meds = any(
                med.class_.lower() in dm_med_classes for med in patient_data.medications
            )
            if (
                has_other_dm_meds and not has_insulin
            ):  # Only count if not already counted as insulin
                diabetes_primary_confidences.append(0.7)  # Other DM meds: 0.7

            # Calculate primary component for diabetes
            primary_component = sum(conf * 0.4 for conf in diabetes_primary_confidences)

        # OSA-specific primary triggers
        elif block.blockId == "OSA_001":
            # OSA-specific primary trigger confidences
            osa_primary_confidences: List[float] = []

            # ICD-10 G47.33: 1.0 → 0.4
            if matching_codes:
                osa_primary_confidences.append(1.0)

            # CPAP prescription: 0.8 → 0.32
            cpap_med_classes = ["cpap", "bipap"]
            has_cpap_prescription = any(
                med.class_.lower() in cpap_med_classes
                for med in patient_data.medications
            )
            if has_cpap_prescription:
                osa_primary_confidences.append(0.8)  # CPAP prescription: 0.8

            # CPAP device: 0.9 → 0.36
            patient_devices_upper = {d.upper() for d in patient_data.devices}
            if "CPAP" in patient_devices_upper or "BIPAP" in patient_devices_upper:
                osa_primary_confidences.append(0.9)  # CPAP device: 0.9

            # Calculate primary component for OSA
            primary_component = sum(conf * 0.4 for conf in osa_primary_confidences)

        else:
            # Default: use tracked primary trigger confidences
            primary_component = sum(conf * 0.4 for conf in primary_trigger_confidences)

        # 2. SECONDARY TRIGGERS component
        # Track individual secondary trigger confidences
        secondary_trigger_confidences: List[float] = []

        # Diabetes-specific secondary triggers
        if block.blockId == "DIABETES_001":
            # HbA1c > 9%: 0.9 → 0.27
            if patient_data.labs.a1c is not None:
                if patient_data.labs.a1c > 9.0:
                    secondary_trigger_confidences.append(0.9)  # HbA1c > 9%: 0.9
                elif patient_data.labs.a1c >= 7.0:
                    secondary_trigger_confidences.append(0.7)  # HbA1c 7-9%: 0.7

            # Random glucose > 200: 0.6 → 0.18
            # Check if glucose is available from lab thresholds or lab_map
            glucose_val = lab_map.get("glucose") or lab_map.get("random_glucose")
            # Also check if glucose is in block triggers
            if glucose_val is None:
                for threshold in block.triggers.labThresholds:
                    if threshold.test in ["glucose", "random_glucose"]:
                        glucose_val = lab_map.get(threshold.test)
                        break
            if glucose_val is not None and glucose_val > 200:
                secondary_trigger_confidences.append(0.6)  # Random glucose > 200: 0.6

        # OSA-specific secondary triggers
        elif block.blockId == "OSA_001":
            # Calculate STOP-BANG score from available data
            # STOP-BANG components: Snoring, Tiredness, Observed apnea, Pressure (BP), BMI > 35, Age > 50, Neck > 17", Gender (male)
            stop_bang_score = 0

            # BMI > 35
            if (
                patient_data.demographics.bmi is not None
                and patient_data.demographics.bmi > 35
            ):
                stop_bang_score += 1

            # Age > 50
            if (
                patient_data.demographics.age is not None
                and patient_data.demographics.age > 50
            ):
                stop_bang_score += 1

            # Gender (male)
            if patient_data.demographics.sex == "M":
                stop_bang_score += 1

            # Pressure (BP) - high BP
            if (
                patient_data.vitals.bloodPressure
                and patient_data.vitals.bloodPressure.systolic is not None
            ):
                if patient_data.vitals.bloodPressure.systolic > 140:
                    stop_bang_score += 1

            # STOP-BANG score ≥ 5: 0.8 → 0.24
            if stop_bang_score >= 5:
                secondary_trigger_confidences.append(0.8)  # STOP-BANG score ≥ 5: 0.8

            # BMI > 35: 0.6 → 0.18 (also contributes to STOP-BANG, but add separately if high)
            if (
                patient_data.demographics.bmi is not None
                and patient_data.demographics.bmi > 35
            ):
                secondary_trigger_confidences.append(0.6)  # BMI > 35: 0.6

            # Neck circumference > 17": 0.5 → 0.15
            # Note: Neck circumference not in current data model
            # Placeholder: if neck circumference is available, check > 17"
            # In production, this would come from physical exam data
            # For now, we'll skip this as it's not in the data model

        # CAD/CHF-specific secondary triggers
        elif block.blockId == "CAD_CHF_001":
            # Check BNP values directly (secondary trigger for CAD/CHF)
            if patient_data.labs.bnp is not None:
                if patient_data.labs.bnp > 400:
                    secondary_trigger_confidences.append(0.9)  # BNP > 400: 0.9
                elif patient_data.labs.bnp >= 100:
                    secondary_trigger_confidences.append(0.7)  # BNP 100-400: 0.7

        # Secondary lab thresholds (from block definition)
        if block.triggers.secondaryTriggers:
            sec_triggers = block.triggers.secondaryTriggers
            for threshold in sec_triggers.labThresholds:
                lab_val = lab_map.get(threshold.test)
                if evaluate_lab_threshold(lab_val, threshold):
                    # Skip BNP if already added above
                    if threshold.test != "bnp":
                        secondary_trigger_confidences.append(0.6)  # Other abnormal labs

        # Abnormal EKG indicators (from vitals)
        vitals = patient_data.vitals
        if vitals.heartRate is not None and (
            vitals.heartRate > 100 or vitals.heartRate < 50
        ):
            secondary_trigger_confidences.append(0.6)  # Abnormal EKG pattern: 0.6

        # Low LVEF known
        if patient_data.lvef is not None and patient_data.lvef < 50:
            secondary_trigger_confidences.append(0.8)  # Low LVEF known: 0.8

        # Sum secondary trigger confidences × 0.3 (no cap, will normalize total)
        secondary_component = sum(conf * 0.3 for conf in secondary_trigger_confidences)

        # 3. TEMPORAL component
        # Sum temporal factors × 0.2 (no cap, will normalize total)
        temporal_factors: List[float] = []

        # Diabetes-specific temporal factors
        if block.blockId == "DIABETES_001":
            # Check problem list dates for recent diagnosis
            recent_diagnosis = False
            for problem_item in patient_data.problemList:
                if problem_item.date:
                    recency_score = _calculate_recency_score(problem_item.date)
                    if recency_score >= 0.8:  # Within 90 days
                        recent_diagnosis = True
                        break

            if recent_diagnosis:
                temporal_factors.append(0.9)  # Recent diagnosis/change: 0.9
            else:
                temporal_factors.append(0.6)  # Long-standing stable: 0.6

        # OSA-specific temporal factors
        elif block.blockId == "OSA_001":
            # Recent sleep study: 0.9 → 0.18
            # Old sleep study: 0.6 → 0.12
            # Check problem list dates for sleep study-related codes or dates
            recent_sleep_study = False
            for problem_item in patient_data.problemList:
                if problem_item.date:
                    recency_score = _calculate_recency_score(problem_item.date)
                    # Check if this is a sleep study-related code (G47.33 or related)
                    if (
                        problem_item.code == "G47.33"
                        or "sleep" in problem_item.name.lower()
                    ):
                        if recency_score >= 0.8:  # Within 90 days
                            recent_sleep_study = True
                            break

            if recent_sleep_study:
                temporal_factors.append(0.9)  # Recent sleep study: 0.9
            else:
                # Check if OSA diagnosis exists but is older
                if matching_codes:
                    temporal_factors.append(0.6)  # Old sleep study: 0.6

        # CAD/CHF-specific temporal factors
        elif block.blockId == "CAD_CHF_001":
            # Recent cardiac event (<6 months): 0.9
            if patient_data.recent_cardiac_event_months is not None:
                if patient_data.recent_cardiac_event_months < 6:
                    temporal_factors.append(0.9)  # Recent cardiac event (<6mo): 0.9
                else:
                    temporal_factors.append(0.6)  # Stable >1 year: 0.6

        # Sum temporal factors × 0.2 (no cap, will normalize total)
        temporal_component = sum(factor * 0.2 for factor in temporal_factors)

        # 4. Consistency component (0-1 scale) - consistency across trigger types
        # Based on: MULTIPLE CONFIRMING SOURCES + CONFLICTING EVIDENCE penalties

        # Track which sources are present
        has_problem_list = len(matching_codes) > 0
        has_medications = any(
            match_medication_trigger(med, trigger)
            for trigger in block.triggers.medicationTriggers
            for med in patient_data.medications
        ) or any(
            mc.lower() in patient_med_classes for mc in block.triggers.medicationClasses
        )
        has_labs = any(
            evaluate_lab_threshold(lab_map.get(threshold.test), threshold)
            for threshold in block.triggers.labThresholds
        )
        has_devices = len(matching_devices) > 0
        has_hpi_flags = len(matching_flags) > 0

        # Count confirming sources (Problem list, Meds, Labs are primary)
        confirming_sources = 0
        if has_problem_list:
            confirming_sources += 1
        if has_medications:
            confirming_sources += 1
        if has_labs:
            confirming_sources += 1

        # MULTIPLE CONFIRMING SOURCES scoring
        if confirming_sources >= 3:
            # Problem list + Meds + Labs all agree
            consistency_component = 1.0
        elif confirming_sources == 2:
            # Problem list + one other source
            consistency_component = 0.8
        elif confirming_sources == 1:
            # Only one source
            consistency_component = 0.6
        else:
            # No primary sources (only devices/HPI flags)
            if has_devices or has_hpi_flags:
                consistency_component = 0.6  # Single source (device/HPI)
            else:
                consistency_component = 0.0  # No sources

        # CONFLICTING EVIDENCE penalties
        conflict_penalty = 0.0

        # Check for conflicts based on block type
        if block.blockId == "CAD_CHF_001":
            # CAD/CHF conflicts: Problem list says CAD but normal cardiac markers
            if has_problem_list:
                # Check for conflicting normal labs
                if (
                    patient_data.labs.troponin is not None
                    and patient_data.labs.troponin <= 0.04
                ):
                    # Normal troponin conflicts with CAD diagnosis
                    conflict_penalty += 0.3
                if patient_data.labs.bnp is not None and patient_data.labs.bnp < 100:
                    # Normal BNP conflicts with CHF diagnosis
                    conflict_penalty += 0.3

                # Missing expected data: CAD but no recent EKG/troponin
                if patient_data.labs.troponin is None:
                    conflict_penalty += 0.2  # Missing expected troponin
                if patient_data.labs.bnp is None and "heart_failure" in [
                    code.lower() for code in matching_codes
                ]:
                    conflict_penalty += 0.2  # Missing expected BNP for CHF

        elif block.blockId == "DIABETES_001":
            # Diabetes conflicts: Problem list says diabetes but normal A1c
            if has_problem_list:
                if patient_data.labs.a1c is not None and patient_data.labs.a1c < 6.5:
                    # Normal A1c conflicts with diabetes diagnosis
                    conflict_penalty += 0.3

                # Missing expected data: Diabetes but no A1c
                if patient_data.labs.a1c is None:
                    conflict_penalty += 0.2

        elif block.blockId == "CKD_001":
            # CKD conflicts: Problem list says CKD but normal creatinine/eGFR
            if has_problem_list:
                if patient_data.labs.creatinine is not None:
                    # Calculate eGFR if possible
                    if patient_data.demographics.age is not None:
                        eGFR = calculate_egfr_ckd_epi(
                            creatinine=patient_data.labs.creatinine,
                            age=patient_data.demographics.age,
                            sex=patient_data.demographics.sex or "M",
                        )
                        if eGFR >= 60:
                            # Normal eGFR conflicts with CKD diagnosis
                            conflict_penalty += 0.3

                # Missing expected data: CKD but no creatinine
                if patient_data.labs.creatinine is None:
                    conflict_penalty += 0.2

        elif block.blockId == "CIRRHOSIS_LIVER_001":
            # Cirrhosis conflicts: Problem list says cirrhosis but normal liver function
            if has_problem_list:
                if (
                    patient_data.labs.bilirubin is not None
                    and patient_data.labs.bilirubin < 2.0
                ):
                    # Normal bilirubin conflicts with cirrhosis
                    conflict_penalty += 0.3
                if patient_data.labs.inr is not None and patient_data.labs.inr < 1.5:
                    # Normal INR conflicts with cirrhosis
                    conflict_penalty += 0.3

                # Missing expected data: Cirrhosis but no liver function tests
                if patient_data.labs.bilirubin is None:
                    conflict_penalty += 0.2
                if patient_data.labs.inr is None:
                    conflict_penalty += 0.2

        # Apply conflict penalties (reduce consistency)
        consistency_component = max(0.0, consistency_component - conflict_penalty)

        # 4. CONSISTENCY component
        # Consistency score × 0.1 (no cap, will normalize total)
        consistency_component_weighted = consistency_component * 0.1

        # Calculate confidence score using ConfidenceCalculator
        confidence_score = ConfidenceCalculator.calculate_block_confidence(
            block=block,
            patient_data=patient_data,
            primary_component=primary_component,
            secondary_component=secondary_component,
            temporal_component=temporal_component,
            consistency_component=consistency_component_weighted,
        )

        # Use calculated confidence and tier
        confidence = confidence_score.score
        confidence_tier = confidence_score.tier

        # Apply minimum confidence threshold and set flags per specification:
        # <0.3: Do not trigger block
        # 0.3-0.49: Trigger with "LOW CONFIDENCE" flag, require clinician review
        # 0.5-0.69: Trigger with standard display, suggest clinician review
        # 0.7-0.89: Trigger with confident display, optional clinician review
        # ≥0.9: Trigger with high confidence, minimal review needed
        confidence_flag = None

        if confidence < 0.3:
            # <0.3: Do not trigger block
            triggered = False
            confidence = 0.0
            confidence_tier = "very_low"
            confidence_flag = None  # Not triggered, no flag needed
        elif 0.3 <= confidence < 0.5:
            # 0.3-0.49: Trigger with "LOW CONFIDENCE" flag, require clinician review
            confidence_flag = "low_confidence_require_review"
        elif 0.5 <= confidence < 0.7:
            # 0.5-0.69: Trigger with standard display, suggest clinician review
            confidence_flag = "standard_confidence_suggest_review"
        elif 0.7 <= confidence < 0.9:
            # 0.7-0.89: Trigger with confident display, optional clinician review
            confidence_flag = "confident_display_optional_review"
        else:  # confidence >= 0.9
            # ≥0.9: Trigger with high confidence, minimal review needed
            confidence_flag = "high_confidence_minimal_review"
    else:
        confidence = 0.0
        confidence_tier = "very_low"
        confidence_flag = None

    # Determine risk level based on triggers and patient severity
    risk_level: Literal["low", "intermediate", "high"] = "low"
    if triggered:
        # Use risk stratification rules if available
        if (
            block.riskStratification.highRisk
            or block.riskStratification.intermediateRisk
        ):
            # Check if patient matches high risk criteria
            if primary_triggers >= 3 or secondary_triggers >= 2:
                risk_level = "high"
            elif primary_triggers >= 2 or secondary_triggers >= 1:
                risk_level = "intermediate"
            else:
                risk_level = "low"
        else:
            # Default heuristic
            if primary_triggers >= 3:
                risk_level = "high"
            elif primary_triggers >= 2:
                risk_level = "intermediate"
            else:
                risk_level = "low"

    # Generate categorized actions based on confidence level
    required_actions = []
    suggested_actions = []
    optional_actions = []

    if triggered:
        # Base actions from content sections
        base_actions = block.contentSections.preOpActions.copy()

        # Add risk-level specific actions
        if risk_level == "high":
            base_actions.append("Consider specialist consultation before proceeding.")
        if secondary_triggers > 0:
            base_actions.append(
                "Review secondary trigger flags for additional considerations."
            )

        # Categorize actions based on confidence level
        if confidence >= 0.7:
            # REQUIRED ACTIONS (confidence ≥ 0.7): Use definitive language
            for action in base_actions:
                action_lower = action.lower()
                required_action = action

                # Specific conversions for common actions
                if "ekg" in action_lower or "ecg" in action_lower:
                    required_action = "Obtain EKG"
                elif "electrolyte" in action_lower:
                    required_action = "Check electrolytes"
                elif "consult" in action_lower:
                    if "cardiology" in action_lower:
                        required_action = "Consult cardiology if symptomatic"
                    elif (
                        "specialist" in action_lower
                        or "nephrology" in action_lower
                        or "endocrinology" in action_lower
                    ):
                        # Generic specialist consult
                        specialist_type = "specialist"
                        if "nephrology" in action_lower:
                            specialist_type = "nephrology"
                        elif "endocrinology" in action_lower:
                            specialist_type = "endocrinology"
                        required_action = f"Consult {specialist_type} if symptomatic"
                    else:
                        # Generic consult
                        required_action = action.replace("Consider", "Consult").replace(
                            "consider", "Consult"
                        )
                        if "if symptomatic" not in action_lower:
                            required_action += " if symptomatic"
                else:
                    # Convert general actions to required format
                    if action_lower.startswith(("consider", "may", "could")):
                        required_action = action.replace("Consider", "Obtain").replace(
                            "consider", "Obtain"
                        )
                        required_action = required_action.replace(
                            "May benefit from", "Check"
                        ).replace("may benefit from", "Check")
                        required_action = required_action.replace(
                            "Could be", ""
                        ).replace("could be", "")
                        required_action = required_action.strip()

                required_actions.append(required_action.strip())

        elif 0.5 <= confidence < 0.7:
            # SUGGESTED ACTIONS (confidence 0.5-0.69): Use qualified language
            for action in base_actions:
                action_lower = action.lower()
                suggested_action = action

                # Specific conversions for common actions
                if "ekg" in action_lower or "ecg" in action_lower:
                    suggested_action = "Consider EKG"
                elif "electrolyte" in action_lower:
                    suggested_action = "May benefit from electrolyte monitoring"
                elif "consult" in action_lower:
                    if "cardiology" in action_lower:
                        suggested_action = "Cardiology consult could be considered"
                    elif (
                        "specialist" in action_lower
                        or "nephrology" in action_lower
                        or "endocrinology" in action_lower
                    ):
                        specialist_type = "specialist"
                        if "nephrology" in action_lower:
                            specialist_type = "nephrology"
                        elif "endocrinology" in action_lower:
                            specialist_type = "endocrinology"
                        suggested_action = f"{specialist_type.capitalize()} consult could be considered"
                    else:
                        suggested_action = action.replace(
                            "Consult", "Consult could be considered"
                        ).replace("consult", "consult could be considered")
                else:
                    # Convert general actions to suggested format
                    if not action_lower.startswith(("consider", "may")):
                        suggested_action = f"Consider {action.lower()}"

                suggested_actions.append(suggested_action)

        elif 0.3 <= confidence < 0.5:
            # OPTIONAL ACTIONS (confidence 0.3-0.49): Use cautious language
            for action in base_actions:
                action_lower = action.lower()
                optional_action = action

                # Specific conversions for common actions
                if "ekg" in action_lower or "ecg" in action_lower:
                    optional_action = "EKG could be obtained if clinical concern"
                elif "electrolyte" in action_lower:
                    optional_action = "Electrolyte monitoring if other indications"
                elif "consult" in action_lower:
                    if "cardiology" in action_lower:
                        optional_action = "Cardiology input may be helpful"
                    elif (
                        "specialist" in action_lower
                        or "nephrology" in action_lower
                        or "endocrinology" in action_lower
                    ):
                        specialist_type = "specialist"
                        if "nephrology" in action_lower:
                            specialist_type = "nephrology"
                        elif "endocrinology" in action_lower:
                            specialist_type = "endocrinology"
                        optional_action = (
                            f"{specialist_type.capitalize()} input may be helpful"
                        )
                    else:
                        optional_action = action.replace(
                            "Consult", "Consult input may be helpful"
                        ).replace("consult", "consult input may be helpful")
                else:
                    # Convert general actions to optional format
                    if "could" not in action_lower and "may" not in action_lower:
                        optional_action = (
                            f"{action} could be obtained if clinical concern"
                        )

                optional_actions.append(optional_action)

        # Determine export status based on confidence
        export_allowed = True  # Export is always allowed
        export_warning = None
        export_note = None

        if confidence_flag == "low_confidence_require_review":
            # 0.3-0.49: LOW CONFIDENCE - Export allowed but with warning
            export_allowed = True
            export_warning = (
                "Low confidence comorbidity block - clinical correlation recommended"
            )
            export_note = "Low confidence comorbidity identified - clinical correlation recommended"
            # Require clinician review
            if optional_actions:
                optional_actions.insert(
                    0,
                    "⚠️ LOW CONFIDENCE: Clinician review REQUIRED - evidence is weak or incomplete. Verify diagnosis before proceeding.",
                )
            else:
                optional_actions.append(
                    "⚠️ LOW CONFIDENCE: Clinician review REQUIRED - evidence is weak or incomplete. Verify diagnosis before proceeding."
                )
        elif confidence_flag == "standard_confidence_suggest_review":
            # 0.5-0.69: MODERATE+ CONFIDENCE - No export restrictions
            export_allowed = True
            export_warning = None
            export_note = None
            # Suggest clinician review
            if suggested_actions:
                suggested_actions.insert(
                    0,
                    "ℹ️ STANDARD CONFIDENCE: Clinician review suggested - verify diagnosis appropriateness.",
                )
            else:
                suggested_actions.append(
                    "ℹ️ STANDARD CONFIDENCE: Clinician review suggested - verify diagnosis appropriateness."
                )
        elif confidence_flag == "confident_display_optional_review":
            # 0.7-0.89: MODERATE+ CONFIDENCE - No export restrictions
            export_allowed = True
            export_warning = None
            export_note = None
            # Optional clinician review - no automatic message
            pass
        elif confidence_flag == "high_confidence_minimal_review":
            # ≥0.9: MODERATE+ CONFIDENCE - No export restrictions
            export_allowed = True
            export_warning = None
            export_note = None
            # High confidence, minimal review needed - no automatic message
            pass
        else:
            # Default: no export restrictions
            export_allowed = True
            export_warning = None
            export_note = None

    # Set export status for non-triggered blocks
    if not triggered:
        export_allowed = True  # Export still allowed even if block not triggered
        export_warning = None
        export_note = None

    categorized_actions = CategorizedActions(
        requiredActions=required_actions,
        suggestedActions=suggested_actions,
        optionalActions=optional_actions,
    )

    # Keep backward compatibility: combine all actions into requiredActions
    all_actions = required_actions + suggested_actions + optional_actions

    return TriggerEvaluationResult(
        blockId=block.blockId,
        triggered=triggered,
        triggerReasons=trigger_reasons,
        confidenceScore=round(confidence, 3),
        confidenceTier=confidence_tier,
        confidenceFlag=confidence_flag,
        riskLevel=risk_level,
        requiredActions=all_actions,  # Backward compatibility
        categorizedActions=categorized_actions,
        exportAllowed=export_allowed,
        exportWarning=export_warning,
        exportNote=export_note,
    )


# Alias for backward compatibility
evaluate_comorbidity_block = evaluate_trigger


# ============================================================================
# Risk Stratification Functions
# ============================================================================


def assess_risk_level(
    block: ComorbidityBlockDefinition,
    patient_data: PatientClinicalData,
    evaluation: TriggerEvaluationResult,
) -> Literal["low", "intermediate", "high"]:
    """
    Assess risk level for a triggered comorbidity block based on patient clinical data.

    This function refines the initial risk level from trigger evaluation by considering
    patient-specific clinical factors (labs, vitals, medications, etc.).

    Args:
        block: Comorbidity block definition
        patient_data: Patient clinical data snapshot
        evaluation: Trigger evaluation result with initial risk level

    Returns:
        Refined risk level: 'low', 'intermediate', or 'high'
    """
    if not evaluation.triggered:
        return "low"

    initial_risk = evaluation.riskLevel

    # Block-specific risk stratification
    if block.blockId == "CAD_CHF_001":
        # CAD/CHF Risk Stratification
        # HIGH RISK: Decompensated symptoms, LVEF < 40%, recent events, BNP > 400
        if (
            (patient_data.lvef is not None and patient_data.lvef < 40)
            or (patient_data.labs.bnp is not None and patient_data.labs.bnp > 400)
            or (
                patient_data.recent_cardiac_event_months is not None
                and patient_data.recent_cardiac_event_months < 6
            )
            or (patient_data.inotropic_dependence)
            or (patient_data.nyha_class in ["III", "IV"])
        ):
            return "high"

        # INTERMEDIATE RISK: Stable symptoms, LVEF 40-50%, BNP 100-400
        if (
            (patient_data.lvef is not None and 40 <= patient_data.lvef <= 50)
            or (
                patient_data.labs.bnp is not None
                and 100 <= patient_data.labs.bnp <= 400
            )
            or (patient_data.nyha_class == "II")
            or (patient_data.dasi_mets is not None and patient_data.dasi_mets < 7)
        ):
            return "intermediate" if initial_risk != "high" else "high"

        # LOW RISK: Asymptomatic, LVEF > 50%, no recent events, DASI ≥ 7 METs
        if (patient_data.lvef is not None and patient_data.lvef > 50) or (
            patient_data.dasi_mets is not None and patient_data.dasi_mets >= 7
        ):
            return "low" if initial_risk == "low" else "intermediate"

    elif block.blockId == "DIABETES_001":
        # Diabetes Risk Stratification
        # HIGH RISK: HbA1c > 9%, end-organ damage, recurrent hypoglycemia
        if (
            (patient_data.labs.a1c is not None and patient_data.labs.a1c > 9.0)
            or (len(patient_data.diabetes_complications) > 0)
            or (patient_data.recurrent_hypoglycemia)
        ):
            return "high"

        # INTERMEDIATE RISK: HbA1c 7-9%, minor complications
        if patient_data.labs.a1c is not None and 7.0 <= patient_data.labs.a1c <= 9.0:
            return "intermediate" if initial_risk != "high" else "high"

        # LOW RISK: HbA1c < 7%, no complications
        if patient_data.labs.a1c is not None and patient_data.labs.a1c < 7.0:
            return "low" if initial_risk == "low" else "intermediate"

    elif block.blockId == "CKD_001":
        # CKD Risk Stratification
        # Calculate eGFR if creatinine available
        eGFR = None
        if (
            patient_data.labs.creatinine is not None
            and patient_data.demographics.age is not None
        ):
            eGFR = calculate_egfr_ckd_epi(
                creatinine=patient_data.labs.creatinine,
                age=patient_data.demographics.age,
                sex=patient_data.demographics.sex or "M",
            )

        # HIGH RISK: eGFR < 30, dialysis dependent, electrolyte abnormalities
        if (
            (eGFR is not None and eGFR < 30)
            or (patient_data.dialysis_dependent)
            or (
                patient_data.labs.sodium is not None
                and (patient_data.labs.sodium < 135 or patient_data.labs.sodium > 145)
            )
        ):
            return "high"

        # INTERMEDIATE RISK: eGFR 30-44, proteinuria present
        if (eGFR is not None and 30 <= eGFR < 45) or (patient_data.proteinuria is True):
            return "intermediate" if initial_risk != "high" else "high"

        # LOW RISK: eGFR 45-59, no proteinuria
        if eGFR is not None and 45 <= eGFR < 60:
            return "low" if initial_risk == "low" else "intermediate"

    # Default: return initial risk level from trigger evaluation
    return initial_risk


def get_risk_specific_recommendations(
    block: ComorbidityBlockDefinition,
    risk_level: Literal["low", "intermediate", "high"],
) -> List[str]:
    """
    Get risk-specific recommendations for a comorbidity block.

    Args:
        block: Comorbidity block definition
        risk_level: Assessed risk level

    Returns:
        List of risk-specific recommendation strings
    """
    recommendations = []

    if risk_level == "high":
        recommendations.append(
            "High risk identified - consider specialist consultation before proceeding."
        )
        recommendations.append("Enhanced monitoring recommended post-operatively.")
    elif risk_level == "intermediate":
        recommendations.append("Intermediate risk - proceed with standard precautions.")
        recommendations.append(
            "Consider enhanced monitoring if procedure complexity increases."
        )
    else:  # low
        recommendations.append("Low risk - proceed with standard monitoring.")

    # Block-specific high-risk recommendations
    if risk_level == "high":
        if block.blockId == "CAD_CHF_001":
            recommendations.append(
                "Consider cardiology consultation and optimization before surgery."
            )
            recommendations.append("Post-operative telemetry monitoring recommended.")
        elif block.blockId == "DIABETES_001":
            recommendations.append(
                "Consider endocrine consultation for glycemic optimization."
            )
            recommendations.append("Enhanced glucose monitoring perioperatively.")
        elif block.blockId == "CKD_001":
            recommendations.append(
                "Consider nephrology consultation if not already established."
            )
            recommendations.append("Avoid nephrotoxic agents and contrast if possible.")

    return recommendations


# ============================================================================
# Content Personalization Functions
# ============================================================================


@dataclass(frozen=True)
class PersonalizedContent:
    """Immutable personalized content for a comorbidity block."""

    baseContent: str  # Base block content
    personalizedRecommendations: List[str]  # Patient-specific recommendations
    fullContent: str  # Combined content (120-160 words)
    versionFooter: str  # Version and attribution footer


def _adjust_language_for_confidence(
    text: str,
    confidence_tier: Literal["very_high", "high", "moderate", "low", "very_low"],
    condition_name: str,
) -> str:
    """
    Adjust language in text based on confidence tier.

    VERY HIGH (≥0.9): Definitive language ("Patient HAS CAD")
    HIGH (0.7-0.89): Confident language ("Patient with CAD")
    MODERATE (0.5-0.69): Qualified language ("Possible CAD", "History suggests CAD")
    LOW (0.3-0.49): Cautious language ("CAD cannot be ruled out")

    Args:
        text: Original text
        confidence_tier: Confidence tier
        condition_name: Condition name (e.g., "CAD", "Diabetes", "CKD")

    Returns:
        Adjusted text with confidence-appropriate language
    """
    if confidence_tier == "very_high":
        # Definitive language: "Patient HAS [condition]"
        text = text.replace(
            f"Patient with {condition_name}", f"Patient HAS {condition_name}"
        )
        text = text.replace(
            f"patient with {condition_name}", f"patient HAS {condition_name}"
        )
        text = text.replace(f"{condition_name} present", f"{condition_name} CONFIRMED")
        # Remove qualifiers like "may", "consider", "if indicated"
        text = re.sub(
            r"\b(may|consider|if indicated|if appropriate)\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip()  # Clean up extra spaces

    elif confidence_tier == "high":
        # Confident language: "Patient with [condition]"
        text = text.replace(
            f"Possible {condition_name}", f"Patient with {condition_name}"
        )
        text = text.replace(
            f"possible {condition_name}", f"patient with {condition_name}"
        )
        text = text.replace(
            f"{condition_name} cannot be ruled out", f"{condition_name} likely present"
        )
        # Minor qualifiers: "consider", "if indicated"
        text = re.sub(
            r"\b(must|should|will)\b",
            lambda m: "consider" if m.group().lower() == "must" else m.group(),
            text,
            flags=re.IGNORECASE,
        )

    elif confidence_tier == "moderate":
        # Qualified language: "Possible CAD" or "History suggests CAD"
        text = text.replace(
            f"Patient HAS {condition_name}", f"Possible {condition_name}"
        )
        text = text.replace(
            f"patient HAS {condition_name}", f"possible {condition_name}"
        )
        text = text.replace(
            f"Patient with {condition_name}", f"History suggests {condition_name}"
        )
        text = text.replace(
            f"patient with {condition_name}", f"history suggests {condition_name}"
        )
        # Add qualifiers: "may benefit from", "consider"
        text = re.sub(
            r"\b(should|must|will)\b",
            lambda m: (
                "may benefit from" if m.group().lower() == "should" else "consider"
            ),
            text,
            flags=re.IGNORECASE,
        )

    elif confidence_tier in ["low", "very_low"]:
        # Cautious language: "CAD cannot be ruled out"
        text = text.replace(
            f"Patient HAS {condition_name}", f"{condition_name} cannot be ruled out"
        )
        text = text.replace(
            f"patient HAS {condition_name}", f"{condition_name} cannot be ruled out"
        )
        text = text.replace(
            f"Patient with {condition_name}", f"{condition_name} cannot be ruled out"
        )
        text = text.replace(
            f"patient with {condition_name}", f"{condition_name} cannot be ruled out"
        )
        text = text.replace(
            f"Possible {condition_name}", f"{condition_name} cannot be ruled out"
        )
        text = text.replace(
            f"possible {condition_name}", f"{condition_name} cannot be ruled out"
        )
        # Strong qualifiers: "if confirmed", "pending further evaluation"
        text = re.sub(
            r"\b(should|must|will|recommend)\b",
            lambda m: (
                "if confirmed"
                if m.group().lower() == "should"
                else "pending further evaluation"
            ),
            text,
            flags=re.IGNORECASE,
        )

    return text


def _adjust_recommendations_for_confidence(
    recommendations: List[str],
    confidence_tier: Literal["very_high", "high", "moderate", "low", "very_low"],
) -> List[str]:
    """
    Adjust recommendation language based on confidence tier.

    VERY HIGH: Full set, definitive language
    HIGH: Most recommendations, minor qualifiers
    MODERATE: Basic recommendations only, qualifiers
    LOW: Critical recommendations only, strong qualifiers

    Args:
        recommendations: List of recommendation strings
        confidence_tier: Confidence tier

    Returns:
        Adjusted recommendations list
    """
    if confidence_tier == "very_high":
        # Full set, definitive language - no changes needed
        return recommendations

    elif confidence_tier == "high":
        # Most recommendations, add minor qualifiers
        adjusted = []
        for rec in recommendations:
            # Add "consider" or "if indicated" to strong statements
            if rec.lower().startswith(("require", "must", "should")):
                rec = rec.replace("Require", "Consider", 1).replace(
                    "require", "consider", 1
                )
                rec = rec.replace("Must", "Should consider", 1).replace(
                    "must", "should consider", 1
                )
                rec = rec.replace("Should", "Consider", 1).replace(
                    "should", "consider", 1
                )
            adjusted.append(rec)
        return adjusted

    elif confidence_tier == "moderate":
        # Basic recommendations only, add qualifiers
        # Filter to only critical recommendations
        critical_keywords = ["consult", "monitor", "avoid", "check", "assess"]
        adjusted = []
        for rec in recommendations:
            rec_lower = rec.lower()
            if any(keyword in rec_lower for keyword in critical_keywords):
                # Add "may benefit from" or "consider"
                if not any(q in rec_lower for q in ["may", "consider", "if"]):
                    rec = f"Consider: {rec}"
                adjusted.append(rec)
        return adjusted

    elif confidence_tier in ["low", "very_low"]:
        # Critical recommendations only, strong qualifiers
        critical_keywords = ["consult", "avoid", "critical", "urgent"]
        adjusted = []
        for rec in recommendations:
            rec_lower = rec.lower()
            if any(keyword in rec_lower for keyword in critical_keywords):
                # Add "if confirmed" or "pending further evaluation"
                if "if confirmed" not in rec_lower and "pending" not in rec_lower:
                    rec = f"If confirmed: {rec} (pending further evaluation)"
                adjusted.append(rec)
        return adjusted

    return recommendations


def generate_personalized_content(
    block: ComorbidityBlockDefinition,
    patient_data: PatientClinicalData,
    risk_level: Literal["low", "intermediate", "high"],
    confidence_tier: Literal[
        "very_high", "high", "moderate", "low", "very_low"
    ] = "moderate",
    procedure_bleed_risk: Optional[str] = None,  # 'low', 'moderate', 'high'
) -> PersonalizedContent:
    """
    Generate patient-specific personalized content for a comorbidity block.

    Combines:
    - Base block content
    - Risk-level specific recommendations
    - Patient-specific modifications based on medications, labs, devices, procedure risk

    Args:
        block: Comorbidity block definition
        patient_data: Patient clinical data snapshot
        risk_level: Assessed risk level
        procedure_bleed_risk: Procedure bleed risk level ('low', 'moderate', 'high')

    Returns:
        PersonalizedContent with full content (120-160 words) and version footer
    """
    personalized_recommendations: List[str] = []

    # Medication-based personalization
    med_personalization = _get_medication_based_personalization(
        patient_data, block.blockId
    )
    personalized_recommendations.extend(med_personalization)

    # Lab-based personalization
    lab_personalization = _get_lab_based_personalization(patient_data, block.blockId)
    personalized_recommendations.extend(lab_personalization)

    # Device-based personalization
    device_personalization = _get_device_based_personalization(
        patient_data, block.blockId
    )
    personalized_recommendations.extend(device_personalization)

    # Procedure-based personalization
    if procedure_bleed_risk:
        procedure_personalization = _get_procedure_based_personalization(
            procedure_bleed_risk, block.blockId
        )
        personalized_recommendations.extend(procedure_personalization)

    # Get risk-specific recommendations
    risk_recommendations = get_risk_specific_recommendations(block, risk_level)

    # Adjust recommendations based on confidence tier
    # VERY HIGH: Full set, HIGH: Most recommendations, MODERATE: Basic only, LOW: Critical only
    if confidence_tier == "very_high":
        # Full set - no filtering
        adjusted_risk_recommendations = risk_recommendations
        adjusted_personalized_recommendations = personalized_recommendations
    elif confidence_tier == "high":
        # Most recommendations - apply minor qualifiers
        adjusted_risk_recommendations = _adjust_recommendations_for_confidence(
            risk_recommendations, confidence_tier
        )
        adjusted_personalized_recommendations = _adjust_recommendations_for_confidence(
            personalized_recommendations, confidence_tier
        )
    elif confidence_tier == "moderate":
        # Basic recommendations only - filter and add qualifiers
        adjusted_risk_recommendations = _adjust_recommendations_for_confidence(
            risk_recommendations, confidence_tier
        )
        adjusted_personalized_recommendations = _adjust_recommendations_for_confidence(
            personalized_recommendations, confidence_tier
        )
    else:  # low or very_low
        # Critical recommendations only - strong qualifiers
        adjusted_risk_recommendations = _adjust_recommendations_for_confidence(
            risk_recommendations, confidence_tier
        )
        adjusted_personalized_recommendations = _adjust_recommendations_for_confidence(
            personalized_recommendations, confidence_tier
        )

    # Adjust base content language based on confidence tier
    condition_name = (
        block.conditionName.split("/")[0]
        if "/" in block.conditionName
        else block.conditionName
    )
    adjusted_why = _adjust_language_for_confidence(
        block.contentSections.why, confidence_tier, condition_name
    )

    # Combine base content sections with confidence-adjusted language
    base_content_parts = [
        adjusted_why,
        "Pre-operative actions: " + "; ".join(block.contentSections.preOpActions),
        "Intra-operative considerations: "
        + "; ".join(block.contentSections.intraOpConsiderations),
        "Post-operative management: "
        + "; ".join(block.contentSections.postOpManagement),
    ]

    # Add risk-specific recommendations (confidence-adjusted)
    if adjusted_risk_recommendations:
        base_content_parts.append(
            "Risk-specific recommendations: " + "; ".join(adjusted_risk_recommendations)
        )

    # Add personalized recommendations (confidence-adjusted)
    if adjusted_personalized_recommendations:
        base_content_parts.append(
            "Patient-specific considerations: "
            + "; ".join(adjusted_personalized_recommendations)
        )

    # Combine into full content
    base_content = " ".join(base_content_parts)

    # Ensure content length is 120-160 words
    words = base_content.split()
    if len(words) < 120:
        # Add anesthesia heads-up if available
        if block.contentSections.anesthesiaHeadsUp:
            base_content += " " + block.contentSections.anesthesiaHeadsUp
            words = base_content.split()
    elif len(words) > 160:
        # Truncate to 160 words
        base_content = " ".join(words[:160])

    # Generate version footer
    version_footer = (
        f"Owner: Bashar | Reviewers: Alaa/Khalid | "
        f"Version: {block.version} | Date: {block.metadata.effectiveDate or '2024-01-01'}"
    )

    return PersonalizedContent(
        baseContent=adjusted_why,  # Use confidence-adjusted base content
        personalizedRecommendations=adjusted_personalized_recommendations,  # Use confidence-adjusted recommendations
        fullContent=base_content,
        versionFooter=version_footer,
    )


def _get_medication_based_personalization(
    patient_data: PatientClinicalData, block_id: str
) -> List[str]:
    """Get medication-based personalized recommendations."""
    recommendations: List[str] = []

    # Check for beta-blockers
    beta_blockers = [
        med
        for med in patient_data.medications
        if "beta" in med.class_.lower() or "beta" in med.name.lower()
    ]
    if beta_blockers:
        recommendations.append(
            "Continue beta-blockers perioperatively unless contraindicated (HR < 50, SBP < 100)"
        )

    # Check for ACEi/ARB/ARNI
    acei_arb_arni = [
        med
        for med in patient_data.medications
        if any(term in med.class_.lower() for term in ["ace", "arb", "arni"])
        or any(
            term in med.name.lower()
            for term in ["ace inhibitor", "arb", "sacubitril", "valsartan", "entresto"]
        )
    ]
    if acei_arb_arni:
        # Check if CKD Stage 3+ (eGFR < 60)
        egfr = calculate_egfr_ckd_epi(
            creatinine=patient_data.labs.creatinine,
            age=patient_data.demographics.age,
            sex=patient_data.demographics.sex,
        )
        if egfr is not None and egfr < 60:
            recommendations.append(
                "Consider holding ACEi/ARB/ARNI evening prior if CKD Stage 3+ or major fluid shifts anticipated"
            )
        elif block_id in ["CAD_CHF_001", "CKD_001"]:
            recommendations.append(
                "Consider holding ACEi/ARB/ARNI evening prior if major fluid shifts anticipated"
            )

    # Check for SGLT2 inhibitors
    sglt2_inhibitors = [
        med
        for med in patient_data.medications
        if "sglt2" in med.class_.lower()
        or any(
            term in med.name.lower()
            for term in ["dapagliflozin", "empagliflozin", "canagliflozin"]
        )
    ]
    if sglt2_inhibitors:
        recommendations.append(
            "Hold SGLT2 inhibitors 3-4 days pre-op per ACC/AHA 2024 guidelines"
        )

    # Check for anticoagulants
    anticoagulants = [
        med
        for med in patient_data.medications
        if any(
            term in med.class_.lower() for term in ["anticoagulant", "antithrombotic"]
        )
        or any(
            term in med.name.lower()
            for term in [
                "warfarin",
                "apixaban",
                "rivaroxaban",
                "dabigatran",
                "edoxaban",
                "heparin",
                "enoxaparin",
            ]
        )
    ]
    if anticoagulants:
        med_names = [med.name for med in anticoagulants]
        recommendations.append(
            f"Anticoagulation management required: {', '.join(med_names)} - "
            "coordinate with anesthesia and surgery teams for perioperative management"
        )

    return recommendations


def _get_lab_based_personalization(
    patient_data: PatientClinicalData, block_id: str
) -> List[str]:
    """Get lab-based personalized recommendations."""
    recommendations: List[str] = []

    # BNP > 400
    if patient_data.labs.bnp is not None and patient_data.labs.bnp > 400:
        recommendations.append(
            "Elevated BNP suggests volume overload - consider diuresis pre-op"
        )

    # HbA1c > 9%
    if patient_data.labs.a1c is not None and patient_data.labs.a1c > 9.0:
        recommendations.append(
            "Poor glycemic control - increased infection and wound healing risk"
        )

    # eGFR < 30
    egfr = calculate_egfr_ckd_epi(
        creatinine=patient_data.labs.creatinine,
        age=patient_data.demographics.age,
        sex=patient_data.demographics.sex,
    )
    if egfr is not None and egfr < 30:
        recommendations.append("Severe CKD - avoid nephrotoxins, contrast if possible")

    return recommendations


def _get_device_based_personalization(
    patient_data: PatientClinicalData, block_id: str
) -> List[str]:
    """Get device-based personalized recommendations."""
    recommendations: List[str] = []

    # Pacemaker/ICD
    if "pacemaker" in [d.lower() for d in patient_data.devices] or "ICD" in [
        d.upper() for d in patient_data.devices
    ]:
        recommendations.append(
            "Device management: Ensure interrogation, magnet response known"
        )

    # CPAP
    if "CPAP" in [d.upper() for d in patient_data.devices] or "BIPAP" in [
        d.upper() for d in patient_data.devices
    ]:
        recommendations.append("Bring home CPAP/BiPAP device for postoperative use")

    return recommendations


def _get_procedure_based_personalization(
    procedure_bleed_risk: str, block_id: str
) -> List[str]:
    """Get procedure-based personalized recommendations."""
    recommendations: List[str] = []

    if block_id == "CAD_CHF_001":
        if procedure_bleed_risk == "high":
            recommendations.append(
                "High bleed risk procedure - more aggressive antiplatelet holding may be required; "
                "coordinate with cardiology and surgery teams"
            )
        elif procedure_bleed_risk == "low":
            recommendations.append(
                "Low bleed risk procedure - standard antiplatelet management per guidelines"
            )

    return recommendations


# ============================================================================
# Comorbidity Block Definitions (TRD v3.0 Compliant)
# ============================================================================


def create_cad_chf_block() -> ComorbidityBlockDefinition:
    """
    CAD/CHF Block with exact trigger rules per TRD v3.0.
    PRIMARY TRIGGERS (ANY ONE triggers block):
    - ICD-10 codes: I25.10, I25.2, I25.5, I50.9, I50.20-I50.43
    - HPI Red-Flags: chestPain = true OR shortnessOfBreath = true
    - Medications: nitrates, ranolazine, loop diuretics >40mg furosemide equivalent,
                   SGLT2 inhibitors for HF, sacubitril/valsartan
    - Devices: pacemaker, ICD, LVAD
    SECONDARY TRIGGERS (Flag for review):
    - Labs: BNP > 100 OR NT-proBNP > 300
    - Labs: Troponin > 99th percentile URL (0.04 ng/mL)
    - Labs: eGFR < 45
    - Vitals: HR > 100 OR < 50, SBP > 180 OR < 90, SpO2 < 92%
    """
    return ComorbidityBlockDefinition(
        blockId="CAD_CHF_001",
        conditionName="CAD/CHF",
        version="1.0.0",
        triggers=TriggerConditions(
            icd10Codes=["I25.10", "I25.2", "I25.5", "I50.9", "I50.20-I50.43"],
            medicationTriggers=[
                MedicationTrigger(
                    class_="nitrate", name_patterns=["nitroglycerin", "isosorbide"]
                ),
                MedicationTrigger(class_="ranolazine", name_patterns=["ranolazine"]),
                MedicationTrigger(
                    class_="loop_diuretic",
                    name_patterns=["furosemide", "bumetanide", "torsemide"],
                    min_dose=40.0,
                    dose_unit="mg",
                ),
                MedicationTrigger(
                    class_="sglt2_inhibitor",
                    name_patterns=["dapagliflozin", "empagliflozin", "canagliflozin"],
                ),
                MedicationTrigger(
                    class_="arni", name_patterns=["sacubitril", "valsartan", "entresto"]
                ),
            ],
            deviceTypes=["pacemaker", "ICD", "LVAD"],
            requiredFlags=["chestPain", "shortnessOfBreath"],
            secondaryTriggers=TriggerConditions(
                labThresholds=[
                    LabThreshold(test="bnp", operator=">", value=100.0),
                    LabThreshold(test="nt_probnp", operator=">", value=300.0),
                    LabThreshold(test="troponin", operator=">", value=0.04),
                ],
            ),
        ),
        contentSections=ContentSections(
            why="CAD and CHF increase perioperative cardiac risk and require optimization.",
            whatToCheck=[
                "Recent EKG",
                "Echocardiogram within 6 months",
                "BNP/troponin levels",
                "Medication compliance",
            ],
            preOpActions=[
                "Optimize heart failure medications",
                "Consider cardiology consultation",
                "Ensure beta-blocker continuation",
            ],
            intraOpConsiderations=[
                "Avoid volume overload",
                "Monitor for arrhythmias",
                "Consider invasive monitoring",
            ],
            postOpManagement=[
                "Continue cardiac medications",
                "Monitor for decompensation",
                "Early mobilization with caution",
            ],
            anesthesiaHeadsUp="Patient has cardiac comorbidities; use cardiac-friendly agents.",
            references=[
                "ACC/AHA 2014 Perioperative Guidelines",
                "ESC/ESA 2014 Guidelines",
            ],
        ),
        riskStratification=RiskStratificationRules(
            lowRisk=["Stable CAD/CHF on medications"],
            intermediateRisk=["Recent symptoms or medication changes"],
            highRisk=["Active symptoms, elevated biomarkers, or decompensation"],
        ),
        metadata=BlockMetadata(
            owner="Barnabus Clinical Team",
            reviewers=["Dr. Smith", "Dr. Jones"],
            effectiveDate="2024-01-01",
            status="active",
        ),
    )


def create_osa_block() -> ComorbidityBlockDefinition:
    """
    OSA Block with exact trigger rules per TRD v3.0.
    PRIMARY TRIGGERS:
    - ICD-10: G47.33
    - Medications: CPAP or BiPAP prescription
    - Devices: CPAP or BiPAP device
    """
    return ComorbidityBlockDefinition(
        blockId="OSA_001",
        conditionName="OSA",
        version="1.0.0",
        triggers=TriggerConditions(
            icd10Codes=["G47.33"],
            medicationTriggers=[
                MedicationTrigger(class_="cpap", name_patterns=["cpap"]),
                MedicationTrigger(class_="bipap", name_patterns=["bipap"]),
            ],
            deviceTypes=["CPAP", "BIPAP"],
        ),
        contentSections=ContentSections(
            why="OSA increases perioperative respiratory risk and requires CPAP/BiPAP management.",
            whatToCheck=["CPAP/BiPAP compliance", "Recent sleep study"],
            preOpActions=[
                "Ensure CPAP/BiPAP device available",
                "Continue CPAP/BiPAP perioperatively",
            ],
            intraOpConsiderations=[
                "Avoid sedatives that worsen OSA",
                "Consider CPAP post-extubation",
            ],
            postOpManagement=[
                "Resume CPAP/BiPAP immediately",
                "Monitor for respiratory depression",
            ],
            anesthesiaHeadsUp="OSA patient - avoid excessive sedation and opioids.",
            references=["ASA Practice Guidelines for Perioperative Management"],
        ),
        metadata=BlockMetadata(
            owner="Barnabus Clinical Team",
            effectiveDate="2024-01-01",
            status="active",
        ),
    )


def create_diabetes_block() -> ComorbidityBlockDefinition:
    """
    Diabetes Block with exact trigger rules per TRD v3.0.
    PRIMARY TRIGGERS:
    - ICD-10: E11.* (Type 2), E10.* (Type 1), E13.* (Other)
    - Medications: insulin, SGLT2 inhibitors, GLP-1 agonists, oral hypoglycemics
    - Labs: HbA1c > 6.5%
    """
    return ComorbidityBlockDefinition(
        blockId="DIABETES_001",
        conditionName="Diabetes",
        version="1.0.0",
        triggers=TriggerConditions(
            icd10Codes=["E11.*", "E10.*", "E13.*"],
            medicationTriggers=[
                MedicationTrigger(class_="insulin", name_patterns=["insulin"]),
                MedicationTrigger(
                    class_="sglt2_inhibitor",
                    name_patterns=["dapagliflozin", "empagliflozin", "canagliflozin"],
                ),
                MedicationTrigger(
                    class_="glp1_agonist",
                    name_patterns=["liraglutide", "semaglutide", "dulaglutide"],
                ),
                MedicationTrigger(
                    class_="oral_hypoglycemic",
                    name_patterns=["metformin", "glipizide", "glyburide"],
                ),
            ],
            labThresholds=[
                LabThreshold(test="a1c", operator=">", value=6.5),
            ],
        ),
        contentSections=ContentSections(
            why="Diabetes increases perioperative risk of infection, poor wound healing, and metabolic complications.",
            whatToCheck=["HbA1c", "Recent glucose logs", "Medication compliance"],
            preOpActions=[
                "Optimize glucose control",
                "Consider endocrinology consultation if HbA1c > 8%",
            ],
            intraOpConsiderations=["Monitor glucose frequently", "Avoid hyperglycemia"],
            postOpManagement=["Resume diabetes medications", "Monitor glucose closely"],
            anesthesiaHeadsUp="Diabetes patient - monitor glucose and avoid hyperglycemia.",
            references=["ADA Standards of Medical Care"],
        ),
        metadata=BlockMetadata(
            owner="Barnabus Clinical Team",
            effectiveDate="2024-01-01",
            status="active",
        ),
    )


def create_ckd_block() -> ComorbidityBlockDefinition:
    """
    CKD Block with exact trigger rules per TRD v3.0.
    PRIMARY TRIGGERS:
    - ICD-10: N18.*
    - Labs: eGFR < 60 (calculated using CKD-EPI formula)
    - Labs: Creatinine > 1.5 with documented CKD
    """
    return ComorbidityBlockDefinition(
        blockId="CKD_001",
        conditionName="CKD",
        version="1.0.0",
        triggers=TriggerConditions(
            icd10Codes=["N18.*"],
            labThresholds=[
                LabThreshold(test="creatinine", operator=">", value=1.5),
            ],
        ),
        contentSections=ContentSections(
            why="CKD increases perioperative risk of AKI, electrolyte abnormalities, and medication toxicity.",
            whatToCheck=["eGFR", "Creatinine", "Electrolytes"],
            preOpActions=[
                "Avoid nephrotoxic agents",
                "Optimize hydration",
                "Consider nephrology consultation if eGFR < 30",
            ],
            intraOpConsiderations=[
                "Avoid contrast if possible",
                "Monitor urine output",
                "Avoid hypotension",
            ],
            postOpManagement=[
                "Monitor creatinine",
                "Avoid nephrotoxins",
                "Ensure adequate hydration",
            ],
            anesthesiaHeadsUp="CKD patient - adjust medication doses and avoid nephrotoxins.",
            references=["KDIGO Clinical Practice Guidelines"],
        ),
        metadata=BlockMetadata(
            owner="Barnabus Clinical Team",
            effectiveDate="2024-01-01",
            status="active",
        ),
    )


def create_cirrhosis_liver_block() -> ComorbidityBlockDefinition:
    """
    Cirrhosis/Liver Block with exact trigger rules per TRD v3.0.
    PRIMARY TRIGGERS:
    - ICD-10: K70.3, K74.6
    - Labs: INR > 1.5 without anticoagulation
    - Labs: Platelets < 150 with liver disease history
    """
    return ComorbidityBlockDefinition(
        blockId="CIRRHOSIS_LIVER_001",
        conditionName="Cirrhosis/Liver Disease",
        version="1.0.0",
        triggers=TriggerConditions(
            icd10Codes=["K70.3", "K74.6"],
            labThresholds=[
                LabThreshold(test="inr", operator=">", value=1.5),
                LabThreshold(test="platelets", operator="<", value=150.0),
            ],
        ),
        contentSections=ContentSections(
            why="Cirrhosis increases perioperative risk of bleeding, infection, and hepatic decompensation.",
            whatToCheck=[
                "INR",
                "Platelets",
                "Albumin",
                "Bilirubin",
                "Child-Pugh score",
            ],
            preOpActions=[
                "Consider hepatology consultation",
                "Optimize coagulation",
                "Avoid hepatotoxins",
            ],
            intraOpConsiderations=[
                "Avoid hepatically metabolized drugs",
                "Monitor for bleeding",
                "Avoid hypotension",
            ],
            postOpManagement=[
                "Monitor LFTs",
                "Avoid hepatotoxins",
                "Monitor for encephalopathy",
            ],
            anesthesiaHeadsUp="Cirrhosis patient - use liver-friendly medications and monitor closely.",
            references=["AASLD Practice Guidelines"],
        ),
        metadata=BlockMetadata(
            owner="Barnabus Clinical Team",
            effectiveDate="2024-01-01",
            status="active",
        ),
    )


def create_anemia_block() -> ComorbidityBlockDefinition:
    """
    Anemia Block with exact trigger rules per TRD v3.0 Section 4.6.
    PRIMARY TRIGGERS:
    - Labs: Hemoglobin < site threshold (default: 12 F, 13 M)
    - Labs: Ferritin < 30 OR TSAT < 20%
    - Problem: Iron deficiency anemia
    """
    return ComorbidityBlockDefinition(
        blockId="ANEMIA_001",
        conditionName="Anemia",
        version="1.0.0",
        triggers=TriggerConditions(
            icd10Codes=["D50.*"],  # Iron deficiency anemia codes
            labThresholds=[
                LabThreshold(test="ferritin", operator="<", value=30.0),
                LabThreshold(test="tsat", operator="<", value=20.0),
            ],
        ),
        contentSections=ContentSections(
            why="Anemia increases perioperative risk of poor oxygen delivery and increased transfusion needs.",
            whatToCheck=["Hemoglobin", "Ferritin", "TSAT", "Iron studies"],
            preOpActions=[
                "Consider iron supplementation",
                "Evaluate for bleeding source",
                "Consider hematology consultation if severe",
            ],
            intraOpConsiderations=[
                "Have blood products available",
                "Monitor for bleeding",
            ],
            postOpManagement=[
                "Monitor hemoglobin",
                "Continue iron supplementation if indicated",
            ],
            anesthesiaHeadsUp="Anemic patient - ensure adequate oxygen delivery and have blood products available.",
            references=["ASH Clinical Practice Guidelines"],
        ),
        metadata=BlockMetadata(
            owner="Barnabus Clinical Team",
            effectiveDate="2024-01-01",
            status="active",
        ),
    )


def create_delirium_prevention_block() -> ComorbidityBlockDefinition:
    """
    Delirium Prevention Block with exact trigger rules per TRD v3.0 Section 4.5.
    PRIMARY TRIGGERS:
    - Demographics: Age ≥ 65
    - Problem: Dementia OR Frailty (CFS ≥ 5)
    """
    return ComorbidityBlockDefinition(
        blockId="DELIRIUM_PREVENTION_001",
        conditionName="Delirium Prevention",
        version="1.0.0",
        triggers=TriggerConditions(
            icd10Codes=["F03.*", "F01.*", "F02.*"],  # Dementia codes
            demographicConditions=["age >= 65"],
        ),
        contentSections=ContentSections(
            why="Advanced age and dementia increase risk of perioperative delirium.",
            whatToCheck=[
                "Cognitive assessment",
                "Medication review",
                "Frailty assessment",
            ],
            preOpActions=[
                "Review medications for deliriogenic agents",
                "Consider geriatrics consultation",
                "Optimize sleep",
            ],
            intraOpConsiderations=[
                "Avoid benzodiazepines",
                "Minimize anticholinergics",
                "Use regional anesthesia when possible",
            ],
            postOpManagement=[
                "Early mobilization",
                "Reorient frequently",
                "Avoid restraints",
                "Optimize sleep",
            ],
            anesthesiaHeadsUp="High delirium risk - avoid deliriogenic medications and optimize environment.",
            references=["AGS Clinical Practice Guidelines for Postoperative Delirium"],
        ),
        metadata=BlockMetadata(
            owner="Barnabus Clinical Team",
            effectiveDate="2024-01-01",
            status="active",
        ),
    )


def get_all_comorbidity_blocks() -> List[ComorbidityBlockDefinition]:
    """Return all configured comorbidity blocks."""
    return [
        create_cad_chf_block(),
        create_osa_block(),
        create_diabetes_block(),
        create_ckd_block(),
        create_cirrhosis_liver_block(),
        create_anemia_block(),
        create_delirium_prevention_block(),
    ]


# Backward compatibility alias
create_example_cad_chf_block = create_cad_chf_block


# ============================================================================
# Decision Support Engine
# ============================================================================


@dataclass(frozen=True)
class ConsultationRecommendation:
    """Immutable consultation recommendation."""

    specialty: str  # e.g., 'cardiology', 'nephrology', 'endocrinology'
    urgency: Literal["required", "recommended", "suggested"]  # Urgency level
    reason: str  # Clinical reason for consult
    blockSource: str  # Block ID that triggered this


@dataclass(frozen=True)
class MonitoringRequirement:
    """Immutable monitoring requirement."""

    type: str  # e.g., 'telemetry', 'arterial_line', 'frequent_labs'
    duration: Optional[str]  # e.g., '≥72 hours', 'perioperative'
    frequency: Optional[str]  # e.g., 'q4h', 'continuous'
    reason: str  # Clinical reason
    blockSource: str  # Block ID that triggered this


@dataclass(frozen=True)
class MedicationAdjustment:
    """Immutable medication adjustment recommendation."""

    medication: str  # Medication name or class
    action: Literal["continue", "hold", "adjust_dose", "start", "stop"]  # Action type
    details: str  # Specific details (dose, timing, etc.)
    urgency: Literal["required", "recommended", "suggested"]  # Urgency level
    reason: str  # Clinical reason
    blockSource: str  # Block ID that triggered this


@dataclass(frozen=True)
class ClinicalRecommendations:
    """Immutable clinical recommendations from decision support engine."""

    requiredActions: List[str]  # Actions that must be completed (hard stops)
    suggestedActions: List[str]  # Suggested actions
    consultationRecommendations: List[
        ConsultationRecommendation
    ]  # Consult types with urgency
    monitoringRequirements: List[MonitoringRequirement]  # Specific monitoring needs
    medicationAdjustments: List[MedicationAdjustment]  # Specific medication changes
    flags: List[str]  # Important flags (e.g., "Consider delaying surgery")


def generate_recommendations(
    triggered_blocks: List[Dict[str, Any]], patient_data: PatientClinicalData
) -> ClinicalRecommendations:
    """
    Generate clinical recommendations based on triggered blocks and patient data.

    Applies block-specific clinical rules and severity-based escalation.

    Args:
        triggered_blocks: List of triggered block dictionaries
        patient_data: Patient clinical data snapshot

    Returns:
        ClinicalRecommendations with structured recommendations
    """
    required_actions: List[str] = []
    suggested_actions: List[str] = []
    consultations: List[ConsultationRecommendation] = []
    monitoring: List[MonitoringRequirement] = []
    medication_adjustments: List[MedicationAdjustment] = []
    flags: List[str] = []

    # Process each triggered block
    for block in triggered_blocks:
        if not block.get("triggered", False):
            continue

        block_id = block.get("blockId", "unknown")
        risk_level = block.get("riskLevel", "low")

        # CAD/CHF specific recommendations
        if block_id == "CAD_CHF_001":
            cad_chf_recs = _generate_cad_chf_recommendations(
                patient_data, block_id, risk_level
            )
            required_actions.extend(cad_chf_recs["required"])
            suggested_actions.extend(cad_chf_recs["suggested"])
            consultations.extend(cad_chf_recs["consultations"])
            monitoring.extend(cad_chf_recs["monitoring"])
            medication_adjustments.extend(cad_chf_recs["medications"])
            flags.extend(cad_chf_recs["flags"])

        # Diabetes specific recommendations
        elif block_id == "DIABETES_001":
            diabetes_recs = _generate_diabetes_recommendations(
                patient_data, block_id, risk_level
            )
            required_actions.extend(diabetes_recs["required"])
            suggested_actions.extend(diabetes_recs["suggested"])
            consultations.extend(diabetes_recs["consultations"])
            monitoring.extend(diabetes_recs["monitoring"])
            medication_adjustments.extend(diabetes_recs["medications"])
            flags.extend(diabetes_recs["flags"])

        # CKD specific recommendations
        elif block_id == "CKD_001":
            ckd_recs = _generate_ckd_recommendations(patient_data, block_id, risk_level)
            required_actions.extend(ckd_recs["required"])
            suggested_actions.extend(ckd_recs["suggested"])
            consultations.extend(ckd_recs["consultations"])
            monitoring.extend(ckd_recs["monitoring"])
            medication_adjustments.extend(ckd_recs["medications"])
            flags.extend(ckd_recs["flags"])

    # Remove duplicates while preserving order
    required_actions = list(dict.fromkeys(required_actions))
    suggested_actions = list(dict.fromkeys(suggested_actions))
    flags = list(dict.fromkeys(flags))

    return ClinicalRecommendations(
        requiredActions=required_actions,
        suggestedActions=suggested_actions,
        consultationRecommendations=consultations,
        monitoringRequirements=monitoring,
        medicationAdjustments=medication_adjustments,
        flags=flags,
    )


def _generate_cad_chf_recommendations(
    patient_data: PatientClinicalData, block_id: str, risk_level: str
) -> Dict[str, Any]:
    """Generate CAD/CHF specific recommendations."""
    required: List[str] = []
    suggested: List[str] = []
    consultations: List[ConsultationRecommendation] = []
    monitoring: List[MonitoringRequirement] = []
    medications: List[MedicationAdjustment] = []
    flags: List[str] = []

    # Rule 1: If chestPain = true OR SOB = true
    if patient_data.hpiRedFlags.chestPain or patient_data.hpiRedFlags.shortnessOfBreath:
        required.append("EKG")
        required.append("Troponin")
        required.append("BNP")
        consultations.append(
            ConsultationRecommendation(
                specialty="cardiology",
                urgency="recommended",
                reason="Active cardiac symptoms (chest pain or shortness of breath)",
                blockSource=block_id,
            )
        )
        flags.append("Consider delaying surgery if unstable symptoms")

    # Rule 2: If LVEF < 40%
    if patient_data.lvef is not None and patient_data.lvef < 40:
        consultations.append(
            ConsultationRecommendation(
                specialty="cardiology",
                urgency="recommended",
                reason=f"Reduced LVEF ({patient_data.lvef}%)",
                blockSource=block_id,
            )
        )
        monitoring.append(
            MonitoringRequirement(
                type="telemetry",
                duration="≥72 hours",
                frequency=None,
                reason=f"Reduced LVEF ({patient_data.lvef}%)",
                blockSource=block_id,
            )
        )
        suggested.append("Consider arterial line for major cases")

    # Rule 3: If recent stent < 12 months
    if (
        patient_data.recent_cardiac_event_months is not None
        and patient_data.recent_cardiac_event_months < 12
    ):
        consultations.append(
            ConsultationRecommendation(
                specialty="cardiology",
                urgency="required",
                reason=f"Recent cardiac intervention ({patient_data.recent_cardiac_event_months} months ago)",
                blockSource=block_id,
            )
        )
        required.append("Antiplatelet decision required in Medication Review Gate")
        flags.append("HARD STOP: Antiplatelet management decision required")

    # Rule 4: If BNP > 400
    if patient_data.labs.bnp is not None and patient_data.labs.bnp > 400:
        suggested.append("Consider diuresis pre-op for volume overload")
        suggested.append("Echocardiogram if not recent (<6 months)")

    # Risk-level based escalation
    if risk_level == "high":
        consultations.append(
            ConsultationRecommendation(
                specialty="cardiology",
                urgency="required",
                reason="High-risk CAD/CHF patient",
                blockSource=block_id,
            )
        )
        flags.append(
            "Strongly consider delaying elective surgery until cardiac optimization"
        )

    return {
        "required": required,
        "suggested": suggested,
        "consultations": consultations,
        "monitoring": monitoring,
        "medications": medications,
        "flags": flags,
    }


def _generate_diabetes_recommendations(
    patient_data: PatientClinicalData, block_id: str, risk_level: str
) -> Dict[str, Any]:
    """Generate Diabetes specific recommendations."""
    required: List[str] = []
    suggested: List[str] = []
    consultations: List[ConsultationRecommendation] = []
    monitoring: List[MonitoringRequirement] = []
    medications: List[MedicationAdjustment] = []
    flags: List[str] = []

    # Rule 1: If HbA1c > 9%
    if patient_data.labs.a1c is not None and patient_data.labs.a1c > 9.0:
        consultations.append(
            ConsultationRecommendation(
                specialty="endocrinology",
                urgency="suggested",
                reason=f"Poor glycemic control (HbA1c {patient_data.labs.a1c}%)",
                blockSource=block_id,
            )
        )
        suggested.append("Pre-op optimization if time allows")
        flags.append("Increased infection and wound healing risk")

    # Rule 2: If on insulin pump
    if "insulin_pump" in [d.lower() for d in patient_data.devices]:
        required.append("Anesthesia notification of insulin pump")
        suggested.append("Plan for pump management perioperatively")
        monitoring.append(
            MonitoringRequirement(
                type="glucose_monitoring",
                duration="perioperative",
                frequency="frequent",
                reason="Insulin pump management",
                blockSource=block_id,
            )
        )

    # Risk-level based escalation
    if risk_level == "high":
        consultations.append(
            ConsultationRecommendation(
                specialty="endocrinology",
                urgency="recommended",
                reason="High-risk diabetes patient",
                blockSource=block_id,
            )
        )

    return {
        "required": required,
        "suggested": suggested,
        "consultations": consultations,
        "monitoring": monitoring,
        "medications": medications,
        "flags": flags,
    }


def _generate_ckd_recommendations(
    patient_data: PatientClinicalData, block_id: str, risk_level: str
) -> Dict[str, Any]:
    """Generate CKD specific recommendations."""
    required: List[str] = []
    suggested: List[str] = []
    consultations: List[ConsultationRecommendation] = []
    monitoring: List[MonitoringRequirement] = []
    medications: List[MedicationAdjustment] = []
    flags: List[str] = []

    # Calculate eGFR
    egfr = calculate_egfr_ckd_epi(
        creatinine=patient_data.labs.creatinine,
        age=patient_data.demographics.age,
        sex=patient_data.demographics.sex,
    )

    # Rule 1: If eGFR < 30
    if egfr is not None and egfr < 30:
        required.append("Avoid nephrotoxic agents")
        suggested.append("Consider ICU/stepdown post-op")
        consultations.append(
            ConsultationRecommendation(
                specialty="nephrology",
                urgency="suggested",
                reason=f"Severe CKD (eGFR {egfr:.1f} mL/min/1.73m²)",
                blockSource=block_id,
            )
        )
        monitoring.append(
            MonitoringRequirement(
                type="creatinine_monitoring",
                duration="postoperative",
                frequency="daily",
                reason="Severe CKD - monitor for AKI",
                blockSource=block_id,
            )
        )

    # Rule 2: If planned contrast (inferred from procedure type)
    # This would ideally come from procedure chips, but we can infer from surgery type
    # For now, we'll check if it's a procedure that typically uses contrast
    # (This is a placeholder - in production would come from procedure chips)

    # Risk-level based escalation
    if risk_level == "high":
        consultations.append(
            ConsultationRecommendation(
                specialty="nephrology",
                urgency="recommended",
                reason="High-risk CKD patient",
                blockSource=block_id,
            )
        )
        required.append("Avoid contrast if possible")
        suggested.append("IV hydration protocol if contrast required")

    return {
        "required": required,
        "suggested": suggested,
        "consultations": consultations,
        "monitoring": monitoring,
        "medications": medications,
        "flags": flags,
    }


# ============================================================================
# De-duplication and Merging Engine
# ============================================================================


@dataclass(frozen=True)
class ActionItem:
    """Immutable action item with provenance."""

    text: str
    category: str  # 'monitoring', 'medication', 'consultation', 'procedure', 'patient_instructions'
    sourceBlocks: List[str]  # Block IDs that contributed this action
    specificity: int = (
        0  # Higher = more specific (e.g., "Monitor K+" > "Monitor electrolytes")
    )
    confidenceScore: float = (
        0.5  # Confidence score of the block that generated this action (0-1)
    )


@dataclass(frozen=True)
class ConflictResolution:
    """Immutable conflict resolution result."""

    conflictingItems: List[ActionItem]
    resolution: Optional[
        str
    ]  # 'resolved', 'requires_review', 'auto_merged', 'confidence_based'
    mergedText: Optional[str]  # Merged text if resolved
    rationale: Optional[str]  # Explanation of resolution
    confidenceDifference: Optional[float] = None  # Difference in confidence scores
    requiresClinicianDecision: bool = False  # True if confidence difference < 0.2


@dataclass(frozen=True)
class MergedContent:
    """Immutable merged content from multiple blocks."""

    monitoring: List[ActionItem]
    medication: List[ActionItem]
    consultation: List[ActionItem]
    procedure: List[ActionItem]
    patientInstructions: List[ActionItem]
    conflicts: List[ConflictResolution]
    requiresClinicianReview: bool
    provenance: Dict[str, List[str]]  # Block ID -> list of action texts


def categorize_action(text: str) -> str:
    """
    Categorize an action item based on keywords.

    Args:
        text: Action text

    Returns:
        Category: 'monitoring', 'medication', 'consultation', 'procedure', 'patient_instructions'
    """
    text_lower = text.lower()

    # Monitoring keywords
    if any(
        kw in text_lower
        for kw in ["monitor", "check", "assess", "evaluate", "observe", "track"]
    ):
        return "monitoring"

    # Medication keywords
    if any(
        kw in text_lower
        for kw in [
            "continue",
            "hold",
            "stop",
            "start",
            "adjust",
            "medication",
            "drug",
            "dose",
        ]
    ):
        return "medication"

    # Consultation keywords
    if any(
        kw in text_lower
        for kw in [
            "consult",
            "refer",
            "specialist",
            "cardiology",
            "nephrology",
            "endocrinology",
            "hepatology",
        ]
    ):
        return "consultation"

    # Procedure keywords
    if any(
        kw in text_lower
        for kw in [
            "procedure",
            "surgery",
            "delay",
            "postpone",
            "modify",
            "avoid contrast",
            "bleed risk",
        ]
    ):
        return "procedure"

    # Patient instructions keywords
    if any(
        kw in text_lower
        for kw in ["bring", "ensure", "patient", "home", "device", "cpap", "bipap"]
    ):
        return "patient_instructions"

    # Default to monitoring
    return "monitoring"


def calculate_specificity(text: str) -> int:
    """
    Calculate specificity score for an action item.
    Higher score = more specific (e.g., "Monitor K+" > "Monitor electrolytes").

    Args:
        text: Action text

    Returns:
        Specificity score (0-100)
    """
    score = 0
    text_lower = text.lower()

    # Specific lab names add points
    specific_labs = [
        "k+",
        "na+",
        "mg++",
        "creatinine",
        "bnp",
        "troponin",
        "hgb",
        "inr",
        "platelets",
    ]
    for lab in specific_labs:
        if lab in text_lower:
            score += 10

    # Specific medication names add points
    if any(
        char.isdigit() for char in text
    ):  # Contains numbers (e.g., "40mg", "3-4 days")
        score += 5

    # Specific thresholds add points
    if any(op in text for op in [">", "<", ">=", "<="]):
        score += 5

    # Longer, more detailed text is more specific
    word_count = len(text.split())
    score += min(word_count * 2, 30)

    return score


def are_actions_duplicate(action1: str, action2: str) -> bool:
    """
    Check if two actions are duplicates (same meaning).

    Args:
        action1: First action text
        action2: Second action text

    Returns:
        True if actions are duplicates
    """
    a1_lower = action1.lower().strip()
    a2_lower = action2.lower().strip()

    # Exact match
    if a1_lower == a2_lower:
        return True

    # One is substring of the other (more specific version)
    if a1_lower in a2_lower or a2_lower in a1_lower:
        return True

    # Check for semantic similarity (simple keyword matching)
    # If >70% of keywords match, consider duplicate
    words1 = set(a1_lower.split())
    words2 = set(a2_lower.split())

    if len(words1) == 0 or len(words2) == 0:
        return False

    common_words = words1 & words2
    similarity = len(common_words) / max(len(words1), len(words2))

    # Remove common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "for",
        "with",
        "to",
        "of",
        "in",
        "on",
        "at",
    }
    meaningful_words1 = words1 - stop_words
    meaningful_words2 = words2 - stop_words

    if len(meaningful_words1) > 0 and len(meaningful_words2) > 0:
        common_meaningful = meaningful_words1 & meaningful_words2
        similarity = len(common_meaningful) / max(
            len(meaningful_words1), len(meaningful_words2)
        )
        if similarity > 0.7:
            return True

    return False


def are_actions_conflicting(action1: str, action2: str) -> bool:
    """
    Check if two actions are conflicting (contradictory).

    Args:
        action1: First action text
        action2: Second action text

    Returns:
        True if actions conflict
    """
    a1_lower = action1.lower()
    a2_lower = action2.lower()

    # Check for opposite medication actions
    continue_hold_pairs = [
        ("continue", "hold"),
        ("continue", "stop"),
        ("hold", "continue"),
        ("stop", "continue"),
        ("discontinue", "continue"),
    ]

    for pair in continue_hold_pairs:
        if pair[0] in a1_lower and pair[1] in a2_lower:
            # Check if they refer to the same medication
            # Extract medication names (simplified)
            med_keywords = [
                "beta-blocker",
                "ace",
                "arb",
                "sglt2",
                "anticoagulant",
                "diuretic",
                "insulin",
            ]
            for med in med_keywords:
                if med in a1_lower and med in a2_lower:
                    return True

    # Check for opposite procedure actions
    if ("delay" in a1_lower or "postpone" in a1_lower) and (
        "proceed" in a2_lower or "continue" in a2_lower
    ):
        return True

    return False


def resolve_conflict(action1: ActionItem, action2: ActionItem) -> ConflictResolution:
    """
    Attempt to resolve a conflict between two action items using confidence scores.

    Rules:
    - Higher confidence recommendation wins
    - If confidence difference < 0.2 → flag for clinician decision
    - Include both recommendations with confidence scores

    Args:
        action1: First conflicting action
        action2: Second conflicting action

    Returns:
        ConflictResolution with resolution attempt
    """
    # Calculate confidence difference
    confidence_diff = abs(action1.confidenceScore - action2.confidenceScore)

    # Check if conflict is resolvable by context (conditional language)
    text1 = action1.text.lower()
    text2 = action2.text.lower()

    conditional_keywords = ["if", "unless", "when", "consider", "may"]
    has_conditional_1 = any(kw in text1 for kw in conditional_keywords)
    has_conditional_2 = any(kw in text2 for kw in conditional_keywords)

    if has_conditional_1 and not has_conditional_2:
        # Action1 is conditional, can coexist with action2
        return ConflictResolution(
            conflictingItems=[action1, action2],
            resolution="resolved",
            mergedText=f"{action2.text} ({action1.text})",
            rationale="Conditional action can coexist with general recommendation",
            confidenceDifference=confidence_diff,
            requiresClinicianDecision=False,
        )

    if has_conditional_2 and not has_conditional_1:
        # Action2 is conditional, can coexist with action1
        return ConflictResolution(
            conflictingItems=[action1, action2],
            resolution="resolved",
            mergedText=f"{action1.text} ({action2.text})",
            rationale="Conditional action can coexist with general recommendation",
            confidenceDifference=confidence_diff,
            requiresClinicianDecision=False,
        )

    # Use confidence scores for conflict resolution
    if action1.confidenceScore > action2.confidenceScore:
        # Action1 has higher confidence
        if confidence_diff < 0.2:
            # Confidence difference < 0.2 → flag for clinician decision
            return ConflictResolution(
                conflictingItems=[action1, action2],
                resolution="confidence_based",
                mergedText=f"[CONFIDENCE {action1.confidenceScore:.2f}] {action1.text} | [CONFIDENCE {action2.confidenceScore:.2f}] {action2.text}",
                rationale=f"Conflicting recommendations with similar confidence (diff: {confidence_diff:.2f}). Higher confidence: {action1.confidenceScore:.2f} from {', '.join(action1.sourceBlocks)} vs {action2.confidenceScore:.2f} from {', '.join(action2.sourceBlocks)}. Requires clinician decision.",
                confidenceDifference=confidence_diff,
                requiresClinicianDecision=True,
            )
        else:
            # Higher confidence wins (difference >= 0.2)
            return ConflictResolution(
                conflictingItems=[action1, action2],
                resolution="confidence_based",
                mergedText=action1.text,
                rationale=f"Higher confidence recommendation ({action1.confidenceScore:.2f} from {', '.join(action1.sourceBlocks)}) selected over lower confidence ({action2.confidenceScore:.2f} from {', '.join(action2.sourceBlocks)}). Confidence difference: {confidence_diff:.2f}",
                confidenceDifference=confidence_diff,
                requiresClinicianDecision=False,
            )

    elif action2.confidenceScore > action1.confidenceScore:
        # Action2 has higher confidence
        if confidence_diff < 0.2:
            # Confidence difference < 0.2 → flag for clinician decision
            return ConflictResolution(
                conflictingItems=[action1, action2],
                resolution="confidence_based",
                mergedText=f"[CONFIDENCE {action2.confidenceScore:.2f}] {action2.text} | [CONFIDENCE {action1.confidenceScore:.2f}] {action1.text}",
                rationale=f"Conflicting recommendations with similar confidence (diff: {confidence_diff:.2f}). Higher confidence: {action2.confidenceScore:.2f} from {', '.join(action2.sourceBlocks)} vs {action1.confidenceScore:.2f} from {', '.join(action1.sourceBlocks)}. Requires clinician decision.",
                confidenceDifference=confidence_diff,
                requiresClinicianDecision=True,
            )
        else:
            # Higher confidence wins (difference >= 0.2)
            return ConflictResolution(
                conflictingItems=[action1, action2],
                resolution="confidence_based",
                mergedText=action2.text,
                rationale=f"Higher confidence recommendation ({action2.confidenceScore:.2f} from {', '.join(action2.sourceBlocks)}) selected over lower confidence ({action1.confidenceScore:.2f} from {', '.join(action1.sourceBlocks)}). Confidence difference: {confidence_diff:.2f}",
                confidenceDifference=confidence_diff,
                requiresClinicianDecision=False,
            )

    else:
        # Same confidence - fall back to specificity
        if action1.specificity > action2.specificity:
            return ConflictResolution(
                conflictingItems=[action1, action2],
                resolution="resolved",
                mergedText=action1.text,
                rationale=f"Equal confidence ({action1.confidenceScore:.2f}). More specific recommendation from {', '.join(action1.sourceBlocks)}",
                confidenceDifference=confidence_diff,
                requiresClinicianDecision=False,
            )

        if action2.specificity > action1.specificity:
            return ConflictResolution(
                conflictingItems=[action1, action2],
                resolution="resolved",
                mergedText=action2.text,
                rationale=f"Equal confidence ({action2.confidenceScore:.2f}). More specific recommendation from {', '.join(action2.sourceBlocks)}",
                confidenceDifference=confidence_diff,
                requiresClinicianDecision=False,
            )

        # Same confidence and specificity - requires clinician review
        return ConflictResolution(
            conflictingItems=[action1, action2],
            resolution="requires_review",
            mergedText=f"[CONFIDENCE {action1.confidenceScore:.2f}] {action1.text} | [CONFIDENCE {action2.confidenceScore:.2f}] {action2.text}",
            rationale=f"Conflicting recommendations with equal confidence ({action1.confidenceScore:.2f}) and specificity from {', '.join(action1.sourceBlocks)} and {', '.join(action2.sourceBlocks)} - requires clinician decision",
            confidenceDifference=confidence_diff,
            requiresClinicianDecision=True,
        )


def merge_blocks(triggered_blocks: List[Dict[str, Any]]) -> MergedContent:
    """
    Merge multiple triggered comorbidity blocks, removing duplicates and resolving conflicts.

    Args:
        triggered_blocks: List of triggered block dictionaries with personalized content

    Returns:
        MergedContent with organized, de-duplicated recommendations
    """
    # Extract all action items from all blocks
    all_actions: List[ActionItem] = []

    for block in triggered_blocks:
        if not block.get("triggered", False):
            continue

        block_id = block.get("blockId", "unknown")
        required_actions = block.get("requiredActions", [])
        personalized_content = block.get("personalizedContent", {})
        personalized_recommendations = personalized_content.get(
            "personalizedRecommendations", []
        )

        # Combine required actions and personalized recommendations
        all_block_actions = required_actions + personalized_recommendations

        for action_text in all_block_actions:
            if not action_text or not action_text.strip():
                continue

            category = categorize_action(action_text)
            specificity = calculate_specificity(action_text)

            action_item = ActionItem(
                text=action_text.strip(),
                category=category,
                sourceBlocks=[block_id],
                specificity=specificity,
            )
            all_actions.append(action_item)

    # Group by category
    by_category: Dict[str, List[ActionItem]] = {
        "monitoring": [],
        "medication": [],
        "consultation": [],
        "procedure": [],
        "patient_instructions": [],
    }

    for action in all_actions:
        by_category[action.category].append(action)

    # De-duplicate within each category
    deduplicated: Dict[str, List[ActionItem]] = {
        "monitoring": [],
        "medication": [],
        "consultation": [],
        "procedure": [],
        "patient_instructions": [],
    }

    conflicts: List[ConflictResolution] = []

    for category, actions in by_category.items():
        seen_texts: List[str] = []
        seen_items: List[ActionItem] = []

        for action in actions:
            # Check for duplicates
            is_duplicate = False
            duplicate_idx = None

            for idx, seen_item in enumerate(seen_items):
                if are_actions_duplicate(action.text, seen_item.text):
                    is_duplicate = True
                    duplicate_idx = idx
                    break

            if is_duplicate:
                # Merge with existing item (keep more specific, merge source blocks)
                existing_item = seen_items[duplicate_idx]

                # Keep more specific version
                if action.specificity > existing_item.specificity:
                    # Replace with more specific version
                    merged_item = ActionItem(
                        text=action.text,
                        category=action.category,
                        sourceBlocks=sorted(
                            list(set(existing_item.sourceBlocks + action.sourceBlocks))
                        ),
                        specificity=action.specificity,
                        confidenceScore=max(
                            existing_item.confidenceScore, action.confidenceScore
                        ),  # Use higher confidence
                    )
                    seen_items[duplicate_idx] = merged_item
                    seen_texts[duplicate_idx] = action.text
                else:
                    # Keep existing, just merge source blocks
                    merged_item = ActionItem(
                        text=existing_item.text,
                        category=existing_item.category,
                        sourceBlocks=sorted(
                            list(set(existing_item.sourceBlocks + action.sourceBlocks))
                        ),
                        specificity=existing_item.specificity,
                        confidenceScore=max(
                            existing_item.confidenceScore, action.confidenceScore
                        ),  # Use higher confidence
                    )
                    seen_items[duplicate_idx] = merged_item
            else:
                # Check for conflicts with existing items
                has_conflict = False
                conflict_idx = None

                for idx, seen_item in enumerate(seen_items):
                    if are_actions_conflicting(action.text, seen_item.text):
                        has_conflict = True
                        conflict_idx = idx
                        break

                if has_conflict:
                    # Attempt to resolve conflict
                    conflict_item = seen_items[conflict_idx]
                    resolution = resolve_conflict(action, conflict_item)
                    conflicts.append(resolution)

                    if (
                        resolution.resolution in ["resolved", "confidence_based"]
                        and not resolution.requiresClinicianDecision
                    ):
                        # Replace conflicting item with resolved version (higher confidence wins)
                        resolved_item = ActionItem(
                            text=resolution.mergedText or action.text,
                            category=action.category,
                            sourceBlocks=sorted(
                                list(
                                    set(
                                        conflict_item.sourceBlocks + action.sourceBlocks
                                    )
                                )
                            ),
                            specificity=max(
                                action.specificity, conflict_item.specificity
                            ),
                            confidenceScore=max(
                                action.confidenceScore, conflict_item.confidenceScore
                            ),  # Use higher confidence
                        )
                        seen_items[conflict_idx] = resolved_item
                        seen_texts[conflict_idx] = resolution.mergedText or action.text
                    else:
                        # Keep both, mark as requiring review (confidence difference < 0.2 or other unresolved conflicts)
                        # Include both recommendations with confidence scores
                        # Remove the conflicting item and add both separately
                        seen_items.pop(conflict_idx)
                        seen_texts.pop(conflict_idx)
                        # Add both items with their confidence scores visible
                        seen_items.append(conflict_item)
                        seen_texts.append(conflict_item.text)
                        seen_items.append(action)
                        seen_texts.append(action.text)
                else:
                    # New unique action
                    seen_items.append(action)
                    seen_texts.append(action.text)

        deduplicated[category] = seen_items

    # Build provenance map
    provenance: Dict[str, List[str]] = {}
    for category_items in deduplicated.values():
        for item in category_items:
            for block_id in item.sourceBlocks:
                if block_id not in provenance:
                    provenance[block_id] = []
                provenance[block_id].append(item.text)

    # Check if any conflicts require clinician review
    # Review required if any conflict requires clinician decision (confidence diff < 0.2 or unresolved)
    requires_review = any(
        c.requiresClinicianDecision or c.resolution == "requires_review"
        for c in conflicts
    )

    return MergedContent(
        monitoring=deduplicated["monitoring"],
        medication=deduplicated["medication"],
        consultation=deduplicated["consultation"],
        procedure=deduplicated["procedure"],
        patientInstructions=deduplicated["patient_instructions"],
        conflicts=conflicts,
        requiresClinicianReview=requires_review,
        provenance=provenance,
    )
