from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


# --- Request Model ---
class RiskAssessmentRequest(BaseModel):
    subject_id: int = Field(..., description="MIMIC-III Subject ID", example=249)
    hadm_id: int = Field(..., description="Hospital Admission ID", example=116935)
    planned_surgery_time: datetime = Field(
        ..., description="ISO 8601 DateTime of planned surgery"
    )


# --- Response Components ---


class ConfidenceMetric(BaseModel):
    score: float
    description: str


class RiskComponent(BaseModel):
    risk_tier: Optional[str]
    score: Optional[float]
    score_percentage: Optional[float]
    factors: Optional[Dict[str, Any]] = None


class CalculatorDetail(BaseModel):
    name: str
    score: Optional[float]
    risk_tier: Optional[str]
    predicted_risk_percent: Optional[float]


class LabValue(BaseModel):
    name: str
    value: Optional[float]
    unit: Optional[str] = None
    normal_range: Optional[Dict[str, Optional[float]]]
    captured_ago: Optional[str]


class VitalSigns(BaseModel):
    heart_rate: Optional[float]
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    spo2: Optional[float]
    temperature: Optional[float]


class HPIFlags(BaseModel):
    chest_pain: bool
    shortness_of_breath: bool
    syncope: bool
    fever: bool


# --- Main Response Sections ---


class PulmonaryRiskResponse(BaseModel):
    risk_tier: Optional[str]
    ariscat_score: Optional[float]
    copd_risk_tier: Optional[str]
    contributors: List[str]
    confidence_score: float


class CardiacRiskResponse(BaseModel):
    overall_risk_tier: Optional[str]
    calculators: List[CalculatorDetail]
    components: Dict[str, Any]
    confidence_score: float


class ReasoningResponse(BaseModel):
    event_context: str
    key_findings: List[str]
    trigger_reasons: List[str]


class PreOpRiskResponse(BaseModel):
    subject_id: int
    hadm_id: int
    planned_surgery_time: datetime

    # The requested services
    pulmonary_risk: PulmonaryRiskResponse
    cardiac_risk: CardiacRiskResponse
    partial_reasoning: ReasoningResponse
    recent_vital_signs: VitalSigns
    lab_summary: Dict[str, LabValue]
    hpi_red_flags: HPIFlags
    recommendations: List[str]
