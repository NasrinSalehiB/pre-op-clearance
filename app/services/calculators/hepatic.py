"""
Hepatic risk calculators for Barnabus-style integration.

This module implements:
  - Child-Pugh score and class
  - MELD-Na score and interpretation
  - VOCAL-Penn-like liver surgical risk (30-day / 90-day mortality)

NOTE:
  This implementation focuses on deterministic, auditable calculations.
  Data sourcing (labs, ICD codes, procedure chips) is handled by higher-level
  orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import math


AsaClass = Literal[1, 2, 3, 4, 5]
SurgicalRisk = Literal["low", "moderate", "high"]


@dataclass
class VocalPennInputs:
    """
    Inputs for VOCAL-Penn-style hepatic surgical risk calculation.

    All numeric fields may be None; completeness checking will determine
    whether a full calculation is possible.
    """

    # VOCAL-Penn core
    asaClass: Optional[AsaClass]
    age: Optional[float]
    ascitesPresent: Optional[bool]
    albumin: Optional[float]  # g/dL
    inr: Optional[float]
    platelets: Optional[float]  # x10^3/µL
    surgicalRisk: Optional[SurgicalRisk]

    # Child-Pugh components
    bilirubin: Optional[float]  # mg/dL
    encephalopathyGrade: Optional[int]  # 0–4, 0 = none

    # MELD-Na components
    creatinine: Optional[float]  # mg/dL
    sodium: Optional[float]  # mmol/L

    # Barnabus metadata
    patientId: str
    caseId: str
    calculatedBy: Optional[Literal["auto", "manual"]] = "auto"


@dataclass
class ChildPughComponent:
    value: Any
    points: int
    grade: Optional[str] = None


@dataclass
class ChildPughResult:
    score: Optional[int]
    class_: Optional[Literal["A", "B", "C"]]
    components: Dict[str, ChildPughComponent]


@dataclass
class MeldNaResult:
    score: Optional[int]
    interpretation: Optional[str]
    components: Dict[str, Optional[float]]


@dataclass
class VocalPennResult:
    # VOCAL-Penn core
    vocalPennScore: Optional[float]
    thirtyDayMortalityPercent: Optional[float]
    ninetyDayMortalityPercent: Optional[float]
    riskCategory: Optional[Literal["Low", "Moderate", "High", "Very High"]]

    # Sub-panels
    childPugh: ChildPughResult
    meldNa: MeldNaResult

    # Provenance
    version: str
    inputs: Dict[str, Any]
    timestamp: str
    calculatedBy: Literal["auto", "manual"]

    # UI display
    pointsBreakdown: Dict[str, Optional[float]]

    # Consensus tier
    normalizedTier: Optional[Literal["Low", "Intermediate", "High"]]

    # Completeness
    completeness: Literal["complete", "incomplete"]
    missingInputs: List[str]


class ValidationError(Exception):
    """Validation error for hepatic risk inputs."""


def _check_completeness(inputs: VocalPennInputs) -> List[str]:
    missing: List[str] = []
    for key in [
        "asaClass",
        "age",
        "ascitesPresent",
        "albumin",
        "inr",
        "platelets",
        "surgicalRisk",
        "bilirubin",
        "encephalopathyGrade",
        "creatinine",
        "sodium",
    ]:
        if getattr(inputs, key) is None:
            missing.append(key)
    return missing


def _validate_inputs(inputs: VocalPennInputs) -> None:
    """Basic sanity checks; do not enforce completeness here."""
    if inputs.age is not None and inputs.age < 18:
        raise ValidationError(f"Invalid age: {inputs.age}")
    if inputs.albumin is not None and inputs.albumin <= 0:
        raise ValidationError(f"Invalid albumin: {inputs.albumin}")
    if inputs.inr is not None and inputs.inr <= 0:
        raise ValidationError(f"Invalid INR: {inputs.inr}")
    if inputs.platelets is not None and inputs.platelets <= 0:
        raise ValidationError(f"Invalid platelets: {inputs.platelets}")
    if inputs.bilirubin is not None and inputs.bilirubin <= 0:
        raise ValidationError(f"Invalid bilirubin: {inputs.bilirubin}")
    if inputs.creatinine is not None and inputs.creatinine <= 0:
        raise ValidationError(f"Invalid creatinine: {inputs.creatinine}")
    if inputs.sodium is not None and inputs.sodium <= 0:
        raise ValidationError(f"Invalid sodium: {inputs.sodium}")


def _calculate_child_pugh(inputs: VocalPennInputs) -> ChildPughResult:
    """
    Child-Pugh scoring based on:
      - Bilirubin (mg/dL)
      - Albumin (g/dL)
      - INR
      - Ascites (bool → none/present)
      - Encephalopathy grade 0–4
    """
    bili = inputs.bilirubin
    alb = inputs.albumin
    inr = inputs.inr
    ascites_present = inputs.ascitesPresent
    enceph_grade = inputs.encephalopathyGrade

    components: Dict[str, ChildPughComponent] = {}

    # Bilirubin: <2 →1; 2–3 →2; >3 →3
    if bili is None:
        bili_pts = 0
    elif bili < 2:
        bili_pts = 1
    elif bili <= 3:
        bili_pts = 2
    else:
        bili_pts = 3
    components["bilirubin"] = ChildPughComponent(value=bili, points=bili_pts)

    # Albumin: >3.5 →1; 2.8–3.5 →2; <2.8 →3
    if alb is None:
        alb_pts = 0
    elif alb > 3.5:
        alb_pts = 1
    elif alb >= 2.8:
        alb_pts = 2
    else:
        alb_pts = 3
    components["albumin"] = ChildPughComponent(value=alb, points=alb_pts)

    # INR: <1.7 →1; 1.7–2.3 →2; >2.3 →3
    if inr is None:
        inr_pts = 0
    elif inr < 1.7:
        inr_pts = 1
    elif inr <= 2.3:
        inr_pts = 2
    else:
        inr_pts = 3
    components["inr"] = ChildPughComponent(value=inr, points=inr_pts)

    # Ascites: None →1; Present →3 (we approximate severity from boolean)
    if ascites_present is None:
        ascites_pts = 0
        ascites_grade = "unknown"
    elif ascites_present:
        ascites_pts = 3
        ascites_grade = "present"
    else:
        ascites_pts = 1
        ascites_grade = "none"
    components["ascites"] = ChildPughComponent(
        value="present" if ascites_present else "none",
        points=ascites_pts,
        grade=ascites_grade,
    )

    # Encephalopathy: None →1; Grade 1–2 →2; Grade 3–4 →3
    if enceph_grade is None:
        enceph_pts = 0
        enceph_label = "unknown"
    elif enceph_grade == 0:
        enceph_pts = 1
        enceph_label = "None"
    elif enceph_grade in (1, 2):
        enceph_pts = 2
        enceph_label = "Grade 1-2"
    else:
        enceph_pts = 3
        enceph_label = "Grade 3-4"
    components["encephalopathy"] = ChildPughComponent(
        value=enceph_grade,
        points=enceph_pts,
        grade=enceph_label,
    )

    # If any core component is missing, we still compute score but may mark completeness upstream
    score = (
        bili_pts
        + alb_pts
        + inr_pts
        + ascites_pts
        + enceph_pts
    )

    if score <= 6:
        klass: Optional[str] = "A"
    elif score <= 9:
        klass = "B"
    else:
        klass = "C"

    return ChildPughResult(score=score, class_=klass, components=components)


def _calculate_meld_na(inputs: VocalPennInputs) -> MeldNaResult:
    """
    MELD-Na calculation:
      MELD = 3.78*ln(bilirubin) + 11.2*ln(INR) + 9.57*ln(creatinine) + 6.43
      MELD-Na = MELD + 1.59*(135 - Na)
    Using clinical constraints:
      bilirubin: 1–20
      INR: 1–10
      creatinine: 0.8–4.0
      sodium: 120–135
      final score: 6–40 (rounded to nearest int)
    """
    bili = inputs.bilirubin
    inr = inputs.inr
    creat = inputs.creatinine
    na = inputs.sodium

    components = {
        "bilirubin": bili,
        "inr": inr,
        "creatinine": creat,
        "sodium": na,
    }

    if any(v is None for v in (bili, inr, creat, na)):
        return MeldNaResult(score=None, interpretation=None, components=components)

    # Apply caps/floors
    bili_c = min(20.0, max(1.0, float(bili)))
    inr_c = min(10.0, max(1.0, float(inr)))
    creat_c = min(4.0, max(0.8, float(creat)))
    na_c = min(135.0, max(120.0, float(na)))

    meld = 3.78 * math.log(bili_c) + 11.2 * math.log(inr_c) + 9.57 * math.log(creat_c) + 6.43
    meld_na = meld + 1.59 * (135.0 - na_c)
    meld_na = max(6.0, min(40.0, meld_na))
    score = int(round(meld_na))

    if score < 10:
        interp = "Low risk"
    elif score < 20:
        interp = "Intermediate risk"
    elif score < 30:
        interp = "High risk"
    else:
        interp = "Very high risk"

    return MeldNaResult(
        score=score,
        interpretation=interp,
        components={
            "bilirubin": bili_c,
            "inr": inr_c,
            "creatinine": creat_c,
            "sodium": na_c,
        },
    )


def calculate_vocal_penn(inputs: VocalPennInputs) -> VocalPennResult:
    """
    Deterministic VOCAL-Penn-style hepatic surgical risk calculator.

    - Computes VOCAL-Penn core score (ASA, age, ascites, albumin, INR,
      platelets, surgical risk).
    - Computes 30-day and 90-day mortality percentages.
    - Computes Child-Pugh and MELD-Na sub-panels.
    - Returns Barnabus-style result object.
    """
    _validate_inputs(inputs)
    missing_inputs = _check_completeness(inputs)

    # Sub-panels
    child_pugh = _calculate_child_pugh(inputs)
    meld_na = _calculate_meld_na(inputs)

    # Core points breakdown (may use defaults for missing values as 0)
    asa = inputs.asaClass or 3
    asa_points = asa - 1  # 1→0, 2→1, 3→2, 4→3, 5→4

    age = inputs.age if inputs.age is not None else 18.0
    age_points = max(0.0, (age - 18.0) * 0.1)

    ascites_present = bool(inputs.ascitesPresent)
    ascites_points = 3.0 if ascites_present else 0.0

    albumin = inputs.albumin if inputs.albumin is not None else 4.0
    albumin_points = max(0.0, min(4.0, (4.5 - albumin) * 2.0))

    inr_val = inputs.inr if inputs.inr is not None else 1.0
    inr_points = max(0.0, (inr_val - 1.0) * 3.0)

    platelets = inputs.platelets if inputs.platelets is not None else 300.0
    platelet_points = max(0.0, min(4.0, (300.0 - platelets) / 50.0))

    surg_risk = inputs.surgicalRisk or "low"
    if surg_risk == "low":
        surgical_points = 0.0
    elif surg_risk == "moderate":
        surgical_points = 2.0
    else:
        surgical_points = 4.0

    points_breakdown = {
        "asaClass": asa_points,
        "age": age_points,
        "ascites": ascites_points,
        "albumin": albumin_points,
        "inr": inr_points,
        "platelets": platelet_points,
        "surgicalRisk": surgical_points,
    }

    vocal_score = (
        asa_points
        + age_points
        + ascites_points
        + albumin_points
        + inr_points
        + platelet_points
        + surgical_points
    )

    # 30-day mortality
    logit = vocal_score - 5.5
    mort30_prob = 1.0 / (1.0 + math.exp(-logit))
    thirty_pct = round(mort30_prob * 100.0, 1)

    # 90-day mortality (simple conversion factor, capped at 100%)
    ninety_pct = round(min(100.0, thirty_pct * 1.5), 1)

    # Risk category from 30-day mortality
    if thirty_pct < 5.0:
        risk_cat: Optional[str] = "Low"
    elif thirty_pct < 15.0:
        risk_cat = "Moderate"
    elif thirty_pct < 30.0:
        risk_cat = "High"
    else:
        risk_cat = "Very High"

    # Map to Barnabus normalized tier
    if risk_cat == "Low":
        tier: Optional[str] = "Low"
    elif risk_cat == "Moderate":
        tier = "Intermediate"
    elif risk_cat in {"High", "Very High"}:
        tier = "High"
    else:
        tier = None

    completeness = "complete" if not missing_inputs else "incomplete"

    return VocalPennResult(
        vocalPennScore=round(vocal_score, 1),
        thirtyDayMortalityPercent=thirty_pct,
        ninetyDayMortalityPercent=ninety_pct,
        riskCategory=risk_cat,
        childPugh=child_pugh,
        meldNa=meld_na,
        version="VOCAL-PENN-1.0",
        inputs=asdict(inputs),
        timestamp=datetime.now().isoformat(),
        calculatedBy=inputs.calculatedBy or "auto",
        pointsBreakdown=points_breakdown,
        normalizedTier=tier,  # Low→Low, Moderate→Intermediate, High/VeryHigh→High
        completeness=completeness,
        missingInputs=missing_inputs,
    )


