import os
from fastapi import FastAPI, HTTPException, Depends
from datetime import datetime
import pandas as pd

# Import the logic from your provided codebase
# Assuming the file structure is restored
from app.services.integration.risk_engine import predict_preop_risk
from app.api.schemas import (
    RiskAssessmentRequest,
    PreOpRiskResponse,
    PulmonaryRiskResponse,
    CardiacRiskResponse,
    CalculatorDetail,
    ReasoningResponse,
    VitalSigns,
    LabValue,
    HPIFlags,
)

app = FastAPI(
    title="Pre-Op Clearance Risk API",
    description="Exposes cardiac and pulmonary risk services based on MIMIC-III data.",
    version="1.0.0",
)

# Configuration for data location
DATA_DIR = os.getenv("MIMIC_DATA_DIR", "./data")


@app.post("/assess/preop", response_model=PreOpRiskResponse)
async def assess_preop_risk(request: RiskAssessmentRequest):
    """
    Generate a full pre-operative risk assessment including:
    - Pulmonary & Cardiac risks with confidence
    - Specific Calculators (RCRI, Gupta, NSQIP, etc.)
    - Reasoning and Context
    - Vitals, Labs, and HPI Flags
    """
    try:
        # Call the core integration function from your codebase
        # Note: We use predict_preop_risk as it enforces the surgery_time logic
        raw_result = predict_preop_risk(
            subject_id=request.subject_id,
            hadm_id=request.hadm_id,
            planned_surgery_time=request.planned_surgery_time,
            data_dir=DATA_DIR,
        )
    except Exception as e:
        # In production, log the full stack trace
        raise HTTPException(status_code=500, detail=f"Risk engine error: {str(e)}")

    # --- Mapping raw logic output to API Schema ---

    # 1. Parse Risk Factors
    calc_risks = raw_result.get("calculated_risk_factors", {})
    cardiac_raw = calc_risks.get("cardiac", {})
    pulm_raw = calc_risks.get("pulmonary", {})
    confidence_raw = raw_result.get("risk_confidence", {})

    # 2. Build Pulmonary Section
    ariscat = pulm_raw.get("ARISCAT", {})
    pulmonary_resp = PulmonaryRiskResponse(
        risk_tier=pulm_raw.get("risk_tier") or ariscat.get("risk_category"),
        ariscat_score=ariscat.get("score"),
        copd_risk_tier=pulm_raw.get("COPD_risk_tier"),
        contributors=ariscat.get("contributors", []),
        confidence_score=confidence_raw.get("pulmonary", 0.0),
    )

    # 3. Build Cardiac Section (Calculators + Components)
    calculators = []
    # RCRI
    if "RCRI" in cardiac_raw:
        rcri = cardiac_raw["RCRI"]
        calculators.append(
            CalculatorDetail(
                name="RCRI",
                score=rcri.get("score"),
                risk_tier=rcri.get("risk_tier"),
                predicted_risk_percent=rcri.get("score_percentage"),
            )
        )

    # Add others (NSQIP, Gupta, AUB)
    for calc_name in ["NSQIP_MACE", "Gupta_MICA", "AUB_HAS2"]:
        if calc_name in cardiac_raw:
            data = cardiac_raw[calc_name]
            calculators.append(
                CalculatorDetail(
                    name=calc_name,
                    score=data.get("score"),
                    risk_tier=data.get("risk_tier"),
                    predicted_risk_percent=data.get("score_percentage"),
                )
            )

    cardiac_resp = CardiacRiskResponse(
        overall_risk_tier=cardiac_raw.get("overall_risk_tier"),
        calculators=calculators,
        components=cardiac_raw.get("lab_risk", {}),  # Contains WBC, BNP, flags
        confidence_score=confidence_raw.get("cardiac", 0.0),
    )

    # 4. Build Reasoning
    # Combine narrative context + comorbidity trigger reasons
    comorbidity_blocks = raw_result.get("comorbidity_blocks", [])
    trigger_reasons = []
    for block in comorbidity_blocks:
        if block.get("triggered"):
            trigger_reasons.extend(block.get("triggerReasons", []))

    reasoning_resp = ReasoningResponse(
        event_context=raw_result.get("event_based_context", ""),
        key_findings=raw_result.get("cardiac_lab_flags", []),
        trigger_reasons=trigger_reasons,
    )

    # 5. Vitals & HPI
    vitals_raw = raw_result.get("recent_vital_signs", {})
    bp_raw = (
        vitals_raw.get("blood_pressure", {}) or {}
    )  # Depending on how extraction normalizes it

    # Handle flat structure vs nested structure from extractor
    if "blood_pressure" in vitals_raw and isinstance(
        vitals_raw["blood_pressure"], dict
    ):
        systolic = vitals_raw["blood_pressure"].get("systolic")
        diastolic = vitals_raw["blood_pressure"].get("diastolic")
    else:
        # Fallback if flat
        systolic = vitals_raw.get("systolic_bp")
        diastolic = vitals_raw.get("diastolic_bp")

    vitals_resp = VitalSigns(
        heart_rate=vitals_raw.get("heart_rate"),
        systolic_bp=systolic,
        diastolic_bp=diastolic,
        spo2=vitals_raw.get("spo2"),
        temperature=vitals_raw.get("temperature"),
    )

    hpi_raw = raw_result.get("hpi_red_flags", {})
    hpi_resp = HPIFlags(
        chest_pain=hpi_raw.get("chestPain", {}).get("present", False),
        shortness_of_breath=hpi_raw.get("shortnessOfBreath", {}).get("present", False),
        syncope=hpi_raw.get(
            "syncope", False
        ),  # Extractor defaults might differ slightly
        fever=hpi_raw.get("fever", False),
    )

    # 6. Labs
    labs_resp = {}
    raw_labs = raw_result.get("lab_summary", {})
    for key, val in raw_labs.items():
        labs_resp[key] = LabValue(
            name=val.get("name", key),
            value=val.get("value"),
            normal_range=val.get("normal_range"),
            captured_ago=val.get("captured_ago"),
        )

    # Final Assembly
    return PreOpRiskResponse(
        subject_id=raw_result["subject_id"],
        hadm_id=raw_result["hadm_id"],
        planned_surgery_time=(
            datetime.fromisoformat(raw_result["surgery_time"])
            if raw_result.get("surgery_time")
            else request.planned_surgery_time
        ),
        pulmonary_risk=pulmonary_resp,
        cardiac_risk=cardiac_resp,
        partial_reasoning=reasoning_resp,
        recent_vital_signs=vitals_resp,
        lab_summary=labs_resp,
        hpi_red_flags=hpi_resp,
        recommendations=raw_result.get("recommendations", []),
    )


# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "active", "icd_version": "Auto-Detect"}
