from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime

# Updated import path
from app.core.definitions import (
    PatientClinicalData,
    HpiRedFlags,
    ConsultationRecommendation,
    MonitoringRequirement,
    MedicationAdjustment,
    ClinicalRecommendations,
    ComorbidityBlockDefinition,
    TriggerEvaluationResult,
    evaluate_trigger,
    assess_risk_level,
    generate_recommendations,
    get_all_comorbidity_blocks,
)

# ============================================================================
# Integration Data Structures
# ============================================================================


@dataclass
class HPIData:
    """HPI Red-Flags data from HPI system."""

    chestPain: bool = False
    shortnessOfBreath: bool = False
    syncope: bool = False
    fever: bool = False
    acknowledgment: Optional[Dict[str, Any]] = None  # Acknowledgment status


@dataclass
class RiskCalculatorResult:
    """Risk calculator result from calculator system."""

    calculatorName: str  # 'DASI', 'RCRI', 'ARISCAT', 'VOCAL-Penn'
    score: Optional[float] = None
    riskCategory: Optional[str] = None
    riskLevel: Optional[Literal["low", "intermediate", "high"]] = None
    components: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MedicationDecision:
    """Medication decision from Medication Review Gate."""

    medication: str
    action: Literal["continue", "hold", "adjust_dose", "start", "stop"]
    details: str
    reason: str
    urgency: Literal["required", "recommended", "suggested"] = "recommended"


@dataclass
class ProcedureData:
    """Procedure data from Procedure Chips system."""

    procedureCode: Optional[str] = None
    procedureName: Optional[str] = None
    bleedRisk: Optional[Literal["low", "moderate", "high"]] = None
    cardiacRisk: Optional[Literal["low", "moderate", "high"]] = None
    neuraxialPlanned: bool = False
    contrastPlanned: bool = False
    durationMinutes: Optional[int] = None


@dataclass
class LabFrameworkData:
    """Lab data from Barnabus Lab Ontology Framework."""

    labs: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # lab_name -> {value, timestamp, trend, etc.}
    derivedIndices: Dict[str, float] = field(
        default_factory=dict
    )  # e.g., {'egfr': 45.2, 'bun_cr_ratio': 20}


@dataclass
class ComorbidityContext:
    """Context data provided to other Barnabus systems."""

    triggeredBlocks: List[str]  # Block IDs
    overallRiskLevel: Literal["low", "intermediate", "high"]
    requiresConsultation: bool
    requiresMonitoring: bool
    medicationAdjustmentsRequired: bool
    flags: List[str]


@dataclass
class MonitoringPlan:
    """Monitoring plan for other systems."""

    type: str
    duration: Optional[str]
    frequency: Optional[str]
    reason: str
    urgency: Literal["required", "recommended", "suggested"]


# ============================================================================
# Comorbidity Integration Manager
# ============================================================================


class ComorbidityIntegrationManager:
    """
    Integration manager for comorbidity block system.

    Handles cross-system communication and state management for comorbidity blocks.
    """

    def __init__(self):
        """Initialize the integration manager."""
        self.patient_data: Optional[PatientClinicalData] = None
        self.triggered_blocks: List[ComorbidityBlockDefinition] = []
        self.evaluation_results: List[TriggerEvaluationResult] = []
        self.clinical_recommendations: Optional[ClinicalRecommendations] = None

        # Integration state
        self.hpi_data: Optional[HPIData] = None
        self.risk_calculators: Dict[str, RiskCalculatorResult] = {}
        self.medication_decisions: List[MedicationDecision] = []
        self.procedure_data: Optional[ProcedureData] = None
        self.lab_framework_data: Optional[LabFrameworkData] = None

        # Load all comorbidity blocks
        self.all_blocks = get_all_comorbidity_blocks()

    # ========================================================================
    # Event Handlers (Receive events from other systems)
    # ========================================================================

    def on_hpi_update(self, hpi_data: HPIData) -> None:
        """
        Handle HPI Red-Flags system update.

        When chestPain or shortnessOfBreath = true → auto-trigger CAD/CHF block.

        Args:
            hpi_data: HPI red-flags data
        """
        self.hpi_data = hpi_data

        # Update patient data HPI flags
        if self.patient_data:
            # Create updated HPI flags
            updated_hpi = HpiRedFlags(
                chestPain=hpi_data.chestPain,
                shortnessOfBreath=hpi_data.shortnessOfBreath,
                syncope=hpi_data.syncope,
                fever=hpi_data.fever,
            )

            # Update patient data (create new instance since it's frozen)
            self.patient_data = PatientClinicalData(
                problemList=self.patient_data.problemList,
                medications=self.patient_data.medications,
                labs=self.patient_data.labs,
                vitals=self.patient_data.vitals,
                devices=self.patient_data.devices,
                demographics=self.patient_data.demographics,
                hpiRedFlags=updated_hpi,
                lvef=self.patient_data.lvef,
                nyha_class=self.patient_data.nyha_class,
                dasi_mets=self.patient_data.dasi_mets,
                recent_cardiac_event_months=self.patient_data.recent_cardiac_event_months,
                inotropic_dependence=self.patient_data.inotropic_dependence,
                proteinuria=self.patient_data.proteinuria,
                dialysis_dependent=self.patient_data.dialysis_dependent,
                diabetes_complications=self.patient_data.diabetes_complications,
                recurrent_hypoglycemia=self.patient_data.recurrent_hypoglycemia,
            )

        # Auto-trigger CAD/CHF block if symptoms present
        if hpi_data.chestPain or hpi_data.shortnessOfBreath:
            # This will be handled in reevaluate_triggers()
            pass

    def on_risk_calculator_update(
        self, calculator_name: str, result: RiskCalculatorResult
    ) -> None:
        """
        Handle risk calculator system update.

        Updates patient context based on calculator results:
        - DASI score < 4 METs → flag as high risk
        - RCRI score ≥ 3 → enhance CAD/CHF recommendations
        - ARISCAT triggered → enhance pulmonary recommendations
        - VOCAL-Penn triggered → enhance cirrhosis recommendations

        Args:
            calculator_name: Name of the calculator ('DASI', 'RCRI', 'ARISCAT', 'VOCAL-Penn')
            result: Calculator result
        """
        self.risk_calculators[calculator_name] = result

        # Update patient data based on calculator results
        if self.patient_data:
            updates: Dict[str, Any] = {}

            # DASI score → update DASI METs
            if calculator_name == "DASI" and result.score is not None:
                updates["dasi_mets"] = result.score

            # RCRI → may influence cardiac risk assessment
            if calculator_name == "RCRI" and result.score is not None:
                # RCRI score ≥ 3 indicates higher cardiac risk
                if result.score >= 3:
                    # This will influence CAD/CHF block risk level
                    pass

            # ARISCAT → pulmonary risk
            if calculator_name == "ARISCAT":
                # ARISCAT results influence pulmonary block
                pass

            # VOCAL-Penn → hepatic risk
            if calculator_name == "VOCAL-Penn":
                # VOCAL-Penn results influence cirrhosis block
                pass

            # Apply updates if any
            if updates:
                self.update_patient_context(updates)

    def on_medication_decision(self, decision: MedicationDecision) -> None:
        """
        Handle medication decision from Medication Review Gate.

        Medication decisions influence comorbidity content personalization.

        Args:
            decision: Medication decision
        """
        self.medication_decisions.append(decision)

        # Medication decisions can influence comorbidity recommendations
        # This is handled in generate_recommendations() via personalization

    def on_procedure_update(self, procedure_data: ProcedureData) -> None:
        """
        Handle procedure update from Procedure Chips system.

        Procedure characteristics influence comorbidity recommendations:
        - Bleed risk → antiplatelet recommendations
        - Cardiac risk → monitoring recommendations
        - Neuraxial planned → anticoagulation recommendations

        Args:
            procedure_data: Procedure data
        """
        self.procedure_data = procedure_data

    def on_lab_framework_update(self, lab_data: LabFrameworkData) -> None:
        """
        Handle lab framework update from Lab Ontology Framework.

        Uses Barnabus Lab Framework for:
        - Latest lab values with timestamps
        - Trend calculations (Δ values)
        - Derived indices (eGFR, BUN/Cr ratio)

        Args:
            lab_data: Lab framework data
        """
        self.lab_framework_data = lab_data

        # Update patient data labs from lab framework
        if self.patient_data and lab_data.labs:
            from app.core.definitions import LabValues

            # Map lab framework data to LabValues
            labs_dict = lab_data.labs
            updated_labs = LabValues(
                hemoglobin=labs_dict.get("hemoglobin", {}).get("value")
                or self.patient_data.labs.hemoglobin,
                a1c=labs_dict.get("a1c", {}).get("value") or self.patient_data.labs.a1c,
                creatinine=labs_dict.get("creatinine", {}).get("value")
                or self.patient_data.labs.creatinine,
                albumin=labs_dict.get("albumin", {}).get("value")
                or self.patient_data.labs.albumin,
                inr=labs_dict.get("inr", {}).get("value") or self.patient_data.labs.inr,
                platelets=labs_dict.get("platelets", {}).get("value")
                or self.patient_data.labs.platelets,
                bilirubin=labs_dict.get("bilirubin", {}).get("value")
                or self.patient_data.labs.bilirubin,
                sodium=labs_dict.get("sodium", {}).get("value")
                or self.patient_data.labs.sodium,
                bnp=labs_dict.get("bnp", {}).get("value") or self.patient_data.labs.bnp,
                troponin=labs_dict.get("troponin", {}).get("value")
                or self.patient_data.labs.troponin,
                nt_probnp=labs_dict.get("nt_probnp", {}).get("value")
                or self.patient_data.labs.nt_probnp,
                ferritin=labs_dict.get("ferritin", {}).get("value")
                or self.patient_data.labs.ferritin,
                tsat=labs_dict.get("tsat", {}).get("value")
                or self.patient_data.labs.tsat,
            )

            # Update patient data with new labs
            self.patient_data = PatientClinicalData(
                problemList=self.patient_data.problemList,
                medications=self.patient_data.medications,
                labs=updated_labs,
                vitals=self.patient_data.vitals,
                devices=self.patient_data.devices,
                demographics=self.patient_data.demographics,
                hpiRedFlags=self.patient_data.hpiRedFlags,
                lvef=self.patient_data.lvef,
                nyha_class=self.patient_data.nyha_class,
                dasi_mets=self.patient_data.dasi_mets,
                recent_cardiac_event_months=self.patient_data.recent_cardiac_event_months,
                inotropic_dependence=self.patient_data.inotropic_dependence,
                proteinuria=self.patient_data.proteinuria,
                dialysis_dependent=self.patient_data.dialysis_dependent,
                diabetes_complications=self.patient_data.diabetes_complications,
                recurrent_hypoglycemia=self.patient_data.recurrent_hypoglycemia,
            )

            # Update derived indices (e.g., eGFR from lab framework)
            if "egfr" in lab_data.derivedIndices:
                # eGFR is calculated, not stored directly in LabValues
                # But we can use it for CKD block evaluation
                pass

    # ========================================================================
    # State Management
    # ========================================================================

    def update_patient_context(self, updates: Dict[str, Any]) -> None:
        """
        Update patient context with partial updates.

        Args:
            updates: Dictionary of updates to apply
        """
        if not self.patient_data:
            return

        # Create updated patient data (since it's frozen, create new instance)
        updated_data = {
            "problemList": self.patient_data.problemList,
            "medications": self.patient_data.medications,
            "labs": self.patient_data.labs,
            "vitals": self.patient_data.vitals,
            "devices": self.patient_data.devices,
            "demographics": self.patient_data.demographics,
            "hpiRedFlags": self.patient_data.hpiRedFlags,
            "lvef": self.patient_data.lvef,
            "nyha_class": self.patient_data.nyha_class,
            "dasi_mets": self.patient_data.dasi_mets,
            "recent_cardiac_event_months": self.patient_data.recent_cardiac_event_months,
            "inotropic_dependence": self.patient_data.inotropic_dependence,
            "proteinuria": self.patient_data.proteinuria,
            "dialysis_dependent": self.patient_data.dialysis_dependent,
            "diabetes_complications": self.patient_data.diabetes_complications,
            "recurrent_hypoglycemia": self.patient_data.recurrent_hypoglycemia,
        }

        # Apply updates
        updated_data.update(updates)

        self.patient_data = PatientClinicalData(**updated_data)

    def initialize_patient(self, patient_data: PatientClinicalData) -> None:
        """
        Initialize patient data for comorbidity evaluation.

        Args:
            patient_data: Initial patient clinical data
        """
        self.patient_data = patient_data
        self.triggered_blocks = []
        self.evaluation_results = []
        self.clinical_recommendations = None

    def reevaluate_triggers(self) -> List[TriggerEvaluationResult]:
        """
        Reevaluate all comorbidity block triggers based on current patient context.

        Returns:
            List of trigger evaluation results
        """
        if not self.patient_data:
            return []

        results: List[TriggerEvaluationResult] = []
        triggered: List[ComorbidityBlockDefinition] = []

        for block in self.all_blocks:
            evaluation = evaluate_trigger(block, self.patient_data)
            results.append(evaluation)

            if evaluation.triggered:
                triggered.append(block)

                # Apply risk stratification
                refined_risk = assess_risk_level(block, self.patient_data, evaluation)
                if refined_risk != evaluation.riskLevel:
                    # Create updated evaluation with refined risk
                    evaluation = TriggerEvaluationResult(
                        blockId=evaluation.blockId,
                        triggered=evaluation.triggered,
                        triggerReasons=evaluation.triggerReasons,
                        confidenceScore=evaluation.confidenceScore,
                        riskLevel=refined_risk,
                        requiredActions=evaluation.requiredActions,
                    )
                    results[-1] = evaluation

        self.triggered_blocks = triggered
        self.evaluation_results = results

        # Generate clinical recommendations
        if triggered:
            triggered_dicts = [
                {
                    "blockId": b.blockId,
                    "triggered": True,
                    "riskLevel": next(
                        (r.riskLevel for r in results if r.blockId == b.blockId), "low"
                    ),
                }
                for b in triggered
            ]
            self.clinical_recommendations = generate_recommendations(
                triggered_dicts, self.patient_data
            )

        return results

    # ========================================================================
    # Data Providers (Provide data to other systems)
    # ========================================================================

    def get_comorbidity_context(self) -> ComorbidityContext:
        """
        Get comorbidity context for other Barnabus systems.

        Returns:
            ComorbidityContext with triggered blocks and risk information
        """
        if not self.patient_data:
            return ComorbidityContext(
                triggeredBlocks=[],
                overallRiskLevel="low",
                requiresConsultation=False,
                requiresMonitoring=False,
                medicationAdjustmentsRequired=False,
                flags=[],
            )

        triggered_block_ids = [b.blockId for b in self.triggered_blocks]

        # Determine overall risk level (highest risk from triggered blocks)
        overall_risk = "low"
        if self.evaluation_results:
            risk_levels = [r.riskLevel for r in self.evaluation_results if r.triggered]
            if "high" in risk_levels:
                overall_risk = "high"
            elif "intermediate" in risk_levels:
                overall_risk = "intermediate"

        # Check if consultations required
        requires_consultation = False
        if self.clinical_recommendations:
            required_consults = [
                c
                for c in self.clinical_recommendations.consultationRecommendations
                if c.urgency == "required"
            ]
            requires_consultation = len(required_consults) > 0

        # Check if monitoring required
        requires_monitoring = False
        if self.clinical_recommendations:
            requires_monitoring = (
                len(self.clinical_recommendations.monitoringRequirements) > 0
            )

        # Check if medication adjustments required
        medication_adjustments_required = False
        if self.clinical_recommendations:
            required_meds = [
                m
                for m in self.clinical_recommendations.medicationAdjustments
                if m.urgency == "required"
            ]
            medication_adjustments_required = len(required_meds) > 0

        # Collect flags
        flags: List[str] = []
        if self.clinical_recommendations:
            flags = self.clinical_recommendations.flags

        return ComorbidityContext(
            triggeredBlocks=triggered_block_ids,
            overallRiskLevel=overall_risk,
            requiresConsultation=requires_consultation,
            requiresMonitoring=requires_monitoring,
            medicationAdjustmentsRequired=medication_adjustments_required,
            flags=flags,
        )

    def get_required_consultations(self) -> List[ConsultationRecommendation]:
        """
        Get required consultation recommendations for other systems.

        Returns:
            List of consultation recommendations
        """
        if not self.clinical_recommendations:
            return []

        return [
            c
            for c in self.clinical_recommendations.consultationRecommendations
            if c.urgency == "required"
        ]

    def get_monitoring_requirements(self) -> List[MonitoringPlan]:
        """
        Get monitoring requirements as MonitoringPlan objects.

        Returns:
            List of monitoring plans
        """
        if not self.clinical_recommendations:
            return []

        return [
            MonitoringPlan(
                type=m.type,
                duration=m.duration,
                frequency=m.frequency,
                reason=m.reason,
                urgency="required" if "required" in m.reason.lower() else "recommended",
            )
            for m in self.clinical_recommendations.monitoringRequirements
        ]

    def get_medication_influences(self) -> List[Dict[str, Any]]:
        """
        Get medication influences for Medication Review Gate.

        Returns:
            List of medication influence dictionaries
        """
        influences: List[Dict[str, Any]] = []

        # Check triggered blocks for medication influences
        for block in self.triggered_blocks:
            if block.blockId == "CAD_CHF_001":
                # CAD/CHF influences beta-blocker, ACEi decisions
                influences.append(
                    {
                        "medication": "beta_blocker",
                        "influence": "continue",
                        "reason": "CAD/CHF block triggered - continue beta-blockers unless contraindicated",
                        "blockSource": block.blockId,
                    }
                )
                influences.append(
                    {
                        "medication": "ace_inhibitor",
                        "influence": "consider_holding",
                        "reason": "CAD/CHF block - consider holding evening prior if CKD Stage 3+",
                        "blockSource": block.blockId,
                    }
                )

            elif block.blockId == "DIABETES_001":
                # Diabetes influences SGLT2, insulin timing
                influences.append(
                    {
                        "medication": "sglt2_inhibitor",
                        "influence": "hold",
                        "reason": "Diabetes block - hold SGLT2 inhibitors 3-4 days pre-op",
                        "blockSource": block.blockId,
                    }
                )

            elif block.blockId == "CKD_001":
                # CKD influences medication dosing
                influences.append(
                    {
                        "medication": "nephrotoxic_agents",
                        "influence": "avoid",
                        "reason": "CKD block - avoid nephrotoxic agents",
                        "blockSource": block.blockId,
                    }
                )

        return influences

    def get_procedure_influences(self) -> Dict[str, Any]:
        """
        Get procedure-specific influences for Procedure Chips system.

        Returns:
            Dictionary of procedure influences
        """
        influences: Dict[str, Any] = {
            "anticoagulationConsiderations": [],
            "monitoringConsiderations": [],
            "contrastConsiderations": [],
        }

        # Check triggered blocks for procedure influences
        for block in self.triggered_blocks:
            if block.blockId == "CAD_CHF_001":
                # CAD/CHF influences monitoring
                influences["monitoringConsiderations"].append(
                    {
                        "type": "telemetry",
                        "reason": "CAD/CHF block triggered",
                        "blockSource": block.blockId,
                    }
                )

            elif block.blockId == "CKD_001":
                # CKD influences contrast
                influences["contrastConsiderations"].append(
                    {
                        "action": "avoid_if_possible",
                        "reason": "CKD block - avoid contrast if possible",
                        "blockSource": block.blockId,
                    }
                )

        return influences
