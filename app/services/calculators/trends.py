"""
Trend-Based Clinical Risk Flags

This module aggregates temporal lab patterns into high-level pulmonary and
cardiac risk flags that can be consumed by the overall risk assessment
and AI narrative layers.

It is intentionally stateless: callers provide temporal_features and
static lab snapshots (already extracted elsewhere).
"""

from typing import Dict, Any, List, Optional


def _mk_flag(
    code: str,
    severity: str,
    condition: str,
    details: Optional[Dict[str, Any]] = None,
    recommended_action: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "condition": condition,
        "details": details or {},
        "recommended_action": recommended_action,
    }


def generate_trend_risk_flags(
    pulmonary_temporal: Optional[Dict[str, Any]] = None,
    pulmonary_lab_snapshot: Optional[Dict[str, Any]] = None,
    cardiac_temporal: Optional[Dict[str, Any]] = None,
    cardiac_lab_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate high-level pulmonary and cardiac trend-based risk flags.

    Inputs:
      - pulmonary_temporal:
            Expected shape from PulmonaryRiskExtractor.analyze_pulmonary_temporal_patterns():
            {
              "lactate": {"lac_rise_3h": float|None, "rising_lactate": bool},
              "crp":     {"slope_48h": float|None, "increasing_crp": bool},
              "albumin": {"slope_7d": float|None, "falling_albumin": bool},
            }

      - pulmonary_lab_snapshot:
            Typically lab_risk dict from enhance_pulmonary_risk_with_labs():
            {
              "crp": float|None,
              "albumin": float|None,
              "lactate": float|None,
              ...
            }

      - cardiac_temporal:
            Expected shape from ScoreBasedRiskExtractor.analyze_cardiac_temporal_patterns():
            {
              "bnp": {
                  "delta_48h": float|None,
                  "worsening_bnp": bool,
              },
              "troponin": {
                  "delta_6h": float|None,
                  "rising_troponin": bool,
              },
              "electrolytes": {
                  "delta_k_24h": float|None,
                  "delta_na_24h": float|None,
                  "unstable_electrolytes": bool,
              },
            }

      - cardiac_lab_snapshot:
            Typically `labs` from ScoreBasedRiskExtractor._compute_lab_intelligence():
            {
              "bnp": {"last": float|None, "delta": float|None},
              "troponin": {"last": float|None, "delta": float|None},
              "electrolytes": {"k": float|None, "mg": float|None, "na": float|None},
              ...
            }

    Returns:
        {
          "pulmonary": [ {flag}, ... ],
          "cardiac":   [ {flag}, ... ],
        }
    """
    pulmonary_flags: List[Dict[str, Any]] = []
    cardiac_flags: List[Dict[str, Any]] = []

    pulmonary_temporal = pulmonary_temporal or {}
    pulmonary_lab_snapshot = pulmonary_lab_snapshot or {}
    cardiac_temporal = cardiac_temporal or {}
    cardiac_lab_snapshot = cardiac_lab_snapshot or {}

    # ------------------------------------------------------------------
    # Pulmonary risk flags
    # ------------------------------------------------------------------
    lactate_tp = pulmonary_temporal.get("lactate", {})
    crp_tp = pulmonary_temporal.get("crp", {})
    alb_tp = pulmonary_temporal.get("albumin", {})

    # 1) PULM_RISK_RISING_LACTATE
    lac_rise = lactate_tp.get("lac_rise_3h")
    if lac_rise is not None and lac_rise > 0.5:
        pulmonary_flags.append(
            _mk_flag(
                code="PULM_RISK_RISING_LACTATE",
                severity="high",
                condition="LacRise_3h > 0.5 mmol/L",
                details={
                    "lac_rise_3h": lac_rise,
                },
                recommended_action=(
                    "Evaluate for perfusion impairment or sepsis; "
                    "optimize hemodynamics and consider delaying high-risk surgery."
                ),
            )
        )

    # 2) PULM_RISK_INFLAMMATION
    crp_slope = crp_tp.get("slope_48h")
    crp_last = pulmonary_lab_snapshot.get("crp")
    if (
        crp_slope is not None
        and crp_slope > 0.0
        and crp_last is not None
        and crp_last > 10.0
    ):
        pulmonary_flags.append(
            _mk_flag(
                code="PULM_RISK_INFLAMMATION",
                severity="moderate",
                condition="CRP_48h_slope > 0 AND CRP > 10 mg/L",
                details={
                    "crp_last": crp_last,
                    "crp_slope_48h": crp_slope,
                },
                recommended_action=(
                    "Assess for ongoing respiratory or systemic infection; "
                    "consider postponing elective surgery or intensifying peri-op antibiotics."
                ),
            )
        )

    # 3) PULM_RISK_POOR_NUTRITION
    alb_slope = alb_tp.get("slope_7d")
    alb_last = pulmonary_lab_snapshot.get("albumin")
    if (
        alb_slope is not None
        and alb_slope < 0.0
        and alb_last is not None
        and alb_last < 3.5
    ):
        pulmonary_flags.append(
            _mk_flag(
                code="PULM_RISK_POOR_NUTRITION",
                severity="moderate",
                condition="Albumin_7d_slope < 0 AND albumin < 3.5 g/dL",
                details={
                    "albumin_last": alb_last,
                    "albumin_slope_7d": alb_slope,
                },
                recommended_action=(
                    "Optimize nutrition and consider delaying non-urgent surgery "
                    "to improve protein status and wound-healing capacity."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Cardiac risk flags
    # ------------------------------------------------------------------
    bnp_tp = cardiac_temporal.get("bnp", {})
    trop_tp = cardiac_temporal.get("troponin", {})
    elec_tp = cardiac_temporal.get("electrolytes", {})

    bnp_snap = cardiac_lab_snapshot.get("bnp", {}) or {}
    trop_snap = cardiac_lab_snapshot.get("troponin", {}) or {}
    elec_snap = cardiac_lab_snapshot.get("electrolytes", {}) or {}

    # 1) CARD_RISK_WORSENING_HF: ΔBNP_48h > 100 AND BNP > 300
    delta_bnp = bnp_tp.get("delta_48h")
    bnp_last = bnp_snap.get("last")
    if (
        delta_bnp is not None
        and delta_bnp > 100.0
        and bnp_last is not None
        and bnp_last > 300.0
    ):
        cardiac_flags.append(
            _mk_flag(
                code="CARD_RISK_WORSENING_HF",
                severity="high",
                condition="ΔBNP_48h > 100 pg/mL AND BNP > 300 pg/mL",
                details={
                    "delta_bnp_48h": delta_bnp,
                    "bnp_last": bnp_last,
                },
                recommended_action=(
                    "Treat and stabilize heart failure before surgery; "
                    "consider cardiology consult and postponing elective procedures."
                ),
            )
        )

    # 2) CARD_RISK_ACTIVE_MI: ΔTrop_6h > 0 AND troponin > ULN
    delta_trop = trop_tp.get("delta_6h")
    trop_last = trop_snap.get("last")
    ULN_TROP = 0.04  # depends on assay; used as generic threshold
    if (
        delta_trop is not None
        and delta_trop > 0.0
        and trop_last is not None
        and trop_last > ULN_TROP
    ):
        cardiac_flags.append(
            _mk_flag(
                code="CARD_RISK_ACTIVE_MI",
                severity="critical",
                condition="ΔTrop_6h > 0 AND troponin > ULN",
                details={
                    "delta_trop_6h": delta_trop,
                    "troponin_last": trop_last,
                    "troponin_ULN": ULN_TROP,
                },
                recommended_action=(
                    "Treat as possible acute coronary syndrome; "
                    "urgent cardiology evaluation and defer non-emergent surgery."
                ),
            )
        )

    # 3) CARD_RISK_ARRHYTHMIA_RISK: |ΔK_24h| > 0.5 OR |ΔNa_24h| > 5
    delta_k = elec_tp.get("delta_k_24h")
    delta_na = elec_tp.get("delta_na_24h")
    if (
        (delta_k is not None and abs(delta_k) > 0.5)
        or (delta_na is not None and abs(delta_na) > 5.0)
    ):
        cardiac_flags.append(
            _mk_flag(
                code="CARD_RISK_ARRHYTHMIA_RISK",
                severity="moderate",
                condition="|ΔK_24h| > 0.5 OR |ΔNa_24h| > 5 mEq/L",
                details={
                    "delta_k_24h": delta_k,
                    "delta_na_24h": delta_na,
                    "k_last": elec_snap.get("k"),
                    "na_last": elec_snap.get("na"),
                },
                recommended_action=(
                    "Correct electrolyte abnormalities and re-check levels "
                    "before anesthesia; monitor closely for arrhythmias."
                ),
            )
        )

    return {
        "pulmonary": pulmonary_flags,
        "cardiac": cardiac_flags,
    }


