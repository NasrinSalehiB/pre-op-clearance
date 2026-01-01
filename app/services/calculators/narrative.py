"""
Temporal Narrative Generator

This module converts quantitative temporal patterns into short, clinical,
human-readable sentences that can be embedded into pre-op risk summaries.

It is template-based and works on top of:
  - Pulmonary temporal patterns from PulmonaryRiskExtractor.analyze_pulmonary_temporal_patterns
  - Cardiac temporal patterns from ScoreBasedRiskExtractor.analyze_cardiac_temporal_patterns
  - Optionally, trend-based risk flags from trend_risk_flags.generate_trend_risk_flags
"""

from typing import Dict, Any, List, Optional


def _fmt_delta(value: Optional[float], units: str) -> str:
    if value is None:
        return "an unquantified change"
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.2f} {units}"


def generate_temporal_narrative(
    pulmonary_temporal: Optional[Dict[str, Any]] = None,
    pulmonary_lab_snapshot: Optional[Dict[str, Any]] = None,
    cardiac_temporal: Optional[Dict[str, Any]] = None,
    cardiac_lab_snapshot: Optional[Dict[str, Any]] = None,
    trend_flags: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Generate a structured temporal narrative.

    Args:
        pulmonary_temporal: e.g. from PulmonaryRiskExtractor.analyze_pulmonary_temporal_patterns
        pulmonary_lab_snapshot: e.g. lab_risk from enhance_pulmonary_risk_with_labs
        cardiac_temporal: e.g. from ScoreBasedRiskExtractor.analyze_cardiac_temporal_patterns
        cardiac_lab_snapshot: e.g. labs from build_score_based_risk_report
        trend_flags: optional flags from trend_risk_flags.generate_trend_risk_flags

    Returns:
        {
          "pulmonary_lines": [...],
          "cardiac_lines": [...],
          "integrated_lines": [...],
        }
    """
    pulmonary_temporal = pulmonary_temporal or {}
    pulmonary_lab_snapshot = pulmonary_lab_snapshot or {}
    cardiac_temporal = cardiac_temporal or {}
    cardiac_lab_snapshot = cardiac_lab_snapshot or {}
    trend_flags = trend_flags or {"pulmonary": [], "cardiac": []}

    pulm_lines: List[str] = []
    card_lines: List[str] = []
    integrated_lines: List[str] = []

    # ------------------------------------------------------------------
    # Pulmonary narratives
    # ------------------------------------------------------------------
    lactate_tp = pulmonary_temporal.get("lactate", {}) or {}
    crp_tp = pulmonary_temporal.get("crp", {}) or {}
    alb_tp = pulmonary_temporal.get("albumin", {}) or {}

    lac_rise = lactate_tp.get("lac_rise_3h")
    if lac_rise is not None and lac_rise > 0.5:
        pulm_lines.append(
            f"Lactate has risen by {lac_rise:.2f} mmol/L over the last 3 hours, "
            "suggesting worsening perfusion or evolving sepsis."
        )

    crp_slope = crp_tp.get("slope_48h")
    crp_last = pulmonary_lab_snapshot.get("crp")
    if crp_slope is not None and crp_slope > 0 and crp_last is not None:
        pulm_lines.append(
            f"CRP is trending upward over 48 hours (slope {crp_slope:.3f} mg/L per hour), "
            f"with a current value of {crp_last:.1f} mg/L, consistent with increasing inflammation."
        )

    alb_slope = alb_tp.get("slope_7d")
    alb_last = pulmonary_lab_snapshot.get("albumin")
    if alb_slope is not None and alb_slope < 0 and alb_last is not None:
        pulm_lines.append(
            f"Albumin has declined over the last 7 days (slope {alb_slope:.4f} g/dL per hour), "
            f"with a latest value of {alb_last:.2f} g/dL, suggesting deteriorating nutritional reserve."
        )

    # ------------------------------------------------------------------
    # Cardiac narratives
    # ------------------------------------------------------------------
    bnp_tp = cardiac_temporal.get("bnp", {}) or {}
    trop_tp = cardiac_temporal.get("troponin", {}) or {}
    elec_tp = cardiac_temporal.get("electrolytes", {}) or {}

    bnp_snap = cardiac_lab_snapshot.get("bnp", {}) or {}
    trop_snap = cardiac_lab_snapshot.get("troponin", {}) or {}
    elec_snap = cardiac_lab_snapshot.get("electrolytes", {}) or {}

    delta_bnp = bnp_tp.get("delta_48h")
    bnp_last = bnp_snap.get("last")
    if delta_bnp is not None and bnp_last is not None:
        card_lines.append(
            f"BNP has changed by {_fmt_delta(delta_bnp, 'pg/mL')} over 48 hours, "
            f"with a latest value of {bnp_last:.1f} pg/mL, informing heart failure risk assessment."
        )

    delta_trop = trop_tp.get("delta_6h")
    trop_last = trop_snap.get("last")
    if delta_trop is not None and trop_last is not None:
        card_lines.append(
            f"Troponin has changed by {_fmt_delta(delta_trop, '')} over 6 hours, "
            f"with a latest value of {trop_last:.4f}, relevant to active myocardial injury risk."
        )

    delta_k = elec_tp.get("delta_k_24h")
    delta_na = elec_tp.get("delta_na_24h")
    if delta_k is not None or delta_na is not None:
        k_part = f"ΔK {delta_k:+.2f} mEq/L" if delta_k is not None else None
        na_part = f"ΔNa {delta_na:+.1f} mEq/L" if delta_na is not None else None
        pieces = [p for p in [k_part, na_part] if p]
        if pieces:
            card_lines.append(
                f"Over 24 hours, electrolyte shifts ({', '.join(pieces)}) may modulate perioperative arrhythmia risk."
            )

    # ------------------------------------------------------------------
    # Integrated narratives (cross-domain combinations)
    # ------------------------------------------------------------------
    # Example: Rising lactate + falling albumin
    if lac_rise is not None and lac_rise > 0.5 and alb_slope is not None and alb_slope < 0:
        integrated_lines.append(
            "Rising lactate together with falling albumin suggests possible sepsis with limited "
            "nutritional reserve, increasing both pulmonary and overall perioperative risk."
        )

    # Example: Rising CRP + BNP trend
    if (
        crp_slope is not None
        and crp_slope > 0
        and delta_bnp is not None
        and bnp_last is not None
        and bnp_last > 300
    ):
        integrated_lines.append(
            "Concordant increases in CRP and BNP over the preoperative period support a picture of "
            "systemic inflammation with hemodynamic strain, reinforcing both pulmonary and cardiac risk concerns."
        )

    # Example: Rising troponin + unstable electrolytes
    if (
        delta_trop is not None
        and delta_trop > 0
        and trop_last is not None
        and (delta_k is not None or delta_na is not None)
    ):
        integrated_lines.append(
            "Rising troponin in the setting of recent electrolyte shifts may reflect an unstable myocardial "
            "substrate with heightened arrhythmia and ischemic risk."
        )

    # Optionally, incorporate explicit flag codes to reinforce narratives
    for flag in trend_flags.get("pulmonary", []):
        # Only add if not already captured; keep single-line phrasing
        code = flag.get("code")
        if code == "PULM_RISK_RISING_LACTATE" and all("Lactate has risen" not in l for l in pulm_lines):
            cond = flag.get("condition", "")
            pulm_lines.append(
                f"Pulmonary trend flag {code}: {cond} – {flag.get('recommended_action', '').rstrip('.') }."
            )
        if code == "PULM_RISK_INFLAMMATION" and all("CRP is trending upward" not in l for l in pulm_lines):
            cond = flag.get("condition", "")
            pulm_lines.append(
                f"Pulmonary trend flag {code}: {cond} – {flag.get('recommended_action', '').rstrip('.') }."
            )
        if code == "PULM_RISK_POOR_NUTRITION" and all("Albumin has declined" not in l for l in pulm_lines):
            cond = flag.get("condition", "")
            pulm_lines.append(
                f"Pulmonary trend flag {code}: {cond} – {flag.get('recommended_action', '').rstrip('.') }."
            )

    for flag in trend_flags.get("cardiac", []):
        code = flag.get("code")
        if code == "CARD_RISK_WORSENING_HF" and all("BNP has changed" not in l for l in card_lines):
            cond = flag.get("condition", "")
            card_lines.append(
                f"Cardiac trend flag {code}: {cond} – {flag.get('recommended_action', '').rstrip('.') }."
            )
        if code == "CARD_RISK_ACTIVE_MI" and all("Troponin has changed" not in l for l in card_lines):
            cond = flag.get("condition", "")
            card_lines.append(
                f"Cardiac trend flag {code}: {cond} – {flag.get('recommended_action', '').rstrip('.') }."
            )
        if code == "CARD_RISK_ARRHYTHMIA_RISK" and all("electrolyte shifts" not in l for l in card_lines):
            cond = flag.get("condition", "")
            card_lines.append(
                f"Cardiac trend flag {code}: {cond} – {flag.get('recommended_action', '').rstrip('.') }."
            )

    return {
        "pulmonary_lines": pulm_lines,
        "cardiac_lines": card_lines,
        "integrated_lines": integrated_lines,
    }


