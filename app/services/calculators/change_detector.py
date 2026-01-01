"""
Surgery-Aware Clinical Change Detection System

A pure data processing engine focused ONLY on clinical change analysis.
This module provides:

1. TIME-BOUNDED CHANGE PROCESSOR:
   - Filters clinical events to pre-operative period only
   - Ensures all events occur before surgeryDateTime

2. CLINICAL CHANGE ANALYZER:
   - Analyzes quantitative deltas in labs/vitals
   - Classifies clinical significance (P0/P1/P2)
   - Detects patterns and trends

3. TREND INTEGRATION ENGINE:
   - Integrates with existing Lab Trend calculations from Barnabus Lab Ontology
   - Combines discrete changes with continuous trends
   - Generates unified change analysis

NOTE: This is a pure data processing module. No user authentication,
RBAC, or case management - these are handled by the backend system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np


# ============================================================================
# Data Models
# ============================================================================

class ClinicalSignificance(Enum):
    """Clinical significance classification (P0/P1/P2)."""
    P0_CRITICAL = "P0"  # Critical - requires immediate attention
    P1_HIGH = "P1"      # High - significant clinical concern
    P2_MODERATE = "P2"  # Moderate - notable but less urgent
    P3_LOW = "P3"       # Low - minor change, monitor


class ChangeType(Enum):
    """Type of clinical change."""
    SYMPTOM = "symptom"
    LAB = "lab"
    MEDICATION = "medication"
    VITAL = "vital"
    DIAGNOSTIC = "diagnostic"


class ChangeCategory(Enum):
    """Clinical category of change."""
    CARDIAC = "cardiac"
    PULMONARY = "pulmonary"
    RENAL = "renal"
    ENDOCRINE = "endocrine"
    HEMATOLOGIC = "hematologic"


class RiskChange(Enum):
    """Direction of risk change."""
    INCREASE = "increase"
    DECREASE = "decrease"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class QuantitativeData:
    """Quantitative data for clinical changes."""
    testName: Optional[str] = None  # For labs: 'creatinine', 'BNP', etc.
    previousValue: Optional[float] = None
    currentValue: Optional[float] = None
    unit: Optional[str] = None
    delta: Optional[float] = None
    percentChange: Optional[float] = None
    rateOfChange: Optional[float] = None  # Change per day


@dataclass(frozen=True)
class ClinicalImpact:
    """Clinical impact assessment."""
    riskChange: RiskChange
    affectsSurgicalPlanning: bool
    requiresAction: bool


@dataclass(frozen=True)
class ClinicalChange:
    """
    Clinical change event structure.
    
    Matches interface ClinicalChange from specification.
    """
    timestamp: str  # ISO 8601 format
    type: ChangeType  # 'symptom' | 'lab' | 'medication' | 'vital' | 'diagnostic'
    category: ChangeCategory  # 'cardiac' | 'pulmonary' | 'renal' | 'endocrine' | 'hematologic'
    
    # CHANGE SPECIFICS
    description: str  # Human-readable change description
    
    # CLINICAL SIGNIFICANCE
    criticality: str  # 'P0' | 'P1' | 'P2' - P0=Critical, P1=Important, P2=Informational
    clinicalImpact: ClinicalImpact
    
    # Optional quantitative data (must come after required fields)
    quantitativeData: Optional[QuantitativeData] = None


@dataclass(frozen=True)
class TemporalFeatures:
    """Temporal features for trend analysis."""
    acceleration: Optional[float] = None  # Rate of rate change
    volatility: Optional[float] = None  # How variable (coefficient of variation)
    seasonality: Optional[Dict[str, Any]] = None  # Time-of-day patterns
    outlierCount: int = 0  # Abnormal spikes


@dataclass(frozen=True)
class IntegratedTrend:
    """Integrated trend data combining multiple sources."""
    # Required fields first
    parameterName: str
    trendDirection: str  # 'increasing', 'decreasing', 'stable'
    confidence: str  # 'high', 'moderate', 'low'
    dataSources: List[str]  # List of source identifiers
    # Optional fields with defaults come after required fields
    trendDetails: Dict[str, Any] = field(default_factory=dict)
    # Enhanced fields for Lab Ontology integration
    test: Optional[str] = None  # Test name (alias for parameterName)
    significance: Optional[str] = None  # Clinical significance
    clinicalImplications: Optional[List[str]] = field(default_factory=list)
    monitoringRecommendations: Optional[List[str]] = field(default_factory=list)
    temporalFeatures: Optional[TemporalFeatures] = None


@dataclass(frozen=True)
class AnalysisWindow:
    """Time boundaries for change analysis."""
    start: str  # ISO 8601 - Last evaluation time
    end: str  # ISO 8601 - Surgery time
    durationDays: float


@dataclass(frozen=True)
class ChangeAnalysisMetrics:
    """Summary metrics for change analysis."""
    totalChanges: int
    criticalChanges: int  # P0 count
    significantLabTrends: int
    stabilityScore: float  # 0-100, higher = more stable


@dataclass(frozen=True)
class ChangeCounts:
    """Counts of changes by criticality level."""
    total: int
    critical: int  # P0 count
    important: int  # P1 count
    informational: int  # P2 count


@dataclass(frozen=True)
class StabilityMetrics:
    """Stability metrics for change analysis."""
    stabilityScore: float  # 0-100, higher = more stable
    volatilityIndex: float  # 0-1, higher = more volatile
    trendStability: str  # 'stable', 'moderate', 'unstable'
    riskLevel: str  # 'low', 'moderate', 'high', 'critical'


@dataclass(frozen=True)
class ChangeSummary:
    """Comprehensive summary of change analysis."""
    analysisWindow: AnalysisWindow
    changeCounts: ChangeCounts
    summaries: Dict[str, str]  # 'clinical' and 'export' summaries
    stability: StabilityMetrics
    detectedPatterns: List[ClinicalPattern] = field(default_factory=list)


@dataclass(frozen=True)
class ChangeAnalysis:
    """
    Change analysis result structure.
    
    Matches interface ChangeAnalysis from specification.
    """
    # TIME BOUNDARIES
    analysisWindow: AnalysisWindow
    
    # SUMMARY METRICS (required field must come before optional fields)
    metrics: ChangeAnalysisMetrics
    
    # DETECTED CHANGES (optional fields with defaults)
    changes: List[ClinicalChange] = field(default_factory=list)
    
    # TREND INTEGRATION (optional fields with defaults)
    integratedTrends: List[IntegratedTrend] = field(default_factory=list)


# ============================================================================
# Legacy data models (kept for backward compatibility)
# ============================================================================

@dataclass(frozen=True)
class QuantitativeDelta:
    """Quantitative change analysis result (legacy)."""
    parameter_name: str
    baseline_value: Optional[float]
    current_value: Optional[float]
    delta_value: Optional[float]  # current - baseline
    delta_percent: Optional[float]  # (delta / baseline) * 100
    time_delta_hours: Optional[float]
    baseline_timestamp: Optional[datetime]
    current_timestamp: Optional[datetime]
    clinical_significance: ClinicalSignificance
    trend_direction: Optional[str]  # 'increasing', 'decreasing', 'stable'
    pattern_type: Optional[str] = None  # 'acute', 'chronic', 'fluctuating'


@dataclass(frozen=True)
class ChangeAnalysisResult:
    """Complete change analysis for a parameter."""
    parameter_name: str
    quantitative_delta: QuantitativeDelta
    trend_integration: Optional[Dict[str, Any]] = None  # Integrated trend data
    pattern_detected: Optional[str] = None
    recommended_action: Optional[str] = None


@dataclass(frozen=True)
class UnifiedChangeAnalysis:
    """Unified change analysis combining discrete changes and trends."""
    subject_id: int
    hadm_id: int
    surgery_datetime: datetime
    analysis_timestamp: datetime
    pre_op_window_hours: float
    changes: List[ChangeAnalysisResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 1. TIME-BOUNDED CHANGE PROCESSOR
# ============================================================================

class TimeBoundedChangeProcessor:
    """
    Filters clinical events to pre-operative period only.
    
    Ensures all events occur before surgeryDateTime to avoid
    including intra-operative or post-operative data in pre-op analysis.
    """
    
    def __init__(self, surgery_datetime: datetime):
        """
        Initialize processor with surgery datetime.
        
        Args:
            surgery_datetime: Surgery date/time - all events must be before this
        """
        self.surgery_datetime = pd.to_datetime(surgery_datetime)
    
    def filter_events(
        self,
        events: List[ClinicalEvent],
        max_hours_before: Optional[int] = None
    ) -> List[ClinicalEvent]:
        """
        Filter events to pre-operative period only.
        
        Args:
            events: List of clinical events with timestamps
            max_hours_before: Optional maximum hours before surgery to include
                            (e.g., 168 for 7 days, None for all pre-op events)
        
        Returns:
            Filtered list of events occurring before surgery_datetime
        """
        filtered = []
        
        for event in events:
            # Handle both ClinicalEvent (legacy) and ClinicalChange (new) formats
            if hasattr(event, 'timestamp'):
                if isinstance(event.timestamp, str):
                    event_time = pd.to_datetime(event.timestamp)
                else:
                    event_time = pd.to_datetime(event.timestamp)
            else:
                continue
            
            # Must be before surgery
            if event_time >= self.surgery_datetime:
                continue
            
            # Optional: filter by maximum hours before surgery
            if max_hours_before is not None:
                cutoff_time = self.surgery_datetime - timedelta(hours=max_hours_before)
                if event_time < cutoff_time:
                    continue
            
            filtered.append(event)
        
        return filtered
    
    def filter_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'CHARTTIME',
        max_hours_before: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter pandas DataFrame to pre-operative period only.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_column: Name of timestamp column (default: 'CHARTTIME')
            max_hours_before: Optional maximum hours before surgery
        
        Returns:
            Filtered DataFrame with only pre-op events
        """
        if df.empty or timestamp_column not in df.columns:
            return pd.DataFrame()
        
        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Filter: must be before surgery
        filtered = df[df[timestamp_column] < self.surgery_datetime].copy()
        
        # Optional: filter by maximum hours before surgery
        if max_hours_before is not None:
            cutoff_time = self.surgery_datetime - timedelta(hours=max_hours_before)
            filtered = filtered[filtered[timestamp_column] >= cutoff_time].copy()
        
        return filtered.sort_values(timestamp_column)
    
    def get_pre_op_window(
        self,
        hours_before: int = 168  # Default: 7 days
    ) -> Tuple[datetime, datetime]:
        """
        Get pre-operative time window.
        
        Args:
            hours_before: Hours before surgery to include
        
        Returns:
            Tuple of (window_start, window_end) where window_end = surgery_datetime
        """
        window_start = self.surgery_datetime - timedelta(hours=hours_before)
        window_end = self.surgery_datetime
        return (window_start, window_end)


# ============================================================================
# 2. CLINICAL CHANGE ANALYZER
# ============================================================================

class ClinicalChangeAnalyzer:
    """
    Analyzes quantitative deltas in labs/vitals and classifies clinical significance.
    
    Provides:
    - Quantitative delta calculations
    - Clinical significance classification (P0/P1/P2)
    - Pattern detection (acute, chronic, fluctuating)
    """
    
    # Clinical significance thresholds for common parameters
    # Format: {parameter: {P0_threshold, P1_threshold, P2_threshold}}
    SIGNIFICANCE_THRESHOLDS: Dict[str, Dict[str, float]] = {
        'creatinine': {
            'P0_delta': 0.5,  # mg/dL
            'P1_delta': 0.3,
            'P2_delta': 0.15,
            'P0_percent': 50.0,  # %
            'P1_percent': 30.0,
            'P2_percent': 15.0,
        },
        'bnp': {
            'P0_delta': 200.0,  # pg/mL
            'P1_delta': 100.0,
            'P2_delta': 50.0,
            'P0_percent': 50.0,
            'P1_percent': 30.0,
            'P2_percent': 15.0,
        },
        'troponin': {
            'P0_delta': 0.1,  # ng/mL
            'P1_delta': 0.05,
            'P2_delta': 0.02,
            'P0_percent': 100.0,  # Troponin changes are often large %
            'P1_percent': 50.0,
            'P2_percent': 25.0,
        },
        'hemoglobin': {
            'P0_delta': 3.0,  # g/dL
            'P1_delta': 2.0,
            'P2_delta': 1.0,
            'P0_percent': 25.0,
            'P1_percent': 15.0,
            'P2_percent': 8.0,
        },
        'sodium': {
            'P0_delta': 10.0,  # mEq/L
            'P1_delta': 5.0,
            'P2_delta': 3.0,
            'P0_percent': 7.0,
            'P1_percent': 4.0,
            'P2_percent': 2.0,
        },
        'potassium': {
            'P0_delta': 1.0,  # mEq/L
            'P1_delta': 0.5,
            'P2_delta': 0.3,
            'P0_percent': 25.0,
            'P1_percent': 15.0,
            'P2_percent': 8.0,
        },
        'heart_rate': {
            'P0_delta': 30.0,  # bpm
            'P1_delta': 20.0,
            'P2_delta': 10.0,
            'P0_percent': 30.0,
            'P1_percent': 20.0,
            'P2_percent': 10.0,
        },
        'systolic_bp': {
            'P0_delta': 30.0,  # mmHg
            'P1_delta': 20.0,
            'P2_delta': 10.0,
            'P0_percent': 20.0,
            'P1_percent': 15.0,
            'P2_percent': 8.0,
        },
    }
    
    def calculate_delta(
        self,
        parameter_name: str,
        baseline_value: Optional[float],
        current_value: Optional[float],
        baseline_timestamp: Optional[datetime],
        current_timestamp: Optional[datetime]
    ) -> QuantitativeDelta:
        """
        Calculate quantitative delta for a parameter.
        
        Args:
            parameter_name: Name of the parameter (e.g., 'creatinine', 'bnp')
            baseline_value: Baseline value
            current_value: Current value
            baseline_timestamp: Timestamp of baseline measurement
            current_timestamp: Timestamp of current measurement
        
        Returns:
            QuantitativeDelta with calculated values and clinical significance
        """
        # Calculate delta values
        delta_value = None
        delta_percent = None
        time_delta_hours = None
        
        if baseline_value is not None and current_value is not None:
            delta_value = current_value - baseline_value
            
            if baseline_value != 0:
                delta_percent = (delta_value / abs(baseline_value)) * 100.0
            elif delta_value != 0:
                # Baseline is 0 but current is not - treat as infinite % change
                delta_percent = float('inf') if delta_value > 0 else float('-inf')
        
        if baseline_timestamp and current_timestamp:
            time_delta = pd.to_datetime(current_timestamp) - pd.to_datetime(baseline_timestamp)
            time_delta_hours = time_delta.total_seconds() / 3600.0
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(delta_value)
        
        # Classify clinical significance
        clinical_significance = self._classify_significance(
            parameter_name,
            delta_value,
            delta_percent
        )
        
        return QuantitativeDelta(
            parameter_name=parameter_name,
            baseline_value=baseline_value,
            current_value=current_value,
            delta_value=delta_value,
            delta_percent=delta_percent,
            time_delta_hours=time_delta_hours,
            baseline_timestamp=baseline_timestamp,
            current_timestamp=current_timestamp,
            clinical_significance=clinical_significance,
            trend_direction=trend_direction,
            pattern_type=None  # Will be determined by trend integration
        )
    
    def analyze_series(
        self,
        parameter_name: str,
        series_df: pd.DataFrame,
        value_column: str = 'VALUENUM',
        timestamp_column: str = 'CHARTTIME'
    ) -> Optional[QuantitativeDelta]:
        """
        Analyze a time series to calculate delta from first to last value.
        
        Args:
            parameter_name: Name of the parameter
            series_df: DataFrame with value and timestamp columns
            value_column: Column name for values
            timestamp_column: Column name for timestamps
        
        Returns:
            QuantitativeDelta or None if insufficient data
        """
        if series_df.empty or value_column not in series_df.columns:
            return None
        
        # Sort by timestamp
        series_df = series_df.copy()
        series_df[timestamp_column] = pd.to_datetime(series_df[timestamp_column])
        series_df = series_df.sort_values(timestamp_column)
        
        # Get valid numeric values
        values = pd.to_numeric(series_df[value_column], errors='coerce').dropna()
        
        if len(values) < 2:
            return None
        
        baseline_value = float(values.iloc[0])
        current_value = float(values.iloc[-1])
        baseline_timestamp = series_df[timestamp_column].iloc[0]
        current_timestamp = series_df[timestamp_column].iloc[-1]
        
        return self.calculate_delta(
            parameter_name=parameter_name,
            baseline_value=baseline_value,
            current_value=current_value,
            baseline_timestamp=baseline_timestamp,
            current_timestamp=current_timestamp
        )
    
    def detect_pattern(
        self,
        series_df: pd.DataFrame,
        value_column: str = 'VALUENUM',
        timestamp_column: str = 'CHARTTIME'
    ) -> Optional[str]:
        """
        Detect pattern type in a time series.
        
        Patterns:
        - 'acute': Rapid change over short time (<24h)
        - 'chronic': Gradual change over longer time (>72h)
        - 'fluctuating': Multiple direction changes
        - 'stable': Minimal variation
        
        Args:
            series_df: DataFrame with value and timestamp columns
            value_column: Column name for values
            timestamp_column: Column name for timestamps
        
        Returns:
            Pattern type string or None
        """
        if series_df.empty or len(series_df) < 2:
            return None
        
        series_df = series_df.copy()
        series_df[timestamp_column] = pd.to_datetime(series_df[timestamp_column])
        series_df = series_df.sort_values(timestamp_column)
        
        values = pd.to_numeric(series_df[value_column], errors='coerce').dropna()
        
        if len(values) < 2:
            return None
        
        # Calculate time span
        time_span = (series_df[timestamp_column].iloc[-1] - 
                    series_df[timestamp_column].iloc[0])
        time_span_hours = time_span.total_seconds() / 3600.0
        
        # Calculate coefficient of variation
        if values.std() > 0:
            cv = values.std() / values.mean()
        else:
            cv = 0.0
        
        # Count direction changes
        diffs = values.diff().dropna()
        direction_changes = (diffs * diffs.shift(1) < 0).sum()
        
        # Classify pattern
        if time_span_hours < 24 and abs(values.iloc[-1] - values.iloc[0]) > values.std() * 2:
            return 'acute'
        elif time_span_hours > 72:
            return 'chronic'
        elif direction_changes >= 2:
            return 'fluctuating'
        elif cv < 0.1:
            return 'stable'
        else:
            return None
    
    def _determine_trend_direction(
        self,
        delta_value: Optional[float]
    ) -> Optional[str]:
        """Determine trend direction from delta value."""
        if delta_value is None:
            return None
        elif delta_value > 0:
            return 'increasing'
        elif delta_value < 0:
            return 'decreasing'
        else:
            return 'stable'
    
    def _classify_significance(
        self,
        parameter_name: str,
        delta_value: Optional[float],
        delta_percent: Optional[float]
    ) -> ClinicalSignificance:
        """
        Classify clinical significance (P0/P1/P2/P3).
        
        Uses both absolute delta and percentage change thresholds.
        """
        if delta_value is None:
            return ClinicalSignificance.P3_LOW
        
        # Get thresholds for this parameter
        thresholds = self.SIGNIFICANCE_THRESHOLDS.get(
            parameter_name.lower(),
            {
                'P0_delta': 999999.0,
                'P1_delta': 999999.0,
                'P2_delta': 999999.0,
                'P0_percent': 999999.0,
                'P1_percent': 999999.0,
                'P2_percent': 999999.0,
            }
        )
        
        abs_delta = abs(delta_value)
        abs_percent = abs(delta_percent) if delta_percent is not None else 0.0
        
        # Check P0 (Critical)
        if (abs_delta >= thresholds.get('P0_delta', 0) or
            abs_percent >= thresholds.get('P0_percent', 0)):
            return ClinicalSignificance.P0_CRITICAL
        
        # Check P1 (High)
        if (abs_delta >= thresholds.get('P1_delta', 0) or
            abs_percent >= thresholds.get('P1_percent', 0)):
            return ClinicalSignificance.P1_HIGH
        
        # Check P2 (Moderate)
        if (abs_delta >= thresholds.get('P2_delta', 0) or
            abs_percent >= thresholds.get('P2_percent', 0)):
            return ClinicalSignificance.P2_MODERATE
        
        # P3 (Low)
        return ClinicalSignificance.P3_LOW


# ============================================================================
# 3. TREND INTEGRATION ENGINE
# ============================================================================

class TrendIntegrationEngine:
    """
    Integrates with existing Lab Trend calculations from Barnabus Lab Ontology.
    
    Combines discrete changes with continuous trends to generate
    unified change analysis.
    """
    
    def __init__(
        self,
        lab_trend_data: Optional[Dict[str, Any]] = None,
        temporal_patterns: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize with existing trend data.
        
        Args:
            lab_trend_data: Lab trend data from Barnabus Lab Ontology
                          Format: {lab_name: {value, timestamp, trend, delta, etc.}}
            temporal_patterns: Temporal pattern data from existing analyzers
                             Format: {parameter: {delta_XXh, slope_XXh, pattern_flags}}
        """
        self.lab_trend_data = lab_trend_data or {}
        self.temporal_patterns = temporal_patterns or {}
    
    def integrate_trends(
        self,
        quantitative_delta: QuantitativeDelta,
        parameter_name: str
    ) -> Dict[str, Any]:
        """
        Integrate quantitative delta with existing trend calculations.
        
        Args:
            quantitative_delta: Calculated quantitative delta
            parameter_name: Name of the parameter
        
        Returns:
            Integrated trend data dictionary
        """
        integrated = {
            'quantitative_delta': {
                'delta_value': quantitative_delta.delta_value,
                'delta_percent': quantitative_delta.delta_percent,
                'time_delta_hours': quantitative_delta.time_delta_hours,
            },
            'lab_trend': None,
            'temporal_pattern': None,
            'unified_trend': None,
        }
        
        # Integrate with lab trend data (from Barnabus Lab Ontology)
        lab_trend = self.lab_trend_data.get(parameter_name.lower())
        if lab_trend:
            integrated['lab_trend'] = {
                'last_value': lab_trend.get('value'),
                'trend_direction': lab_trend.get('trend'),
                'delta': lab_trend.get('delta'),
                'timestamp': lab_trend.get('timestamp'),
            }
        
        # Integrate with temporal patterns (from existing analyzers)
        temporal_pattern = self.temporal_patterns.get(parameter_name.lower())
        if temporal_pattern:
            integrated['temporal_pattern'] = temporal_pattern
        
        # Generate unified trend assessment
        integrated['unified_trend'] = self._generate_unified_trend(
            quantitative_delta,
            lab_trend,
            temporal_pattern
        )
        
        return integrated
    
    def _generate_unified_trend(
        self,
        quantitative_delta: QuantitativeDelta,
        lab_trend: Optional[Dict[str, Any]],
        temporal_pattern: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate unified trend assessment combining all sources.
        
        Args:
            quantitative_delta: Calculated delta
            lab_trend: Lab trend data
            temporal_pattern: Temporal pattern data
        
        Returns:
            Unified trend assessment
        """
        unified = {
            'consensus_direction': quantitative_delta.trend_direction,
            'consensus_significance': quantitative_delta.clinical_significance.value,
            'data_sources': [],
            'confidence': 'moderate',
        }
        
        # Collect data sources
        if quantitative_delta.delta_value is not None:
            unified['data_sources'].append('quantitative_delta')
        
        if lab_trend:
            unified['data_sources'].append('lab_trend')
            # Check for agreement
            lab_trend_dir = lab_trend.get('trend')
            if lab_trend_dir and lab_trend_dir == quantitative_delta.trend_direction:
                unified['confidence'] = 'high'
        
        if temporal_pattern:
            unified['data_sources'].append('temporal_pattern')
            # Check for agreement
            pattern_delta = temporal_pattern.get('delta_48h') or temporal_pattern.get('delta_24h')
            if pattern_delta:
                pattern_dir = 'increasing' if pattern_delta > 0 else 'decreasing' if pattern_delta < 0 else 'stable'
                if pattern_dir == quantitative_delta.trend_direction:
                    unified['confidence'] = 'high'
        
        # Multiple confirming sources increase confidence
        if len(unified['data_sources']) >= 2:
            if unified['confidence'] == 'moderate':
                unified['confidence'] = 'high'
        
        return unified
    
    def update_lab_trend_data(self, lab_trend_data: Dict[str, Any]) -> None:
        """Update lab trend data from Barnabus Lab Ontology."""
        self.lab_trend_data.update(lab_trend_data)
    
    def update_temporal_patterns(self, temporal_patterns: Dict[str, Any]) -> None:
        """Update temporal pattern data from existing analyzers."""
        self.temporal_patterns.update(temporal_patterns)


# ============================================================================
# LAB ONTOLOGY TREND INTEGRATION
# ============================================================================

@dataclass(frozen=True)
class LabTrend:
    """
    Lab trend data from Barnabus Lab Ontology.
    
    Represents trend information for a single lab test.
    """
    testName: str
    previousValue: Optional[float]
    currentValue: Optional[float]
    unit: Optional[str]
    delta: Optional[float]
    percentChange: Optional[float]
    rateOfChange: Optional[float]  # Change per day
    previousTimestamp: Optional[str]  # ISO 8601
    currentTimestamp: str  # ISO 8601
    trend: str  # 'increasing', 'decreasing', 'stable'
    significance: Optional[str] = None  # Clinical significance
    values: Optional[List[float]] = None  # Full time series values
    timestamps: Optional[List[str]] = None  # Full time series timestamps


@dataclass(frozen=True)
class LabValueSeries:
    """Lab value time series for temporal feature analysis."""
    testName: str
    values: List[float]
    timestamps: List[str]  # ISO 8601 format
    unit: Optional[str] = None


@dataclass(frozen=True)
class EnhancedChange:
    """ClinicalChange enhanced with temporal features."""
    change: ClinicalChange
    temporalFeatures: Optional[TemporalFeatures] = None
    rateVsNormal: Optional[float] = None  # Rate relative to normal range


@dataclass(frozen=True)
class ClinicalPattern:
    """Detected clinical pattern across changes and trends."""
    type: str  # Pattern type identifier
    description: str  # Human-readable description
    changes: List[ClinicalChange]  # Related changes
    significance: str  # 'critical', 'high', 'moderate', 'low'
    clinicalImplication: str  # Clinical interpretation
    confidence: float = 0.5  # Confidence score 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional pattern data


@dataclass(frozen=True)
class IntegratedAnalysis:
    """Integrated analysis combining discrete changes and lab trends."""
    changes: List[ClinicalChange]
    integratedTrends: List[IntegratedTrend]
    analysis: Dict[str, Any]  # Pattern analysis results


def integrate_trends_with_changes(
    discrete_changes: List[ClinicalChange],
    lab_trends: List[LabTrend],
    surgery_datetime: str
) -> IntegratedAnalysis:
    """
    Integrate Barnabus Lab Ontology trend calculations with change detection.
    
    Args:
        discrete_changes: List of discrete clinical changes
        lab_trends: List of lab trends from Barnabus Lab Ontology
        surgery_datetime: Surgery datetime in ISO 8601 format
    
    Returns:
        IntegratedAnalysis with combined changes and integrated trends
    """
    # 1. Map lab trends to change events
    trend_based_changes: List[ClinicalChange] = [
        _map_lab_trend_to_change(trend) for trend in lab_trends
    ]
    
    # 2. Combine with discrete changes (de-duplicate)
    all_changes = _merge_changes(discrete_changes, trend_based_changes)
    
    # 3. Apply surgery time filter
    surgery_dt = pd.to_datetime(surgery_datetime)
    pre_op_changes = [
        change for change in all_changes
        if pd.to_datetime(change.timestamp) < surgery_dt
    ]
    
    # 4. Generate integrated trends
    integrated_trends: List[IntegratedTrend] = [
        _create_integrated_trend(trend) for trend in lab_trends
    ]
    
    # 5. Analyze integrated patterns
    analysis = _analyze_integrated_patterns(pre_op_changes, integrated_trends)
    
    return IntegratedAnalysis(
        changes=pre_op_changes,
        integratedTrends=integrated_trends,
        analysis=analysis
    )


def _map_lab_to_category(test_name: str) -> ChangeCategory:
    """Map lab test name to clinical category."""
    test_lower = test_name.lower()
    
    # Cardiac
    if any(term in test_lower for term in ['bnp', 'troponin', 'ck', 'ck-mb', 'nt-probnp']):
        return ChangeCategory.CARDIAC
    
    # Pulmonary
    if any(term in test_lower for term in ['lactate', 'crp', 'albumin', 'spo2', 'pco2', 'po2']):
        return ChangeCategory.PULMONARY
    
    # Renal
    if any(term in test_lower for term in ['creatinine', 'bun', 'egfr', 'gfr', 'urine']):
        return ChangeCategory.RENAL
    
    # Endocrine
    if any(term in test_lower for term in ['glucose', 'a1c', 'hba1c', 'insulin', 'tsh', 't4']):
        return ChangeCategory.ENDOCRINE
    
    # Hematologic
    if any(term in test_lower for term in ['hemoglobin', 'hgb', 'hct', 'platelet', 'wbc', 'inr', 'ptt']):
        return ChangeCategory.HEMATOLOGIC
    
    # Default to cardiac for unknown
    return ChangeCategory.CARDIAC


def _generate_trend_description(trend: LabTrend) -> str:
    """
    Generate human-readable description with quantitative details.
    
    Different descriptions based on significance level.
    """
    if trend.delta is None or trend.previousValue is None or trend.currentValue is None:
        return f"{trend.testName} trend: {trend.trend}"
    
    direction = 'increased' if trend.delta > 0 else 'decreased' if trend.delta < 0 else 'stable'
    abs_delta = abs(trend.delta)
    percent = abs(trend.percentChange) if trend.percentChange is not None else 0.0
    unit = trend.unit or ''
    
    # Different descriptions based on significance
    if trend.significance and 'critical' in trend.significance.lower():
        return (
            f"CRITICAL: {trend.testName} {direction} from {trend.previousValue} to "
            f"{trend.currentValue} {unit} (Δ{abs_delta:.2f} {unit}, {percent:.1f}%)"
        )
    elif trend.significance and 'significant' in trend.significance.lower():
        return (
            f"{trend.testName} {direction} from {trend.previousValue} to "
            f"{trend.currentValue} {unit} ({percent:.1f}% change)"
        )
    else:
        return f"{trend.testName}: {trend.previousValue} → {trend.currentValue} {unit}"


def generate_trend_description(trend: LabTrend) -> str:
    """
    Public function to generate trend description with quantitative details.
    
    Wrapper around _generate_trend_description for external use.
    """
    return _generate_trend_description(trend)


# ============================================================================
# PURE CLINICAL CRITICALITY RULES
# ============================================================================

def classify_criticality(change: ClinicalChange) -> str:
    """
    Classify criticality level (P0/P1/P2) based on pure clinical rules.
    
    P0: CRITICAL - Immediate action required
    P1: IMPORTANT - Affects management
    P2: INFORMATIONAL - No immediate action
    
    Args:
        change: ClinicalChange to classify
    
    Returns:
        'P0', 'P1', or 'P2'
    """
    # P0: CRITICAL - Immediate action required
    if is_critical_change(change):
        return 'P0'
    
    # P1: IMPORTANT - Affects management
    if is_important_change(change):
        return 'P1'
    
    # P2: INFORMATIONAL - No immediate action
    return 'P2'


def is_critical_change(change: ClinicalChange) -> bool:
    """
    Determine if change is P0 CRITICAL - requires immediate action.
    
    P0 Critical changes include:
    - Critical symptoms (chest pain, SOB, syncope, active bleeding)
    - Labs with critical deltas (creatinine ≥0.3, troponin ≥0.01, BNP ≥100, etc.)
    - Starting critical medications (warfarin, antiplatelets, insulin)
    
    Args:
        change: ClinicalChange to evaluate
    
    Returns:
        True if change is P0 critical
    """
    # Symptoms
    if change.type == ChangeType.SYMPTOM:
        critical_symptoms = ['chest pain', 'shortness of breath', 'syncope', 'active bleeding']
        desc_lower = change.description.lower()
        return any(symptom in desc_lower for symptom in critical_symptoms)
    
    # Labs with critical deltas
    if change.type == ChangeType.LAB and change.quantitativeData:
        qd = change.quantitativeData
        test_name = qd.testName.lower() if qd.testName else ''
        delta = qd.delta
        percent_change = qd.percentChange
        
        # Critical thresholds by test
        critical_thresholds = {
            'creatinine': {'absolute': 0.3, 'percent': 30},
            'troponin': {'absolute': 0.01, 'percent': 50},
            'bnp': {'absolute': 100, 'percent': 50},
            'nt-probnp': {'absolute': 100, 'percent': 50},
            'inr': {'absolute': 0.5, 'percent': 30},
            'potassium': {'absolute': 0.5, 'percent': 15},
            'k+': {'absolute': 0.5, 'percent': 15},
        }
        
        # Check if test matches any threshold key
        for test_key, thresholds in critical_thresholds.items():
            if test_key in test_name:
                abs_delta = abs(delta) if delta is not None else 0.0
                abs_percent = abs(percent_change) if percent_change is not None else 0.0
                
                if abs_delta >= thresholds['absolute'] or abs_percent >= thresholds['percent']:
                    return True
                break
    
    # Medication: Starting critical drugs
    if change.type == ChangeType.MEDICATION:
        critical_meds = [
            'warfarin', 'clopidogrel', 'ticagrelor', 'prasugrel', 
            'insulin', 'heparin', 'enoxaparin', 'rivaroxaban', 
            'apixaban', 'dabigatran'
        ]
        desc_lower = change.description.lower()
        return any(med in desc_lower for med in critical_meds)
    
    # Vitals: Critical vital sign changes
    if change.type == ChangeType.VITAL:
        desc_lower = change.description.lower()
        critical_vitals = [
            'hypotension', 'hypertension', 'tachycardia', 'bradycardia',
            'hypoxia', 'respiratory distress', 'altered mental status'
        ]
        return any(vital in desc_lower for vital in critical_vitals)
    
    return False


def is_important_change(change: ClinicalChange) -> bool:
    """
    Determine if change is P1 IMPORTANT - affects management.
    
    P1 Important changes include:
    - Labs with significant but not critical changes
    - Medication adjustments (furosemide, beta-blockers, ACE inhibitors, SGLT2)
    - Moderate vital sign changes
    
    Args:
        change: ClinicalChange to evaluate
    
    Returns:
        True if change is P1 important
    """
    # Labs with significant but not critical changes
    if change.type == ChangeType.LAB and change.quantitativeData:
        qd = change.quantitativeData
        test_name = qd.testName.lower() if qd.testName else ''
        delta = qd.delta
        percent_change = qd.percentChange
        
        # Important thresholds (lower than critical)
        important_thresholds = {
            'creatinine': {'absolute': 0.1, 'percent': 10},
            'hba1c': {'absolute': 0.5, 'percent': 8},
            'a1c': {'absolute': 0.5, 'percent': 8},
            'hemoglobin': {'absolute': 1.0, 'percent': 10},
            'hgb': {'absolute': 1.0, 'percent': 10},
            'platelets': {'absolute': 50, 'percent': 20},
            'platelet': {'absolute': 50, 'percent': 20},
            'bnp': {'absolute': 50, 'percent': 25},
            'nt-probnp': {'absolute': 50, 'percent': 25},
            'sodium': {'absolute': 3.0, 'percent': 2},
            'na+': {'absolute': 3.0, 'percent': 2},
        }
        
        # Check if test matches any threshold key
        for test_key, thresholds in important_thresholds.items():
            if test_key in test_name:
                abs_delta = abs(delta) if delta is not None else 0.0
                abs_percent = abs(percent_change) if percent_change is not None else 0.0
                
                # Must meet threshold but be below critical threshold (3x multiplier)
                critical_multiplier = 3.0
                critical_abs = thresholds['absolute'] * critical_multiplier
                critical_pct = thresholds['percent'] * critical_multiplier
                
                meets_important = (
                    (abs_delta >= thresholds['absolute'] and abs_delta < critical_abs) or
                    (abs_percent >= thresholds['percent'] and abs_percent < critical_pct)
                )
                
                if meets_important:
                    return True
                break
    
    # Medication adjustments
    if change.type == ChangeType.MEDICATION:
        important_meds = [
            'furosemide', 'lasix', 'beta-blocker', 'beta blocker',
            'ace inhibitor', 'acei', 'arb', 'angiotensin',
            'sglt2', 'sglt-2', 'metformin', 'diuretic',
            'statin', 'aspirin', 'clopidogrel'
        ]
        desc_lower = change.description.lower()
        return any(med in desc_lower for med in important_meds)
    
    # Vitals: Moderate vital sign changes
    if change.type == ChangeType.VITAL:
        desc_lower = change.description.lower()
        important_vitals = [
            'elevated blood pressure', 'mild tachycardia', 'mild bradycardia',
            'mild hypoxia', 'fever', 'tachypnea'
        ]
        return any(vital in desc_lower for vital in important_vitals)
    
    # Diagnostic: New significant diagnoses
    if change.type == ChangeType.DIAGNOSTIC:
        desc_lower = change.description.lower()
        important_diagnoses = [
            'new diagnosis', 'new finding', 'abnormal', 'elevated',
            'worsening', 'progression'
        ]
        return any(diag in desc_lower for diag in important_diagnoses)
    
    return False


def _calculate_trend_criticality(trend: LabTrend) -> str:
    """
    Calculate criticality level (P0/P1/P2) for a trend.
    
    Uses pure clinical criticality rules by converting trend to ClinicalChange
    and applying classify_criticality().
    """
    if trend.significance:
        # Use provided significance if available
        if 'critical' in trend.significance.lower() or 'P0' in trend.significance:
            return 'P0'
        elif 'high' in trend.significance.lower() or 'P1' in trend.significance:
            return 'P1'
        elif 'moderate' in trend.significance.lower() or 'P2' in trend.significance:
            return 'P2'
    
    # Convert trend to ClinicalChange for classification
    # Create a temporary ClinicalChange to use pure clinical rules
    quantitative_data = QuantitativeData(
        testName=trend.testName,
        previousValue=trend.previousValue,
        currentValue=trend.currentValue,
        unit=trend.unit,
        delta=trend.delta,
        percentChange=trend.percentChange,
        rateOfChange=trend.rateOfChange
    )
    
    # Create temporary ClinicalChange for classification
    temp_change = ClinicalChange(
        timestamp=trend.currentTimestamp,
        type=ChangeType.LAB,
        category=_map_lab_to_category(trend.testName),
        description=_generate_trend_description(trend),
        quantitativeData=quantitative_data,
        criticality='P2',  # Temporary, will be overwritten
        clinicalImpact=ClinicalImpact(
            riskChange=RiskChange.NEUTRAL,
            affectsSurgicalPlanning=False,
            requiresAction=False
        )
    )
    
    # Use pure clinical criticality rules
    return classify_criticality(temp_change)


def convert_trend_to_changes(lab_trends: List[LabTrend]) -> List[ClinicalChange]:
    """
    Convert LabTrend[] to ClinicalChange[].
    
    Maps each lab trend to a clinical change event with proper categorization,
    criticality classification, and clinical impact assessment.
    
    Args:
        lab_trends: List of lab trends from Barnabus Lab Ontology
    
    Returns:
        List of ClinicalChange events
    """
    return [_map_lab_trend_to_change(trend) for trend in lab_trends]


def _does_affect_surgical_planning(trend: LabTrend) -> bool:
    """Determine if trend affects surgical planning."""
    criticality = _calculate_trend_criticality(trend)
    return criticality in ['P0', 'P1']


def _assess_trend_impact(trend: LabTrend) -> ClinicalImpact:
    """Assess clinical impact of a trend."""
    trend_dir = trend.trend
    criticality = _calculate_trend_criticality(trend)
    
    # Determine risk change direction
    test_lower = trend.testName.lower()
    
    # For most labs, increasing values indicate increased risk
    # Exceptions: hemoglobin, albumin (decreasing = increased risk)
    if trend_dir == 'increasing':
        if any(term in test_lower for term in ['hemoglobin', 'hgb', 'albumin', 'sodium']):
            risk_change = RiskChange.DECREASE  # Decreasing values = increased risk
        else:
            risk_change = RiskChange.INCREASE
    elif trend_dir == 'decreasing':
        if any(term in test_lower for term in ['hemoglobin', 'hgb', 'albumin', 'sodium']):
            risk_change = RiskChange.INCREASE  # Decreasing values = increased risk
        else:
            risk_change = RiskChange.DECREASE
    else:
        risk_change = RiskChange.NEUTRAL
    
    # Determine if affects surgical planning
    affects_surgical_planning = criticality in ['P0', 'P1']
    
    # Determine if requires action
    requires_action = criticality == 'P0' or (criticality == 'P1' and trend_dir != 'stable')
    
    return ClinicalImpact(
        riskChange=risk_change,
        affectsSurgicalPlanning=affects_surgical_planning,
        requiresAction=requires_action
    )


def _map_lab_trend_to_change(trend: LabTrend) -> ClinicalChange:
    """Map LabTrend to ClinicalChange event using pure clinical criticality rules."""
    quantitative_data = QuantitativeData(
        testName=trend.testName,
        previousValue=trend.previousValue,
        currentValue=trend.currentValue,
        unit=trend.unit,
        delta=trend.delta,
        percentChange=trend.percentChange,
        rateOfChange=trend.rateOfChange
    )
    
    # Create temporary ClinicalChange for classification
    temp_change = ClinicalChange(
        timestamp=trend.currentTimestamp,
        type=ChangeType.LAB,
        category=_map_lab_to_category(trend.testName),
        description=_generate_trend_description(trend),
        quantitativeData=quantitative_data,
        criticality='P2',  # Temporary, will be overwritten
        clinicalImpact=ClinicalImpact(
            riskChange=RiskChange.NEUTRAL,
            affectsSurgicalPlanning=False,
            requiresAction=False
        )
    )
    
    # Use pure clinical criticality rules
    criticality = classify_criticality(temp_change)
    
    return ClinicalChange(
        timestamp=trend.currentTimestamp,
        type=ChangeType.LAB,
        category=_map_lab_to_category(trend.testName),
        description=_generate_trend_description(trend),
        quantitativeData=quantitative_data,
        criticality=criticality,
        clinicalImpact=_assess_trend_impact(trend)
    )


def _merge_changes(
    discrete_changes: List[ClinicalChange],
    trend_based_changes: List[ClinicalChange]
) -> List[ClinicalChange]:
    """
    Merge discrete changes with trend-based changes, de-duplicating.
    
    If the same test appears in both lists, prefer the more recent one
    or the one with higher criticality.
    """
    # Group by test name
    changes_by_test: Dict[str, List[ClinicalChange]] = {}
    
    for change in discrete_changes + trend_based_changes:
        test_name = None
        if change.quantitativeData and change.quantitativeData.testName:
            test_name = change.quantitativeData.testName.lower()
        else:
            # Use description as fallback identifier
            test_name = change.description.lower()
        
        if test_name not in changes_by_test:
            changes_by_test[test_name] = []
        changes_by_test[test_name].append(change)
    
    # For each test, keep the most significant change
    merged: List[ClinicalChange] = []
    for test_name, changes in changes_by_test.items():
        if len(changes) == 1:
            merged.append(changes[0])
        else:
            # Prefer higher criticality, then more recent timestamp
            changes.sort(
                key=lambda c: (
                    {'P0': 0, 'P1': 1, 'P2': 2}.get(c.criticality, 3),
                    pd.to_datetime(c.timestamp)
                ),
                reverse=True
            )
            merged.append(changes[0])  # Keep most significant
    
    return merged


def _calculate_acceleration(values: List[float]) -> Optional[float]:
    """Calculate acceleration (rate of rate change) from time series."""
    if not values or len(values) < 3:
        return None
    
    # Calculate first derivatives (rates of change)
    first_derivatives = [values[i+1] - values[i] for i in range(len(values) - 1)]
    
    if len(first_derivatives) < 2:
        return None
    
    # Calculate second derivative (acceleration)
    accelerations = [first_derivatives[i+1] - first_derivatives[i] 
                     for i in range(len(first_derivatives) - 1)]
    
    if not accelerations:
        return None
    
    # Return mean acceleration
    return float(np.mean(accelerations))


def _calculate_volatility(values: List[float]) -> Optional[float]:
    """Calculate volatility (coefficient of variation) from time series."""
    if not values or len(values) < 2:
        return None
    
    values_array = np.array(values)
    mean_val = np.mean(values_array)
    
    if mean_val == 0:
        return None
    
    std_val = np.std(values_array)
    cv = std_val / abs(mean_val)  # Coefficient of variation
    
    return float(cv)


def _detect_diurnal_pattern(timestamps: List[str]) -> Optional[Dict[str, Any]]:
    """Detect diurnal (time-of-day) patterns in timestamps."""
    if not timestamps or len(timestamps) < 3:
        return None
    
    # Parse timestamps and extract hours
    hours = []
    for ts in timestamps:
        try:
            dt = pd.to_datetime(ts)
            hours.append(dt.hour)
        except:
            continue
    
    if len(hours) < 3:
        return None
    
    # Group by time of day (morning: 6-12, afternoon: 12-18, evening: 18-24, night: 0-6)
    time_periods = {
        'morning': sum(1 for h in hours if 6 <= h < 12),
        'afternoon': sum(1 for h in hours if 12 <= h < 18),
        'evening': sum(1 for h in hours if 18 <= h < 24),
        'night': sum(1 for h in hours if h < 6 or h >= 24)
    }
    
    # Find dominant time period
    dominant_period = max(time_periods.items(), key=lambda x: x[1])
    
    return {
        'hasPattern': dominant_period[1] > len(hours) * 0.4,  # >40% in one period
        'dominantPeriod': dominant_period[0],
        'distribution': time_periods
    }


def _count_outliers(values: List[float], method: str = 'iqr') -> int:
    """Count outliers in time series using IQR method."""
    if not values or len(values) < 4:
        return 0
    
    values_array = np.array(values)
    
    if method == 'iqr':
        Q1 = np.percentile(values_array, 25)
        Q3 = np.percentile(values_array, 75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return 0
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = np.sum((values_array < lower_bound) | (values_array > upper_bound))
        return int(outliers)
    
    return 0


def _get_normal_range(test_name: str) -> Optional[Dict[str, float]]:
    """
    Get normal range for a lab test.
    
    Returns:
        Dict with 'min' and 'max' values, or None if unknown
    """
    test_lower = test_name.lower()
    
    # Common normal ranges (approximate, may vary by institution)
    normal_ranges = {
        'creatinine': {'min': 0.6, 'max': 1.2},
        'bnp': {'min': 0, 'max': 100},
        'nt-probnp': {'min': 0, 'max': 300},
        'troponin': {'min': 0, 'max': 0.04},
        'hemoglobin': {'min': 12.0, 'max': 16.0},
        'hgb': {'min': 12.0, 'max': 16.0},
        'platelets': {'min': 150, 'max': 450},
        'sodium': {'min': 135, 'max': 145},
        'potassium': {'min': 3.5, 'max': 5.0},
        'glucose': {'min': 70, 'max': 100},
        'inr': {'min': 0.9, 'max': 1.1},
        'albumin': {'min': 3.5, 'max': 5.0},
    }
    
    # Check if test matches any key
    for key, range_dict in normal_ranges.items():
        if key in test_lower:
            return range_dict
    
    return None


def _calculate_rate_vs_normal(
    values: List[float],
    normal_range: Optional[Dict[str, float]]
) -> Optional[float]:
    """
    Calculate rate of change relative to normal range.
    
    Returns:
        Rate as multiple of normal range width, or None if cannot calculate
    """
    if not values or len(values) < 2 or not normal_range:
        return None
    
    # Calculate rate of change per day (simplified - assumes uniform spacing)
    rate = (values[-1] - values[0]) / len(values)  # Approximate rate
    
    # Normal range width
    range_width = normal_range['max'] - normal_range['min']
    
    if range_width == 0:
        return None
    
    # Rate relative to normal range width
    rate_vs_normal = rate / range_width
    
    return float(rate_vs_normal)


def _detect_diurnal_pattern_enhanced(
    timestamps: List[str],
    values: List[float]
) -> Optional[Dict[str, Any]]:
    """
    Enhanced diurnal pattern detection with value correlation.
    
    Detects if values vary by time of day.
    """
    if not timestamps or not values or len(timestamps) < 3 or len(values) < 3:
        return None
    
    # Parse timestamps and extract hours
    hours = []
    valid_values = []
    for i, ts in enumerate(timestamps):
        try:
            dt = pd.to_datetime(ts)
            hours.append(dt.hour)
            if i < len(values):
                valid_values.append(values[i])
        except:
            continue
    
    if len(hours) < 3:
        return None
    
    # Group by time of day
    time_periods = {
        'morning': {'hours': [], 'values': []},  # 6-12
        'afternoon': {'hours': [], 'values': []},  # 12-18
        'evening': {'hours': [], 'values': []},  # 18-24
        'night': {'hours': [], 'values': []}  # 0-6
    }
    
    for h, val in zip(hours, valid_values):
        if 6 <= h < 12:
            time_periods['morning']['hours'].append(h)
            time_periods['morning']['values'].append(val)
        elif 12 <= h < 18:
            time_periods['afternoon']['hours'].append(h)
            time_periods['afternoon']['values'].append(val)
        elif 18 <= h < 24:
            time_periods['evening']['hours'].append(h)
            time_periods['evening']['values'].append(val)
        else:
            time_periods['night']['hours'].append(h)
            time_periods['night']['values'].append(val)
    
    # Calculate mean values per period
    period_means = {}
    for period, data in time_periods.items():
        if data['values']:
            period_means[period] = np.mean(data['values'])
    
    # Check for significant variation (>10% of mean)
    if period_means:
        overall_mean = np.mean(list(period_means.values()))
        if overall_mean > 0:
            max_variation = max(abs(v - overall_mean) for v in period_means.values())
            variation_pct = (max_variation / overall_mean) * 100
            
            dominant_period = max(period_means.items(), key=lambda x: len(time_periods[x[0]]['values']))
            
            return {
                'hasPattern': variation_pct > 10.0,
                'dominantPeriod': dominant_period[0],
                'variationPercent': variation_pct,
                'periodMeans': {k: float(v) for k, v in period_means.items()},
                'distribution': {
                    k: len(data['values'])
                    for k, data in time_periods.items()
                }
            }
    
    return None


def enhance_with_temporal_features(
    changes: List[ClinicalChange],
    lab_value_series: List[LabValueSeries]
) -> List[EnhancedChange]:
    """
    Enhance ClinicalChange[] with temporal features from lab value series.
    
    Integrates Barnabus temporal features including:
    - Acceleration (rate of rate change)
    - Volatility (standard deviation of changes)
    - Diurnal pattern detection
    - Outlier analysis
    - Rate relative to normal range
    
    Args:
        changes: List of ClinicalChange events
        lab_value_series: List of lab value time series
    
    Returns:
        List of EnhancedChange with temporal features
    """
    enhanced_changes = []
    
    for change in changes:
        enhanced_change = EnhancedChange(change=change)
        
        # Only enhance lab changes with quantitative data
        if change.type == ChangeType.LAB and change.quantitativeData:
            test_name = change.quantitativeData.testName
            if test_name:
                # Find matching lab value series
                series = next(
                    (s for s in lab_value_series if s.testName.lower() == test_name.lower()),
                    None
                )
                
                if series and len(series.values) >= 2:
                    # Calculate temporal features
                    acceleration = _calculate_acceleration(series.values)
                    volatility = _calculate_volatility(series.values)
                    diurnal_pattern = _detect_diurnal_pattern_enhanced(
                        series.timestamps,
                        series.values
                    )
                    outlier_count = _count_outliers(series.values)
                    
                    # Calculate rate vs normal
                    normal_range = _get_normal_range(test_name)
                    rate_vs_normal = None
                    if normal_range:
                        rate_vs_normal = _calculate_rate_vs_normal(
                            series.values,
                            normal_range
                        )
                    
                    # Create temporal features
                    temporal_features = TemporalFeatures(
                        acceleration=acceleration,
                        volatility=volatility,
                        seasonality=diurnal_pattern,
                        outlierCount=outlier_count
                    )
                    
                    enhanced_change = EnhancedChange(
                        change=change,
                        temporalFeatures=temporal_features,
                        rateVsNormal=rate_vs_normal
                    )
        
        enhanced_changes.append(enhanced_change)
    
    return enhanced_changes


# ============================================================================
# CLINICAL PATTERN DETECTION
# ============================================================================

def detect_clinical_patterns(
    changes: List[ClinicalChange],
    integrated_trends: List[IntegratedTrend]
) -> List[ClinicalPattern]:
    """
    Detect clinical patterns across changes and integrated trends.
    
    Identifies:
    - Symptom-lab correlations (temporally related)
    - Multi-system worsening (trend convergence)
    - Accelerating trends
    - Other clinically significant patterns
    
    Args:
        changes: List of ClinicalChange events
        integrated_trends: List of IntegratedTrend data
    
    Returns:
        List of detected ClinicalPattern objects
    """
    patterns: List[ClinicalPattern] = []
    
    # 1. CORRELATION PATTERNS: Lab changes with symptoms
    symptom_changes = [c for c in changes if c.type == ChangeType.SYMPTOM]
    lab_changes = [c for c in changes if c.type == ChangeType.LAB]
    
    for symptom in symptom_changes:
        # Find temporally related lab changes
        related_labs = [
            lab for lab in lab_changes
            if _is_temporally_related(symptom.timestamp, lab.timestamp, hours=24)
            and _is_clinically_related(symptom, lab)
        ]
        
        if related_labs:
            patterns.append(ClinicalPattern(
                type='symptom_lab_correlation',
                description=f"{symptom.description} with related lab changes",
                changes=[symptom] + related_labs,
                significance='high',
                clinicalImplication='Consider causal relationship between symptom and lab abnormalities',
                confidence=0.7,
                metadata={
                    'symptom': symptom.description,
                    'relatedLabs': [lab.quantitativeData.testName for lab in related_labs if lab.quantitativeData],
                    'timeWindowHours': 24
                }
            ))
    
    # 2. TREND CONVERGENCE: Multiple trends pointing same direction
    worsening_trends = [
        t for t in integrated_trends
        if t.trendDirection in ['increasing', 'worsening', 'decreasing']
        and t.significance and 'significant' in t.significance.lower()
    ]
    
    # Group by direction
    increasing_trends = [t for t in worsening_trends if t.trendDirection == 'increasing']
    decreasing_trends = [t for t in worsening_trends if t.trendDirection == 'decreasing']
    
    # Multi-system worsening (increasing trends in multiple systems)
    if len(increasing_trends) >= 2:
        # Create synthetic changes for trends
        trend_changes = [
            ClinicalChange(
                timestamp=datetime.now().isoformat(),
                type=ChangeType.LAB,
                category=_map_lab_to_category(trend.test or trend.parameterName),
                description=f"{trend.test or trend.parameterName} showing {trend.trendDirection} trend",
                quantitativeData=None,
                criticality='P1',
                clinicalImpact=ClinicalImpact(
                    riskChange=RiskChange.INCREASE,
                    affectsSurgicalPlanning=True,
                    requiresAction=True
                )
            )
            for trend in increasing_trends
        ]
        
        patterns.append(ClinicalPattern(
            type='multi_system_worsening',
            description=f"Multiple systems showing worsening trends ({len(increasing_trends)} systems)",
            changes=trend_changes,
            significance='critical',
            clinicalImplication='Possible systemic decompensation - requires comprehensive evaluation',
            confidence=0.8,
            metadata={
                'affectedSystems': [t.test or t.parameterName for t in increasing_trends],
                'trendCount': len(increasing_trends)
            }
        ))
    
    # Multi-system improvement (decreasing trends in multiple systems)
    if len(decreasing_trends) >= 2:
        trend_changes = [
            ClinicalChange(
                timestamp=datetime.now().isoformat(),
                type=ChangeType.LAB,
                category=_map_lab_to_category(trend.test or trend.parameterName),
                description=f"{trend.test or trend.parameterName} showing {trend.trendDirection} trend",
                quantitativeData=None,
                criticality='P2',
                clinicalImpact=ClinicalImpact(
                    riskChange=RiskChange.DECREASE,
                    affectsSurgicalPlanning=False,
                    requiresAction=False
                )
            )
            for trend in decreasing_trends
        ]
        
        patterns.append(ClinicalPattern(
            type='multi_system_improvement',
            description=f"Multiple systems showing improving trends ({len(decreasing_trends)} systems)",
            changes=trend_changes,
            significance='moderate',
            clinicalImplication='Positive trend across multiple systems - patient may be stabilizing',
            confidence=0.7,
            metadata={
                'affectedSystems': [t.test or t.parameterName for t in decreasing_trends],
                'trendCount': len(decreasing_trends)
            }
        ))
    
    # 3. RATE CHANGE PATTERNS: Acceleration in trends
    for trend in integrated_trends:
        if trend.temporalFeatures and trend.temporalFeatures.acceleration is not None:
            acceleration = trend.temporalFeatures.acceleration
            
            # Threshold for significant acceleration (configurable)
            acceleration_threshold = 0.5
            
            if abs(acceleration) > acceleration_threshold:
                test_name = trend.test or trend.parameterName
                
                trend_change = ClinicalChange(
                    timestamp=datetime.now().isoformat(),
                    type=ChangeType.LAB,
                    category=_map_lab_to_category(test_name),
                    description=f"{test_name} showing accelerating {trend.trendDirection}",
                    quantitativeData=None,
                    criticality=trend.significance or 'P1',
                    clinicalImpact=ClinicalImpact(
                        riskChange=RiskChange.INCREASE if acceleration > 0 else RiskChange.DECREASE,
                        affectsSurgicalPlanning=True,
                        requiresAction=True
                    )
                )
                
                patterns.append(ClinicalPattern(
                    type='accelerating_trend',
                    description=f"{test_name} showing accelerating {trend.trendDirection} (acceleration: {acceleration:.2f})",
                    changes=[trend_change],
                    significance=trend.significance or 'high',
                    clinicalImplication='Trend may be worsening/improving faster than expected - monitor closely',
                    confidence=0.75,
                    metadata={
                        'testName': test_name,
                        'acceleration': acceleration,
                        'trendDirection': trend.trendDirection,
                        'threshold': acceleration_threshold
                    }
                ))
    
    # 4. VOLATILITY PATTERNS: High variability in trends
    for trend in integrated_trends:
        if trend.temporalFeatures and trend.temporalFeatures.volatility is not None:
            volatility = trend.temporalFeatures.volatility
            
            # High volatility threshold (coefficient of variation > 0.3)
            if volatility > 0.3:
                test_name = trend.test or trend.parameterName
                
                trend_change = ClinicalChange(
                    timestamp=datetime.now().isoformat(),
                    type=ChangeType.LAB,
                    category=_map_lab_to_category(test_name),
                    description=f"{test_name} showing high variability",
                    quantitativeData=None,
                    criticality='P1',
                    clinicalImpact=ClinicalImpact(
                        riskChange=RiskChange.NEUTRAL,
                        affectsSurgicalPlanning=True,
                        requiresAction=True
                    )
                )
                
                patterns.append(ClinicalPattern(
                    type='high_volatility',
                    description=f"{test_name} showing high variability (volatility: {volatility:.2f})",
                    changes=[trend_change],
                    significance='moderate',
                    clinicalImplication='High variability may indicate unstable condition or measurement issues',
                    confidence=0.65,
                    metadata={
                        'testName': test_name,
                        'volatility': volatility,
                        'threshold': 0.3
                    }
                ))
    
    # 5. CROSS-CATEGORY PATTERNS: Changes across multiple clinical categories
    changes_by_category: Dict[str, List[ClinicalChange]] = {}
    for change in changes:
        cat = change.category.value
        if cat not in changes_by_category:
            changes_by_category[cat] = []
        changes_by_category[cat].append(change)
    
    # Find categories with multiple significant changes
    multi_category_changes = {
        cat: changes_list
        for cat, changes_list in changes_by_category.items()
        if len([c for c in changes_list if c.criticality in ['P0', 'P1']]) >= 2
    }
    
    if len(multi_category_changes) >= 2:
        all_multi_changes = []
        for changes_list in multi_category_changes.values():
            all_multi_changes.extend(changes_list)
        
        patterns.append(ClinicalPattern(
            type='cross_category_instability',
            description=f"Significant changes across {len(multi_category_changes)} clinical categories",
            changes=all_multi_changes,
            significance='critical',
            clinicalImplication='Multi-system instability - requires comprehensive evaluation and may affect surgical planning',
            confidence=0.85,
            metadata={
                'affectedCategories': list(multi_category_changes.keys()),
                'categoryCount': len(multi_category_changes),
                'totalChanges': len(all_multi_changes)
            }
        ))
    
    return patterns


def _is_temporally_related(
    timestamp1: str,
    timestamp2: str,
    hours: int = 24
) -> bool:
    """
    Check if two timestamps are temporally related (within specified hours).
    
    Args:
        timestamp1: First timestamp (ISO 8601)
        timestamp2: Second timestamp (ISO 8601)
        hours: Maximum hours apart to be considered related
    
    Returns:
        True if timestamps are within specified hours
    """
    try:
        dt1 = pd.to_datetime(timestamp1)
        dt2 = pd.to_datetime(timestamp2)
        
        time_diff = abs((dt2 - dt1).total_seconds() / 3600.0)
        return time_diff <= hours
    except:
        return False


def _is_clinically_related(
    change1: ClinicalChange,
    change2: ClinicalChange
) -> bool:
    """
    Check if two changes are clinically related.
    
    Args:
        change1: First clinical change
        change2: Second clinical change
    
    Returns:
        True if changes are clinically related
    """
    # Symptom-lab relationships
    if change1.type == ChangeType.SYMPTOM and change2.type == ChangeType.LAB:
        symptom_desc = change1.description.lower()
        lab_test = change2.quantitativeData.testName.lower() if change2.quantitativeData else ''
        
        # Chest pain / SOB related to cardiac labs
        if any(term in symptom_desc for term in ['chest pain', 'shortness of breath', 'sob']):
            if any(term in lab_test for term in ['troponin', 'bnp', 'ck', 'ck-mb']):
                return True
        
        # Syncope related to cardiac/neurologic labs
        if 'syncope' in symptom_desc:
            if any(term in lab_test for term in ['troponin', 'glucose', 'sodium', 'potassium']):
                return True
        
        # Active bleeding related to hematologic labs
        if 'bleeding' in symptom_desc or 'hemorrhage' in symptom_desc:
            if any(term in lab_test for term in ['hemoglobin', 'hgb', 'platelet', 'inr', 'ptt']):
                return True
    
    # Lab-lab relationships (same category)
    if change1.type == ChangeType.LAB and change2.type == ChangeType.LAB:
        if change1.category == change2.category:
            return True
    
    return False


# ============================================================================
# CLINICAL SUMMARIZATION
# ============================================================================

def generate_change_summary(analysis: ChangeAnalysis) -> ChangeSummary:
    """
    Generate comprehensive change summary from ChangeAnalysis.
    
    Groups changes by criticality, generates human-readable summaries,
    calculates stability metrics, and detects clinical patterns.
    
    Args:
        analysis: ChangeAnalysis result
    
    Returns:
        ChangeSummary with all summary information
    """
    # Group by criticality
    by_criticality = {
        'P0': [c for c in analysis.changes if c.criticality == 'P0'],
        'P1': [c for c in analysis.changes if c.criticality == 'P1'],
        'P2': [c for c in analysis.changes if c.criticality == 'P2']
    }
    
    # Generate human-readable summaries
    clinical_summary = generate_clinical_summary_text(by_criticality)
    export_summary = generate_export_summary_text(by_criticality)
    
    # Calculate stability metrics
    stability_metrics = calculate_stability_metrics(analysis)
    
    # Detect clinical patterns
    detected_patterns = detect_clinical_patterns(
        analysis.changes,
        analysis.integratedTrends
    )
    
    # Create change counts
    change_counts = ChangeCounts(
        total=len(analysis.changes),
        critical=len(by_criticality['P0']),
        important=len(by_criticality['P1']),
        informational=len(by_criticality['P2'])
    )
    
    return ChangeSummary(
        analysisWindow=analysis.analysisWindow,
        changeCounts=change_counts,
        summaries={
            'clinical': clinical_summary,
            'export': export_summary
        },
        stability=stability_metrics,
        detectedPatterns=detected_patterns
    )


def generate_clinical_summary_text(
    by_criticality: Dict[str, List[ClinicalChange]]
) -> str:
    """
    Generate human-readable clinical summary text grouped by criticality.
    
    Args:
        by_criticality: Dictionary of changes grouped by criticality level
    
    Returns:
        Formatted clinical summary text
    """
    sections: List[str] = []
    
    # P0: Critical Changes
    if by_criticality.get('P0'):
        sections.append('🚨 CRITICAL CHANGES (Require Immediate Action):')
        for change in by_criticality['P0']:
            sections.append(f"• {change.description}")
            
            # Add quantitative details if available
            if change.quantitativeData:
                q = change.quantitativeData
                delta_str = f"{abs(q.delta):.2f}" if q.delta is not None else "N/A"
                percent_str = f"{abs(q.percentChange):.1f}" if q.percentChange is not None else "N/A"
                unit_str = q.unit or ""
                
                if q.previousValue is not None and q.currentValue is not None:
                    sections.append(
                        f"  {q.testName or 'Lab'}: {q.previousValue} → {q.currentValue} {unit_str} "
                        f"(Δ{delta_str} {unit_str}, {percent_str}%)"
                    )
                elif q.delta is not None:
                    sections.append(
                        f"  {q.testName or 'Lab'}: Change of {delta_str} {unit_str} ({percent_str}%)"
                    )
            
            # Add clinical impact if significant
            if change.clinicalImpact.requiresAction:
                action_msg = 'May affect surgical planning' if change.clinicalImpact.affectsSurgicalPlanning else 'Review recommended'
                sections.append(f"  ⚠️ Action Required: {action_msg}")
    
    # P1: Important Changes
    if by_criticality.get('P1'):
        sections.append('\n⚠️ IMPORTANT CHANGES (Affect Management):')
        for change in by_criticality['P1']:
            sections.append(f"• {change.description}")
            
            # For labs, show the trend
            if change.type == ChangeType.LAB and change.quantitativeData:
                q = change.quantitativeData
                if q.delta is not None:
                    trend_symbol = '↑' if q.delta > 0 else '↓' if q.delta < 0 else '→'
                    delta_abs = abs(q.delta)
                    unit_str = q.unit or ""
                    sections.append(f"  Trend: {trend_symbol} {delta_abs:.2f} {unit_str} over period")
    
    # P2: Informational Changes
    if by_criticality.get('P2'):
        sections.append('\nℹ️ INFORMATIONAL CHANGES:')
        for change in by_criticality['P2']:
            sections.append(f"• {change.description}")
    
    # If no changes, add message
    if not sections:
        sections.append('No significant clinical changes detected in the analysis window.')
    
    return '\n'.join(sections)


def generate_export_summary_text(
    by_criticality: Dict[str, List[ClinicalChange]]
) -> str:
    """
    Generate export-friendly summary text (concise, structured).
    
    Args:
        by_criticality: Dictionary of changes grouped by criticality level
    
    Returns:
        Formatted export summary text
    """
    sections: List[str] = []
    
    total = sum(len(changes) for changes in by_criticality.values())
    
    if total == 0:
        return "No clinical changes detected."
    
    # Summary header
    sections.append(f"CHANGE ANALYSIS SUMMARY ({total} total changes)")
    sections.append("=" * 50)
    
    # P0: Critical
    p0_count = len(by_criticality.get('P0', []))
    if p0_count > 0:
        sections.append(f"\nCRITICAL (P0): {p0_count} changes")
        for i, change in enumerate(by_criticality['P0'], 1):
            sections.append(f"  {i}. {change.description}")
            if change.quantitativeData and change.quantitativeData.testName:
                q = change.quantitativeData
                if q.delta is not None:
                    sections.append(f"     {q.testName}: Δ{q.delta:.2f} {q.unit or ''}")
    
    # P1: Important
    p1_count = len(by_criticality.get('P1', []))
    if p1_count > 0:
        sections.append(f"\nIMPORTANT (P1): {p1_count} changes")
        for i, change in enumerate(by_criticality['P1'], 1):
            sections.append(f"  {i}. {change.description}")
    
    # P2: Informational
    p2_count = len(by_criticality.get('P2', []))
    if p2_count > 0:
        sections.append(f"\nINFORMATIONAL (P2): {p2_count} changes")
        # Only list first 5 for brevity
        for i, change in enumerate(by_criticality['P2'][:5], 1):
            sections.append(f"  {i}. {change.description}")
        if p2_count > 5:
            sections.append(f"  ... and {p2_count - 5} more")
    
    return '\n'.join(sections)


def calculate_stability_metrics(analysis: ChangeAnalysis) -> StabilityMetrics:
    """
    Calculate stability metrics from change analysis.
    
    Args:
        analysis: ChangeAnalysis result
    
    Returns:
        StabilityMetrics with calculated values
    """
    total_changes = len(analysis.changes)
    critical_changes = len([c for c in analysis.changes if c.criticality == 'P0'])
    important_changes = len([c for c in analysis.changes if c.criticality == 'P1'])
    
    # Calculate stability score (0-100)
    # Base score starts at 100, decreases with changes
    stability_score = 100.0
    
    # Deduct points for changes
    stability_score -= critical_changes * 30  # -30 per critical change
    stability_score -= important_changes * 10  # -10 per important change
    
    # Consider volatility from trends
    if analysis.integratedTrends:
        high_volatility_count = sum(
            1 for t in analysis.integratedTrends
            if t.temporalFeatures and t.temporalFeatures.volatility
            and t.temporalFeatures.volatility > 0.3
        )
        stability_score -= high_volatility_count * 5  # -5 per high volatility trend
    
    # Ensure score stays in valid range
    stability_score = max(0.0, min(100.0, stability_score))
    
    # Calculate volatility index (0-1)
    # Based on number of changes and their criticality
    volatility_index = 0.0
    if total_changes > 0:
        volatility_index = (
            critical_changes * 0.5 +
            important_changes * 0.3 +
            (total_changes - critical_changes - important_changes) * 0.1
        ) / max(total_changes, 1)
        volatility_index = min(1.0, volatility_index)
    
    # Determine trend stability
    if stability_score >= 80:
        trend_stability = 'stable'
    elif stability_score >= 50:
        trend_stability = 'moderate'
    else:
        trend_stability = 'unstable'
    
    # Determine risk level
    if critical_changes > 0:
        risk_level = 'critical'
    elif important_changes >= 3 or stability_score < 50:
        risk_level = 'high'
    elif important_changes > 0 or stability_score < 70:
        risk_level = 'moderate'
    else:
        risk_level = 'low'
    
    return StabilityMetrics(
        stabilityScore=round(stability_score, 1),
        volatilityIndex=round(volatility_index, 2),
        trendStability=trend_stability,
        riskLevel=risk_level
    )


# ============================================================================
# INTEGRATION POINTS
# ============================================================================

@dataclass(frozen=True)
class LabValuePoint:
    """Single lab value with timestamp."""
    timestamp: str  # ISO 8601
    value: float


@dataclass(frozen=True)
class NormalRange:
    """Normal range for a lab test."""
    low: float
    high: float


@dataclass(frozen=True)
class LabTrendInput:
    """
    Lab trend input from Barnabus Lab Ontology.
    
    Data interface for integration with Lab Ontology system.
    """
    testName: str
    values: List[LabValuePoint]
    unit: str
    normalRange: NormalRange


@dataclass(frozen=True)
class ComorbidityConfidence:
    """Comorbidity confidence score."""
    condition: str  # Condition name (e.g., 'CAD', 'Diabetes', 'CKD')
    score: float  # Confidence score 0-1
    lastUpdated: str  # ISO 8601 timestamp
    changeReason: Optional[str] = None


@dataclass(frozen=True)
class UpdatedConfidence:
    """Updated comorbidity confidence after change analysis."""
    condition: str
    score: float  # Updated confidence score 0-1
    lastUpdated: str  # ISO 8601 timestamp
    changeReason: str  # Reason for confidence adjustment
    previousScore: float  # Previous confidence score


@dataclass(frozen=True)
class RiskScores:
    """Current risk scores from calculators."""
    rcri: Optional[float] = None
    aubHas2: Optional[float] = None
    ariscat: Optional[float] = None
    vocalPenn: Optional[float] = None
    lastUpdated: Optional[str] = None  # ISO 8601 timestamp


@dataclass(frozen=True)
class RiskImpact:
    """Impact of changes on risk scores."""
    rcriAdjustment: Optional[float] = None
    aubHas2Adjustment: Optional[float] = None
    ariscatAdjustment: Optional[float] = None
    vocalPennAdjustment: Optional[float] = None
    overallImpact: str = 'neutral'  # 'increase', 'decrease', 'neutral'


@dataclass(frozen=True)
class UpdatedRiskScores:
    """Updated risk scores after change analysis."""
    # Required fields first
    adjustments: RiskImpact
    updatedAt: str  # ISO 8601 timestamp
    changeSummary: str  # Summary of changes
    # Optional fields with defaults come after required fields
    rcri: Optional[float] = None
    aubHas2: Optional[float] = None
    ariscat: Optional[float] = None
    vocalPenn: Optional[float] = None


def process_lab_trends(trends: List[LabTrendInput]) -> List[IntegratedTrend]:
    """
    Process lab trends from Barnabus Lab Ontology.
    
    Converts LabTrendInput[] to IntegratedTrend[] format for change analysis.
    
    Args:
        trends: List of lab trend inputs from Lab Ontology
    
    Returns:
        List of IntegratedTrend objects
    """
    integrated_trends = []
    
    for trend_input in trends:
        if not trend_input.values or len(trend_input.values) < 2:
            continue
        
        # Extract values and timestamps
        values = [vp.value for vp in trend_input.values]
        timestamps = [vp.timestamp for vp in trend_input.values]
        
        # Calculate trend metrics
        previous_value = values[0]
        current_value = values[-1]
        delta = current_value - previous_value
        
        # Calculate percent change
        percent_change = None
        if previous_value != 0:
            percent_change = (delta / abs(previous_value)) * 100.0
        
        # Calculate rate of change (per day)
        rate_of_change = None
        if len(timestamps) >= 2:
            try:
                first_time = pd.to_datetime(timestamps[0])
                last_time = pd.to_datetime(timestamps[-1])
                days_diff = (last_time - first_time).total_seconds() / (24 * 3600)
                if days_diff > 0:
                    rate_of_change = delta / days_diff
            except:
                pass
        
        # Determine trend direction
        if delta > 0:
            trend_direction = 'increasing'
        elif delta < 0:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        # Calculate temporal features
        acceleration = _calculate_acceleration(values)
        volatility = _calculate_volatility(values)
        diurnal_pattern = _detect_diurnal_pattern_enhanced(timestamps, values)
        outlier_count = _count_outliers(values)
        
        temporal_features = TemporalFeatures(
            acceleration=acceleration,
            volatility=volatility,
            seasonality=diurnal_pattern,
            outlierCount=outlier_count
        )
        
        # Determine significance based on change magnitude
        significance = None
        if abs(percent_change) >= 50 if percent_change else False:
            significance = 'critical'
        elif abs(percent_change) >= 25 if percent_change else False:
            significance = 'significant'
        
        # Create IntegratedTrend
        integrated_trend = IntegratedTrend(
            parameterName=trend_input.testName,
            test=trend_input.testName,
            trendDirection=trend_direction,
            confidence='high' if len(values) >= 3 else 'moderate',
            dataSources=['lab_ontology'],
            significance=significance,
            clinicalImplications=[],
            monitoringRecommendations=[],
            temporalFeatures=temporal_features,
            trendDetails={
                'delta': delta,
                'percentChange': percent_change,
                'rateOfChange': rate_of_change,
                'normalRange': {
                    'low': trend_input.normalRange.low,
                    'high': trend_input.normalRange.high
                }
            }
        )
        
        integrated_trends.append(integrated_trend)
    
    return integrated_trends


def update_comorbidity_confidence(
    changes: List[ClinicalChange],
    current_confidence: List[ComorbidityConfidence]
) -> List[UpdatedConfidence]:
    """
    Update comorbidity confidence scores based on detected changes.
    
    Adjusts confidence scores when relevant changes are detected.
    
    Args:
        changes: List of detected clinical changes
        current_confidence: Current comorbidity confidence scores
    
    Returns:
        List of updated confidence scores
    """
    updated_confidence = []
    
    for confidence in current_confidence:
        # Find relevant changes for this comorbidity
        relevant_changes = [
            c for c in changes
            if _is_change_relevant_to_comorbidity(c, confidence.condition)
        ]
        
        if relevant_changes:
            # Calculate confidence adjustment
            adjustment = _calculate_confidence_adjustment(relevant_changes)
            
            # Update score (clamped to 0-1)
            new_score = max(0.0, min(1.0, confidence.score + adjustment))
            
            updated_confidence.append(UpdatedConfidence(
                condition=confidence.condition,
                score=round(new_score, 3),
                lastUpdated=datetime.now().isoformat(),
                changeReason=f"Adjusted based on {len(relevant_changes)} relevant changes",
                previousScore=confidence.score
            ))
        else:
            # No relevant changes, keep current confidence
            updated_confidence.append(UpdatedConfidence(
                condition=confidence.condition,
                score=confidence.score,
                lastUpdated=confidence.lastUpdated,
                changeReason="No relevant changes detected",
                previousScore=confidence.score
            ))
    
    return updated_confidence


def _is_change_relevant_to_comorbidity(
    change: ClinicalChange,
    condition: str
) -> bool:
    """
    Check if a change is relevant to a specific comorbidity.
    
    Args:
        change: Clinical change to check
        condition: Comorbidity condition name
    
    Returns:
        True if change is relevant to the condition
    """
    condition_lower = condition.lower()
    
    # Map conditions to relevant categories and tests
    condition_mappings = {
        'cad': ['cardiac', 'troponin', 'bnp', 'ck', 'chest pain', 'sob'],
        'chf': ['cardiac', 'bnp', 'nt-probnp', 'shortness of breath', 'sob'],
        'diabetes': ['endocrine', 'glucose', 'a1c', 'hba1c', 'insulin'],
        'ckd': ['renal', 'creatinine', 'egfr', 'bun'],
        'osa': ['pulmonary', 'cpap', 'bipap'],
        'cirrhosis': ['hematologic', 'inr', 'bilirubin', 'albumin', 'platelets'],
        'anemia': ['hematologic', 'hemoglobin', 'hgb', 'ferritin', 'tsat']
    }
    
    # Check if condition matches any mapping
    relevant_terms = condition_mappings.get(condition_lower, [])
    if not relevant_terms:
        return False
    
    # Check category match
    if change.category.value in relevant_terms:
        return True
    
    # Check test name match (for lab changes)
    if change.type == ChangeType.LAB and change.quantitativeData:
        test_name = change.quantitativeData.testName.lower() if change.quantitativeData.testName else ''
        if any(term in test_name for term in relevant_terms):
            return True
    
    # Check description match (for symptoms)
    if change.type == ChangeType.SYMPTOM:
        desc_lower = change.description.lower()
        if any(term in desc_lower for term in relevant_terms):
            return True
    
    return False


def _calculate_confidence_adjustment(changes: List[ClinicalChange]) -> float:
    """
    Calculate confidence adjustment based on changes.
    
    Positive adjustment increases confidence, negative decreases.
    
    Args:
        changes: List of relevant clinical changes
    
    Returns:
        Confidence adjustment value (-1 to +1)
    """
    adjustment = 0.0
    
    for change in changes:
        # P0 changes significantly increase confidence
        if change.criticality == 'P0':
            adjustment += 0.2
        # P1 changes moderately increase confidence
        elif change.criticality == 'P1':
            adjustment += 0.1
        # P2 changes slightly increase confidence
        elif change.criticality == 'P2':
            adjustment += 0.05
        
        # Consider clinical impact
        if change.clinicalImpact.affectsSurgicalPlanning:
            adjustment += 0.05
    
    # Cap adjustment to reasonable range
    return max(-0.5, min(0.5, adjustment))


def update_risk_scores(
    changes: List[ClinicalChange],
    current_scores: RiskScores
) -> UpdatedRiskScores:
    """
    Update risk scores based on detected clinical changes.
    
    Adjusts risk calculator scores when relevant changes are detected.
    
    Args:
        changes: List of detected clinical changes
        current_scores: Current risk scores from calculators
    
    Returns:
        Updated risk scores with adjustments
    """
    # Calculate risk impact
    impact = _calculate_risk_impact(changes)
    
    # Apply adjustments to current scores
    updated_rcri = current_scores.rcri
    updated_aub_has2 = current_scores.aubHas2
    updated_ariscat = current_scores.ariscat
    updated_vocal_penn = current_scores.vocalPenn
    
    if impact.rcriAdjustment is not None and updated_rcri is not None:
        updated_rcri = max(0, min(10, updated_rcri + impact.rcriAdjustment))
    
    if impact.aubHas2Adjustment is not None and updated_aub_has2 is not None:
        updated_aub_has2 = max(0, min(10, updated_aub_has2 + impact.aubHas2Adjustment))
    
    if impact.ariscatAdjustment is not None and updated_ariscat is not None:
        updated_ariscat = max(0, min(10, updated_ariscat + impact.ariscatAdjustment))
    
    if impact.vocalPennAdjustment is not None and updated_vocal_penn is not None:
        updated_vocal_penn = max(0, min(10, updated_vocal_penn + impact.vocalPennAdjustment))
    
    return UpdatedRiskScores(
        adjustments=impact,
        updatedAt=datetime.now().isoformat(),
        changeSummary=f"Risk scores updated based on {len(changes)} clinical changes",
        rcri=updated_rcri,
        aubHas2=updated_aub_has2,
        ariscat=updated_ariscat,
        vocalPenn=updated_vocal_penn
    )


def _calculate_risk_impact(changes: List[ClinicalChange]) -> RiskImpact:
    """
    Calculate impact of changes on risk scores.
    
    Args:
        changes: List of clinical changes
    
    Returns:
        RiskImpact with adjustments for each calculator
    """
    rcri_adjustment = 0.0
    aub_has2_adjustment = 0.0
    ariscat_adjustment = 0.0
    vocal_penn_adjustment = 0.0
    
    for change in changes:
        # Cardiac changes affect RCRI and AUB-HAS2
        if change.category == ChangeCategory.CARDIAC:
            if change.criticality == 'P0':
                rcri_adjustment += 0.5
                aub_has2_adjustment += 0.5
            elif change.criticality == 'P1':
                rcri_adjustment += 0.3
                aub_has2_adjustment += 0.3
        
        # Pulmonary changes affect ARISCAT
        if change.category == ChangeCategory.PULMONARY:
            if change.criticality == 'P0':
                ariscat_adjustment += 0.5
            elif change.criticality == 'P1':
                ariscat_adjustment += 0.3
        
        # Hepatic changes affect VOCAL-Penn
        if change.category == ChangeCategory.HEMATOLOGIC:
            # Check if it's liver-related
            if change.type == ChangeType.LAB and change.quantitativeData:
                test_name = change.quantitativeData.testName.lower() if change.quantitativeData.testName else ''
                if any(term in test_name for term in ['bilirubin', 'inr', 'albumin']):
                    if change.criticality == 'P0':
                        vocal_penn_adjustment += 0.5
                    elif change.criticality == 'P1':
                        vocal_penn_adjustment += 0.3
        
        # Renal changes affect multiple calculators
        if change.category == ChangeCategory.RENAL:
            if change.criticality == 'P0':
                rcri_adjustment += 0.3
                aub_has2_adjustment += 0.3
    
    # Determine overall impact
    total_adjustment = abs(rcri_adjustment) + abs(aub_has2_adjustment) + abs(ariscat_adjustment) + abs(vocal_penn_adjustment)
    if total_adjustment > 1.0:
        overall_impact = 'increase'
    elif total_adjustment > 0.5:
        overall_impact = 'increase'
    else:
        overall_impact = 'neutral'
    
    return RiskImpact(
        rcriAdjustment=round(rcri_adjustment, 2) if rcri_adjustment != 0 else None,
        aubHas2Adjustment=round(aub_has2_adjustment, 2) if aub_has2_adjustment != 0 else None,
        ariscatAdjustment=round(ariscat_adjustment, 2) if ariscat_adjustment != 0 else None,
        vocalPennAdjustment=round(vocal_penn_adjustment, 2) if vocal_penn_adjustment != 0 else None,
        overallImpact=overall_impact
    )


def _create_integrated_trend(trend: LabTrend) -> IntegratedTrend:
    """Create IntegratedTrend from LabTrend with temporal features."""
    # Calculate temporal features if values available
    temporal_features = None
    if trend.values and len(trend.values) >= 2:
        acceleration = _calculate_acceleration(trend.values)
        volatility = _calculate_volatility(trend.values)
        seasonality = None
        if trend.timestamps:
            seasonality = _detect_diurnal_pattern(trend.timestamps)
        outlier_count = _count_outliers(trend.values)
        
        temporal_features = TemporalFeatures(
            acceleration=acceleration,
            volatility=volatility,
            seasonality=seasonality,
            outlierCount=outlier_count
        )
    
    # Map trend to clinical implications
    clinical_implications = _map_trend_to_clinical_implications(trend)
    
    # Generate monitoring recommendations
    monitoring_recommendations = _generate_monitoring_recommendations(trend)
    
    return IntegratedTrend(
        parameterName=trend.testName,
        test=trend.testName,
        trendDirection=trend.trend,
        confidence='high' if trend.significance else 'moderate',
        dataSources=['lab_ontology'],
        significance=trend.significance,
        clinicalImplications=clinical_implications,
        monitoringRecommendations=monitoring_recommendations,
        temporalFeatures=temporal_features,
        trendDetails={
            'delta': trend.delta,
            'percentChange': trend.percentChange,
            'rateOfChange': trend.rateOfChange,
        }
    )


def _map_trend_to_clinical_implications(trend: LabTrend) -> List[str]:
    """Map trend to clinical implications."""
    implications = []
    test_lower = trend.testName.lower()
    trend_dir = trend.trend
    
    # Test-specific implications
    if 'creatinine' in test_lower:
        if trend_dir == 'increasing':
            implications.append("Worsening renal function - risk of AKI")
            implications.append("Consider nephrology consultation if eGFR declining")
        elif trend_dir == 'decreasing':
            implications.append("Improving renal function")
    
    elif 'bnp' in test_lower or 'nt-probnp' in test_lower:
        if trend_dir == 'increasing':
            implications.append("Worsening heart failure - volume overload")
            implications.append("Consider cardiology consultation")
            implications.append("May require diuresis before surgery")
        elif trend_dir == 'decreasing':
            implications.append("Improving heart failure status")
    
    elif 'troponin' in test_lower:
        if trend_dir == 'increasing':
            implications.append("Possible active myocardial injury")
            implications.append("URGENT: Cardiology evaluation required")
            implications.append("Consider delaying non-emergent surgery")
    
    elif 'hemoglobin' in test_lower or 'hgb' in test_lower:
        if trend_dir == 'decreasing':
            implications.append("Worsening anemia - increased bleeding risk")
            implications.append("Consider transfusion threshold planning")
    
    elif 'glucose' in test_lower or 'a1c' in test_lower:
        if trend_dir == 'increasing':
            implications.append("Worsening glycemic control")
            implications.append("Increased infection and wound healing risk")
    
    return implications


def _generate_monitoring_recommendations(trend: LabTrend) -> List[str]:
    """Generate monitoring recommendations based on trend."""
    recommendations = []
    test_lower = trend.testName.lower()
    criticality = _calculate_trend_criticality(trend)
    
    if criticality == 'P0':
        recommendations.append(f"CRITICAL: Monitor {trend.testName} closely - repeat within 24 hours")
        recommendations.append("Consider delaying surgery if trend continues")
    elif criticality == 'P1':
        recommendations.append(f"Monitor {trend.testName} - repeat within 48 hours")
    else:
        recommendations.append(f"Continue monitoring {trend.testName} as clinically indicated")
    
    # Test-specific recommendations
    if 'creatinine' in test_lower and trend.trend == 'increasing':
        recommendations.append("Monitor urine output and avoid nephrotoxins")
    
    if 'bnp' in test_lower and trend.trend == 'increasing':
        recommendations.append("Monitor fluid status and consider diuresis")
    
    if 'troponin' in test_lower and trend.trend == 'increasing':
        recommendations.append("URGENT: Repeat troponin and EKG immediately")
    
    return recommendations


def _analyze_integrated_patterns(
    changes: List[ClinicalChange],
    integrated_trends: List[IntegratedTrend]
) -> Dict[str, Any]:
    """Analyze integrated patterns across changes and trends."""
    analysis = {
        'totalChanges': len(changes),
        'criticalChanges': sum(1 for c in changes if c.criticality == 'P0'),
        'trendsRequiringAction': sum(
            1 for t in integrated_trends
            if t.significance and 'critical' in t.significance.lower()
        ),
        'crossCategoryPatterns': {},
        'temporalPatterns': {},
    }
    
    # Analyze cross-category patterns
    changes_by_category: Dict[str, List[ClinicalChange]] = {}
    for change in changes:
        cat = change.category.value
        if cat not in changes_by_category:
            changes_by_category[cat] = []
        changes_by_category[cat].append(change)
    
    analysis['crossCategoryPatterns'] = {
        cat: len(changes_list)
        for cat, changes_list in changes_by_category.items()
    }
    
    # Analyze temporal patterns
    trends_with_features = [t for t in integrated_trends if t.temporalFeatures]
    if trends_with_features:
        analysis['temporalPatterns'] = {
            'highVolatility': sum(
                1 for t in trends_with_features
                if t.temporalFeatures and t.temporalFeatures.volatility
                and t.temporalFeatures.volatility > 0.3
            ),
            'acceleratingTrends': sum(
                1 for t in trends_with_features
                if t.temporalFeatures and t.temporalFeatures.acceleration
                and abs(t.temporalFeatures.acceleration) > 0.1
            ),
            'outlierCount': sum(
                t.temporalFeatures.outlierCount
                for t in trends_with_features
                if t.temporalFeatures
            ),
        }
    
    return analysis


# ============================================================================
# MAIN CHANGE DETECTION ENGINE
# ============================================================================

class SurgeryAwareChangeDetector:
    """
    Main engine that orchestrates all change detection components.
    
    Provides a unified interface for surgery-aware clinical change analysis.
    """
    
    def __init__(
        self,
        surgery_datetime: datetime,
        lab_trend_data: Optional[Dict[str, Any]] = None,
        temporal_patterns: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize change detector.
        
        Args:
            surgery_datetime: Surgery date/time
            lab_trend_data: Optional lab trend data from Barnabus Lab Ontology
            temporal_patterns: Optional temporal pattern data
        """
        self.surgery_datetime = pd.to_datetime(surgery_datetime)
        self.time_processor = TimeBoundedChangeProcessor(surgery_datetime)
        self.change_analyzer = ClinicalChangeAnalyzer()
        self.trend_integrator = TrendIntegrationEngine(lab_trend_data, temporal_patterns)
    
    def analyze_changes(
        self,
        subject_id: int,
        hadm_id: int,
        lab_series: Dict[str, pd.DataFrame],
        max_hours_before: Optional[int] = 168,  # Default: 7 days
    ) -> UnifiedChangeAnalysis:
        """
        Perform complete change analysis for all parameters.
        
        Args:
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            lab_series: Dictionary of {parameter_name: DataFrame} with lab/vital data
            max_hours_before: Maximum hours before surgery to analyze
        
        Returns:
            UnifiedChangeAnalysis with all detected changes
        """
        changes: List[ChangeAnalysisResult] = []
        
        # Process each parameter
        for parameter_name, series_df in lab_series.items():
            # Step 1: Filter to pre-op period
            filtered_df = self.time_processor.filter_dataframe(
                series_df,
                timestamp_column='CHARTTIME',
                max_hours_before=max_hours_before
            )
            
            if filtered_df.empty:
                continue
            
            # Step 2: Calculate quantitative delta
            quantitative_delta = self.change_analyzer.analyze_series(
                parameter_name=parameter_name,
                series_df=filtered_df,
                value_column='VALUENUM',
                timestamp_column='CHARTTIME'
            )
            
            if quantitative_delta is None:
                continue
            
            # Step 3: Detect pattern
            pattern = self.change_analyzer.detect_pattern(
                series_df=filtered_df,
                value_column='VALUENUM',
                timestamp_column='CHARTTIME'
            )
            
            # Step 4: Integrate with trends
            trend_integration = self.trend_integrator.integrate_trends(
                quantitative_delta=quantitative_delta,
                parameter_name=parameter_name
            )
            
            # Step 5: Generate recommended action
            recommended_action = self._generate_recommended_action(
                quantitative_delta,
                pattern,
                trend_integration
            )
            
            # Create change analysis result
            change_result = ChangeAnalysisResult(
                parameter_name=parameter_name,
                quantitative_delta=quantitative_delta,
                trend_integration=trend_integration,
                pattern_detected=pattern,
                recommended_action=recommended_action
            )
            
            changes.append(change_result)
        
        # Generate summary
        summary = self._generate_summary(changes)
        
        return UnifiedChangeAnalysis(
            subject_id=subject_id,
            hadm_id=hadm_id,
            surgery_datetime=self.surgery_datetime,
            analysis_timestamp=datetime.now(),
            pre_op_window_hours=max_hours_before or 168.0,
            changes=changes,
            summary=summary
        )
    
    def _generate_recommended_action(
        self,
        quantitative_delta: QuantitativeDelta,
        pattern: Optional[str],
        trend_integration: Dict[str, Any]
    ) -> Optional[str]:
        """Generate recommended action based on change analysis."""
        significance = quantitative_delta.clinical_significance
        
        if significance == ClinicalSignificance.P0_CRITICAL:
            return f"CRITICAL: {quantitative_delta.parameter_name} change requires immediate attention. Consider delaying surgery."
        elif significance == ClinicalSignificance.P1_HIGH:
            return f"HIGH: Significant {quantitative_delta.parameter_name} change. Review before proceeding."
        elif significance == ClinicalSignificance.P2_MODERATE:
            return f"MODERATE: Notable {quantitative_delta.parameter_name} change. Monitor closely."
        else:
            return None
    
    def _generate_summary(
        self,
        changes: List[ChangeAnalysisResult]
    ) -> Dict[str, Any]:
        """Generate summary statistics for all changes."""
        if not changes:
            return {
                'total_changes': 0,
                'critical_changes': 0,
                'high_changes': 0,
                'moderate_changes': 0,
            }
        
        summary = {
            'total_changes': len(changes),
            'critical_changes': sum(
                1 for c in changes
                if c.quantitative_delta.clinical_significance == ClinicalSignificance.P0_CRITICAL
            ),
            'high_changes': sum(
                1 for c in changes
                if c.quantitative_delta.clinical_significance == ClinicalSignificance.P1_HIGH
            ),
            'moderate_changes': sum(
                1 for c in changes
                if c.quantitative_delta.clinical_significance == ClinicalSignificance.P2_MODERATE
            ),
            'parameters_with_changes': [c.parameter_name for c in changes],
        }
        
        return summary


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
Example usage of SurgeryAwareChangeDetector:

```python
from datetime import datetime
from clinical_change_detector import SurgeryAwareChangeDetector
import pandas as pd

# 1. Initialize with surgery datetime
surgery_time = datetime(2024, 1, 15, 10, 0, 0)
detector = SurgeryAwareChangeDetector(
    surgery_datetime=surgery_time,
    lab_trend_data={
        'creatinine': {'value': 1.8, 'trend': 'increasing', 'delta': 0.3},
        'bnp': {'value': 450, 'trend': 'increasing', 'delta': 100},
    },
    temporal_patterns={
        'creatinine': {'delta_48h': 0.3, 'slope_48h': 0.15},
        'bnp': {'delta_48h': 100, 'worsening_bnp': True},
    }
)

# 2. Prepare lab series DataFrames (from existing extractors)
lab_series = {
    'creatinine': pd.DataFrame({
        'CHARTTIME': [
            datetime(2024, 1, 10, 8, 0),
            datetime(2024, 1, 12, 8, 0),
            datetime(2024, 1, 14, 8, 0),
        ],
        'VALUENUM': [1.2, 1.5, 1.8],
    }),
    'bnp': pd.DataFrame({
        'CHARTTIME': [
            datetime(2024, 1, 13, 10, 0),
            datetime(2024, 1, 14, 10, 0),
        ],
        'VALUENUM': [350, 450],
    }),
}

# 3. Analyze changes
analysis = detector.analyze_changes(
    subject_id=12345,
    hadm_id=67890,
    lab_series=lab_series,
    max_hours_before=168,  # 7 days
)

# 4. Access results
print(f"Total changes detected: {analysis.summary['total_changes']}")
print(f"Critical changes: {analysis.summary['critical_changes']}")

for change in analysis.changes:
    delta = change.quantitative_delta
    print(f"{delta.parameter_name}:")
    print(f"  Delta: {delta.delta_value} ({delta.delta_percent:.1f}%)")
    print(f"  Significance: {delta.clinical_significance.value}")
    print(f"  Trend: {delta.trend_direction}")
    if change.recommended_action:
        print(f"  Action: {change.recommended_action}")
```

Integration with existing systems:

```python
# Integrate with ScoreBasedRiskExtractor
from score_based_risk import ScoreBasedRiskExtractor
from clinical_change_detector import SurgeryAwareChangeDetector

cardiac_extractor = ScoreBasedRiskExtractor()
cardiac_extractor.load_data(...)

# Get lab series from existing extractor
creat_series = cardiac_extractor._get_lab_series_by_time_window(
    subject_id=12345,
    lab_name='creatinine',
    anchor_time=surgery_time,
    hours_before=168
)

# Use with change detector
detector = SurgeryAwareChangeDetector(surgery_datetime=surgery_time)
analysis = detector.analyze_changes(
    subject_id=12345,
    hadm_id=67890,
    lab_series={'creatinine': creat_series},
)
```
"""

