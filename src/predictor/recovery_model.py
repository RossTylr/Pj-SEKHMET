"""
Recovery Prediction Model
=========================
Core logic for predicting recovery trajectory.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from .config import (
        RecoveryConfig, DEFAULT_CONFIG,
        InjuryType, BodyRegion, JMESStatus, RecoveryBand, Trade,
        get_age_band, get_prior_injury_band, get_recovery_band
    )
except ImportError:
    from config import (
        RecoveryConfig, DEFAULT_CONFIG,
        InjuryType, BodyRegion, JMESStatus, RecoveryBand, Trade,
        get_age_band, get_prior_injury_band, get_recovery_band
    )


@dataclass
class CaseInput:
    """Input case for prediction"""
    age: int
    trade: Trade
    injury_type: InjuryType
    body_region: BodyRegion
    severity_score: int  # 1-10
    prior_injury_count: int
    prior_same_region: bool
    current_jmes: JMESStatus
    months_since_injury: int = 0
    receiving_treatment: bool = True


@dataclass
class RecoveryPrediction:
    """Output prediction"""
    # Point estimates
    expected_recovery_months: float
    recovery_band: RecoveryBand
    
    # Probability distribution
    prob_recovery_3mo: float
    prob_recovery_6mo: float
    prob_recovery_12mo: float
    prob_recovery_24mo: float
    
    # JMES trajectory
    prob_full_recovery: float  # Return to MFD
    prob_partial_recovery: float  # MND→MLD or stay MLD
    prob_medical_discharge: float
    
    # Confidence
    confidence_level: str  # Low, Medium, High
    
    # Factors
    contributing_factors: Dict[str, float]  # Factor name → impact
    
    # Timeline
    optimistic_months: float  # 25th percentile
    realistic_months: float  # 50th percentile
    pessimistic_months: float  # 75th percentile


class RecoveryPredictor:
    """
    Predicts recovery trajectory for injured service personnel.
    
    Uses configurable parameters that can be adjusted via UI.
    """
    
    def __init__(self, config: RecoveryConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    def predict(self, case: CaseInput) -> RecoveryPrediction:
        """Generate recovery prediction for a case"""
        
        # Get base recovery time from injury type
        injury_profile = self.config.injury_profiles.get(
            case.injury_type.value,
            self.config.injury_profiles["Other"]
        )
        
        base_min, base_max = injury_profile.base_recovery_months
        base_recovery = (base_min + base_max) / 2
        
        # Calculate modifiers
        factors = {}
        total_modifier = 1.0
        
        # 1. Age modifier
        age_band = get_age_band(case.age)
        age_mod = self.config.age_modifiers.get(age_band, 1.0)
        factors["Age"] = age_mod
        total_modifier *= age_mod
        
        # 2. Body region modifier
        region_mod = self.config.body_region_modifiers.get(
            case.body_region.value, 1.0
        )
        factors["Body Region"] = region_mod
        total_modifier *= region_mod
        
        # 3. Trade physical demand
        trade_demand = self.config.trade_physical_demand.get(
            case.trade.value, "Medium"
        )
        demand_mod = self.config.physical_demand_modifiers.get(trade_demand, 1.0)
        factors["Trade Demand"] = demand_mod
        total_modifier *= demand_mod
        
        # 4. Prior injuries
        prior_band = get_prior_injury_band(case.prior_injury_count)
        prior_mod = self.config.prior_injury_modifiers.get(prior_band, 1.0)
        factors["Prior Injuries"] = prior_mod
        total_modifier *= prior_mod
        
        # 5. Recurrence
        if case.prior_same_region:
            recur_mod = self.config.recurrence_modifier
            factors["Recurrence"] = recur_mod
            total_modifier *= recur_mod
        
        # 6. Severity score (1-10 → 0.8-1.4 modifier)
        severity_mod = 0.8 + (case.severity_score - 1) * 0.067
        factors["Severity"] = severity_mod
        total_modifier *= severity_mod
        
        # 7. Treatment effect
        if case.receiving_treatment:
            treatment_mod = 0.85
            factors["Treatment"] = treatment_mod
            total_modifier *= treatment_mod
        
        # Calculate final recovery estimate
        expected_months = base_recovery * total_modifier
        
        # Adjust for time already elapsed
        remaining_months = max(0, expected_months - case.months_since_injury)
        
        # Calculate variance based on injury profile
        variance_map = {"Low": 0.15, "Medium": 0.25, "High": 0.35, "Very High": 0.50}
        variance = variance_map.get(injury_profile.variance, 0.25)
        
        # Percentiles
        optimistic = remaining_months * (1 - variance)
        realistic = remaining_months
        pessimistic = remaining_months * (1 + variance)
        
        # Recovery probabilities at time points
        prob_3mo = self._calc_recovery_prob(remaining_months, 3, variance)
        prob_6mo = self._calc_recovery_prob(remaining_months, 6, variance)
        prob_12mo = self._calc_recovery_prob(remaining_months, 12, variance)
        prob_24mo = self._calc_recovery_prob(remaining_months, 24, variance)
        
        # JMES trajectory probabilities
        prob_full, prob_partial, prob_discharge = self._calc_jmes_probs(
            case, injury_profile, total_modifier
        )
        
        # Confidence level
        if variance <= 0.20 and case.prior_injury_count <= 2:
            confidence = "High"
        elif variance >= 0.40 or case.prior_injury_count >= 5:
            confidence = "Low"
        else:
            confidence = "Medium"
        
        # Recovery band
        band = get_recovery_band(remaining_months, self.config)
        
        return RecoveryPrediction(
            expected_recovery_months=round(remaining_months, 1),
            recovery_band=band,
            prob_recovery_3mo=round(prob_3mo, 2),
            prob_recovery_6mo=round(prob_6mo, 2),
            prob_recovery_12mo=round(prob_12mo, 2),
            prob_recovery_24mo=round(prob_24mo, 2),
            prob_full_recovery=round(prob_full, 2),
            prob_partial_recovery=round(prob_partial, 2),
            prob_medical_discharge=round(prob_discharge, 2),
            confidence_level=confidence,
            contributing_factors={k: round(v, 2) for k, v in factors.items()},
            optimistic_months=round(optimistic, 1),
            realistic_months=round(realistic, 1),
            pessimistic_months=round(pessimistic, 1)
        )
    
    def _calc_recovery_prob(
        self, 
        expected_months: float, 
        target_months: float,
        variance: float
    ) -> float:
        """Calculate probability of recovery by target month"""
        if expected_months <= 0:
            return 1.0
        
        # Use log-normal approximation
        # If target > expected, higher probability
        ratio = target_months / expected_months
        
        if ratio >= 2.0:
            return min(0.98, 0.85 + variance * 0.5)
        elif ratio >= 1.5:
            return min(0.95, 0.70 + variance * 0.3)
        elif ratio >= 1.0:
            return 0.50 + (ratio - 1.0) * 0.4
        elif ratio >= 0.5:
            return 0.20 + (ratio - 0.5) * 0.6
        else:
            return max(0.05, ratio * 0.4)
    
    def _calc_jmes_probs(
        self,
        case: CaseInput,
        injury_profile,
        total_modifier: float
    ) -> Tuple[float, float, float]:
        """Calculate JMES outcome probabilities"""
        
        # Base probabilities from injury profile
        mld_prob = injury_profile.mld_probability
        mnd_prob = injury_profile.mnd_probability
        
        # Adjust for current JMES
        if case.current_jmes == JMESStatus.MFD:
            # Minor injury, likely stays MFD
            prob_full = 1 - mld_prob
            prob_partial = mld_prob * 0.8
            prob_discharge = mld_prob * 0.2 * mnd_prob
        
        elif case.current_jmes == JMESStatus.MLD:
            # Already downgraded
            base_recovery = 0.70  # Base 70% recover to MFD
            modifier_effect = max(0.3, 1.2 - total_modifier * 0.3)
            
            prob_full = base_recovery * modifier_effect
            prob_partial = (1 - prob_full) * 0.7
            prob_discharge = (1 - prob_full) * 0.3
        
        else:  # MND
            # Serious - harder to recover
            base_recovery = 0.40
            modifier_effect = max(0.2, 1.3 - total_modifier * 0.4)
            
            prob_full = base_recovery * modifier_effect * 0.5
            prob_partial = base_recovery * modifier_effect * 0.5
            prob_discharge = 1 - prob_full - prob_partial
        
        # Normalize
        total = prob_full + prob_partial + prob_discharge
        return (
            prob_full / total,
            prob_partial / total,
            prob_discharge / total
        )
    
    def predict_cohort(
        self, 
        cases: List[CaseInput]
    ) -> Dict[str, any]:
        """Predict for a cohort and return summary stats"""
        
        predictions = [self.predict(c) for c in cases]
        
        # Aggregate stats
        recovery_times = [p.expected_recovery_months for p in predictions]
        
        band_counts = {}
        for p in predictions:
            band = p.recovery_band.value
            band_counts[band] = band_counts.get(band, 0) + 1
        
        return {
            "total_cases": len(cases),
            "mean_recovery_months": np.mean(recovery_times),
            "median_recovery_months": np.median(recovery_times),
            "std_recovery_months": np.std(recovery_times),
            "band_distribution": band_counts,
            "avg_prob_full_recovery": np.mean([p.prob_full_recovery for p in predictions]),
            "avg_prob_medical_discharge": np.mean([p.prob_medical_discharge for p in predictions]),
            "predictions": predictions
        }
    
    def generate_recovery_curve(
        self,
        case: CaseInput,
        months: int = 24
    ) -> List[Dict]:
        """Generate month-by-month recovery probability curve"""
        
        prediction = self.predict(case)
        expected = prediction.expected_recovery_months
        
        # Variance from injury profile
        injury_profile = self.config.injury_profiles.get(
            case.injury_type.value,
            self.config.injury_profiles["Other"]
        )
        variance_map = {"Low": 0.15, "Medium": 0.25, "High": 0.35, "Very High": 0.50}
        variance = variance_map.get(injury_profile.variance, 0.25)
        
        curve = []
        for m in range(1, months + 1):
            prob = self._calc_recovery_prob(expected, m, variance)
            curve.append({
                "month": m,
                "cumulative_recovery_prob": round(prob, 3),
                "still_recovering_prob": round(1 - prob, 3)
            })
        
        return curve
