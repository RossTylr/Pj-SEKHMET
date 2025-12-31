"""
SEKHMET Recovery Predictor - Cox Proportional Hazards Model
============================================================

This module implements a Cox Proportional Hazards survival model for predicting
time-to-recovery in military personnel. The model is calibrated to published
clinical evidence from the evidence_base.yaml.

Theoretical Background
----------------------
The Cox PH model estimates the hazard (instantaneous risk of recovery) as:

    h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₖXₖ)

Where:
- h₀(t) is the baseline hazard function (injury-specific)
- X₁...Xₖ are covariates (age, prior injury, etc.)
- β₁...βₖ are log hazard ratios (from evidence base)

Key Assumption: Proportional Hazards
------------------------------------
The Cox model assumes hazard ratios are CONSTANT over time. This means if
smoking increases hazard by 43% (HR=1.43), it does so at all time points.

This assumption is reasonable for most SEKHMET covariates:
- Age effect: Older personnel recover slower throughout (plausible)
- Prior injury: Effect persists over recovery period (plausible)
- Smoking: Tissue healing impaired throughout (plausible)

BUT may be violated for:
- Mental health comorbidity: Effect may diminish if treated
- Supervised rehab: Effect may be strongest early in recovery

For now, we accept this limitation. Future work could use time-varying
coefficients or stratified models.

Baseline Hazard Estimation
--------------------------
Without individual patient data, we cannot fit a true Cox model. Instead,
we derive baseline survival curves from published median recovery times
using a Weibull approximation:

    S(t) = exp(-(t/λ)^k)

Where:
- λ (scale) is derived from median: λ = median / (ln(2))^(1/k)
- k (shape) controls the hazard shape:
  - k < 1: Hazard decreases over time (early failures)
  - k = 1: Constant hazard (exponential)
  - k > 1: Hazard increases over time (wear-out)

For recovery, k > 1 is typical: the longer you've been recovering,
the more likely you are to recover in the next interval (healing accumulates).

References
----------
- Cox DR (1972). Regression Models and Life-Tables. JRSS-B.
- Collett D (2015). Modelling Survival Data in Medical Research. CRC Press.
- Evidence base: evidence_base.yaml v1.1.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
import logging

from config import (
    InjuryType, BodyRegion, Trade, TradeCategory, JMESStatus, RecoveryBand,
    EvidenceBase, InjuryParameters, RiskFactor,
    get_trade_category, TRADE_CATEGORY_RTD_MODIFIER
)

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

# Weibull shape parameter by injury category
# Derived from clinical knowledge of recovery trajectories
WEIBULL_SHAPE_PARAMS: Dict[str, float] = {
    # MSKI: k > 1, increasing hazard (healing accumulates)
    # Rationale: Tissue repair is progressive; longer time = more healing done
    'MSKI': 1.5,

    # Mental Health: k ≈ 1, roughly constant hazard
    # Rationale: Recovery less predictable, therapy response variable
    # Some recover quickly, others plateau
    'MH': 1.1,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CaseInput:
    """
    Input data for a single prediction case.
    
    Attributes:
        age: Patient age in years
        trade: Military trade/role
        injury_type: Type and severity of injury
        body_region: Anatomical location
        severity_score: 1-10 clinical severity (optional refinement)
        prior_injury_count: Number of prior injuries (any region)
        prior_same_region: Whether prior injury was to same region
        current_jmes: Current JMES status
        months_since_injury: Time since injury onset
        receiving_treatment: Whether in active supervised treatment
        is_smoker: Current smoking status
        has_mh_comorbidity: Mental health comorbidity present
        multiple_tbi_history: ≥3 lifetime TBIs (for TBI cases)
        is_female: Sex (for incidence adjustment, limited recovery data)
    """
    age: int
    trade: Trade
    injury_type: InjuryType
    body_region: BodyRegion
    severity_score: int = 5  # 1-10, default moderate
    prior_injury_count: int = 0
    prior_same_region: bool = False
    current_jmes: JMESStatus = JMESStatus.MLD
    months_since_injury: float = 0.0
    receiving_treatment: bool = True
    is_smoker: bool = False
    has_mh_comorbidity: bool = False
    multiple_tbi_history: bool = False  # ≥3 TBIs
    is_female: bool = False


@dataclass
class CoxPrediction:
    """
    Output from Cox model prediction.
    
    Provides both point estimates and uncertainty quantification.
    All probabilities are cumulative (by that time point).
    """
    # Point estimates (months)
    median_recovery_months: float
    time_to_fitness_months: float  # Clinical milestone
    time_to_rtd_months: float      # Occupational clearance
    
    # Uncertainty bounds (90% prediction interval)
    recovery_lower_90: float
    recovery_upper_90: float
    
    # Cumulative recovery probabilities
    prob_recovery_3mo: float
    prob_recovery_6mo: float
    prob_recovery_12mo: float
    prob_recovery_24mo: float
    
    # Outcome probabilities
    prob_full_recovery: float      # Return to MFD
    prob_partial_recovery: float   # RTD with restrictions
    prob_medical_discharge: float  # MU outcome
    
    # Risk score (linear predictor)
    risk_score: float
    
    # Recovery band for workforce planning
    recovery_band: RecoveryBand
    
    # Contributing factors (for explainability)
    hazard_ratios: Dict[str, float]
    
    # Confidence level
    confidence: str  # "high", "moderate", "low"
    confidence_rationale: str
    
    # Citations
    primary_sources: List[str]


# =============================================================================
# COX MODEL IMPLEMENTATION
# =============================================================================

class CoxRecoveryModel:
    """
    Cox Proportional Hazards model for recovery prediction.
    
    This model combines:
    1. Baseline survival curves from published median recovery times
    2. Hazard ratio modifiers from systematic reviews
    3. Weibull parametric approximation for smooth survival functions
    
    The model is NOT fitted to individual patient data (which we don't have).
    Instead, it synthesises published aggregate evidence into a coherent
    predictive framework.
    
    Limitations:
    - Assumes proportional hazards (constant HR over time)
    - Baseline curves from medians, not full Kaplan-Meier data
    - Limited validation against UK military population
    """
    
    def __init__(self, evidence_base: Optional[EvidenceBase] = None):
        """
        Initialise model with evidence base.
        
        Args:
            evidence_base: EvidenceBase instance. If None, loads default.
        """
        self.evidence = evidence_base or EvidenceBase()
        self._validate_evidence_base()
        
        logger.info(f"CoxRecoveryModel initialised with evidence base v{self.evidence.version}")
    
    def _validate_evidence_base(self):
        """Check evidence base has required parameters."""
        required_factors = ['age', 'prior_same_region_injury', 'smoking']
        for factor in required_factors:
            if self.evidence.get_risk_factor(factor) is None:
                logger.warning(f"Risk factor '{factor}' not found in evidence base")
    
    def predict(self, case: CaseInput) -> CoxPrediction:
        """
        Generate recovery prediction for a case.
        
        This is the main entry point. Steps:
        1. Get baseline parameters for injury type/region
        2. Calculate cumulative hazard ratio from covariates
        3. Adjust baseline survival curve
        4. Extract point estimates and probabilities
        5. Classify into recovery band
        
        Args:
            case: CaseInput with patient/injury details
            
        Returns:
            CoxPrediction with estimates and uncertainty
        """
        # Step 1: Get baseline injury parameters
        # -----------------------------------------
        # These come from published literature (see evidence_base.yaml)
        baseline_params = self.evidence.get_injury_params(
            case.injury_type, 
            case.body_region
        )
        
        if baseline_params is None:
            logger.warning(
                f"No evidence for {case.injury_type.value}/{case.body_region.value}, "
                "using defaults"
            )
            baseline_params = self._get_default_params(case.injury_type)
        
        # Step 2: Calculate cumulative hazard ratio
        # ------------------------------------------
        # HR > 1 means SLOWER recovery (higher hazard of NOT recovering)
        # HR < 1 means FASTER recovery
        # 
        # We accumulate log(HR) and exponentiate at the end:
        #   Total HR = exp(Σ log(HRᵢ)) = Π HRᵢ
        
        hazard_ratios = {}
        total_log_hr = 0.0
        
        # Age effect: HR 1.15 per decade over 25
        # Rationale: Older tissue heals slower, reduced physiological reserve
        # Source: anderson_2023, wiggins_2016
        if case.age > 25:
            decades_over_25 = (case.age - 25) / 10.0
            age_hr = self._get_hr('age', default=1.15) ** decades_over_25
            hazard_ratios['age'] = age_hr
            total_log_hr += np.log(age_hr)
            logger.debug(f"Age {case.age}: HR = {age_hr:.3f}")
        else:
            hazard_ratios['age'] = 1.0
        
        # Prior same-region injury: HR 1.80
        # Rationale: Previous injury indicates vulnerability, incomplete healing
        # Source: wiggins_2016, olivotto_2025
        if case.prior_same_region:
            prior_hr = self._get_hr('prior_same_region_injury', default=1.80)
            hazard_ratios['prior_same_region'] = prior_hr
            total_log_hr += np.log(prior_hr)
            logger.debug(f"Prior same region injury: HR = {prior_hr:.3f}")
        else:
            hazard_ratios['prior_same_region'] = 1.0
        
        # Smoking: HR 1.43
        # Rationale: Impaired tissue oxygenation, delayed wound healing
        # Source: anderson_2023
        if case.is_smoker:
            smoke_hr = self._get_hr('smoking', default=1.43)
            hazard_ratios['smoking'] = smoke_hr
            total_log_hr += np.log(smoke_hr)
            logger.debug(f"Smoker: HR = {smoke_hr:.3f}")
        else:
            hazard_ratios['smoking'] = 1.0
        
        # Supervised rehabilitation: HR 0.75 (protective)
        # Rationale: Structured progression, professional oversight, adherence
        # Source: olivotto_2025
        if case.receiving_treatment:
            rehab_hr = self._get_hr('supervised_rehabilitation', default=0.75)
            hazard_ratios['supervised_rehab'] = rehab_hr
            total_log_hr += np.log(rehab_hr)
            logger.debug(f"Supervised rehab: HR = {rehab_hr:.3f}")
        else:
            hazard_ratios['supervised_rehab'] = 1.0
        
        # Mental health comorbidity: RR 6.0 for poor outcome
        # This is a RISK RATIO for outcome, not a hazard ratio for time
        # We convert to approximate HR using log-linear scaling
        # Rationale: Psychological factors are strongest predictor of chronicity
        # Source: olivotto_2025, kcmhr_2024
        if case.has_mh_comorbidity:
            # RR 6.0 for poor outcome → approximate HR for delayed recovery
            # Use conservative HR of 2.0 for time effect (RR affects probability)
            mh_hr = 2.0
            hazard_ratios['mh_comorbidity'] = mh_hr
            total_log_hr += np.log(mh_hr)
            logger.debug(f"MH comorbidity: HR = {mh_hr:.3f}")
        else:
            hazard_ratios['mh_comorbidity'] = 1.0
        
        # Multiple TBI history (≥3): HR 1.80 (for TBI cases only)
        # Rationale: Cumulative neurological burden, reduced cognitive reserve
        # Source: kennedy_2018, tbicohe_2023
        if case.multiple_tbi_history and case.injury_type in [
            InjuryType.TBI_MILD, InjuryType.TBI_MODERATE, InjuryType.TBI_SEVERE
        ]:
            tbi_hr = self._get_hr('multiple_tbi_history', default=1.80)
            hazard_ratios['multiple_tbi'] = tbi_hr
            total_log_hr += np.log(tbi_hr)
            logger.debug(f"Multiple TBI history: HR = {tbi_hr:.3f}")
        else:
            hazard_ratios['multiple_tbi'] = 1.0
        
        # Trade category modifier (for RTD only, not fitness)
        # Rationale: Combat roles require higher fitness threshold
        # Source: Expert estimate pending calibration
        trade_category = get_trade_category(case.trade)
        trade_hr = TRADE_CATEGORY_RTD_MODIFIER[trade_category]
        hazard_ratios['trade_category'] = trade_hr
        # Note: Applied to RTD calculation separately, not total HR
        
        # Calculate total HR
        total_hr = np.exp(total_log_hr)
        logger.info(f"Total hazard ratio: {total_hr:.3f}")
        
        # Step 3: Adjust baseline survival curve
        # ----------------------------------------
        # Under proportional hazards:
        #   S(t|X) = S₀(t)^exp(βX) = S₀(t)^HR
        #
        # For median time:
        #   New median = Baseline median × HR^(1/k)
        # Where k is the Weibull shape parameter
        
        shape_k = self._get_weibull_shape(case.injury_type)
        
        # Adjusted median recovery (time to fitness)
        baseline_median = baseline_params.median_recovery_months
        adjusted_median = baseline_median * (total_hr ** (1 / shape_k))
        
        # Time to fitness (clinical milestone) - uses total HR
        time_to_fitness = baseline_params.time_to_fitness_months * (total_hr ** (1 / shape_k))
        
        # Time to RTD (occupational clearance) - also applies trade modifier
        # RTD takes longer for combat roles even after clinical recovery
        baseline_rtd = baseline_params.time_to_rtd_months
        time_to_rtd = baseline_rtd * (total_hr ** (1 / shape_k)) * trade_hr
        
        # Step 4: Calculate survival probabilities
        # -----------------------------------------
        # Using Weibull: S(t) = exp(-(t/λ)^k)
        # Where λ (scale) is derived from adjusted median
        
        scale_lambda = adjusted_median / (np.log(2) ** (1 / shape_k))
        
        def survival_prob(t: float) -> float:
            """Probability of NOT recovering by time t."""
            return np.exp(-((t / scale_lambda) ** shape_k))
        
        def recovery_prob(t: float) -> float:
            """Probability of having recovered by time t (= 1 - S(t))."""
            return 1.0 - survival_prob(t)
        
        # Recovery probabilities at key time points
        prob_3mo = recovery_prob(3.0)
        prob_6mo = recovery_prob(6.0)
        prob_12mo = recovery_prob(12.0)
        prob_24mo = recovery_prob(24.0)
        
        # Step 5: Calculate 90% prediction interval
        # ------------------------------------------
        # For Weibull, the p-th quantile is: t_p = λ × (-ln(1-p))^(1/k)
        
        lower_90 = scale_lambda * ((-np.log(0.95)) ** (1 / shape_k))
        upper_90 = scale_lambda * ((-np.log(0.05)) ** (1 / shape_k))
        
        # Step 6: Adjust outcome probabilities
        # -------------------------------------
        # Base probabilities from evidence, adjusted for MH comorbidity
        
        prob_full = baseline_params.prob_full_recovery
        prob_partial = baseline_params.prob_partial_recovery
        prob_discharge = baseline_params.prob_not_recovered
        
        # Mental health comorbidity shifts probabilities toward worse outcomes
        # RR 6.0 for poor outcome (olivotto_2025)
        if case.has_mh_comorbidity:
            # Shift probability mass from full recovery to partial/discharge
            shift = prob_full * 0.3  # 30% of full recovery probability shifts
            prob_full -= shift
            prob_partial += shift * 0.5
            prob_discharge += shift * 0.5
        
        # Multiple TBI shifts toward worse outcomes
        if case.multiple_tbi_history and case.injury_type in [
            InjuryType.TBI_MILD, InjuryType.TBI_MODERATE
        ]:
            shift = prob_full * 0.2
            prob_full -= shift
            prob_discharge += shift
        
        # Ensure probabilities sum to 1.0
        total_prob = prob_full + prob_partial + prob_discharge
        prob_full /= total_prob
        prob_partial /= total_prob
        prob_discharge /= total_prob
        
        # Step 7: Classify recovery band
        # --------------------------------
        recovery_band = self._classify_recovery_band(adjusted_median)
        
        # Step 8: Assess confidence level
        # ---------------------------------
        confidence, rationale = self._assess_confidence(
            baseline_params, case, total_hr
        )
        
        # Step 9: Compile prediction
        # ---------------------------
        return CoxPrediction(
            median_recovery_months=round(adjusted_median, 1),
            time_to_fitness_months=round(time_to_fitness, 1),
            time_to_rtd_months=round(time_to_rtd, 1),
            recovery_lower_90=round(lower_90, 1),
            recovery_upper_90=round(upper_90, 1),
            prob_recovery_3mo=round(prob_3mo, 3),
            prob_recovery_6mo=round(prob_6mo, 3),
            prob_recovery_12mo=round(prob_12mo, 3),
            prob_recovery_24mo=round(prob_24mo, 3),
            prob_full_recovery=round(prob_full, 3),
            prob_partial_recovery=round(prob_partial, 3),
            prob_medical_discharge=round(prob_discharge, 3),
            risk_score=round(total_log_hr, 3),
            recovery_band=recovery_band,
            hazard_ratios=hazard_ratios,
            confidence=confidence,
            confidence_rationale=rationale,
            primary_sources=baseline_params.sources or []
        )
    
    def _get_hr(self, factor_name: str, default: float = 1.0) -> float:
        """Get hazard ratio from evidence base with fallback."""
        rf = self.evidence.get_risk_factor(factor_name)
        if rf is None:
            return default
        return rf.value
    
    def _get_weibull_shape(self, injury_type: InjuryType) -> float:
        """
        Get Weibull shape parameter for injury type.
        
        Shape k controls hazard trajectory:
        - k < 1: Decreasing hazard (early failures more likely)
        - k = 1: Constant hazard (exponential distribution)
        - k > 1: Increasing hazard (recovery more likely over time)
        
        For recovery, k > 1 is appropriate: the longer you recover,
        the more likely you are to fully recover (healing accumulates).
        """
        if injury_type in [InjuryType.MSKI_MINOR, InjuryType.MSKI_MODERATE,
                          InjuryType.MSKI_MAJOR, InjuryType.MSKI_SEVERE]:
            return WEIBULL_SHAPE_PARAMS['MSKI']

        elif injury_type in [InjuryType.MH_MILD, InjuryType.MH_MODERATE,
                            InjuryType.MH_SEVERE]:
            return WEIBULL_SHAPE_PARAMS['MH']

        else:
            return 1.5  # Default: increasing hazard
    
    def _get_default_params(self, injury_type: InjuryType) -> InjuryParameters:
        """Return default parameters when evidence base has no match."""
        defaults = {
            InjuryType.MSKI_MINOR: (2.0, 1.0, 4.0, 0.8, 0.15, 0.05),
            InjuryType.MSKI_MODERATE: (6.0, 3.0, 12.0, 0.6, 0.25, 0.15),
            InjuryType.MSKI_MAJOR: (12.0, 6.0, 18.0, 0.5, 0.30, 0.20),
            InjuryType.MSKI_SEVERE: (18.0, 12.0, 36.0, 0.4, 0.30, 0.30),
            InjuryType.MH_MILD: (3.0, 1.0, 6.0, 0.7, 0.20, 0.10),
            InjuryType.MH_MODERATE: (8.0, 6.0, 18.0, 0.35, 0.30, 0.35),
            InjuryType.MH_SEVERE: (18.0, 12.0, 36.0, 0.2, 0.30, 0.50),
        }
        
        d = defaults.get(injury_type, (6.0, 3.0, 12.0, 0.5, 0.3, 0.2))
        
        return InjuryParameters(
            median_recovery_months=d[0],
            recovery_range_months=(d[1], d[2]),
            time_to_fitness_months=d[0] * 0.7,
            time_to_rtd_months=d[0],
            prob_full_recovery=d[3],
            prob_partial_recovery=d[4],
            prob_not_recovered=d[5],
            evidence_grade="Low",
            sources=["default_estimate"],
            stakeholder_explainer="Default estimates used - specific evidence not available."
        )
    
    def _classify_recovery_band(self, median_months: float) -> RecoveryBand:
        """
        Classify into workforce planning band.
        
        Bands:
        - FAST: <3 months - can plan short-term cover
        - MEDIUM: 3-6 months - medium-term workforce adjustment
        - SLOW: 6-12 months - long-term planning needed
        - COMPLEX: >12 months - may need permanent replacement planning
        """
        if median_months < 3:
            return RecoveryBand.FAST
        elif median_months < 6:
            return RecoveryBand.MEDIUM
        elif median_months < 12:
            return RecoveryBand.SLOW
        else:
            return RecoveryBand.COMPLEX
    
    def _assess_confidence(
        self, 
        params: InjuryParameters, 
        case: CaseInput,
        total_hr: float
    ) -> Tuple[str, str]:
        """
        Assess confidence in prediction based on evidence quality.
        
        Factors reducing confidence:
        - Low evidence grade
        - Many risk factors compounding
        - Unusual covariate combinations
        """
        issues = []
        
        # Evidence grade
        if params.evidence_grade == "Low" or params.evidence_grade == "Low-Moderate":
            issues.append("limited evidence for this injury type")
        
        # High total HR indicates many risk factors stacking
        if total_hr > 2.5:
            issues.append("multiple compounding risk factors")
        
        # Unusual combinations
        if case.has_mh_comorbidity and case.multiple_tbi_history:
            issues.append("complex comorbidity pattern")
        
        # Very young or old
        if case.age < 20 or case.age > 50:
            issues.append("age at edge of typical military range")
        
        # Determine confidence level
        if len(issues) == 0:
            return ("high", "Good evidence match with typical case profile")
        elif len(issues) == 1:
            return ("moderate", f"Some uncertainty: {issues[0]}")
        else:
            return ("low", f"High uncertainty: {'; '.join(issues)}")
    
    def get_survival_curve(
        self, 
        case: CaseInput, 
        max_months: int = 36
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full survival curve for plotting.
        
        Returns:
            Tuple of (time_points, survival_probabilities)
            Where survival = probability of NOT YET recovering
        """
        prediction = self.predict(case)
        
        # Derive Weibull parameters from prediction
        shape_k = self._get_weibull_shape(case.injury_type)
        scale_lambda = prediction.median_recovery_months / (np.log(2) ** (1 / shape_k))
        
        t = np.linspace(0, max_months, 100)
        survival = np.exp(-((t / scale_lambda) ** shape_k))
        
        return t, survival
    
    def explain_prediction(self, prediction: CoxPrediction) -> str:
        """
        Generate stakeholder-friendly explanation of prediction.
        
        Returns plain-language text suitable for clinical reports.
        """
        lines = []
        
        lines.append(f"PREDICTED RECOVERY TIMELINE")
        lines.append(f"=" * 40)
        lines.append(f"")
        lines.append(f"Expected time to full recovery: {prediction.median_recovery_months} months")
        lines.append(f"Range (90% of similar cases): {prediction.recovery_lower_90} - {prediction.recovery_upper_90} months")
        lines.append(f"")
        lines.append(f"MILESTONES:")
        lines.append(f"  • Return to Fitness (clinical): {prediction.time_to_fitness_months} months")
        lines.append(f"  • Return to Duty (occupational): {prediction.time_to_rtd_months} months")
        lines.append(f"")
        lines.append(f"RECOVERY PROBABILITIES:")
        lines.append(f"  • By 3 months:  {prediction.prob_recovery_3mo:.0%}")
        lines.append(f"  • By 6 months:  {prediction.prob_recovery_6mo:.0%}")
        lines.append(f"  • By 12 months: {prediction.prob_recovery_12mo:.0%}")
        lines.append(f"")
        lines.append(f"EXPECTED OUTCOMES:")
        lines.append(f"  • Full recovery (MFD):     {prediction.prob_full_recovery:.0%}")
        lines.append(f"  • Partial (RTD w/ limits): {prediction.prob_partial_recovery:.0%}")
        lines.append(f"  • Medical discharge:       {prediction.prob_medical_discharge:.0%}")
        lines.append(f"")
        lines.append(f"WORKFORCE PLANNING: {prediction.recovery_band.value.upper()} recovery trajectory")
        lines.append(f"")
        lines.append(f"FACTORS AFFECTING THIS PREDICTION:")
        
        for factor, hr in prediction.hazard_ratios.items():
            if hr != 1.0:
                direction = "↑ slower" if hr > 1 else "↓ faster"
                effect = abs(hr - 1) * 100
                lines.append(f"  • {factor}: {direction} by {effect:.0f}%")
        
        lines.append(f"")
        lines.append(f"Confidence: {prediction.confidence.upper()}")
        lines.append(f"({prediction.confidence_rationale})")
        lines.append(f"")
        lines.append(f"Evidence sources: {', '.join(prediction.primary_sources)}")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_model() -> CoxRecoveryModel:
    """Factory function to create model with default evidence base."""
    return CoxRecoveryModel()


def quick_predict(
    injury_type: InjuryType,
    body_region: BodyRegion,
    age: int = 30,
    trade: Trade = Trade.INFANTRY,
    prior_same_region: bool = False,
    is_smoker: bool = False,
    has_mh_comorbidity: bool = False
) -> CoxPrediction:
    """
    Quick prediction with minimal inputs.
    
    Convenience wrapper for common use cases.
    """
    model = CoxRecoveryModel()
    
    case = CaseInput(
        age=age,
        trade=trade,
        injury_type=injury_type,
        body_region=body_region,
        prior_same_region=prior_same_region,
        is_smoker=is_smoker,
        has_mh_comorbidity=has_mh_comorbidity
    )
    
    return model.predict(case)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: 35yo Infantry soldier with moderate knee injury
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Infantry, ACL injury, prior knee injury, smoker")
    print("=" * 60)
    
    case1 = CaseInput(
        age=35,
        trade=Trade.INFANTRY,
        injury_type=InjuryType.MSKI_MODERATE,
        body_region=BodyRegion.KNEE,
        prior_same_region=True,
        is_smoker=True,
        receiving_treatment=True
    )
    
    model = CoxRecoveryModel()
    pred1 = model.predict(case1)
    print(model.explain_prediction(pred1))
    
    # Example 2: Logistics clerk with mTBI + MH comorbidity
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Logistics, mTBI with PTSD comorbidity")
    print("=" * 60)
    
    case2 = CaseInput(
        age=28,
        trade=Trade.LOGISTICS,
        injury_type=InjuryType.TBI_MILD,
        body_region=BodyRegion.BRAIN,
        has_mh_comorbidity=True,
        multiple_tbi_history=True,
        receiving_treatment=True
    )
    
    pred2 = model.predict(case2)
    print(model.explain_prediction(pred2))
    
    # Example 3: Young Medic with PTSD
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Medic, moderate PTSD, no comorbidities")
    print("=" * 60)
    
    case3 = CaseInput(
        age=24,
        trade=Trade.MEDIC,
        injury_type=InjuryType.MH_MODERATE,
        body_region=BodyRegion.MENTAL,
        receiving_treatment=True
    )
    
    pred3 = model.predict(case3)
    print(model.explain_prediction(pred3))
