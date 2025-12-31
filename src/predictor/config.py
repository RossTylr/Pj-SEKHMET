"""
SEKHMET Recovery Predictor - Configuration & Domain Types
=========================================================

This module defines the core domain types for the SEKHMET military personnel
recovery prediction system. All enums and configurations are aligned with:
- UK Defence Medical Services terminology
- Joint Medical Employment Standard (JMES) framework
- Clinical evidence base (evidence_base.yaml v1.1.0)

Trade Categories
----------------
Following NATO/UK doctrine, trades are grouped into three categories based on
physical demands and operational role. This affects Return to Duty (RTD)
thresholds - a Combat role requires higher fitness than Combat Service Support.

Reference: JSP 950 Medical Policy, Chapter 6 - Medical Employment Standards
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import yaml
from pathlib import Path


# =============================================================================
# TRADE / ROLE CLASSIFICATION
# =============================================================================

class TradeCategory(Enum):
    """
    Trade categories based on physical demands and operational role.

    These map to different RTD fitness thresholds:
    - COMBAT: Highest physical demands, requires full JMES MFD for deployment
    - COMBAT_SUPPORT: Moderate demands, some role restrictions acceptable
    - COMBAT_SERVICE_SUPPORT: Lower physical demands, wider range of MES acceptable

    Evidence basis:
    - High-demand trades have HR ~1.30 for delayed RTD (expert estimate,
      pending calibration with MOD MES data)
    - Infantry/Para/RM have highest re-injury rates post-RTD

    Reference: JSP 822 (Defence Direction and Guidance for Training and Education)
    """
    COMBAT = "combat"
    COMBAT_SUPPORT = "combat_support"
    COMBAT_SERVICE_SUPPORT = "combat_service_support"


class Trade(Enum):
    """
    Specific military trades/roles mapped to categories.

    This is not exhaustive - represents common trades for modelling.
    Each trade maps to a TradeCategory for applying the appropriate
    recovery time modifier.
    """
    # Combat Arms (TradeCategory.COMBAT)
    INFANTRY = "infantry"
    ROYAL_MARINES = "royal_marines"
    PARACHUTE_REGIMENT = "para"
    ARMOUR = "armour"
    ARTILLERY = "artillery"
    COMBAT_ENGINEER = "combat_engineer"

    # Combat Support (TradeCategory.COMBAT_SUPPORT)
    SIGNALS = "signals"
    INTELLIGENCE = "intelligence"
    REME = "reme"  # Royal Electrical and Mechanical Engineers
    MEDIC = "medic"
    MILITARY_POLICE = "military_police"

    # Combat Service Support (TradeCategory.COMBAT_SERVICE_SUPPORT)
    LOGISTICS = "logistics"
    AGC = "agc"  # Adjutant General's Corps (admin, HR, legal)
    DENTAL = "dental"
    VETERINARY = "veterinary"
    CHAPLAIN = "chaplain"
    GENERIC = "generic"  # Default/unknown

    # Legacy aliases for backward compatibility with heuristic model
    ENGINEER = "combat_engineer"
    CMT = "medic"
    PARAMEDIC = "medic"
    NURSE = "medic"
    AHP = "medic"
    MENTAL_HEALTH = "medic"
    ADMIN = "agc"
    OTHER = "generic"


# Mapping from specific trades to categories
TRADE_CATEGORY_MAP: Dict[Trade, TradeCategory] = {
    # Combat
    Trade.INFANTRY: TradeCategory.COMBAT,
    Trade.ROYAL_MARINES: TradeCategory.COMBAT,
    Trade.PARACHUTE_REGIMENT: TradeCategory.COMBAT,
    Trade.ARMOUR: TradeCategory.COMBAT,
    Trade.ARTILLERY: TradeCategory.COMBAT,
    Trade.COMBAT_ENGINEER: TradeCategory.COMBAT,

    # Combat Support
    Trade.SIGNALS: TradeCategory.COMBAT_SUPPORT,
    Trade.INTELLIGENCE: TradeCategory.COMBAT_SUPPORT,
    Trade.REME: TradeCategory.COMBAT_SUPPORT,
    Trade.MEDIC: TradeCategory.COMBAT_SUPPORT,
    Trade.MILITARY_POLICE: TradeCategory.COMBAT_SUPPORT,

    # Combat Service Support
    Trade.LOGISTICS: TradeCategory.COMBAT_SERVICE_SUPPORT,
    Trade.AGC: TradeCategory.COMBAT_SERVICE_SUPPORT,
    Trade.DENTAL: TradeCategory.COMBAT_SERVICE_SUPPORT,
    Trade.VETERINARY: TradeCategory.COMBAT_SERVICE_SUPPORT,
    Trade.CHAPLAIN: TradeCategory.COMBAT_SERVICE_SUPPORT,
    Trade.GENERIC: TradeCategory.COMBAT_SERVICE_SUPPORT,
}


def get_trade_category(trade: Trade) -> TradeCategory:
    """Get the category for a specific trade."""
    return TRADE_CATEGORY_MAP.get(trade, TradeCategory.COMBAT_SERVICE_SUPPORT)


# =============================================================================
# INJURY CLASSIFICATION
# =============================================================================

class InjuryType(Enum):
    """
    Injury types aligned with evidence base categories.

    Severity levels (MINOR/MODERATE/MAJOR/SEVERE) map to clinical grading:
    - MINOR: Self-limiting, <4 weeks expected recovery
    - MODERATE: Requires treatment, 1-6 months recovery
    - MAJOR: Significant intervention, 6-12 months recovery
    - SEVERE: Complex/surgical, >12 months or uncertain recovery

    Note: MH removed in V2 - MSKI only.
    """
    # Musculoskeletal (MSKI) only
    MSKI_MINOR = "mski_minor"
    MSKI_MODERATE = "mski_moderate"
    MSKI_MAJOR = "mski_major"
    MSKI_SEVERE = "mski_severe"

    # Legacy values for backward compatibility
    ACUTE_MEDICAL = "mski_minor"
    OTHER = "mski_moderate"

    @property
    def display_name(self) -> str:
        """Human-readable name for UI with proper capitalisation."""
        # Split into category and severity
        parts = self.value.split('_')
        if len(parts) == 2:
            category = parts[0].upper()  # MSKI
            severity = parts[1].title()  # Minor, Moderate, Major, Severe
            return f"{category} {severity}"
        return self.value.replace('_', ' ').title()


class BodyRegion(Enum):
    """
    Anatomical regions for MSKI injury localisation.

    Maps to evidence base parameters (e.g., injuries.MSKI_moderate.knee_acl).
    Note: MH removed in V2 - MSKI body regions only.
    """
    # Upper body
    SHOULDER = "shoulder"
    WRIST_HAND = "wrist_hand"

    # Spine
    CERVICAL_SPINE = "cervical_spine"
    LOWER_BACK = "lower_back"  # Lumbar

    # Lower body
    HIP_GROIN = "hip_groin"
    KNEE = "knee"
    ANKLE_FOOT = "ankle_foot"

    # Legacy values for backward compatibility
    NECK = "cervical_spine"
    HIP = "hip_groin"


# =============================================================================
# JMES STATUS
# =============================================================================

class JMESStatus(Enum):
    """
    Joint Medical Employment Standard grades.

    JMES defines employability in UK Armed Forces:
    - MFD: Medically Fully Deployable - no restrictions
    - MLD: Medically Limited Deployable - some restrictions, can deploy
    - MND: Medically Non-Deployable - cannot deploy, may be fit for home duties
    - MNDPR: MND Permanent - permanent downgrade
    - MU: Medically Unfit - being processed for medical discharge

    Reference: JSP 950 Leaflet 6-7-1
    """
    MFD = "mfd"      # Fully fit
    MLD = "mld"      # Limited deployable
    MND = "mnd"      # Non-deployable (temporary)
    MNDPR = "mnd_permanent"  # Non-deployable permanent
    MU = "mu"        # Medically unfit / discharge


class RecoveryBand(Enum):
    """
    Recovery trajectory classification for workforce planning.

    Used to bucket personnel into planning categories:
    - FAST: Expected RTD <3 months
    - MEDIUM: Expected RTD 3-6 months
    - SLOW: Expected RTD 6-12 months
    - COMPLEX: Expected RTD >12 months or high uncertainty
    """
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    COMPLEX = "complex"


# =============================================================================
# EVIDENCE BASE CONFIGURATION (Cox Model)
# =============================================================================

@dataclass
class InjuryParameters:
    """
    Evidence-based parameters for a specific injury type.

    All parameters sourced from evidence_base.yaml with citations.
    Times in months, probabilities as decimals (0-1).
    """
    # Recovery timeline
    median_recovery_months: float
    recovery_range_months: tuple  # (lower_90, upper_90)
    time_to_fitness_months: float
    time_to_rtd_months: float

    # Outcome probabilities
    prob_full_recovery: float
    prob_partial_recovery: float
    prob_not_recovered: float

    # Additional rates
    reinjury_rate: Optional[float] = None
    recurrence_rate_12mo: Optional[float] = None

    # Metadata
    evidence_grade: str = "Unknown"
    sources: List[str] = None
    stakeholder_explainer: str = ""


@dataclass
class RiskFactor:
    """
    Risk factor modifier from evidence base.

    Effect types:
    - hazard_ratio: Multiplicative effect on time-to-event
    - risk_ratio: Multiplicative effect on probability
    - prevalence_modifier: Adjusts baseline prevalence
    """
    name: str
    effect_type: str  # hazard_ratio, risk_ratio, prevalence_modifier
    value: float      # HR, RR, or prevalence
    ci_95: tuple      # (lower, upper)
    applies_to: List[str]  # injury types this applies to
    direction: str    # delays_recovery, accelerates_recovery, increases_reinjury
    sources: List[str]
    stakeholder_explainer: str = ""


class EvidenceBase:
    """
    Loader and accessor for clinical evidence parameters.

    Loads from evidence_base.yaml and provides typed access to:
    - Injury-specific parameters
    - Risk factor modifiers
    - Source citations
    """

    def __init__(self, yaml_path: Optional[Path] = None):
        """
        Load evidence base from YAML file.

        Args:
            yaml_path: Path to evidence_base.yaml. If None, uses default.
        """
        if yaml_path is None:
            yaml_path = Path(__file__).parent / "evidence_base.yaml"

        with open(yaml_path, 'r') as f:
            self._data = yaml.safe_load(f)

        self.version = self._data.get('metadata', {}).get('version', 'unknown')
        self._injuries = self._data.get('injuries', {})
        self._risk_factors = self._data.get('risk_factors', {})
        self._sources = self._data.get('sources', {})

    def get_injury_params(
        self,
        injury_type: InjuryType,
        body_region: BodyRegion
    ) -> Optional[InjuryParameters]:
        """
        Get evidence-based parameters for an injury.

        Args:
            injury_type: The type/severity of injury
            body_region: Anatomical location

        Returns:
            InjuryParameters if found, None otherwise
        """
        # Map enum to YAML keys
        type_key = self._injury_type_to_yaml_key(injury_type)
        region_key = self._body_region_to_yaml_key(body_region)

        if type_key not in self._injuries:
            return None

        injury_data = self._injuries[type_key].get(region_key)
        if injury_data is None:
            return None

        return InjuryParameters(
            median_recovery_months=injury_data.get('median_recovery_months', 6.0),
            recovery_range_months=tuple(injury_data.get('recovery_range_months', [3.0, 12.0])),
            time_to_fitness_months=injury_data.get('time_to_fitness_months', 3.0),
            time_to_rtd_months=injury_data.get('time_to_rtd_months', 6.0),
            prob_full_recovery=injury_data.get('prob_full_recovery', 0.5),
            prob_partial_recovery=injury_data.get('prob_partial_recovery', 0.3),
            prob_not_recovered=injury_data.get('prob_not_recovered', 0.2),
            reinjury_rate=injury_data.get('reinjury_rate'),
            recurrence_rate_12mo=injury_data.get('recurrence_rate_12mo'),
            evidence_grade=injury_data.get('evidence_grade', 'Unknown'),
            sources=injury_data.get('sources', []),
            stakeholder_explainer=injury_data.get('stakeholder_explainer', '')
        )

    def get_risk_factor(self, factor_name: str) -> Optional[RiskFactor]:
        """Get a risk factor modifier by name."""
        rf_data = self._risk_factors.get(factor_name)
        if rf_data is None:
            return None

        return RiskFactor(
            name=factor_name,
            effect_type=rf_data.get('effect_type', 'hazard_ratio'),
            value=rf_data.get('hr', rf_data.get('rr', rf_data.get('value', 1.0))),
            ci_95=tuple(rf_data.get('ci_95', [0.8, 1.2])),
            applies_to=rf_data.get('applies_to', ['all']),
            direction=rf_data.get('direction', 'unknown'),
            sources=rf_data.get('sources', []),
            stakeholder_explainer=rf_data.get('stakeholder_explainer', '')
        )

    def get_source_citation(self, source_id: str) -> Optional[Dict]:
        """Get full citation for a source."""
        return self._sources.get(source_id)

    def _injury_type_to_yaml_key(self, injury_type: InjuryType) -> str:
        """Map InjuryType enum to YAML section key (MSKI only - V2)."""
        mapping = {
            InjuryType.MSKI_MINOR: 'MSKI_minor',
            InjuryType.MSKI_MODERATE: 'MSKI_moderate',
            InjuryType.MSKI_MAJOR: 'MSKI_major',
            InjuryType.MSKI_SEVERE: 'MSKI_severe',
        }
        return mapping.get(injury_type, 'MSKI_moderate')

    def _body_region_to_yaml_key(self, body_region: BodyRegion) -> str:
        """Map BodyRegion enum to YAML section key (MSKI only - V2)."""
        mapping = {
            BodyRegion.KNEE: 'knee_acl',
            BodyRegion.LOWER_BACK: 'lower_back',
            BodyRegion.SHOULDER: 'shoulder',
            BodyRegion.ANKLE_FOOT: 'ankle_foot',
            BodyRegion.HIP_GROIN: 'hip_groin',
            BodyRegion.CERVICAL_SPINE: 'cervical_spine',
            BodyRegion.WRIST_HAND: 'wrist_hand',
        }
        key = body_region.value if hasattr(body_region, 'value') else str(body_region)
        return mapping.get(body_region, key)


# =============================================================================
# TRADE CATEGORY MODIFIERS
# =============================================================================

# Hazard ratios for RTD by trade category
# Combat roles require higher fitness threshold, so RTD takes longer
# even when Return to Fitness milestone is reached
TRADE_CATEGORY_RTD_MODIFIER: Dict[TradeCategory, float] = {
    # Combat: HR 1.30 - takes 30% longer to achieve RTD vs CSS baseline
    # Rationale: Infantry/Para require full MFD, high physical demands
    # Evidence: Expert estimate, pending calibration (evidence_base.yaml)
    TradeCategory.COMBAT: 1.30,

    # Combat Support: HR 1.15 - moderate additional time
    # Rationale: Field roles but lower sustained physical demands
    TradeCategory.COMBAT_SUPPORT: 1.15,

    # Combat Service Support: HR 1.00 - baseline
    # Rationale: Office/technical roles, lower physical RTD threshold
    TradeCategory.COMBAT_SERVICE_SUPPORT: 1.00,
}


# Note: Return to Fitness is NOT modified by trade - it's a clinical milestone
# Only Return to Duty (occupational clearance) varies by role demands


# =============================================================================
# LEGACY HEURISTIC MODEL CONFIGURATION
# =============================================================================
# These classes maintain backward compatibility with recovery_model.py

@dataclass
class InjuryProfile:
    """Recovery profile for an injury type (legacy heuristic model)"""
    base_recovery_months: Tuple[float, float]  # (min, max)
    variance: str  # Low, Medium, High, Very High
    recurrence_risk: float  # 0-1
    mld_probability: float  # Probability of causing MLD
    mnd_probability: float  # Probability of causing MND
    description: str


@dataclass
class RecoveryConfig:
    """Master configuration for recovery prediction (legacy heuristic model)"""

    # Recovery band thresholds (months)
    band_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Fast": (0, 3),
        "Medium": (3, 6),
        "Slow": (6, 12),
        "Complex": (12, 36)
    })

    # Injury type profiles
    injury_profiles: Dict[str, InjuryProfile] = field(default_factory=lambda: {
        "mski_minor": InjuryProfile(
            base_recovery_months=(1, 3),
            variance="Low",
            recurrence_risk=0.15,
            mld_probability=0.10,
            mnd_probability=0.01,
            description="Sprains, strains, minor soft tissue"
        ),
        "mski_moderate": InjuryProfile(
            base_recovery_months=(3, 6),
            variance="Medium",
            recurrence_risk=0.25,
            mld_probability=0.45,
            mnd_probability=0.05,
            description="Fractures, ligament tears, disc issues"
        ),
        "mski_major": InjuryProfile(
            base_recovery_months=(6, 12),
            variance="High",
            recurrence_risk=0.35,
            mld_probability=0.70,
            mnd_probability=0.20,
            description="Surgery required, complex fractures"
        ),
        "mski_severe": InjuryProfile(
            base_recovery_months=(12, 24),
            variance="Very High",
            recurrence_risk=0.50,
            mld_probability=0.85,
            mnd_probability=0.50,
            description="Career-threatening, multiple surgeries"
        ),
    })

    # Body region modifiers (multiplier on recovery time) - MSKI only
    body_region_modifiers: Dict[str, float] = field(default_factory=lambda: {
        "lower_back": 1.4,
        "knee": 1.3,
        "ankle_foot": 1.1,
        "shoulder": 1.2,
        "cervical_spine": 1.3,
        "hip_groin": 1.25,
        "wrist_hand": 1.0,
    })

    # Age modifiers (multiplier on recovery time)
    age_modifiers: Dict[str, float] = field(default_factory=lambda: {
        "18-25": 0.85,
        "26-30": 0.95,
        "31-35": 1.0,
        "36-40": 1.1,
        "41-45": 1.25,
        "46-50": 1.4,
        "51+": 1.6
    })

    # Trade physical demand (affects MSKI recovery)
    trade_physical_demand: Dict[str, str] = field(default_factory=lambda: {
        "infantry": "Very High",
        "royal_marines": "Very High",
        "para": "Very High",
        "armour": "High",
        "artillery": "High",
        "combat_engineer": "High",
        "signals": "Medium",
        "intelligence": "Low",
        "reme": "High",
        "medic": "High",
        "military_police": "Medium",
        "logistics": "Medium",
        "agc": "Low",
        "dental": "Low",
        "veterinary": "Low",
        "chaplain": "Low",
        "generic": "Medium",
    })

    # Physical demand recovery modifier
    physical_demand_modifiers: Dict[str, float] = field(default_factory=lambda: {
        "Very High": 1.3,
        "High": 1.15,
        "Medium": 1.0,
        "Low": 0.9
    })

    # Prior injury count modifier
    prior_injury_modifiers: Dict[str, float] = field(default_factory=lambda: {
        "0": 1.0,
        "1-2": 1.1,
        "3-4": 1.25,
        "5+": 1.5
    })

    # Recurrence modifier (if same body region injured before)
    recurrence_modifier: float = 1.4

    # JMES transition probabilities (monthly)
    jmes_transitions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "MND_to_MLD": {"base": 0.08, "with_treatment": 0.12},
        "MND_to_MFD": {"base": 0.02, "with_treatment": 0.04},
        "MLD_to_MFD": {"base": 0.15, "with_treatment": 0.22},
        "MLD_to_MND": {"base": 0.03, "with_treatment": 0.02},
        "MFD_to_MLD": {"base": 0.005, "with_treatment": 0.003}
    })


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_age_band(age: int) -> str:
    """Convert age to band"""
    if age <= 25:
        return "18-25"
    elif age <= 30:
        return "26-30"
    elif age <= 35:
        return "31-35"
    elif age <= 40:
        return "36-40"
    elif age <= 45:
        return "41-45"
    elif age <= 50:
        return "46-50"
    else:
        return "51+"


def get_prior_injury_band(count: int) -> str:
    """Convert prior injury count to band"""
    if count == 0:
        return "0"
    elif count <= 2:
        return "1-2"
    elif count <= 4:
        return "3-4"
    else:
        return "5+"


def get_recovery_band(months: float, config: RecoveryConfig) -> RecoveryBand:
    """Determine recovery band from predicted months"""
    for band, (lo, hi) in config.band_thresholds.items():
        if lo <= months < hi:
            return RecoveryBand(band.lower())
    return RecoveryBand.COMPLEX


# Default config instance
DEFAULT_CONFIG = RecoveryConfig()
