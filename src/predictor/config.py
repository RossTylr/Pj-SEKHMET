"""
Recovery Predictor Configuration
================================
All configurable parameters for recovery prediction.
Editable via Streamlit UI.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class InjuryType(str, Enum):
    MSKI_MINOR = "MSKI_minor"
    MSKI_MODERATE = "MSKI_moderate"
    MSKI_MAJOR = "MSKI_major"
    MSKI_SEVERE = "MSKI_severe"
    MH_MILD = "MH_mild"
    MH_MODERATE = "MH_moderate"
    MH_SEVERE = "MH_severe"
    ACUTE_MEDICAL = "Acute_medical"
    OTHER = "Other"


class BodyRegion(str, Enum):
    LOWER_BACK = "Lower_back"
    KNEE = "Knee"
    ANKLE_FOOT = "Ankle_foot"
    SHOULDER = "Shoulder"
    NECK = "Neck"
    HIP = "Hip"
    WRIST_HAND = "Wrist_hand"
    UPPER_BACK = "Upper_back"
    HEAD = "Head"
    MENTAL = "Mental"
    SYSTEMIC = "Systemic"
    OTHER = "Other"


class JMESStatus(str, Enum):
    MFD = "MFD"  # Medically Fully Deployable
    MLD = "MLD"  # Medically Limited Deployable
    MND = "MND"  # Medically Non-Deployable


class RecoveryBand(str, Enum):
    FAST = "Fast"
    MEDIUM = "Medium"
    SLOW = "Slow"
    COMPLEX = "Complex"


class Trade(str, Enum):
    INFANTRY = "Infantry"
    ARMOUR = "Armour"
    ARTILLERY = "Artillery"
    ENGINEER = "Engineer"
    SIGNALS = "Signals"
    LOGISTICS = "Logistics"
    INTELLIGENCE = "Intelligence"
    CMT = "CMT"
    PARAMEDIC = "Paramedic"
    NURSE = "Nurse"
    AHP = "AHP"
    MENTAL_HEALTH = "Mental_Health"
    ADMIN = "Admin"
    OTHER = "Other"


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

@dataclass
class InjuryProfile:
    """Recovery profile for an injury type"""
    base_recovery_months: Tuple[float, float]  # (min, max)
    variance: str  # Low, Medium, High, Very High
    recurrence_risk: float  # 0-1
    mld_probability: float  # Probability of causing MLD
    mnd_probability: float  # Probability of causing MND
    description: str


@dataclass
class RecoveryConfig:
    """Master configuration for recovery prediction"""
    
    # Recovery band thresholds (months)
    band_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Fast": (0, 3),
        "Medium": (3, 6),
        "Slow": (6, 12),
        "Complex": (12, 36)
    })
    
    # Injury type profiles
    injury_profiles: Dict[str, InjuryProfile] = field(default_factory=lambda: {
        "MSKI_minor": InjuryProfile(
            base_recovery_months=(1, 3),
            variance="Low",
            recurrence_risk=0.15,
            mld_probability=0.10,
            mnd_probability=0.01,
            description="Sprains, strains, minor soft tissue"
        ),
        "MSKI_moderate": InjuryProfile(
            base_recovery_months=(3, 6),
            variance="Medium",
            recurrence_risk=0.25,
            mld_probability=0.45,
            mnd_probability=0.05,
            description="Fractures, ligament tears, disc issues"
        ),
        "MSKI_major": InjuryProfile(
            base_recovery_months=(6, 12),
            variance="High",
            recurrence_risk=0.35,
            mld_probability=0.70,
            mnd_probability=0.20,
            description="Surgery required, complex fractures"
        ),
        "MSKI_severe": InjuryProfile(
            base_recovery_months=(12, 24),
            variance="Very High",
            recurrence_risk=0.50,
            mld_probability=0.85,
            mnd_probability=0.50,
            description="Career-threatening, multiple surgeries"
        ),
        "MH_mild": InjuryProfile(
            base_recovery_months=(2, 4),
            variance="Medium",
            recurrence_risk=0.30,
            mld_probability=0.15,
            mnd_probability=0.02,
            description="Adjustment disorders, mild anxiety"
        ),
        "MH_moderate": InjuryProfile(
            base_recovery_months=(4, 9),
            variance="High",
            recurrence_risk=0.45,
            mld_probability=0.55,
            mnd_probability=0.15,
            description="Depression, anxiety disorders"
        ),
        "MH_severe": InjuryProfile(
            base_recovery_months=(9, 18),
            variance="Very High",
            recurrence_risk=0.55,
            mld_probability=0.80,
            mnd_probability=0.40,
            description="PTSD, complex trauma, severe depression"
        ),
        "Acute_medical": InjuryProfile(
            base_recovery_months=(1, 4),
            variance="Medium",
            recurrence_risk=0.10,
            mld_probability=0.20,
            mnd_probability=0.05,
            description="Illness, infection, acute conditions"
        ),
        "Other": InjuryProfile(
            base_recovery_months=(2, 6),
            variance="Medium",
            recurrence_risk=0.15,
            mld_probability=0.25,
            mnd_probability=0.05,
            description="Other injuries not classified"
        )
    })
    
    # Body region modifiers (multiplier on recovery time)
    body_region_modifiers: Dict[str, float] = field(default_factory=lambda: {
        "Lower_back": 1.4,
        "Knee": 1.3,
        "Ankle_foot": 1.1,
        "Shoulder": 1.2,
        "Neck": 1.3,
        "Hip": 1.25,
        "Wrist_hand": 1.0,
        "Upper_back": 1.15,
        "Head": 1.2,
        "Mental": 1.5,
        "Systemic": 1.1,
        "Other": 1.0
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
        "Infantry": "Very High",
        "Armour": "High",
        "Artillery": "High",
        "Engineer": "High",
        "Signals": "Medium",
        "Logistics": "Medium",
        "Intelligence": "Low",
        "CMT": "High",
        "Paramedic": "High",
        "Nurse": "Medium",
        "AHP": "Medium",
        "Mental_Health": "Low",
        "Admin": "Low",
        "Other": "Medium"
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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
            return RecoveryBand(band)
    return RecoveryBand.COMPLEX


# Default config instance
DEFAULT_CONFIG = RecoveryConfig()
