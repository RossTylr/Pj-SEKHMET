"""
JMES Enhanced Data Models
=========================
Extended models with ethnicity, multiple injuries, pregnancy history,
and contingency modelling support.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import uuid
from datetime import datetime


# ============================================================
# EXTENDED ENUMERATIONS
# ============================================================

class ServiceBranch(str, Enum):
    ARMY = "Army"
    RN = "RN"
    RAF = "RAF"


class ServiceType(str, Enum):
    REGULAR = "Regular"
    RESERVE = "Reserve"


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class Ethnicity(str, Enum):
    WHITE_BRITISH = "White_British"
    WHITE_OTHER = "White_Other"
    BLACK_CARIBBEAN = "Black_Caribbean"
    BLACK_AFRICAN = "Black_African"
    ASIAN_INDIAN = "Asian_Indian"
    ASIAN_PAKISTANI = "Asian_Pakistani"
    ASIAN_BANGLADESHI = "Asian_Bangladeshi"
    ASIAN_CHINESE = "Asian_Chinese"
    ASIAN_OTHER = "Asian_Other"
    MIXED_WHITE_BLACK = "Mixed_White_Black"
    MIXED_WHITE_ASIAN = "Mixed_White_Asian"
    MIXED_OTHER = "Mixed_Other"
    ARAB = "Arab"
    OTHER = "Other"


class RankBand(str, Enum):
    OR1 = "OR1"
    OR2 = "OR2"
    OR3 = "OR3"
    OR4 = "OR4"
    OR5 = "OR5"
    OR6 = "OR6"
    OR7 = "OR7"
    OR8 = "OR8"
    OR9 = "OR9"
    OF1 = "OF1"
    OF2 = "OF2"
    OF3 = "OF3"
    OF4 = "OF4"
    OF5 = "OF5"
    OF6 = "OF6"
    OF7_PLUS = "OF7_plus"


class RankCategory(str, Enum):
    JUNIOR_ENLISTED = "Junior_Enlisted"   # OR1-OR4
    SENIOR_ENLISTED = "Senior_Enlisted"   # OR5-OR7
    WARRANT_OFFICER = "Warrant_Officer"   # OR8-OR9
    JUNIOR_OFFICER = "Junior_Officer"     # OF1-OF3
    SENIOR_OFFICER = "Senior_Officer"     # OF4-OF5
    FLAG_OFFICER = "Flag_Officer"         # OF6+


class Trade(str, Enum):
    # Medical
    CMT = "CMT"
    PARAMEDIC = "Paramedic"
    NURSE = "Nurse"
    AHP = "AHP"
    ODP = "ODP"
    MED_ADMIN = "MedAdmin"
    DENTAL = "Dental"
    PHARMACY = "Pharmacy"
    MENTAL_HEALTH = "Mental_Health"
    RADIOLOGY = "Radiology"
    # Combat
    INFANTRY = "Infantry"
    ARMOUR = "Armour"
    ARTILLERY = "Artillery"
    ENGINEER = "Engineer"
    # Support
    SIGNALS = "Signals"
    LOGISTICS = "Logistics"
    INTELLIGENCE = "Intelligence"
    OTHER = "Other"


class JMESStatus(str, Enum):
    MFD = "MFD"   # Medically Fully Deployable
    MLD = "MLD"   # Medically Limited Deployable
    MND = "MND"   # Medically Non-Deployable


class InjuryType(str, Enum):
    NONE = "None"
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
    LOWER_BACK = "lower_back"
    KNEE = "knee"
    ANKLE_FOOT = "ankle_foot"
    SHOULDER = "shoulder"
    NECK = "neck"
    HIP = "hip"
    WRIST_HAND = "wrist_hand"
    UPPER_BACK = "upper_back"
    HEAD = "head"
    MENTAL = "mental"  # For MH conditions
    SYSTEMIC = "systemic"  # For medical conditions
    OTHER = "other"


class DeploymentType(str, Enum):
    NOT_DEPLOYED = "Not_deployed"
    OPERATIONAL_HIGH = "Operational_high_intensity"
    OPERATIONAL_LOW = "Operational_low_intensity"
    TRAINING_OVERSEAS = "Training_overseas"
    HUMANITARIAN = "Humanitarian"
    OTHER = "Other_deployment"


class TrainingPhase(str, Enum):
    NONE = "None"
    BASIC = "Basic_training"
    SPECIALIST = "Specialist_training"
    CAREER_COURSE = "Career_course"
    EXERCISE_MAJOR = "Exercise_major"
    EXERCISE_MINOR = "Exercise_minor"
    ADVENTURE = "Adventure_training"


class PregnancyStatus(str, Enum):
    NOT_PREGNANT = "Not_pregnant"
    NOT_APPLICABLE = "Not_applicable"  # Male
    T1 = "Trimester_1"
    T2 = "Trimester_2"
    T3 = "Trimester_3"
    MATERNITY_LEAVE = "Maternity_leave"
    POSTPARTUM = "Postpartum"


class EngagementType(str, Enum):
    OPEN = "Open_Engagement"
    FIXED = "Fixed_Commission"
    SHORT = "Short_Career"
    FULL = "Full_Career"
    RESERVE = "Reserve"


class UnitEnvType(str, Enum):
    STANDARD = "Standard"
    HIGH_READINESS = "High_readiness"
    TRAINING_EST = "Training_establishment"
    HEADQUARTERS = "Headquarters"
    REMOTE_HARSH = "Remote_harsh"


class HealthRisk(str, Enum):
    VERY_LOW = "Very_Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very_High"


class OutflowReason(str, Enum):
    VOLUNTARY_EARLY = "Voluntary_early"
    END_OF_ENGAGEMENT = "End_of_engagement"
    MEDICAL_DISCHARGE = "Medical_discharge"
    ADMINISTRATIVE = "Administrative"
    RETIREMENT = "Retirement"
    FAMILY_REASONS = "Family_reasons"
    CAREER_CHANGE = "Career_change"
    MISCONDUCT = "Misconduct"


class ChronicCondition(str, Enum):
    NONE = "None"
    CHRONIC_PAIN = "Chronic_pain"
    MOBILITY_LIMITATION = "Mobility_limitation"
    PTSD = "PTSD"
    DEPRESSION = "Depression"
    ANXIETY = "Anxiety"
    HEARING_LOSS = "Hearing_loss"
    RESPIRATORY = "Respiratory"
    CARDIOVASCULAR = "Cardiovascular"
    MUSCULOSKELETAL = "Musculoskeletal_chronic"
    OTHER = "Other_chronic"


# ============================================================
# INJURY HISTORY RECORD
# ============================================================

@dataclass
class InjuryRecord:
    """Single injury event record"""
    injury_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    month_occurred: int = 0
    injury_type: InjuryType = InjuryType.NONE
    body_region: BodyRegion = BodyRegion.OTHER
    severity_score: int = 0
    recovery_days: int = 0
    led_to_jmes_change: bool = False
    context: str = "routine"  # deployment, training, exercise, routine
    is_recurrence: bool = False
    
    def to_dict(self) -> dict:
        return {
            'injury_id': self.injury_id,
            'month_occurred': self.month_occurred,
            'injury_type': self.injury_type.value,
            'body_region': self.body_region.value,
            'severity_score': self.severity_score,
            'recovery_days': self.recovery_days,
            'led_to_jmes_change': int(self.led_to_jmes_change),
            'context': self.context,
            'is_recurrence': int(self.is_recurrence)
        }


# ============================================================
# PREGNANCY HISTORY RECORD
# ============================================================

@dataclass
class PregnancyRecord:
    """Single pregnancy event record"""
    pregnancy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_month: int = 0
    end_month: int = 0
    outcome: str = "normal"  # normal, complication, early_return
    career_break_taken: bool = False
    
    def to_dict(self) -> dict:
        return {
            'pregnancy_id': self.pregnancy_id,
            'start_month': self.start_month,
            'end_month': self.end_month,
            'outcome': self.outcome,
            'career_break_taken': int(self.career_break_taken)
        }


# ============================================================
# DEPLOYMENT HISTORY RECORD
# ============================================================

@dataclass
class DeploymentRecord:
    """Single deployment event record"""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_month: int = 0
    end_month: int = 0
    deployment_type: DeploymentType = DeploymentType.NOT_DEPLOYED
    location_risk: str = "medium"  # low, medium, high, very_high
    injuries_during: int = 0
    
    def to_dict(self) -> dict:
        return {
            'deployment_id': self.deployment_id,
            'start_month': self.start_month,
            'end_month': self.end_month,
            'deployment_type': self.deployment_type.value,
            'location_risk': self.location_risk,
            'injuries_during': self.injuries_during
        }


# ============================================================
# ENHANCED PERSONNEL MASTER (TABLE A)
# ============================================================

@dataclass
class EnhancedPersonnelMaster:
    """
    Enhanced personnel master record with full demographic
    and contingency modelling attributes.
    """
    # Core identifiers
    person_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Service attributes
    service_branch: ServiceBranch = ServiceBranch.ARMY
    service_type: ServiceType = ServiceType.REGULAR
    
    # Demographics
    age_at_entry: int = 20
    age_start: int = 25  # Age at simulation start
    gender: Gender = Gender.MALE
    ethnicity: Ethnicity = Ethnicity.WHITE_BRITISH
    
    # Career attributes
    rank: RankBand = RankBand.OR2
    rank_category: RankCategory = RankCategory.JUNIOR_ENLISTED
    trade: Trade = Trade.OTHER
    length_of_service_start: int = 0
    engagement_type: EngagementType = EngagementType.OPEN
    
    # Medical baseline
    baseline_jmes: JMESStatus = JMESStatus.MFD
    baseline_health_risk: HealthRisk = HealthRisk.LOW
    has_chronic_condition: bool = False
    chronic_conditions: List[ChronicCondition] = field(default_factory=list)
    
    # Environment
    unit_env_type: UnitEnvType = UnitEnvType.STANDARD
    initial_deployability: bool = True
    
    # History summaries (computed at generation)
    prior_injury_count: int = 0
    prior_deployment_count: int = 0
    prior_pregnancy_count: int = 0
    
    # Performance proxy (affects promotion, retention)
    performance_band: str = "average"  # low, below_average, average, above_average, high
    
    # Metadata
    cohort_month: int = 0  # Month entered simulation (0 = initial cohort)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate constraints"""
        assert 18 <= self.age_start <= 55, f"Age {self.age_start} out of range [18, 55]"
        assert 0 <= self.length_of_service_start <= 37, f"LoS out of range"
        assert self.length_of_service_start <= self.age_start - 18, "LoS cannot exceed age - 18"
        
        # Set rank category based on rank
        self._set_rank_category()
    
    def _set_rank_category(self):
        """Derive rank category from rank"""
        rank_val = self.rank.value
        if rank_val in ['OR1', 'OR2', 'OR3', 'OR4']:
            self.rank_category = RankCategory.JUNIOR_ENLISTED
        elif rank_val in ['OR5', 'OR6', 'OR7']:
            self.rank_category = RankCategory.SENIOR_ENLISTED
        elif rank_val in ['OR8', 'OR9']:
            self.rank_category = RankCategory.WARRANT_OFFICER
        elif rank_val in ['OF1', 'OF2', 'OF3']:
            self.rank_category = RankCategory.JUNIOR_OFFICER
        elif rank_val in ['OF4', 'OF5']:
            self.rank_category = RankCategory.SENIOR_OFFICER
        else:
            self.rank_category = RankCategory.FLAG_OFFICER
    
    def to_dict(self) -> dict:
        return {
            'person_id': self.person_id,
            'service_branch': self.service_branch.value,
            'service_type': self.service_type.value,
            'age_at_entry': self.age_at_entry,
            'age_start': self.age_start,
            'gender': self.gender.value,
            'ethnicity': self.ethnicity.value,
            'rank': self.rank.value,
            'rank_category': self.rank_category.value,
            'trade': self.trade.value,
            'length_of_service_start': self.length_of_service_start,
            'engagement_type': self.engagement_type.value,
            'baseline_jmes': self.baseline_jmes.value,
            'baseline_health_risk': self.baseline_health_risk.value,
            'has_chronic_condition': int(self.has_chronic_condition),
            'chronic_conditions': ','.join([c.value for c in self.chronic_conditions]) if self.chronic_conditions else '',
            'unit_env_type': self.unit_env_type.value,
            'initial_deployability': int(self.initial_deployability),
            'prior_injury_count': self.prior_injury_count,
            'prior_deployment_count': self.prior_deployment_count,
            'prior_pregnancy_count': self.prior_pregnancy_count,
            'performance_band': self.performance_band,
            'cohort_month': self.cohort_month,
            'created_at': self.created_at.isoformat()
        }


# ============================================================
# ENHANCED PERSON-MONTH (TABLE B)
# ============================================================

@dataclass
class EnhancedPersonMonth:
    """
    Enhanced person-month record with full event tracking
    and contingency state.
    """
    # Identifiers
    person_id: str = ""
    month: int = 1
    
    # Current state
    age: float = 25.0
    current_los: float = 0.0
    current_rank: RankBand = RankBand.OR2
    
    # JMES
    jmes_current: JMESStatus = JMESStatus.MFD
    jmes_event_this_month: bool = False
    jmes_direction: str = "stable"  # deterioration, improvement, stable
    
    # Injury this month
    injury_occurred: bool = False
    injury_type: InjuryType = InjuryType.NONE
    injury_body_region: BodyRegion = BodyRegion.OTHER
    injury_severity: int = 0
    injury_is_recurrence: bool = False
    injury_context: str = "routine"
    
    # Cumulative injury state
    total_injury_count: int = 0
    injuries_last_12_months: int = 0
    active_recovery: bool = False
    recovery_days_remaining: int = 0
    
    # Deployment
    deployment_status: DeploymentType = DeploymentType.NOT_DEPLOYED
    deployment_month_count: int = 0  # Months into current deployment
    total_deployment_months: int = 0  # Lifetime
    
    # Training
    training_phase: TrainingPhase = TrainingPhase.NONE
    
    # Pregnancy (females only)
    pregnancy_status: PregnancyStatus = PregnancyStatus.NOT_APPLICABLE
    pregnancy_month: int = 0  # Month of current pregnancy
    total_pregnancies: int = 0
    
    # Work impact
    sick_days: int = 0
    limited_duties: bool = False
    
    # Turnover
    outflow_flag: bool = False
    outflow_reason: Optional[OutflowReason] = None
    inflow_flag: bool = False
    
    # Survival analysis
    survival_time: float = 0.0
    event_flag: bool = False  # Had JMES deterioration
    censor_flag: bool = False
    
    # Chronic condition development
    new_chronic_condition: bool = False
    chronic_condition_type: ChronicCondition = ChronicCondition.NONE
    
    # Promotion
    promoted_this_month: bool = False
    
    def __post_init__(self):
        assert self.month >= 1, "Month must be >= 1"
        assert 0 <= self.sick_days <= 31, "Sick days out of range"
    
    def to_dict(self) -> dict:
        return {
            'person_id': self.person_id,
            'month': self.month,
            'age': round(self.age, 4),
            'current_los': round(self.current_los, 4),
            'current_rank': self.current_rank.value,
            'jmes_current': self.jmes_current.value,
            'jmes_event_this_month': int(self.jmes_event_this_month),
            'jmes_direction': self.jmes_direction,
            'injury_occurred': int(self.injury_occurred),
            'injury_type': self.injury_type.value,
            'injury_body_region': self.injury_body_region.value,
            'injury_severity': self.injury_severity,
            'injury_is_recurrence': int(self.injury_is_recurrence),
            'injury_context': self.injury_context,
            'total_injury_count': self.total_injury_count,
            'injuries_last_12_months': self.injuries_last_12_months,
            'active_recovery': int(self.active_recovery),
            'recovery_days_remaining': self.recovery_days_remaining,
            'deployment_status': self.deployment_status.value,
            'deployment_month_count': self.deployment_month_count,
            'total_deployment_months': self.total_deployment_months,
            'training_phase': self.training_phase.value,
            'pregnancy_status': self.pregnancy_status.value,
            'pregnancy_month': self.pregnancy_month,
            'total_pregnancies': self.total_pregnancies,
            'sick_days': self.sick_days,
            'limited_duties': int(self.limited_duties),
            'outflow_flag': int(self.outflow_flag),
            'outflow_reason': self.outflow_reason.value if self.outflow_reason else None,
            'inflow_flag': int(self.inflow_flag),
            'survival_time': round(self.survival_time, 4),
            'event_flag': int(self.event_flag),
            'censor_flag': int(self.censor_flag),
            'new_chronic_condition': int(self.new_chronic_condition),
            'chronic_condition_type': self.chronic_condition_type.value,
            'promoted_this_month': int(self.promoted_this_month)
        }


# ============================================================
# INJURY HISTORY TABLE (TABLE C)
# ============================================================

@dataclass
class InjuryHistoryRecord:
    """
    Separate table for full injury history.
    Links to person_id, allows multiple injuries per person.
    """
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    person_id: str = ""
    injury_number: int = 1  # 1st, 2nd, 3rd injury for this person
    month_occurred: int = 0
    injury_type: InjuryType = InjuryType.NONE
    body_region: BodyRegion = BodyRegion.OTHER
    severity_score: int = 0
    recovery_days: int = 0
    sick_days_taken: int = 0
    led_to_jmes_change: bool = False
    new_jmes_status: Optional[JMESStatus] = None
    context: str = "routine"
    is_recurrence: bool = False
    same_region_as_previous: bool = False
    led_to_chronic: bool = False
    chronic_condition_type: ChronicCondition = ChronicCondition.NONE
    
    def to_dict(self) -> dict:
        return {
            'record_id': self.record_id,
            'person_id': self.person_id,
            'injury_number': self.injury_number,
            'month_occurred': self.month_occurred,
            'injury_type': self.injury_type.value,
            'body_region': self.body_region.value,
            'severity_score': self.severity_score,
            'recovery_days': self.recovery_days,
            'sick_days_taken': self.sick_days_taken,
            'led_to_jmes_change': int(self.led_to_jmes_change),
            'new_jmes_status': self.new_jmes_status.value if self.new_jmes_status else None,
            'context': self.context,
            'is_recurrence': int(self.is_recurrence),
            'same_region_as_previous': int(self.same_region_as_previous),
            'led_to_chronic': int(self.led_to_chronic),
            'chronic_condition_type': self.chronic_condition_type.value
        }


# ============================================================
# SCHEMA DEFINITIONS
# ============================================================

ENHANCED_MASTER_SCHEMA = {
    'person_id': 'string',
    'service_branch': 'category',
    'service_type': 'category',
    'age_at_entry': 'int16',
    'age_start': 'int16',
    'gender': 'category',
    'ethnicity': 'category',
    'rank': 'category',
    'rank_category': 'category',
    'trade': 'category',
    'length_of_service_start': 'int16',
    'engagement_type': 'category',
    'baseline_jmes': 'category',
    'baseline_health_risk': 'category',
    'has_chronic_condition': 'int8',
    'chronic_conditions': 'string',
    'unit_env_type': 'category',
    'initial_deployability': 'int8',
    'prior_injury_count': 'int16',
    'prior_deployment_count': 'int16',
    'prior_pregnancy_count': 'int8',
    'performance_band': 'category',
    'cohort_month': 'int16',
    'created_at': 'datetime64[ns]'
}

ENHANCED_MONTH_SCHEMA = {
    'person_id': 'string',
    'month': 'int16',
    'age': 'float32',
    'current_los': 'float32',
    'current_rank': 'category',
    'jmes_current': 'category',
    'jmes_event_this_month': 'int8',
    'jmes_direction': 'category',
    'injury_occurred': 'int8',
    'injury_type': 'category',
    'injury_body_region': 'category',
    'injury_severity': 'int8',
    'injury_is_recurrence': 'int8',
    'injury_context': 'category',
    'total_injury_count': 'int16',
    'injuries_last_12_months': 'int8',
    'active_recovery': 'int8',
    'recovery_days_remaining': 'int16',
    'deployment_status': 'category',
    'deployment_month_count': 'int8',
    'total_deployment_months': 'int16',
    'training_phase': 'category',
    'pregnancy_status': 'category',
    'pregnancy_month': 'int8',
    'total_pregnancies': 'int8',
    'sick_days': 'int8',
    'limited_duties': 'int8',
    'outflow_flag': 'int8',
    'outflow_reason': 'category',
    'inflow_flag': 'int8',
    'survival_time': 'float32',
    'event_flag': 'int8',
    'censor_flag': 'int8',
    'new_chronic_condition': 'int8',
    'chronic_condition_type': 'category',
    'promoted_this_month': 'int8'
}

INJURY_HISTORY_SCHEMA = {
    'record_id': 'string',
    'person_id': 'string',
    'injury_number': 'int16',
    'month_occurred': 'int16',
    'injury_type': 'category',
    'body_region': 'category',
    'severity_score': 'int8',
    'recovery_days': 'int16',
    'sick_days_taken': 'int8',
    'led_to_jmes_change': 'int8',
    'new_jmes_status': 'category',
    'context': 'category',
    'is_recurrence': 'int8',
    'same_region_as_previous': 'int8',
    'led_to_chronic': 'int8',
    'chronic_condition_type': 'category'
}
