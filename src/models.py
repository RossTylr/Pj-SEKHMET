"""
JMES Synthetic Workforce - Data Models & Schemas
=================================================
Defines Pydantic models for type safety and validation.
Implements Chain-of-Verification through schema constraints.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import uuid
from datetime import datetime
import pandas as pd
import numpy as np


# ============================================================
# ENUMERATIONS - Constrained categorical values
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


class RankBand(str, Enum):
    OR2_OR4 = "OR2-OR4"
    OR5_OR7 = "OR5-OR7"
    OR8_OR9 = "OR8-OR9"
    OF1_OF3 = "OF1-OF3"
    OF4_OF5 = "OF4-OF5"


class Trade(str, Enum):
    CMT = "CMT"
    PARAMEDIC = "Paramedic"
    AHP = "AHP"
    ODP = "ODP"
    OTHER = "Other"


class JMESStatus(str, Enum):
    MFD = "MFD"  # Medically Fully Deployable
    MLD = "MLD"  # Medically Limited Deployable
    MND = "MND"  # Medically Non-Deployable


class InjuryType(str, Enum):
    NONE = "None"
    MSKI_MINOR = "MSKI-minor"
    MSKI_MAJOR = "MSKI-major"
    MH_EPISODE = "MH-episode"
    OTHER = "Other"


class DeploymentStatus(str, Enum):
    NOT_DEPLOYED = "Not_deployed"
    LOW_TEMPO = "Low_tempo"
    HIGH_TEMPO = "High_tempo"


class TrainingPhase(str, Enum):
    NONE = "None"
    LOW_RISK = "Low_risk"
    HIGH_RISK = "High_risk"


class PregnancyStatus(str, Enum):
    NOT_PREGNANT = "Not_pregnant"
    T1 = "T1"  # Trimester 1
    T2 = "T2"
    T3 = "T3"
    POSTPARTUM = "Postpartum"


class EngagementType(str, Enum):
    PC = "PC"
    IC = "IC"
    FE = "FE"
    UCM_H = "UCM-H"


class UnitEnvType(str, Enum):
    STANDARD = "Standard"
    HIGH_RISK = "High-Risk"
    HOT_COLD = "Hot/Cold"


class HealthRisk(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class OutflowReason(str, Enum):
    PREMATURE_VOL = "Premature_voluntary"
    END_ENGAGEMENT = "End_of_engagement"
    MEDICAL_DISCHARGE = "Medical_discharge"
    NORMAL_TERM = "Normal_termination"
    ADMINISTRATIVE = "Administrative"


# ============================================================
# DATACLASS MODELS - With validation
# ============================================================

@dataclass
class PersonnelMaster:
    """
    Table A: Personnel Master Record
    One row per synthetic individual
    """
    person_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    service_branch: ServiceBranch = ServiceBranch.ARMY
    regular_reserve: ServiceType = ServiceType.REGULAR
    age_start: int = 25
    gender: Gender = Gender.MALE
    rank_band: RankBand = RankBand.OR2_OR4
    trade: Trade = Trade.OTHER
    length_of_service_start: int = 0
    engagement_type: EngagementType = EngagementType.FE
    baseline_jmes: JMESStatus = JMESStatus.MFD
    baseline_health_risk: HealthRisk = HealthRisk.LOW
    unit_env_type: UnitEnvType = UnitEnvType.STANDARD
    initial_deployability: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Chain-of-Verification: Validate constraints"""
        assert 18 <= self.age_start <= 55, f"Age {self.age_start} out of range [18, 55]"
        assert 0 <= self.length_of_service_start <= 30, f"LoS {self.length_of_service_start} out of range"
        
    def to_dict(self) -> dict:
        return {
            'person_id': self.person_id,
            'service_branch': self.service_branch.value,
            'regular_reserve': self.regular_reserve.value,
            'age_start': self.age_start,
            'gender': self.gender.value,
            'rank_band': self.rank_band.value,
            'trade': self.trade.value,
            'length_of_service_start': self.length_of_service_start,
            'engagement_type': self.engagement_type.value,
            'baseline_jmes': self.baseline_jmes.value,
            'baseline_health_risk': self.baseline_health_risk.value,
            'unit_env_type': self.unit_env_type.value,
            'initial_deployability': self.initial_deployability,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class PersonMonth:
    """
    Table B: Person-Month Longitudinal Record
    One row per individual per month
    """
    person_id: str
    month: int
    age: float
    jmes_current: JMESStatus = JMESStatus.MFD
    jmes_event_this_month: bool = False
    injury_type: InjuryType = InjuryType.NONE
    injury_severity_score: int = 0
    deployment_status: DeploymentStatus = DeploymentStatus.NOT_DEPLOYED
    training_phase: TrainingPhase = TrainingPhase.NONE
    pregnancy_status: PregnancyStatus = PregnancyStatus.NOT_PREGNANT
    sick_days: int = 0
    outflow_flag: bool = False
    outflow_reason: Optional[OutflowReason] = None
    inflow_flag: bool = False
    survival_time: float = 0.0
    censor_flag: bool = False
    
    def __post_init__(self):
        """Chain-of-Verification: Validate constraints"""
        assert self.month >= 1, f"Month {self.month} must be >= 1"
        assert 0 <= self.sick_days <= 30, f"Sick days {self.sick_days} out of range"
        assert self.injury_severity_score >= 0, "Severity must be non-negative"
        
    def to_dict(self) -> dict:
        return {
            'person_id': self.person_id,
            'month': self.month,
            'age': round(self.age, 4),
            'jmes_current': self.jmes_current.value,
            'jmes_event_this_month': int(self.jmes_event_this_month),
            'injury_type': self.injury_type.value,
            'injury_severity_score': self.injury_severity_score,
            'deployment_status': self.deployment_status.value,
            'training_phase': self.training_phase.value,
            'pregnancy_status': self.pregnancy_status.value,
            'sick_days': self.sick_days,
            'outflow_flag': int(self.outflow_flag),
            'outflow_reason': self.outflow_reason.value if self.outflow_reason else None,
            'inflow_flag': int(self.inflow_flag),
            'survival_time': round(self.survival_time, 4),
            'censor_flag': int(self.censor_flag)
        }


# ============================================================
# SCHEMA DEFINITIONS - For DataFrame validation
# ============================================================

PERSONNEL_MASTER_SCHEMA = {
    'person_id': 'string',
    'service_branch': 'category',
    'regular_reserve': 'category',
    'age_start': 'int32',
    'gender': 'category',
    'rank_band': 'category',
    'trade': 'category',
    'length_of_service_start': 'int32',
    'engagement_type': 'category',
    'baseline_jmes': 'category',
    'baseline_health_risk': 'category',
    'unit_env_type': 'category',
    'initial_deployability': 'bool',
    'created_at': 'datetime64[ns]'
}

PERSON_MONTH_SCHEMA = {
    'person_id': 'string',
    'month': 'int32',
    'age': 'float32',
    'jmes_current': 'category',
    'jmes_event_this_month': 'int8',
    'injury_type': 'category',
    'injury_severity_score': 'int16',
    'deployment_status': 'category',
    'training_phase': 'category',
    'pregnancy_status': 'category',
    'sick_days': 'int16',
    'outflow_flag': 'int8',
    'outflow_reason': 'category',
    'inflow_flag': 'int8',
    'survival_time': 'float32',
    'censor_flag': 'int8'
}


def validate_dataframe(df: pd.DataFrame, schema: dict, table_name: str) -> list:
    """
    Chain-of-Verification: Validate DataFrame against schema
    Returns list of validation errors
    """
    errors = []
    
    # Check required columns
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        errors.append(f"{table_name}: Missing columns: {missing_cols}")
    
    # Check for unexpected columns
    extra_cols = set(df.columns) - set(schema.keys())
    if extra_cols:
        errors.append(f"{table_name}: Unexpected columns: {extra_cols}")
    
    # Check for nulls in required fields
    for col in ['person_id', 'month'] if 'month' in schema else ['person_id']:
        if col in df.columns and df[col].isnull().any():
            errors.append(f"{table_name}: Null values in {col}")
    
    return errors


def apply_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Apply schema types to DataFrame for memory optimization"""
    for col, dtype in schema.items():
        if col in df.columns:
            try:
                if dtype == 'category':
                    df[col] = df[col].astype('category')
                elif dtype.startswith('datetime'):
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert {col} to {dtype}: {e}")
    return df
