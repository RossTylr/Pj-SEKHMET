"""
JMES Synthetic Workforce - Validation Module
=============================================
Chain-of-Verification for synthetic data quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    PASS = "✅ PASS"
    WARNING = "⚠️ WARNING"
    FAIL = "❌ FAIL"
    INFO = "ℹ️ INFO"


@dataclass
class ValidationResult:
    name: str
    severity: ValidationSeverity
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


class SyntheticDataValidator:
    """Chain-of-Verification for synthetic data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.results: List[ValidationResult] = []
    
    def validate_all(self, master_df: pd.DataFrame, month_df: pd.DataFrame) -> List[ValidationResult]:
        self.results = []
        self._validate_schema(master_df, "master")
        self._validate_schema(month_df, "monthly")
        self._validate_distributions(master_df)
        self._validate_temporal(month_df)
        self._validate_business_rules(master_df, month_df)
        return self.results
    
    def _validate_schema(self, df: pd.DataFrame, table_name: str):
        required = ['person_id', 'month', 'jmes_current'] if table_name == 'monthly' else ['person_id', 'service_branch', 'age_start']
        missing = [c for c in required if c not in df.columns]
        
        self.results.append(ValidationResult(
            name=f"Schema: {table_name}",
            severity=ValidationSeverity.PASS if not missing else ValidationSeverity.FAIL,
            message="All required columns present" if not missing else f"Missing: {missing}"
        ))
    
    def _validate_distributions(self, master_df: pd.DataFrame):
        cfg = self.config
        
        # Service mix
        actual = master_df['service_branch'].value_counts(normalize=True).to_dict()
        expected = cfg['service_mix']
        ok = all(abs(actual.get(k, 0) - v) < 0.03 for k, v in expected.items())
        
        self.results.append(ValidationResult(
            name="Distribution: Service",
            severity=ValidationSeverity.PASS if ok else ValidationSeverity.WARNING,
            message=f"Actual: {actual}",
            expected=str(expected)
        ))
        
        # Gender
        female_rate = (master_df['gender'] == 'Female').mean()
        expected_female = cfg['gender']['overall_female_rate']
        
        self.results.append(ValidationResult(
            name="Distribution: Gender",
            severity=ValidationSeverity.PASS if abs(female_rate - expected_female) < 0.03 else ValidationSeverity.WARNING,
            message=f"Female: {female_rate:.1%}",
            expected=f"{expected_female:.1%}"
        ))
    
    def _validate_temporal(self, month_df: pd.DataFrame):
        months = sorted(month_df['month'].unique())
        expected = list(range(1, max(months) + 1))
        
        self.results.append(ValidationResult(
            name="Temporal: Month sequence",
            severity=ValidationSeverity.PASS if months == expected else ValidationSeverity.FAIL,
            message=f"Months 1-{max(months)}"
        ))
    
    def _validate_business_rules(self, master_df: pd.DataFrame, month_df: pd.DataFrame):
        # Age >= 18
        invalid = (master_df['age_start'] < 18).sum()
        self.results.append(ValidationResult(
            name="Rule: Min age 18",
            severity=ValidationSeverity.PASS if invalid == 0 else ValidationSeverity.FAIL,
            message=f"{invalid} violations"
        ))
        
        # LoS constraint
        invalid_los = (master_df['length_of_service_start'] > master_df['age_start'] - 18).sum()
        self.results.append(ValidationResult(
            name="Rule: LoS <= age-18",
            severity=ValidationSeverity.PASS if invalid_los == 0 else ValidationSeverity.FAIL,
            message=f"{invalid_los} violations"
        ))
    
    def get_summary(self) -> Dict:
        return {
            'total': len(self.results),
            'passed': sum(1 for r in self.results if r.severity == ValidationSeverity.PASS),
            'warnings': sum(1 for r in self.results if r.severity == ValidationSeverity.WARNING),
            'failed': sum(1 for r in self.results if r.severity == ValidationSeverity.FAIL)
        }
