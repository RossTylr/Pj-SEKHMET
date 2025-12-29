"""
JMES Synthetic Workforce - Unit Tests
=====================================
Chain-of-Verification test suite.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import (
    PersonnelMaster, PersonMonth,
    ServiceBranch, Gender, JMESStatus, Trade,
    validate_dataframe, PERSONNEL_MASTER_SCHEMA
)


class TestPersonnelMaster:
    """Test PersonnelMaster dataclass"""
    
    def test_default_creation(self):
        """Test default instance creation"""
        person = PersonnelMaster()
        assert person.person_id is not None
        assert person.age_start == 25
        assert person.service_branch == ServiceBranch.ARMY
    
    def test_age_validation(self):
        """Test age constraint enforcement"""
        with pytest.raises(AssertionError):
            PersonnelMaster(age_start=15)  # Under 18
        
        with pytest.raises(AssertionError):
            PersonnelMaster(age_start=60)  # Over 55
    
    def test_los_validation(self):
        """Test length of service constraint"""
        with pytest.raises(AssertionError):
            PersonnelMaster(length_of_service_start=35)  # Over 30
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        person = PersonnelMaster(
            age_start=30,
            gender=Gender.FEMALE,
            trade=Trade.CMT
        )
        d = person.to_dict()
        
        assert d['age_start'] == 30
        assert d['gender'] == 'Female'
        assert d['trade'] == 'CMT'


class TestPersonMonth:
    """Test PersonMonth dataclass"""
    
    def test_creation(self):
        """Test basic creation"""
        pm = PersonMonth(
            person_id="test-123",
            month=1,
            age=25.0
        )
        assert pm.month == 1
        assert pm.jmes_current == JMESStatus.MFD
    
    def test_month_validation(self):
        """Test month must be >= 1"""
        with pytest.raises(AssertionError):
            PersonMonth(person_id="test", month=0, age=25.0)
    
    def test_sick_days_validation(self):
        """Test sick days constraint"""
        with pytest.raises(AssertionError):
            PersonMonth(person_id="test", month=1, age=25.0, sick_days=35)


class TestSchemaValidation:
    """Test schema validation functions"""
    
    def test_valid_master_schema(self):
        """Test validation of compliant DataFrame"""
        df = pd.DataFrame({
            'person_id': ['a', 'b'],
            'service_branch': ['Army', 'RN'],
            'regular_reserve': ['Regular', 'Regular'],
            'age_start': [25, 30],
            'gender': ['Male', 'Female'],
            'rank_band': ['OR2-OR4', 'OR5-OR7'],
            'trade': ['CMT', 'AHP'],
            'length_of_service_start': [2, 5],
            'engagement_type': ['FE', 'PC'],
            'baseline_jmes': ['MFD', 'MLD'],
            'baseline_health_risk': ['Low', 'Medium'],
            'unit_env_type': ['Standard', 'High-Risk'],
            'initial_deployability': [True, False],
            'created_at': pd.to_datetime(['2024-01-01', '2024-01-01'])
        })
        
        errors = validate_dataframe(df, PERSONNEL_MASTER_SCHEMA, "test")
        assert len(errors) == 0
    
    def test_missing_columns(self):
        """Test detection of missing columns"""
        df = pd.DataFrame({'person_id': ['a']})
        errors = validate_dataframe(df, PERSONNEL_MASTER_SCHEMA, "test")
        assert any('Missing columns' in e for e in errors)


class TestDistributions:
    """Test distribution generation"""
    
    def test_service_mix_approximately_correct(self):
        """Test service branch distribution"""
        # This would test the generator, placeholder for now
        pass
    
    def test_age_range_valid(self):
        """Test age stays within bounds"""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
