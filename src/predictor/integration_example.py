"""
Integration Example: Using Synthetic Data with Recovery Predictor
================================================================
Shows how to integrate the synthetic data generator with the recovery predictor.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from predictor import RecoveryPredictor, CaseInput, RecoveryConfig
from predictor.config import InjuryType, BodyRegion, JMESStatus, Trade

# Optional: Import synthetic generator if available
try:
    from enhanced_generator import EnhancedSyntheticGenerator
    SYNTHETIC_AVAILABLE = True
except ImportError:
    try:
        from generator import SyntheticDataGenerator
        SYNTHETIC_AVAILABLE = True
    except ImportError:
        SYNTHETIC_AVAILABLE = False
        print("⚠️  Synthetic generator not available - using standalone predictor")


def predict_from_synthetic_data(master_df, monthly_df, injury_df=None):
    """
    Use recovery predictor on synthetic data.
    
    Args:
        master_df: Master personnel table
        monthly_df: Monthly person-month table
        injury_df: Optional injury table
    
    Returns:
        DataFrame with predictions for each injury case
    """
    predictor = RecoveryPredictor(RecoveryConfig())
    predictions = []
    
    # Example: Process injuries from monthly data
    injured = monthly_df[monthly_df['injury_type'] != 'None'].copy()
    
    for idx, row in injured.head(100).iterrows():  # Limit for demo
        try:
            # Map synthetic data to predictor input
            case = CaseInput(
                age=int(row.get('age', 30)),
                trade=Trade.OTHER,  # Map from row['trade'] if available
                injury_type=_map_injury_type(row.get('injury_type', 'MSKI-minor')),
                body_region=BodyRegion.OTHER,  # Map from injury data if available
                severity_score=_estimate_severity(row),
                prior_injury_count=0,  # Could track from history
                prior_same_region=False,
                current_jmes=_map_jmes(row.get('jmes_current', 'MFD')),
                months_since_injury=0,
                receiving_treatment=True
            )
            
            prediction = predictor.predict(case)
            
            predictions.append({
                'person_id': row.get('person_id'),
                'month': row.get('month'),
                'expected_recovery_months': prediction.expected_recovery_months,
                'recovery_band': prediction.recovery_band.value,
                'prob_full_recovery': prediction.prob_full_recovery,
                'confidence': prediction.confidence_level
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    return pd.DataFrame(predictions)


def _map_injury_type(injury_str):
    """Map synthetic injury types to predictor types"""
    mapping = {
        'MSKI-minor': InjuryType.MSKI_MINOR,
        'MSKI-major': InjuryType.MSKI_MAJOR,
        'MH-episode': InjuryType.MH_MODERATE,
    }
    return mapping.get(injury_str, InjuryType.MSKI_MINOR)


def _map_jmes(jmes_str):
    """Map JMES status"""
    mapping = {
        'MFD': JMESStatus.MFD,
        'MLD': JMESStatus.MLD,
        'MND': JMESStatus.MND,
    }
    return mapping.get(jmes_str, JMESStatus.MFD)


def _estimate_severity(row):
    """Estimate severity from synthetic data"""
    # Simple heuristic - could be improved
    if row.get('sick_days', 0) > 10:
        return 7
    elif row.get('sick_days', 0) > 5:
        return 5
    else:
        return 3


if __name__ == "__main__":
    print("✅ Recovery Predictor Integration Example")
    print(f"   Synthetic generator available: {SYNTHETIC_AVAILABLE}")
    print("\n   To use:")
    print("   1. Generate synthetic data")
    print("   2. Load into DataFrames")
    print("   3. Call predict_from_synthetic_data()")
    print("\n   Example:")
    print("   ```python")
    print("   from integration_example import predict_from_synthetic_data")
    print("   predictions = predict_from_synthetic_data(master_df, monthly_df)")
    print("   ```")

