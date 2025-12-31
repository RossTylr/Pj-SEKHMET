"""
SEKHMET XGBoost Survival Model
==============================

XGBoost model with SHAP explainability for MSKI recovery prediction.

RESEARCH DEMONSTRATION
This model is trained on SYNTHETIC data derived from published literature.
It has NOT been validated against real UK military outcomes.
For clinical decisions, prefer the Cox PH model.

The value of this model:
1. Demonstrates non-linear interaction capture
2. Provides SHAP-based explainability
3. Prepares infrastructure for real data calibration
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


@dataclass
class XGBPrediction:
    """Output from XGBoost survival prediction."""
    predicted_time_months: float
    lower_bound_months: float
    upper_bound_months: float
    recovery_band: str  # FAST/MEDIUM/SLOW/COMPLEX

    # SHAP explainability
    shap_values: Optional[Dict[str, float]] = None
    top_positive_factors: Optional[List[tuple]] = None  # Slows recovery
    top_negative_factors: Optional[List[tuple]] = None  # Speeds recovery

    # Metadata
    model_version: str = "xgb_synthetic_v1"
    is_validated: bool = False  # Always False until real data calibration

    disclaimer: str = (
        "RESEARCH DEMONSTRATION: This prediction uses a model trained on "
        "synthetic data. NOT validated against real outcomes. "
        "For clinical decisions, use Cox PH model."
    )


class FeatureEncoder:
    """Encode case inputs into features for XGBoost."""

    SEVERITY_MAP = {
        'mski_minor': 1,
        'mski_moderate': 2,
        'mski_major': 3,
        'mski_severe': 4,
    }

    REGION_COMPLEXITY = {
        'knee': 3,
        'lower_back': 2,
        'shoulder': 2,
        'ankle_foot': 1,
        'hip_groin': 2,
        'cervical_spine': 3,
        'wrist_hand': 1,
    }

    OH_RISK_MAP = {
        'Low': 0,
        'Moderate': 1,
        'High': 2,
    }

    @classmethod
    def encode_case(cls, case_dict: Dict[str, Any]) -> Dict[str, float]:
        """Encode a single case into feature dictionary."""
        features = {}

        # Age features
        age = float(case_dict.get('age', 30))
        features['age'] = age
        features['age_over_35'] = 1.0 if age > 35 else 0.0
        features['age_squared'] = (age ** 2) / 100  # Scaled

        # Injury severity
        injury_type = str(case_dict.get('injury_type', 'mski_moderate')).lower()
        if hasattr(case_dict.get('injury_type'), 'value'):
            injury_type = case_dict['injury_type'].value
        features['injury_severity'] = cls.SEVERITY_MAP.get(injury_type, 2)

        # Body region complexity
        body_region = str(case_dict.get('body_region', 'knee')).lower()
        if hasattr(case_dict.get('body_region'), 'value'):
            body_region = case_dict['body_region'].value
        features['region_complexity'] = cls.REGION_COMPLEXITY.get(body_region, 2)

        # OH/Occupational risk
        oh_risk = case_dict.get('oh_risk', 'Moderate')
        features['oh_risk'] = cls.OH_RISK_MAP.get(oh_risk, 1)

        # Binary risk factors
        features['prior_same_region'] = float(case_dict.get('prior_same_region', 0))
        features['is_smoker'] = float(case_dict.get('is_smoker', 0))
        features['high_alcohol'] = float(case_dict.get('high_alcohol', 0))
        features['poor_sleep'] = float(case_dict.get('poor_sleep', 0))
        features['receiving_treatment'] = float(case_dict.get('receiving_treatment', 1))

        # BMI
        bmi = float(case_dict.get('bmi', 25.0))
        features['bmi'] = bmi
        features['bmi_obese'] = 1.0 if bmi >= 30 else 0.0
        features['bmi_severely_obese'] = 1.0 if bmi >= 35 else 0.0

        # Interaction terms
        features['age_x_severity'] = features['age'] * features['injury_severity'] / 10
        features['bmi_x_severity'] = features['bmi'] * features['injury_severity'] / 100
        features['prior_x_severity'] = features['prior_same_region'] * features['injury_severity']

        return features

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get ordered list of feature names."""
        dummy = cls.encode_case({})
        return list(dummy.keys())


class XGBSurvivalModel:
    """
    XGBoost model for MSKI recovery prediction.

    Trained on synthetic data derived from evidence_base.yaml parameters.
    Uses regression on log(time) as simplified survival approach.
    """

    DEFAULT_PARAMS = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
    }

    def __init__(self, params: Optional[Dict] = None):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost is required for XGBSurvivalModel. Install with: pip install xgboost")
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBRegressor] = None
        self.explainer = None
        self.feature_names: List[str] = FeatureEncoder.get_feature_names()
        self.is_fitted = False

    def train(self, df: Optional[pd.DataFrame] = None, n_synthetic: int = 5000):
        """
        Train model on synthetic data.

        If df is None, generates synthetic training data.
        """
        if df is None:
            logger.info(f"Generating {n_synthetic} synthetic training samples...")
            df = self._generate_synthetic_data(n_synthetic)

        logger.info(f"Training XGBoost on {len(df)} samples...")

        # Encode features
        X = pd.DataFrame([FeatureEncoder.encode_case(row.to_dict()) for _, row in df.iterrows()])
        self.feature_names = list(X.columns)

        # Target: log of recovery time
        y = np.log(df['time_to_event'].values + 0.1)

        # Train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, eval_set=[(X, y)], verbose=False)

        self.is_fitted = True

        # Initialize SHAP explainer
        if SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP explainer initialized")

        # Training metrics
        y_pred = self.model.predict(X)
        mae_months = np.mean(np.abs(np.exp(y) - np.exp(y_pred)))
        logger.info(f"Training complete. MAE: {mae_months:.2f} months")

        return {'mae_months': mae_months, 'n_samples': len(df)}

    def predict(self, case_dict: Dict[str, Any]) -> XGBPrediction:
        """Predict recovery time for a single case."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        # Encode features
        features = FeatureEncoder.encode_case(case_dict)
        X = pd.DataFrame([features])

        # Predict log-time
        log_time_pred = self.model.predict(X)[0]
        time_months = float(np.exp(log_time_pred))

        # Uncertainty estimate (simplified)
        uncertainty = time_months * 0.3
        lower_bound = max(0.5, time_months - 1.96 * uncertainty)
        upper_bound = time_months + 1.96 * uncertainty

        # SHAP values
        shap_values = None
        top_positive = None
        top_negative = None

        if SHAP_AVAILABLE and self.explainer is not None:
            shap_vals = self.explainer.shap_values(X)[0]
            shap_values = dict(zip(self.feature_names, shap_vals))

            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            top_positive = [(k, v) for k, v in sorted_shap if v > 0][:3]
            top_negative = [(k, v) for k, v in sorted_shap if v < 0][:3]

        # Recovery band
        recovery_band = self._classify_band(time_months)

        return XGBPrediction(
            predicted_time_months=round(time_months, 1),
            lower_bound_months=round(lower_bound, 1),
            upper_bound_months=round(upper_bound, 1),
            recovery_band=recovery_band,
            shap_values=shap_values,
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
        )

    def _classify_band(self, months: float) -> str:
        """Classify into recovery band."""
        if months < 3:
            return "FAST"
        elif months < 6:
            return "MEDIUM"
        elif months < 12:
            return "SLOW"
        else:
            return "COMPLEX"

    def _generate_synthetic_data(self, n: int) -> pd.DataFrame:
        """Generate synthetic training data from evidence base parameters."""
        np.random.seed(42)

        # Injury distribution
        injuries = np.random.choice(
            ['mski_minor', 'mski_moderate', 'mski_major', 'mski_severe'],
            size=n,
            p=[0.25, 0.45, 0.20, 0.10]
        )

        # Body regions
        regions = np.random.choice(
            ['knee', 'lower_back', 'shoulder', 'ankle_foot', 'hip_groin', 'cervical_spine', 'wrist_hand'],
            size=n,
            p=[0.20, 0.25, 0.15, 0.15, 0.10, 0.10, 0.05]
        )

        # Baseline recovery times by injury type (from evidence base)
        baseline_times = {
            'mski_minor': 2.0,
            'mski_moderate': 6.0,
            'mski_major': 9.0,
            'mski_severe': 18.0,
        }

        # Region modifiers
        region_mods = {
            'knee': 1.3,
            'lower_back': 1.4,
            'shoulder': 1.2,
            'ankle_foot': 1.1,
            'hip_groin': 1.25,
            'cervical_spine': 1.3,
            'wrist_hand': 1.0,
        }

        # Generate covariates
        ages = np.clip(np.random.normal(28, 6, n), 18, 55).astype(int)
        oh_risk = np.random.choice(['Low', 'Moderate', 'High'], n, p=[0.3, 0.5, 0.2])
        prior_injury = (np.random.random(n) < 0.15).astype(int)
        smokers = (np.random.random(n) < 0.20).astype(int)
        high_alcohol = (np.random.random(n) < 0.10).astype(int)
        poor_sleep = (np.random.random(n) < 0.15).astype(int)
        supervised = (np.random.random(n) < 0.75).astype(int)
        bmi = np.clip(np.random.normal(26, 4, n), 18, 45)

        # Calculate recovery times
        times = []
        for i in range(n):
            base = baseline_times[injuries[i]]

            # Apply modifiers
            t = base * region_mods[regions[i]]

            # Age effect
            if ages[i] > 25:
                t *= 1.15 ** ((ages[i] - 25) / 10)

            # OH risk
            if oh_risk[i] == 'Moderate':
                t *= 1.15
            elif oh_risk[i] == 'High':
                t *= 1.30

            # Other risk factors
            if prior_injury[i]:
                t *= 1.80
            if smokers[i]:
                t *= 1.43
            if high_alcohol[i]:
                t *= 1.25
            if poor_sleep[i]:
                t *= 1.30
            if supervised[i]:
                t *= 0.75
            if bmi[i] >= 35:
                t *= 1.40
            elif bmi[i] >= 30:
                t *= 1.20

            # Add noise
            t *= np.random.lognormal(0, 0.2)
            times.append(max(0.5, t))

        return pd.DataFrame({
            'injury_type': injuries,
            'body_region': regions,
            'age': ages,
            'oh_risk': oh_risk,
            'prior_same_region': prior_injury,
            'is_smoker': smokers,
            'high_alcohol': high_alcohol,
            'poor_sleep': poor_sleep,
            'receiving_treatment': supervised,
            'bmi': bmi,
            'time_to_event': times,
            'event': np.ones(n, dtype=int),
        })

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


# Convenience function
def train_xgboost_model(n_samples: int = 5000) -> XGBSurvivalModel:
    """Train XGBoost model on synthetic data."""
    model = XGBSurvivalModel()
    model.train(n_synthetic=n_samples)
    return model


if __name__ == "__main__":
    # Quick test
    if XGB_AVAILABLE:
        print("Testing XGBoost model...")
        model = XGBSurvivalModel()
        model.train(n_synthetic=1000)
        pred = model.predict({
            'age': 35,
            'injury_type': 'mski_moderate',
            'body_region': 'knee',
            'oh_risk': 'High',
            'bmi': 28
        })
        print(f'XGBoost prediction: {pred.predicted_time_months} months')
        print(f'SHAP available: {pred.shap_values is not None}')
        print('PASS: XGBoost working')
    else:
        print("XGBoost not installed - skipping test")
