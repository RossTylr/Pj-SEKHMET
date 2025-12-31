# SEKHMET Revision V2 - Instructions for Claude Code

## Context

You are updating the Pj-SEKHMET repository with these changes:
1. **Remove MH completely** - all MH injury types, MH tab, MH risk factors, MH sections in YAML
2. **Remove Trade/Physical Demand** - replace with OH/Occupational risk factor
3. **Add XGBoost model** - third model option with SHAP explainability
4. **Keep synthetic data warning** on XGBoost

Repository: https://github.com/RossTylr/Pj-SEKHMET

---

## SUMMARY OF CHANGES

| Remove | Add/Update |
|--------|------------|
| All MH injury types | XGBoost model with SHAP |
| MH tab in UI | Three model options in UI |
| MH risk factors | OH/Occupational risk factor |
| Trade/Physical Demand dropdown | - |
| MH sections in evidence_base.yaml | XGBoost synthetic training |

---

## STEP 1: Remove MH Completely

### In config.py

**Remove MH from InjuryType enum:**
```python
class InjuryType(Enum):
    # MSKI only - MH REMOVED
    MSKI_MINOR = "mski_minor"
    MSKI_MODERATE = "mski_moderate"
    MSKI_MAJOR = "mski_major"
    MSKI_SEVERE = "mski_severe"
    
    # DELETE these:
    # MH_MILD = "mh_mild"
    # MH_MODERATE = "mh_moderate"
    # MH_SEVERE = "mh_severe"
```

**Remove BodyRegion.MENTAL:**
```python
class BodyRegion(Enum):
    KNEE = "knee"
    LOWER_BACK = "lower_back"
    SHOULDER = "shoulder"
    ANKLE_FOOT = "ankle_foot"
    HIP_GROIN = "hip_groin"
    CERVICAL_SPINE = "cervical_spine"
    WRIST_HAND = "wrist_hand"
    
    # DELETE this:
    # MENTAL = "mental"
```

### In evidence_base.yaml

**Delete all MH sections:**
```yaml
# DELETE entire sections:
# MH_mild:
#   ptsd:
#     ...
#   depression:
#     ...
# MH_moderate:
#   ...
# MH_severe:
#   ...
```

**Delete MH-specific risk factors:**
```yaml
# DELETE these risk factors:
# mh_comorbidity:
#   ...
# trauma_exposure:
#   ...
# low_social_support:
#   ...
# prior_mh_history:
#   ...
# substance_use:
#   ...
# poor_treatment_engagement:
#   ...
```

### In app.py

**Remove MH tab entirely:**
```python
# BEFORE:
# tab_mski, tab_mh, tab_cohort, tab_settings = st.tabs([...])

# AFTER:
tab_mski, tab_cohort, tab_settings = st.tabs([
    "🦴 Individual Prediction",
    "📊 Cohort Planning",
    "⚙️ Model Settings"
])
```

**Delete the entire `render_mh_prediction()` function.**

**Remove any MH options from dropdowns:**
```python
# DELETE any MH-related selectbox options
# DELETE has_mh_comorbidity checkbox
```

---

## STEP 2: Remove Trade/Physical Demand, Add OH/Occupational Risk

### In config.py

**Remove Trade enum and related functions** (or keep for backend but remove from UI):
```python
# Can comment out or delete:
# class Trade(Enum):
#     INFANTRY = "infantry"
#     ...

# class TradeCategory(Enum):
#     COMBAT = "combat"
#     ...

# def get_trade_category(trade: Trade) -> TradeCategory:
#     ...

# TRADE_CATEGORY_RTD_MODIFIER = {...}
```

### In app.py

**Remove Trade dropdown from sidebar:**
```python
# DELETE:
# trade_options = {...}
# trade = st.sidebar.selectbox("Trade", ...)
# st.sidebar.caption(f"Category: {trade_cat.value...}")
```

**Add OH/Occupational Risk Factor:**
```python
# Risk Factors - OCCUPATION (replaces Trade)
st.sidebar.subheader("💼 Occupational Factors")

oh_risk = st.sidebar.select_slider(
    "OH/Occupational Risk",
    options=["Low", "Moderate", "High"],
    value="Moderate",
    help="Occupational health risk assessment for RTD"
)

# Map to hazard ratio
oh_risk_modifier = {
    "Low": 1.0,
    "Moderate": 1.15,
    "High": 1.30
}[oh_risk]
```

### In evidence_base.yaml

**Update risk factors section:**
```yaml
risk_factors:
  # LIFESTYLE
  smoking:
    effect_type: "hazard_ratio"
    hr: 1.43
    ci_95: [1.17, 1.74]
    category: "lifestyle"
    direction: "delays_recovery"
    applies_to: ["MSKI_minor", "MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["anderson_2023"]
    stakeholder_explainer: "Smoking impairs tissue oxygenation and delays wound healing."
  
  alcohol_high:
    effect_type: "hazard_ratio"
    hr: 1.25
    ci_95: [1.05, 1.48]
    category: "lifestyle"
    direction: "delays_recovery"
    applies_to: ["MSKI_minor", "MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["olivotto_2025"]
    stakeholder_explainer: "High alcohol intake associated with poorer recovery outcomes."
  
  poor_sleep:
    effect_type: "hazard_ratio"
    hr: 1.30
    ci_95: [1.10, 1.54]
    category: "lifestyle"
    direction: "delays_recovery"
    applies_to: ["MSKI_minor", "MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["olivotto_2025"]
    stakeholder_explainer: "Poor sleep quality delays tissue repair and recovery."
  
  # OCCUPATION (replaces Trade/Physical Demand)
  oh_risk_moderate:
    effect_type: "hazard_ratio"
    hr: 1.15
    ci_95: [1.00, 1.32]
    category: "occupation"
    direction: "delays_recovery"
    applies_to: ["MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["olivotto_2025", "shaw_2019"]
    stakeholder_explainer: "Moderate occupational risk - standard RTD pathway."
  
  oh_risk_high:
    effect_type: "hazard_ratio"
    hr: 1.30
    ci_95: [1.10, 1.54]
    category: "occupation"
    direction: "delays_recovery"
    applies_to: ["MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["olivotto_2025", "shaw_2019"]
    stakeholder_explainer: "High occupational risk requires longer RTD even after clinical recovery."
  
  # BMI
  obesity_class1:
    effect_type: "hazard_ratio"
    hr: 1.20
    ci_95: [1.05, 1.38]
    category: "bmi"
    applies_to_bmi: [30, 35]
    direction: "delays_recovery"
    applies_to: ["MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["olivotto_2025"]
    stakeholder_explainer: "Obesity (BMI 30-35) associated with 20% longer recovery times."
  
  obesity_class2plus:
    effect_type: "hazard_ratio"
    hr: 1.40
    ci_95: [1.15, 1.70]
    category: "bmi"
    applies_to_bmi: [35, 100]
    direction: "delays_recovery"
    applies_to: ["MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["olivotto_2025"]
    stakeholder_explainer: "Severe obesity (BMI 35+) significantly delays MSKI recovery."
  
  # CLINICAL
  age:
    effect_type: "hazard_ratio"
    per_unit: "per decade over 25"
    hr: 1.15
    ci_95: [1.08, 1.23]
    category: "demographic"
    direction: "delays_recovery"
    applies_to: ["MSKI_minor", "MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["anderson_2023", "wiggins_2016"]
    stakeholder_explainer: "Each decade over 25 adds ~15% to recovery time."
  
  prior_same_region_injury:
    effect_type: "hazard_ratio"
    hr: 1.80
    ci_95: [1.40, 2.30]
    category: "clinical"
    direction: "delays_recovery"
    applies_to: ["MSKI_minor", "MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["wiggins_2016", "olivotto_2025"]
    stakeholder_explainer: "Prior injury to same region nearly doubles recovery time."
  
  supervised_rehabilitation:
    effect_type: "hazard_ratio"
    hr: 0.75
    ci_95: [0.65, 0.87]
    category: "treatment"
    direction: "accelerates_recovery"
    applies_to: ["MSKI_minor", "MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["olivotto_2025"]
    stakeholder_explainer: "Supervised rehab reduces recovery time by ~25%."
```

---

## STEP 3: Add XGBoost Model with SHAP

### Create new file: xgb_model.py

```python
"""
SEKHMET XGBoost Survival Model
==============================

XGBoost model with SHAP explainability for MSKI recovery prediction.

⚠️ RESEARCH DEMONSTRATION
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
import xgboost as xgb
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
        "⚠️ RESEARCH DEMONSTRATION: This prediction uses a model trained on "
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
```

---

## STEP 4: Update app.py with Three Models

### Model Selection in Sidebar

```python
# Model Selection
st.sidebar.header("🔬 Model Selection")

model_choice = st.sidebar.radio(
    "Select Model",
    [
        "Cox PH (Evidence-based)",
        "Bayesian (Clinician-adjustable)",
        "XGBoost (ML/SHAP)"
    ],
    index=0,
    help="""
    • **Cox PH**: Published hazard ratios, clinical gold standard
    • **Bayesian**: Adjustable parameters for local calibration
    • **XGBoost**: ML model with SHAP explainability (research only)
    """
)

# Show warning for XGBoost
if model_choice == "XGBoost (ML/SHAP)":
    st.sidebar.warning("⚠️ Trained on synthetic data - research use only")
```

### Updated Risk Factors Sidebar

```python
st.sidebar.header("📋 Case Input")

# Demographics
st.sidebar.subheader("👤 Demographics")
age = st.sidebar.slider("Age", min_value=18, max_value=55, value=30)
sex = st.sidebar.radio("Sex", ["Male", "Female"], horizontal=True)

# Injury Details - Body Region FIRST
st.sidebar.subheader("🦴 Injury Details")

body_region_options = {
    "Knee": "knee",
    "Lower Back": "lower_back",
    "Shoulder": "shoulder",
    "Ankle/Foot": "ankle_foot",
    "Hip/Groin": "hip_groin",
    "Cervical Spine": "cervical_spine",
    "Wrist/Hand": "wrist_hand",
}
body_region_name = st.sidebar.selectbox("Body Region", list(body_region_options.keys()))
body_region = body_region_options[body_region_name]

severity_options = {
    "Minor": "mski_minor",
    "Moderate": "mski_moderate",
    "Major": "mski_major",
    "Severe": "mski_severe",
}
severity_name = st.sidebar.selectbox("Severity", list(severity_options.keys()))
injury_type = severity_options[severity_name]

prior_same_region = st.sidebar.checkbox("Prior injury to same region", help="HR 1.80")

# Risk Factors - LIFESTYLE
st.sidebar.subheader("🚬 Lifestyle Factors")
is_smoker = st.sidebar.checkbox("Current smoker", help="HR 1.43")
high_alcohol = st.sidebar.checkbox("High alcohol intake", help="HR 1.25")
sleep_quality = st.sidebar.select_slider(
    "Sleep quality",
    options=["Poor", "Fair", "Good"],
    value="Good",
    help="Poor sleep HR 1.30"
)
poor_sleep = (sleep_quality == "Poor")

# Risk Factors - OCCUPATION
st.sidebar.subheader("💼 Occupational Factors")
oh_risk = st.sidebar.select_slider(
    "OH/Occupational Risk",
    options=["Low", "Moderate", "High"],
    value="Moderate",
    help="Low: 1.0x | Moderate: 1.15x | High: 1.30x"
)

# Risk Factors - BMI
st.sidebar.subheader("⚖️ BMI")
bmi = st.sidebar.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.5)
if bmi < 18.5:
    bmi_cat = "Underweight"
elif bmi < 25:
    bmi_cat = "Normal"
elif bmi < 30:
    bmi_cat = "Overweight"
elif bmi < 35:
    bmi_cat = "Obese (Class 1)"
else:
    bmi_cat = "Obese (Class 2+)"
st.sidebar.caption(f"Category: {bmi_cat}")

# Treatment
st.sidebar.subheader("💊 Treatment")
receiving_treatment = st.sidebar.checkbox("Supervised rehabilitation", value=True, help="HR 0.75")
```

### Model Loading (Cached)

```python
@st.cache_resource
def load_models():
    """Load all models (cached for performance)."""
    from cox_model import CoxRecoveryModel
    from bayesian_model import BayesianRecoveryModel
    from xgb_model import XGBSurvivalModel
    
    cox_model = CoxRecoveryModel()
    bayesian_model = BayesianRecoveryModel()
    
    xgb_model = XGBSurvivalModel()
    xgb_model.train(n_synthetic=5000)
    
    return cox_model, bayesian_model, xgb_model


# In main():
cox_model, bayesian_model, xgb_model = load_models()
```

### Prediction Logic

```python
# Build case dict for all models
case_dict = {
    'age': age,
    'body_region': body_region,
    'injury_type': injury_type,
    'prior_same_region': prior_same_region,
    'is_smoker': is_smoker,
    'high_alcohol': high_alcohol,
    'poor_sleep': poor_sleep,
    'oh_risk': oh_risk,
    'bmi': bmi,
    'receiving_treatment': receiving_treatment,
}

# Generate prediction based on model choice
if model_choice == "Cox PH (Evidence-based)":
    # Convert to CaseInput for Cox model
    from config import InjuryType, BodyRegion
    case = CaseInput(
        age=age,
        injury_type=InjuryType(injury_type),
        body_region=BodyRegion(body_region),
        prior_same_region=prior_same_region,
        is_smoker=is_smoker,
        receiving_treatment=receiving_treatment,
        # Pass other factors via custom_hrs or handle in model
    )
    pred = cox_model.predict(case)
    recovery_months = pred.median_recovery_months
    model_label = "Cox PH (Evidence-based)"
    show_shap = False

elif model_choice == "Bayesian (Clinician-adjustable)":
    pred = bayesian_model.predict(case_dict)
    recovery_months = pred.predicted_months
    model_label = "Bayesian (Clinician-adjustable)"
    show_shap = False

else:  # XGBoost
    pred = xgb_model.predict(case_dict)
    recovery_months = pred.predicted_time_months
    model_label = "XGBoost (ML/SHAP)"
    show_shap = True
```

### XGBoost SHAP Display

```python
# In results section, add SHAP visualization for XGBoost
if show_shap and hasattr(pred, 'shap_values') and pred.shap_values:
    st.subheader("🔍 SHAP Feature Contributions")
    
    # Warning banner
    st.warning(pred.disclaimer)
    
    # SHAP waterfall-style display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Factors Slowing Recovery** ↑")
        if pred.top_positive_factors:
            for factor, value in pred.top_positive_factors:
                factor_display = factor.replace('_', ' ').title()
                st.markdown(f"• {factor_display}: +{value:.3f}")
        else:
            st.caption("None significant")
    
    with col2:
        st.markdown("**Factors Speeding Recovery** ↓")
        if pred.top_negative_factors:
            for factor, value in pred.top_negative_factors:
                factor_display = factor.replace('_', ' ').title()
                st.markdown(f"• {factor_display}: {value:.3f}")
        else:
            st.caption("None significant")
    
    # Full SHAP bar chart
    if pred.shap_values:
        import plotly.graph_objects as go
        
        sorted_shap = sorted(pred.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        factors = [s[0].replace('_', ' ').title() for s in sorted_shap]
        values = [s[1] for s in sorted_shap]
        colors = ['#c62828' if v > 0 else '#2e7d32' for v in values]
        
        fig = go.Figure(data=[
            go.Bar(y=factors, x=values, orientation='h', marker_color=colors)
        ])
        fig.update_layout(
            title="SHAP Values (Top 10 Features)",
            xaxis_title="Impact on Prediction (log-months)",
            height=350,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
```

---

## STEP 5: Update requirements.txt

```txt
# Core
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0

# ML Models
xgboost>=2.0.0
shap>=0.42.0

# Web Interface
streamlit>=1.28.0
plotly>=5.18.0
```

---

## STEP 6: Updated Tab Structure

```python
def main():
    st.title("🏥 SEKHMET Recovery Predictor")
    st.markdown("*Evidence-based MSKI recovery prediction for Defence workforce planning*")
    
    # Load models
    cox_model, bayesian_model, xgb_model = load_models()
    
    # Tabs - MH REMOVED
    tab_predict, tab_cohort, tab_settings = st.tabs([
        "🦴 Individual Prediction",
        "📊 Cohort Planning",
        "⚙️ Model Settings"
    ])
    
    with tab_predict:
        render_prediction(cox_model, bayesian_model, xgb_model)
    
    with tab_cohort:
        render_cohort_planning()
    
    with tab_settings:
        render_model_settings()
```

---

## TESTING CHECKLIST

```bash
# 1. Verify MH removed
python -c "
from config import InjuryType, BodyRegion
try:
    _ = InjuryType.MH_MILD
    print('FAIL: MH still exists')
except AttributeError:
    print('PASS: MH removed from InjuryType')

try:
    _ = BodyRegion.MENTAL
    print('FAIL: MENTAL still exists')
except AttributeError:
    print('PASS: MENTAL removed from BodyRegion')
"

# 2. Test XGBoost model
python -c "
from xgb_model import XGBSurvivalModel
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
"

# 3. Run Streamlit
streamlit run app.py

# Manual checks:
# - [ ] Only one prediction tab (no MH tab)
# - [ ] Three model options: Cox PH, Bayesian, XGBoost
# - [ ] XGBoost shows SHAP values
# - [ ] XGBoost shows synthetic data warning
# - [ ] OH/Occupational Risk slider present
# - [ ] No Trade dropdown
# - [ ] No MH comorbidity checkbox
# - [ ] Body Region → Severity order
```

---

## COMMIT MESSAGE

```bash
git add -A
git commit -m "feat: Add XGBoost, remove MH, restructure risk factors

REMOVED:
- All MH injury types and MH tab
- MH-specific risk factors
- Trade/Physical Demand dropdown

ADDED:
- XGBoost model with SHAP explainability
- Three model options: Cox PH, Bayesian, XGBoost
- OH/Occupational risk factor (Low/Moderate/High)
- Synthetic data warning for XGBoost

UPDATED:
- Risk factors: Lifestyle (smoking, alcohol, sleep), Occupation (OH risk), BMI
- evidence_base.yaml - MSKI only
- requirements.txt - added xgboost, shap

Note: XGBoost trained on synthetic data - research demonstration only"

git push
```

---

## FILE CHECKLIST

- [ ] `config.py` - Remove MH enums, remove Trade if not needed
- [ ] `evidence_base.yaml` - Remove MH sections, update risk factors
- [ ] `xgb_model.py` - NEW: XGBoost with SHAP
- [ ] `app.py` - Remove MH tab, add XGBoost option, update risk factors
- [ ] `requirements.txt` - Add xgboost, shap
- [ ] `cox_model.py` - Remove MH handling if present
- [ ] `bayesian_model.py` - Remove MH handling if present

---

## SUMMARY

| Before | After |
|--------|-------|
| MSKI + MH | MSKI only |
| Trade dropdown | OH/Occupational Risk slider |
| 2 models (Cox, Bayesian) | 3 models (Cox, Bayesian, XGBoost) |
| No SHAP | SHAP explainability for XGBoost |
| Physical Demand | Removed |
| MH risk factors | Removed |
