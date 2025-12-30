# SEKHMET Cox Model Integration - Instructions for Claude Code

## Context

You are updating the Pj-SEKHMET repository to integrate evidence-based Cox proportional hazards modelling. This replaces/augments the existing heuristic predictor with clinically-validated parameters.

Repository: https://github.com/RossTylr/Pj-SEKHMET

## Summary of Changes

1. **Update Trade enum** → Combat / Combat Support / Combat Service Support categories
2. **Add evidence_base.yaml** → Clinical parameters with citations (22 sources)
3. **Add cox_model.py** → Cox PH survival model with Weibull baseline
4. **Update config.py** → New enums, EvidenceBase loader class
5. **Update any existing code** that references old Trade values

---

## Step 1: Understand Current State

First, explore the existing codebase:

```bash
# Check current structure
ls -la src/predictor/
cat src/predictor/config.py
cat src/predictor/recovery_model.py
```

Note which Trade enum values are currently used and where they're referenced.

---

## Step 2: Update config.py

Replace or merge with the following key changes:

### 2.1 New Trade Categories

```python
from enum import Enum

class TradeCategory(Enum):
    """
    Trade categories based on physical demands and operational role.
    
    These map to different RTD fitness thresholds:
    - COMBAT: Highest physical demands, requires full JMES MFD for deployment
    - COMBAT_SUPPORT: Moderate demands, some role restrictions acceptable
    - COMBAT_SERVICE_SUPPORT: Lower physical demands, wider range of MES acceptable
    
    Evidence basis: High-demand trades have HR ~1.30 for delayed RTD
    Reference: JSP 822, JSP 950 Medical Policy
    """
    COMBAT = "combat"
    COMBAT_SUPPORT = "combat_support"
    COMBAT_SERVICE_SUPPORT = "combat_service_support"


class Trade(Enum):
    """Specific military trades mapped to categories."""
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
    REME = "reme"
    MEDIC = "medic"
    MILITARY_POLICE = "military_police"
    
    # Combat Service Support (TradeCategory.COMBAT_SERVICE_SUPPORT)
    LOGISTICS = "logistics"
    AGC = "agc"
    DENTAL = "dental"
    VETERINARY = "veterinary"
    CHAPLAIN = "chaplain"
    GENERIC = "generic"


# Mapping from specific trades to categories
TRADE_CATEGORY_MAP = {
    Trade.INFANTRY: TradeCategory.COMBAT,
    Trade.ROYAL_MARINES: TradeCategory.COMBAT,
    Trade.PARACHUTE_REGIMENT: TradeCategory.COMBAT,
    Trade.ARMOUR: TradeCategory.COMBAT,
    Trade.ARTILLERY: TradeCategory.COMBAT,
    Trade.COMBAT_ENGINEER: TradeCategory.COMBAT,
    Trade.SIGNALS: TradeCategory.COMBAT_SUPPORT,
    Trade.INTELLIGENCE: TradeCategory.COMBAT_SUPPORT,
    Trade.REME: TradeCategory.COMBAT_SUPPORT,
    Trade.MEDIC: TradeCategory.COMBAT_SUPPORT,
    Trade.MILITARY_POLICE: TradeCategory.COMBAT_SUPPORT,
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


# RTD modifiers by trade category
TRADE_CATEGORY_RTD_MODIFIER = {
    TradeCategory.COMBAT: 1.30,              # 30% longer RTD
    TradeCategory.COMBAT_SUPPORT: 1.15,      # 15% longer RTD
    TradeCategory.COMBAT_SERVICE_SUPPORT: 1.00,  # Baseline
}
```

### 2.2 Add EvidenceBase Loader Class

Add this class to config.py to load parameters from evidence_base.yaml:

```python
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class InjuryParameters:
    """Evidence-based parameters for a specific injury type."""
    median_recovery_months: float
    recovery_range_months: tuple
    time_to_fitness_months: float
    time_to_rtd_months: float
    prob_full_recovery: float
    prob_partial_recovery: float
    prob_not_recovered: float
    reinjury_rate: Optional[float] = None
    recurrence_rate_12mo: Optional[float] = None
    evidence_grade: str = "Unknown"
    sources: List[str] = None
    stakeholder_explainer: str = ""


@dataclass
class RiskFactor:
    """Risk factor modifier from evidence base."""
    name: str
    effect_type: str  # hazard_ratio, risk_ratio
    value: float
    ci_95: tuple
    applies_to: List[str]
    direction: str
    sources: List[str]
    stakeholder_explainer: str = ""


class EvidenceBase:
    """Loader for clinical evidence parameters."""
    
    def __init__(self, yaml_path: Optional[Path] = None):
        if yaml_path is None:
            yaml_path = Path(__file__).parent / "evidence_base.yaml"
        
        with open(yaml_path, 'r') as f:
            self._data = yaml.safe_load(f)
        
        self.version = self._data.get('metadata', {}).get('version', 'unknown')
        self._injuries = self._data.get('injuries', {})
        self._risk_factors = self._data.get('risk_factors', {})
        self._sources = self._data.get('sources', {})
    
    def get_injury_params(self, injury_type, body_region) -> Optional[InjuryParameters]:
        """Get evidence-based parameters for an injury."""
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
    
    def _injury_type_to_yaml_key(self, injury_type) -> str:
        """Map InjuryType enum to YAML section key."""
        mapping = {
            'mski_minor': 'MSKI_minor',
            'mski_moderate': 'MSKI_moderate',
            'mski_major': 'MSKI_major',
            'mski_severe': 'MSKI_severe',
            'mh_mild': 'MH_mild',
            'mh_moderate': 'MH_moderate',
            'mh_severe': 'MH_severe',
            'tbi_mild': 'TBI_mild',
            'tbi_moderate': 'TBI_moderate',
            'tbi_severe': 'TBI_severe',
        }
        key = injury_type.value if hasattr(injury_type, 'value') else str(injury_type)
        return mapping.get(key, 'MSKI_moderate')
    
    def _body_region_to_yaml_key(self, body_region) -> str:
        """Map BodyRegion enum to YAML section key."""
        mapping = {
            'knee': 'knee_acl',
            'lower_back': 'lower_back',
            'shoulder': 'shoulder',
            'mental': 'ptsd',
            'brain': 'mtbi',
        }
        key = body_region.value if hasattr(body_region, 'value') else str(body_region)
        return mapping.get(key, key)
```

### 2.3 Backward Compatibility Aliases

If the old code used different Trade values, add aliases:

```python
# Backward compatibility for old Trade values
# Add at the end of Trade enum section if needed
Trade.PARA = Trade.PARACHUTE_REGIMENT  # Alias
Trade.RM = Trade.ROYAL_MARINES         # Alias
```

---

## Step 3: Create evidence_base.yaml

Create `src/predictor/evidence_base.yaml` with the full clinical evidence.

This file is ~750 lines. Key sections:

```yaml
metadata:
  version: "1.1.0"
  last_updated: "2025-01-01"
  total_sources: 22

injuries:
  MSKI_moderate:
    knee_acl:
      median_recovery_months: 9.0
      recovery_range_months: [6.0, 12.0]
      time_to_fitness_months: 6.0
      time_to_rtd_months: 9.0
      prob_full_recovery: 0.50
      prob_partial_recovery: 0.30
      prob_not_recovered: 0.20
      reinjury_rate: 0.15
      sources: [antosh_2018, marquina_2024, anderson_2023, wiggins_2016, piussi_2024]
      evidence_grade: "Moderate"
      stakeholder_explainer: |
        ACL injuries have one of the longest recovery trajectories.
        About half return to unrestricted duty.
    
    lower_back:
      median_recovery_months: 6.0
      # ... etc

  TBI_mild:
    mtbi:
      median_recovery_months: 1.0
      prob_full_recovery: 0.85
      ptsd_comorbidity_rate: 0.482  # 48.2% military vs 15% civilian
      # ...

  MH_moderate:
    ptsd:
      median_recovery_months: 8.0
      prob_full_recovery: 0.35  # UK-specific from KCMHR
      # ...

risk_factors:
  age:
    effect_type: "hazard_ratio"
    per_unit: "per decade over 25"
    hr: 1.15
    ci_95: [1.08, 1.23]
    sources: ["anderson_2023", "wiggins_2016"]
  
  prior_same_region_injury:
    hr: 1.80
    ci_95: [1.40, 2.30]
  
  smoking:
    hr: 1.43
    ci_95: [1.17, 1.74]
  
  supervised_rehabilitation:
    hr: 0.75  # Protective
    ci_95: [0.65, 0.87]
  
  mental_health_comorbidity:
    effect_type: "risk_ratio"
    rr: 6.0
    ci_95: [3.0, 10.0]

sources:
  antosh_2018:
    authors: "Antosh IJ et al."
    title: "Return to Military Duty After ACL Reconstruction"
    journal: "Military Medicine"
    year: 2018
    doi: "10.1093/milmed/usx007"
  # ... 21 more sources
```

**IMPORTANT**: Get the full evidence_base.yaml from the provided files - it contains all 22 sources with complete citations.

---

## Step 4: Create cox_model.py

Create `src/predictor/cox_model.py` with the Cox PH model.

Key components:

### 4.1 Model Theory (Document in Docstring)

```python
"""
Cox Proportional Hazards Model for Recovery Prediction
======================================================

The Cox PH model estimates hazard as:
    h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₖXₖ)

Where:
- h₀(t) is baseline hazard (injury-specific, from evidence_base.yaml)
- X are covariates (age, prior injury, smoking, etc.)
- β are log hazard ratios (from systematic reviews)

Baseline Hazard Estimation:
Without individual patient data, we use Weibull approximation:
    S(t) = exp(-(t/λ)^k)

Where:
- λ (scale) derived from median recovery time
- k (shape) controls hazard trajectory:
  - k > 1: Increasing hazard (recovery more likely over time) - used for MSKI
  - k ≈ 1: Constant hazard - used for MH
  - k = 2: Steep early hazard - used for mTBI
"""
```

### 4.2 Weibull Shape Parameters (With Reasoning)

```python
WEIBULL_SHAPE_PARAMS = {
    # MSKI: k=1.5, increasing hazard
    # Rationale: Tissue repair is progressive; longer time = more healing
    'MSKI': 1.5,
    
    # Mental Health: k≈1, roughly constant hazard
    # Rationale: Recovery less predictable, therapy response variable
    'MH': 1.1,
    
    # mTBI: k=2, steep early hazard
    # Rationale: 90% recover within 3 months, rapid early resolution
    'TBI_mild': 2.0,
    
    # Moderate/severe TBI: k≈1.2, less predictable
    'TBI_moderate': 1.2,
}
```

### 4.3 Risk Factor Application (With Clinical Reasoning)

```python
def _calculate_hazard_ratios(self, case: CaseInput) -> Dict[str, float]:
    """
    Calculate cumulative HR from covariates.
    
    HR > 1 = slower recovery
    HR < 1 = faster recovery
    """
    hazard_ratios = {}
    
    # Age: HR 1.15 per decade over 25
    # Rationale: Older tissue heals slower, reduced physiological reserve
    # Source: anderson_2023, wiggins_2016
    if case.age > 25:
        decades = (case.age - 25) / 10.0
        hazard_ratios['age'] = 1.15 ** decades
    
    # Prior same-region injury: HR 1.80
    # Rationale: Previous injury indicates vulnerability, incomplete healing
    # Source: wiggins_2016, olivotto_2025
    if case.prior_same_region:
        hazard_ratios['prior_same_region'] = 1.80
    
    # Smoking: HR 1.43
    # Rationale: Impaired tissue oxygenation, delayed wound healing
    # Source: anderson_2023
    if case.is_smoker:
        hazard_ratios['smoking'] = 1.43
    
    # Supervised rehab: HR 0.75 (protective)
    # Rationale: Structured progression, professional oversight
    # Source: olivotto_2025
    if case.receiving_treatment:
        hazard_ratios['supervised_rehab'] = 0.75
    
    # MH comorbidity: HR 2.0 (approximated from RR 6.0)
    # Rationale: Strongest predictor of poor MSKI outcomes
    # Source: olivotto_2025, kcmhr_2024
    if case.has_mh_comorbidity:
        hazard_ratios['mh_comorbidity'] = 2.0
    
    # Multiple TBI (≥3): HR 1.80
    # Rationale: Cumulative neurological burden
    # Source: kennedy_2018, tbicohe_2023
    if case.multiple_tbi_history and 'tbi' in case.injury_type.value:
        hazard_ratios['multiple_tbi'] = 1.80
    
    return hazard_ratios
```

### 4.4 Trade Category Modifier (RTD Only)

```python
# Trade affects RTD but NOT Return to Fitness
# Combat roles require higher fitness threshold for occupational clearance

trade_category = get_trade_category(case.trade)
trade_hr = TRADE_CATEGORY_RTD_MODIFIER[trade_category]

# Apply to RTD calculation only
time_to_rtd = baseline_rtd * (total_hr ** (1/k)) * trade_hr
# Return to Fitness is NOT modified by trade
time_to_fitness = baseline_fitness * (total_hr ** (1/k))
```

---

## Step 5: Update Existing recovery_model.py

If there's an existing heuristic model, either:

**Option A**: Keep it as fallback, add model selector:
```python
class RecoveryPredictor:
    def __init__(self, model_type: str = "cox"):
        if model_type == "cox":
            self.model = CoxRecoveryModel()
        else:
            self.model = HeuristicModel()  # Existing
```

**Option B**: Replace entirely with Cox model

---

## Step 6: Update Streamlit App (if exists)

Add model selection and citation display:

```python
import streamlit as st
from cox_model import CoxRecoveryModel, CaseInput

# Model selection
model_type = st.selectbox("Model", ["Cox PH (Evidence-based)", "Heuristic"])

# After prediction, show citations
if prediction.primary_sources:
    with st.expander("📚 Evidence Sources"):
        for source_id in prediction.primary_sources:
            citation = model.evidence.get_source_citation(source_id)
            if citation:
                st.markdown(f"**{citation['authors']}** ({citation['year']}). "
                           f"*{citation['title']}*. {citation['journal']}. "
                           f"doi:{citation.get('doi', 'N/A')}")
```

---

## Step 7: Update requirements.txt

Add if not present:
```
pyyaml>=6.0
numpy>=1.24.0
```

---

## Step 8: Test

```bash
cd src/predictor
python -c "
from config import Trade, TradeCategory, get_trade_category, EvidenceBase
from cox_model import CoxRecoveryModel, CaseInput, InjuryType, BodyRegion

# Test trade categories
print(f'Infantry -> {get_trade_category(Trade.INFANTRY)}')
print(f'Logistics -> {get_trade_category(Trade.LOGISTICS)}')

# Test evidence base
eb = EvidenceBase()
print(f'Evidence base version: {eb.version}')

# Test prediction
model = CoxRecoveryModel()
case = CaseInput(
    age=35,
    trade=Trade.INFANTRY,
    injury_type=InjuryType.MSKI_MODERATE,
    body_region=BodyRegion.KNEE,
    prior_same_region=True,
    is_smoker=True
)
pred = model.predict(case)
print(f'Recovery: {pred.median_recovery_months} months')
print(f'Sources: {pred.primary_sources}')
"
```

---

## Step 9: Commit

```bash
git add src/predictor/config.py
git add src/predictor/evidence_base.yaml
git add src/predictor/cox_model.py
git commit -m "feat: Cox PH model with evidence base v1.1.0

- Updated Trade enum to Combat/Combat Support/CSS categories
- Added evidence_base.yaml with 22 clinical sources
- Added Cox model with Weibull baseline hazards
- Includes mTBI, UK-specific PTSD parameters
- All HRs documented with clinical reasoning and citations"

git push
```

---

## File Checklist

- [ ] `src/predictor/config.py` - Updated with TradeCategory, EvidenceBase
- [ ] `src/predictor/evidence_base.yaml` - New file, 22 sources
- [ ] `src/predictor/cox_model.py` - New file, Cox PH model
- [ ] `src/predictor/recovery_model.py` - Updated or kept as fallback
- [ ] `requirements.txt` - Add pyyaml if needed
- [ ] `app.py` - Update Streamlit UI (if exists)

---

## Questions to Resolve

1. **Old Trade values**: Check if INFANTRY, PARA etc. are used elsewhere and need aliases
2. **Existing tests**: Update any unit tests for new enum values
3. **Streamlit app**: Does it need model selector UI?
4. **Data files**: Any existing YAML/JSON configs that reference old Trade values?

---

## Source Files

The complete source files (config.py, cox_model.py, evidence_base.yaml) are available in the Claude.ai conversation. Copy them directly rather than retyping.
