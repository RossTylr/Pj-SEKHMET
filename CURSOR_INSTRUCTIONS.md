# JMES Synthetic Workforce Simulation - Cursor/Claude Build Instructions

## 🎯 MISSION BRIEF

You are building a **synthetic Defence workforce data generator** for JMES (Joint Medical Employment Standard) prediction modelling. This simulates 100,000 tri-service personnel over 5-10 years with realistic turnover, injuries, and medical status transitions.

---

## 📋 PRE-BUILD CHECKLIST

Before starting, verify you understand these requirements:

```
□ Target: 100,000 synthetic personnel
□ Duration: 60-120 months longitudinal
□ Turnover: 10% annual (~0.88% monthly)
□ Two tables: Master (personnel) + Monthly (person-month)
□ Key outcome: JMES deterioration (MFD → MLD → MND)
□ Stack: Python + Streamlit + Pandas + Plotly
```

---

## 🏗️ BUILD SEQUENCE

Execute these steps in order. After each step, run the self-check before proceeding.

### STEP 1: Project Structure

Create this folder structure:

```
jmes_mvp/
├── configs/
│   └── simulation_config.yaml
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── generator.py
│   ├── validation.py
│   └── app.py
├── tests/
│   └── test_generator.py
├── data/
├── requirements.txt
└── README.md
```

**SELF-CHECK 1:**
```python
import os
required = ['configs', 'src', 'tests', 'data']
assert all(os.path.isdir(d) for d in required), "Missing directories"
print("✅ Step 1: Structure verified")
```

---

### STEP 2: Configuration File

Create `configs/simulation_config.yaml` with this exact structure:

```yaml
# ===========================================
# JMES SIMULATION CONFIGURATION
# ===========================================

population:
  initial_size: 100000
  simulation_months: 60

turnover:
  annual_rate: 0.10
  monthly_rate: 0.00877
  los_weights:
    "0-4": 1.3
    "5-12": 0.9
    "13+": 0.8

service_mix:
  Army: 0.60
  RN: 0.20
  RAF: 0.20

gender:
  overall_female_rate: 0.175
  trade_adjustments:
    CMT: 0.25
    Paramedic: 0.22
    AHP: 0.35
    ODP: 0.30
    Other: 0.12

age:
  min: 18
  max: 55
  mean: 32.5
  std: 8.0
  inflow:
    min: 18
    max: 35
    mean: 22
    std: 4.0

length_of_service:
  bands:
    "0-4": 0.40
    "5-12": 0.35
    "13-25": 0.25
  max_years: 30

rank_bands:
  OR2-OR4: 0.45
  OR5-OR7: 0.30
  OR8-OR9: 0.10
  OF1-OF3: 0.10
  OF4-OF5: 0.05

trades:
  CMT: 0.15
  Paramedic: 0.10
  AHP: 0.12
  ODP: 0.08
  Other: 0.55

jmes:
  baseline_distribution:
    MFD: 0.85
    MLD: 0.12
    MND: 0.03
  transitions:
    MFD_to_MLD: 0.008
    MFD_to_MND: 0.001
    MLD_to_MND: 0.015
    MLD_to_MFD: 0.025
    MND_to_MLD: 0.010
    MND_to_MFD: 0.002

injuries:
  baseline_monthly_mski: 0.02
  trade_multipliers:
    CMT: 1.8
    Paramedic: 1.6
    AHP: 1.0
    ODP: 1.2
    Other: 1.3
  deployment_multiplier: 1.5
  high_risk_training_multiplier: 3.0
  types:
    MSKI-minor: 0.50
    MSKI-major: 0.25
    MH-episode: 0.15
    Other: 0.10

deployment:
  baseline_monthly_rate: 0.08
  service_rates:
    Army: 1.2
    RN: 1.1
    RAF: 0.8
  intensity:
    Low_tempo: 0.60
    High_tempo: 0.40

training:
  monthly_rate: 0.15
  intensity:
    Low_risk: 0.70
    High_risk: 0.30

pregnancy:
  annual_conception_rate: 0.065
  monthly_conception_rate: 0.0055
  duration:
    pregnancy_months: 9
    postpartum_months: 4

engagement_types:
  PC: 0.35
  IC: 0.25
  FE: 0.30
  UCM-H: 0.10

unit_environment:
  Standard: 0.70
  High-Risk: 0.20
  Hot/Cold: 0.10

validation:
  turnover_tolerance: 0.02
  population_stability: 0.05
  jmes_mfd_floor: 0.80
  injury_rate_ceiling: 0.10
```

**SELF-CHECK 2:**
```python
import yaml
with open('configs/simulation_config.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['population']['initial_size'] == 100000
assert abs(sum(cfg['service_mix'].values()) - 1.0) < 0.01
assert abs(sum(cfg['jmes']['baseline_distribution'].values()) - 1.0) < 0.01
print("✅ Step 2: Config verified")
```

---

### STEP 3: Data Models

Create `src/models.py` with these specifications:

```python
"""
JMES Data Models - Type-safe schemas with validation
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import uuid
from datetime import datetime

# ============================================================
# ENUMERATIONS - Use these exact values
# ============================================================

class ServiceBranch(str, Enum):
    ARMY = "Army"
    RN = "RN"
    RAF = "RAF"

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
    T1 = "T1"
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

class ServiceType(str, Enum):
    REGULAR = "Regular"
    RESERVE = "Reserve"

# ============================================================
# DATACLASS: Personnel Master (Table A)
# ============================================================

@dataclass
class PersonnelMaster:
    """One row per synthetic individual"""
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
        """Validate constraints"""
        assert 18 <= self.age_start <= 55, f"Age {self.age_start} out of range"
        assert 0 <= self.length_of_service_start <= 30, f"LoS out of range"
        
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

# ============================================================
# DATACLASS: Person-Month (Table B)
# ============================================================

@dataclass
class PersonMonth:
    """One row per individual per month"""
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
        assert self.month >= 1, "Month must be >= 1"
        assert 0 <= self.sick_days <= 30, "Sick days out of range"
        
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
```

**SELF-CHECK 3:**
```python
from src.models import PersonnelMaster, PersonMonth, JMESStatus, Gender

# Test valid creation
p = PersonnelMaster(age_start=30, gender=Gender.FEMALE)
assert p.to_dict()['gender'] == 'Female'

pm = PersonMonth(person_id="test", month=1, age=30.0)
assert pm.to_dict()['jmes_current'] == 'MFD'

# Test validation
try:
    PersonnelMaster(age_start=15)  # Should fail
    assert False, "Should have raised"
except AssertionError:
    pass

print("✅ Step 3: Models verified")
```

---

### STEP 4: Generator Engine

Create `src/generator.py` implementing:

**CRITICAL IMPLEMENTATION NOTES:**

1. **DO NOT use `rng.choice()` directly on Enum lists** - NumPy returns numpy.str_ not Enum
   
   ```python
   # ❌ WRONG - causes AttributeError
   result = self.rng.choice([JMESStatus.MFD, JMESStatus.MLD], p=[0.8, 0.2])
   
   # ✅ CORRECT - use index selection
   options = [JMESStatus.MFD, JMESStatus.MLD]
   idx = self.rng.choice(len(options), p=[0.8, 0.2])
   result = options[idx]
   ```

2. **Handle LoS edge cases** - prevent `low >= high` in integers()
   
   ```python
   # ❌ WRONG - fails when age=30, max_los=13
   los = self.rng.integers(13, min(26, age - 17))
   
   # ✅ CORRECT - validate range first
   max_los = min(26, age - 17)
   if max_los > 13:
       los = self.rng.integers(13, max_los)
   else:
       los = self.rng.integers(5, 13)  # Fallback
   ```

3. **Monthly simulation loop structure:**
   ```python
   def simulate_month(self, month: int) -> List[Dict]:
       for person_id, state in self.active_personnel.items():
           # 1. Update age
           # 2. JMES transition
           # 3. Injury event
           # 4. Deployment
           # 5. Training
           # 6. Pregnancy (female only)
           # 7. Sick days
           # 8. Outflow check
           # 9. Create PersonMonth record
       
       # After loop: remove outflows, add inflows
       return month_records
   ```

**SELF-CHECK 4:**
```python
from src.generator import SyntheticDataGenerator

gen = SyntheticDataGenerator('configs/simulation_config.yaml', seed=42)
gen.config['population']['initial_size'] = 100
gen.config['population']['simulation_months'] = 6

master_df, month_df = gen.run_simulation()

assert len(master_df) >= 100, "Master table too small"
assert len(month_df) >= 500, "Monthly table too small"
assert 'person_id' in master_df.columns
assert 'jmes_current' in month_df.columns

print(f"✅ Step 4: Generator verified")
print(f"   Master: {len(master_df)} rows")
print(f"   Monthly: {len(month_df)} rows")
```

---

### STEP 5: Streamlit Dashboard

Create `src/app.py` with these tabs:

| Tab | Components |
|-----|------------|
| Overview | 4 KPI cards, pie chart (service), histogram (age), bar charts (trade, JMES) |
| JMES Monitoring | Stacked area (JMES over time), line (deterioration events), grouped bar (by service) |
| Turnover | Dual line (inflow/outflow), population trend, pie (outflow reasons) |
| Injuries | Monthly injury line, pie (injury types), bar (rate by trade) |
| Export | Download buttons for CSV, config preview |

**Key Streamlit patterns:**

```python
# Session state for persistence
if 'master_df' not in st.session_state:
    st.session_state.master_df = None

# Progress tracking during generation
progress = st.progress(0)
for month in range(1, n_months + 1):
    # ... simulate
    progress.progress(month / n_months)

# Plotly integration
fig = px.pie(df, names='column', title='Title')
st.plotly_chart(fig, use_container_width=True)
```

**SELF-CHECK 5:**
```python
import ast
with open('src/app.py') as f:
    tree = ast.parse(f.read())

functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

required = ['render_sidebar', 'render_overview_tab', 'render_jmes_tab', 'main']
missing = [f for f in required if f not in functions]
assert not missing, f"Missing functions: {missing}"

print("✅ Step 5: App structure verified")
```

---

### STEP 6: Requirements

Create `requirements.txt`:

```
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
streamlit>=1.28.0
plotly>=5.18.0
pytest>=7.4.0
```

**SELF-CHECK 6:**
```bash
pip install -r requirements.txt
python -c "import streamlit; import plotly; import yaml; print('✅ Step 6: Dependencies verified')"
```

---

## 🔬 FINAL INTEGRATION TEST

Run this complete verification:

```python
"""
FINAL INTEGRATION TEST
Run from project root: python -m pytest tests/ -v
Or run this script directly
"""

import sys
sys.path.insert(0, 'src')

from generator import SyntheticDataGenerator
from validation import SyntheticDataValidator

print("=" * 60)
print("JMES MVP - FINAL INTEGRATION TEST")
print("=" * 60)

# 1. Generate small dataset
print("\n[1/4] Generating test dataset...")
gen = SyntheticDataGenerator('configs/simulation_config.yaml', seed=42)
gen.config['population']['initial_size'] = 1000
gen.config['population']['simulation_months'] = 12

master_df, month_df = gen.run_simulation()

# 2. Validate distributions
print("\n[2/4] Validating distributions...")
checks = {
    'Service mix': abs((master_df['service_branch'] == 'Army').mean() - 0.60) < 0.05,
    'Gender ratio': abs((master_df['gender'] == 'Female').mean() - 0.175) < 0.05,
    'Age range': master_df['age_start'].between(18, 55).all(),
    'JMES baseline': (master_df['baseline_jmes'] == 'MFD').mean() >= 0.80,
}

for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {check}")

# 3. Validate temporal consistency
print("\n[3/4] Validating temporal consistency...")
months = sorted(month_df['month'].unique())
assert months == list(range(1, 13)), "Month sequence broken"
print("  ✅ Month sequence 1-12")

# 4. Validate business rules
print("\n[4/4] Validating business rules...")
assert (master_df['age_start'] >= 18).all(), "Age < 18 found"
assert (master_df['length_of_service_start'] <= master_df['age_start'] - 18).all(), "LoS > age-18"
print("  ✅ Age >= 18")
print("  ✅ LoS <= age - 18")

# Summary
print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print(f"  Master table: {len(master_df):,} rows, {len(master_df.columns)} cols")
print(f"  Monthly table: {len(month_df):,} rows, {len(month_df.columns)} cols")
print("=" * 60)

all_passed = all(checks.values())
if all_passed:
    print("\n🎉 ALL CHECKS PASSED - MVP READY FOR ITERATION")
else:
    print("\n⚠️  SOME CHECKS FAILED - REVIEW AND FIX")
```

---

## 📊 EXPECTED OUTPUTS

After successful build, you should see:

```
=== Chain-of-Verification: Initial Cohort ===
  ✓ Army: ~60% (expected 60%)
  ✓ RN: ~20% (expected 20%)
  ✓ RAF: ~20% (expected 20%)
  ✓ Female rate: ~17.5%
  ✓ Mean age: 31-34
  ✓ MFD rate: ≥80%
==================================================

Running 12-month simulation...
  Year 1: Pop=~1000, MFD=~83%

============================================================
FINAL CHAIN-OF-VERIFICATION
============================================================
✓ Population stability: within ±5%
✓ Implied annual turnover: ~10% (±2%)
ℹ JMES deterioration events: ~150
ℹ Total injuries: ~400
ℹ Total pregnancies: ~15
============================================================
```

---

## 🔄 SPIRAL ITERATION PRIORITIES

After MVP validation, iterate in this order:

1. **Tune turnover rate** - Adjust `monthly_rate` if >12% or <8%
2. **Add injury-JMES correlation** - Injuries should increase deterioration risk
3. **Implement Cox survival prep** - Verify survival_time and censor_flag
4. **Scale test** - Run 100K/60mo and monitor memory
5. **Add scenario builder** - UI for counterfactual analysis

---

## 🚨 COMMON ISSUES & FIXES

| Issue | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'numpy.str_' has no attribute 'value'` | Using `rng.choice()` on Enum list | Use index-based selection |
| `ValueError: low >= high` | LoS range invalid for young ages | Add bounds checking |
| Population exploding | Inflow >> Outflow | Balance inflow to match outflow |
| Memory error at 100K | No chunking | Process in batches of 10K |

---

## ✅ BUILD COMPLETE CHECKLIST

```
□ All 6 self-checks pass
□ Integration test passes
□ Streamlit app launches: streamlit run src/app.py
□ Can generate 1000 personnel / 12 months in <30s
□ Distributions within tolerance of config
□ CSV export works
```

**You are now ready to iterate!**
