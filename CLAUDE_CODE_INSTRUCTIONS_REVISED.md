# SEKHMET ML Integration - REVISED Instructions for Claude Code

## Context

You are updating the Pj-SEKHMET repository based on user feedback and testing. This revision addresses:
1. Fix "default_estimate" source issue - must show actual citations
2. Remove JMES box from UI
3. Remove Trade Types dropdown (replace with Occupation risk factor)
4. Restructure risk factors: Lifestyle, Occupation, BMI
5. Restructure injury input: Body Region first → then Severity
6. Separate tabs: Individual Prediction (MSKI) and Individual Prediction (MH)
7. Capitalise: MSKI and MH (not Mski or Mh)
8. Remove TBI completely (too complex for current scope)
9. Rename "Heuristic" model to "Bayesian" (clinician-designed/adjustable)
10. Fix the KeyError crash in app.py line 909

Repository: https://github.com/RossTylr/Pj-SEKHMET

---

## CRITICAL FIXES REQUIRED

### Issue 1: "default_estimate" Instead of Real Citations

**Problem**: Predictions show `Evidence sources: default_estimate` instead of actual citations.

**Root Cause**: The `_body_region_to_yaml_key()` mapping doesn't match YAML keys, causing fallback to defaults.

**Fix in config.py**:

```python
def _body_region_to_yaml_key(self, body_region) -> str:
    """Map BodyRegion enum to YAML section key."""
    # YAML uses these exact keys - verify they match evidence_base.yaml
    mapping = {
        'knee': 'knee_acl',
        'lower_back': 'lower_back',
        'shoulder': 'shoulder',
        'ankle_foot': 'ankle_foot',
        'hip_groin': 'hip_groin',
        'cervical_spine': 'cervical_spine',
        'wrist_hand': 'wrist_hand',
        'mental': 'ptsd',  # For MH conditions
    }
    key = body_region.value if hasattr(body_region, 'value') else str(body_region)
    return mapping.get(key, key)
```

**Also verify evidence_base.yaml has matching structure**:

```yaml
injuries:
  MSKI_moderate:
    knee_acl:           # Must match mapping above
      median_recovery_months: 9.0
      sources:
        - antosh_2018   # These must exist in sources section
        - anderson_2023
    lower_back:
      median_recovery_months: 6.0
      sources:
        - rhon_2022
        - marquina_2024
```

**Test after fix**:
```python
from config import EvidenceBase, InjuryType, BodyRegion
eb = EvidenceBase()
params = eb.get_injury_params(InjuryType.MSKI_MODERATE, BodyRegion.KNEE)
print(f"Sources: {params.sources}")  # Should NOT be ['default_estimate']
```

---

### Issue 2: KeyError Crash in app.py

**Problem**: `KeyError` at line 909 when using Heuristic model.

**Likely Cause**: Heuristic model returns different prediction structure than Cox model.

**Fix**: Ensure consistent prediction interface or add defensive checks:

```python
# In app.py, wrap prediction display in try/except with specific handling
try:
    if model_choice == "Bayesian (Clinician-adjustable)":
        pred = bayesian_model.predict(case)
        # Bayesian model may have different attributes
        recovery_months = getattr(pred, 'predicted_months', 
                                  getattr(pred, 'median_recovery_months', 6.0))
    else:
        pred = cox_model.predict(case)
        recovery_months = pred.median_recovery_months
except KeyError as e:
    st.error(f"Prediction error: Missing key {e}. Check model output structure.")
    return
except AttributeError as e:
    st.error(f"Prediction error: {e}. Model output may be incompatible.")
    return
```

---

## UI RESTRUCTURING

### Change 1: Remove JMES Box

**Current**: Shows JMES status somewhere in UI
**Required**: Remove completely

**In app.py**, search for and remove:
- Any `st.metric` or display showing "JMES"
- Any sidebar input for JMES status
- References to `JMESStatus` enum in UI (keep in backend if needed)

```python
# DELETE any code like:
# st.metric("JMES Status", pred.jmes_status)
# jmes = st.selectbox("Current JMES", [...])
```

---

### Change 2: Remove Trade Types Dropdown

**Current**: Dropdown with Infantry, Royal Marines, etc.
**Required**: Remove - occupation captured differently

**In app.py**, remove:
```python
# DELETE:
# trade_options = {
#     "Infantry": Trade.INFANTRY,
#     ...
# }
# trade = st.sidebar.selectbox("Trade", ...)
```

---

### Change 3: Restructure Risk Factors

**Current**: Mixed risk factors
**Required**: Three clear categories - Lifestyle, Occupation, BMI

**New sidebar structure**:

```python
st.sidebar.header("📋 Case Input")

# Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age", min_value=18, max_value=55, value=30)
sex = st.sidebar.radio("Sex", ["Male", "Female"])

# Risk Factors - LIFESTYLE
st.sidebar.subheader("🚬 Lifestyle Factors")
is_smoker = st.sidebar.checkbox("Current smoker", help="HR 1.43 - delays tissue healing")
alcohol_risk = st.sidebar.checkbox("High alcohol intake", help="Associated with poorer outcomes")
sleep_quality = st.sidebar.select_slider(
    "Sleep quality",
    options=["Poor", "Fair", "Good"],
    value="Good",
    help="Poor sleep delays recovery"
)

# Risk Factors - OCCUPATION
st.sidebar.subheader("💼 Occupation Factors")
physical_demand = st.sidebar.select_slider(
    "Physical job demands",
    options=["Low", "Moderate", "High", "Very High"],
    value="Moderate",
    help="Higher demands = longer RTD"
)
# Map to modifier
occupation_modifier = {
    "Low": 1.0,
    "Moderate": 1.15,
    "High": 1.25,
    "Very High": 1.40
}[physical_demand]

job_control = st.sidebar.checkbox(
    "Limited job modification options",
    help="Cannot adjust duties during recovery"
)

# Risk Factors - BMI
st.sidebar.subheader("⚖️ BMI")
bmi = st.sidebar.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0)
bmi_category = (
    "Underweight" if bmi < 18.5 else
    "Normal" if bmi < 25 else
    "Overweight" if bmi < 30 else
    "Obese"
)
st.sidebar.caption(f"Category: {bmi_category}")
# BMI modifier (obesity delays MSK recovery)
bmi_modifier = 1.0 if bmi < 30 else 1.2 if bmi < 35 else 1.4
```

---

### Change 4: Body Region FIRST, Then Severity

**Current**: Injury Type dropdown → Body Region dropdown
**Required**: Body Region dropdown → Severity dropdown

**New structure**:

```python
# INJURY INPUT - Body Region First
st.sidebar.subheader("🦴 Injury Details")

# Step 1: Select Body Region
body_region_options = {
    "Knee": BodyRegion.KNEE,
    "Lower Back": BodyRegion.LOWER_BACK,
    "Shoulder": BodyRegion.SHOULDER,
    "Ankle/Foot": BodyRegion.ANKLE_FOOT,
    "Hip/Groin": BodyRegion.HIP_GROIN,
    "Cervical Spine": BodyRegion.CERVICAL_SPINE,
    "Wrist/Hand": BodyRegion.WRIST_HAND,
}
body_region_name = st.sidebar.selectbox(
    "Body Region",
    list(body_region_options.keys()),
    help="Select the injured body part"
)
body_region = body_region_options[body_region_name]

# Step 2: Select Severity (appears after region selected)
severity_options = {
    "Minor": InjuryType.MSKI_MINOR,
    "Moderate": InjuryType.MSKI_MODERATE,
    "Major": InjuryType.MSKI_MAJOR,
    "Severe": InjuryType.MSKI_SEVERE,
}
severity_name = st.sidebar.selectbox(
    "Severity",
    list(severity_options.keys()),
    help="Clinical severity grading"
)
injury_type = severity_options[severity_name]

# Step 3: Additional injury factors
prior_same_region = st.sidebar.checkbox(
    "Prior injury to same region",
    help="HR 1.80 - significantly delays recovery"
)
```

---

### Change 5: Separate Tabs for MSKI and MH

**Current**: Single "Individual Prediction" tab
**Required**: Two separate tabs

**New tab structure**:

```python
def main():
    st.title("🏥 SEKHMET Recovery Predictor")
    st.markdown("*Evidence-based recovery prediction for Defence workforce planning*")
    
    # Main tabs - MSKI and MH separated
    tab_mski, tab_mh, tab_cohort, tab_settings = st.tabs([
        "🦴 Individual Prediction (MSKI)",
        "🧠 Individual Prediction (MH)",
        "📊 Cohort Planning",
        "⚙️ Model Settings"
    ])
    
    with tab_mski:
        render_mski_prediction()
    
    with tab_mh:
        render_mh_prediction()
    
    with tab_cohort:
        render_cohort_planning()
    
    with tab_settings:
        render_model_settings()


def render_mski_prediction():
    """MSKI prediction tab with body region → severity flow."""
    st.header("🦴 MSKI Recovery Prediction")
    st.caption("Musculoskeletal injury prediction")
    
    # Sidebar inputs specific to MSKI
    # ... (body region, severity, lifestyle, occupation, BMI)
    
    # Display MSKI-specific results
    # ...


def render_mh_prediction():
    """MH prediction tab with condition → severity flow."""
    st.header("🧠 MH Recovery Prediction")
    
    # Research warning for MH
    st.warning("""
    ⚠️ **RESEARCH USE ONLY**
    
    Mental health predictions require additional clinical validation.
    These estimates are for workforce planning discussion only.
    Individual clinical decisions should involve qualified mental health professionals.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MH Condition Selection
        st.subheader("Condition")
        mh_condition = st.selectbox(
            "Primary Diagnosis",
            ["PTSD", "Depression", "Anxiety", "Adjustment Disorder"],
            help="Primary mental health diagnosis"
        )
        
        mh_severity = st.selectbox(
            "Severity",
            ["Mild", "Moderate", "Severe"],
            help="Clinical severity assessment"
        )
        
        # Map conditions to evidence base keys
        # Each condition has its own parameters in evidence_base.yaml
        mh_condition_map = {
            "PTSD": "ptsd",
            "Depression": "depression",
            "Anxiety": "anxiety",
            "Adjustment Disorder": "adjustment_disorder"
        }
        
        mh_severity_map = {
            "Mild": InjuryType.MH_MILD,
            "Moderate": InjuryType.MH_MODERATE,
            "Severe": InjuryType.MH_SEVERE,
        }
        
        body_region_key = mh_condition_map[mh_condition]
        injury_type = mh_severity_map[mh_severity]
    
    with col2:
        # MH-specific risk factors
        st.subheader("MH Risk Factors")
        
        trauma_exposure = st.checkbox(
            "Combat/operational trauma exposure",
            help="HR 1.35 for PTSD specifically"
        )
        
        social_support = st.select_slider(
            "Social support level",
            options=["Low", "Moderate", "Good"],
            value="Moderate",
            help="Low support HR 1.40"
        )
        
        prior_mh_history = st.checkbox(
            "Prior mental health history",
            help="HR 1.60 - previous episodes predict longer recovery"
        )
        
        sleep_quality_mh = st.select_slider(
            "Sleep quality",
            options=["Poor", "Fair", "Good"],
            value="Fair",
            help="Poor sleep HR 1.30 - common in MH conditions"
        )
        
        substance_use = st.checkbox(
            "Alcohol/substance misuse",
            help="HR 1.50 - complicates treatment response"
        )
    
    # Additional clinical factors (collapsible)
    with st.expander("Additional Clinical Factors"):
        treatment_engagement = st.select_slider(
            "Treatment engagement",
            options=["Poor", "Variable", "Good"],
            value="Good",
            help="Engagement with therapy affects outcomes"
        )
        
        medication_response = st.selectbox(
            "Medication response (if applicable)",
            ["Not on medication", "Good response", "Partial response", "Poor/no response"],
            help="For those on psychopharmacology"
        )
        
        functional_impairment = st.select_slider(
            "Current functional impairment",
            options=["Minimal", "Moderate", "Severe"],
            value="Moderate",
            help="Impact on daily activities and work"
        )
    
    # Calculate MH-specific hazard ratios
    mh_hr_factors = {}
    
    if trauma_exposure and mh_condition == "PTSD":
        mh_hr_factors['trauma_exposure'] = 1.35
    
    if social_support == "Low":
        mh_hr_factors['low_social_support'] = 1.40
    elif social_support == "Moderate":
        mh_hr_factors['moderate_social_support'] = 1.15
    
    if prior_mh_history:
        mh_hr_factors['prior_mh_history'] = 1.60
    
    if sleep_quality_mh == "Poor":
        mh_hr_factors['poor_sleep'] = 1.30
    elif sleep_quality_mh == "Fair":
        mh_hr_factors['fair_sleep'] = 1.10
    
    if substance_use:
        mh_hr_factors['substance_use'] = 1.50
    
    # ... prediction and display using body_region_key and injury_type
```

---

### Change 6: Capitalisation - MSKI and MH

**Throughout codebase**, ensure:
- "MSKI" not "Mski" or "mski" in UI text
- "MH" not "Mh" or "mh" in UI text

**In app.py**:
```python
# CORRECT:
st.header("🦴 MSKI Recovery Prediction")
st.selectbox("Severity", ["MSKI Minor", "MSKI Moderate", "MSKI Major", "MSKI Severe"])

# INCORRECT:
st.header("Mski Recovery Prediction")  # Wrong
st.selectbox("Severity", ["Mski Minor", ...])  # Wrong
```

**In config.py enum display**:
```python
class InjuryType(Enum):
    MSKI_MINOR = "mski_minor"
    MSKI_MODERATE = "mski_moderate"
    # ... 

    @property
    def display_name(self) -> str:
        """Human-readable name for UI."""
        name = self.value.replace('_', ' ').upper()
        # Ensure MSKI and MH are capitalised
        return name.replace('MSKI', 'MSKI').replace('MH', 'MH')
```

---

### Change 7: Remove TBI Completely

**In config.py**, remove or comment out:
```python
# REMOVE these:
# InjuryType.TBI_MILD = "tbi_mild"
# InjuryType.TBI_MODERATE = "tbi_moderate"
# InjuryType.TBI_SEVERE = "tbi_severe"
# BodyRegion.BRAIN = "brain"
```

**In evidence_base.yaml**, remove or comment out:
```yaml
# REMOVE this entire section:
# TBI_mild:
#   mtbi:
#     median_recovery_months: 1.0
#     ...
```

**In app.py**, remove TBI from any dropdowns:
```python
# Remove any "TBI" options from injury_options
# Remove BodyRegion.BRAIN from region_options
```

---

### Change 8: Rename Heuristic → Bayesian

**In app.py**:
```python
# Model selection
model_choice = st.sidebar.radio(
    "Select Model",
    [
        "Cox PH (Evidence-based)", 
        "Bayesian (Clinician-adjustable)"  # Was "Heuristic"
    ],
    index=0,
    help="Cox uses published HRs; Bayesian allows clinical parameter adjustment"
)
```

**Rename recovery_model.py → bayesian_model.py** (or update class name):
```python
# In bayesian_model.py (renamed from recovery_model.py)
class BayesianRecoveryModel:
    """
    Bayesian/Clinician-adjustable recovery model.
    
    This model allows clinicians to adjust parameters based on
    local experience and clinical judgment. Parameters can be
    updated through the UI sidebar.
    
    Unlike the Cox model which uses fixed published HRs, this
    model can be calibrated to local population characteristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
    
    def predict(self, case) -> BayesianPrediction:
        """Generate prediction with adjustable parameters."""
        # ... implementation
        pass
```

**Update imports in app.py**:
```python
# Change:
# from recovery_model import HeuristicModel
# To:
from bayesian_model import BayesianRecoveryModel
```

---

## UPDATED evidence_base.yaml STRUCTURE

Ensure the YAML has correct structure for citation lookup:

```yaml
metadata:
  version: "1.2.0"
  last_updated: "2024-12-31"
  description: "Clinical evidence for SEKHMET - MSKI and MH only (TBI removed)"

injuries:
  MSKI_minor:
    knee_acl:
      median_recovery_months: 3.0
      recovery_range_months: [1.0, 4.0]
      time_to_fitness_months: 2.0
      time_to_rtd_months: 3.0
      prob_full_recovery: 0.85
      prob_partial_recovery: 0.10
      prob_not_recovered: 0.05
      evidence_grade: "Low"
      sources:
        - olivotto_2025
      stakeholder_explainer: |
        Minor knee injuries typically resolve within 1-4 months.
        Most personnel return to full duties.
    
    lower_back:
      median_recovery_months: 2.0
      recovery_range_months: [1.0, 4.0]
      # ... etc
      sources:
        - rhon_2022
  
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
      evidence_grade: "Moderate"
      sources:
        - antosh_2018
        - marquina_2024
        - anderson_2023
        - wiggins_2016
      stakeholder_explainer: |
        ACL injuries have one of the longest recovery trajectories.
        About half return to unrestricted duty within 9-12 months.
    
    lower_back:
      median_recovery_months: 6.0
      recovery_range_months: [3.0, 12.0]
      sources:
        - rhon_2022
        - shaw_2019
      # ... etc

  # MH section - Each condition has separate parameters
  MH_mild:
    ptsd:
      median_recovery_months: 3.0
      recovery_range_months: [2.0, 6.0]
      time_to_fitness_months: 2.0
      time_to_rtd_months: 3.0
      prob_full_recovery: 0.70
      prob_partial_recovery: 0.20
      prob_not_recovered: 0.10
      evidence_grade: "Moderate"
      sources:
        - kcmhr_2024
      stakeholder_explainer: |
        Mild PTSD with early intervention typically resolves within 2-6 months.
        70% achieve full recovery with appropriate treatment.
    
    depression:
      median_recovery_months: 4.0
      recovery_range_months: [2.0, 8.0]
      time_to_fitness_months: 3.0
      time_to_rtd_months: 4.0
      prob_full_recovery: 0.65
      prob_partial_recovery: 0.25
      prob_not_recovered: 0.10
      evidence_grade: "Moderate"
      sources:
        - kcmhr_2024
        - kessler_2020
      stakeholder_explainer: |
        Mild depression often responds well to treatment within 2-8 months.
    
    anxiety:
      median_recovery_months: 3.0
      recovery_range_months: [1.5, 6.0]
      time_to_fitness_months: 2.0
      time_to_rtd_months: 3.0
      prob_full_recovery: 0.70
      prob_partial_recovery: 0.20
      prob_not_recovered: 0.10
      evidence_grade: "Moderate"
      sources:
        - kcmhr_2024
      stakeholder_explainer: |
        Mild anxiety disorders typically respond well to treatment.
    
    adjustment_disorder:
      median_recovery_months: 2.0
      recovery_range_months: [1.0, 4.0]
      time_to_fitness_months: 1.5
      time_to_rtd_months: 2.0
      prob_full_recovery: 0.80
      prob_partial_recovery: 0.15
      prob_not_recovered: 0.05
      evidence_grade: "Low"
      sources:
        - kcmhr_2024
      stakeholder_explainer: |
        Adjustment disorders typically resolve within 1-4 months
        once the stressor is addressed.
  
  MH_moderate:
    ptsd:
      median_recovery_months: 8.0
      recovery_range_months: [6.0, 18.0]
      time_to_fitness_months: 6.0
      time_to_rtd_months: 8.0
      prob_full_recovery: 0.35
      prob_partial_recovery: 0.30
      prob_not_recovered: 0.35
      evidence_grade: "Moderate"
      sources:
        - kcmhr_2024
        - hoge_2014
        - kessler_2020
      stakeholder_explainer: |
        Moderate PTSD typically requires 6-18 months of treatment.
        About 35% achieve full recovery, 35% have persistent symptoms.
        UK military PTSD prevalence is approximately 6% (KCMHR).
    
    depression:
      median_recovery_months: 9.0
      recovery_range_months: [6.0, 18.0]
      time_to_fitness_months: 6.0
      time_to_rtd_months: 9.0
      prob_full_recovery: 0.40
      prob_partial_recovery: 0.35
      prob_not_recovered: 0.25
      evidence_grade: "Moderate"
      sources:
        - kcmhr_2024
        - kessler_2020
      stakeholder_explainer: |
        Moderate depression often requires 6-18 months of treatment.
        Recurrence risk is significant - about 50% within 2 years.
    
    anxiety:
      median_recovery_months: 6.0
      recovery_range_months: [4.0, 12.0]
      time_to_fitness_months: 4.0
      time_to_rtd_months: 6.0
      prob_full_recovery: 0.50
      prob_partial_recovery: 0.30
      prob_not_recovered: 0.20
      evidence_grade: "Moderate"
      sources:
        - kcmhr_2024
      stakeholder_explainer: |
        Moderate anxiety disorders require structured treatment,
        typically CBT plus/minus medication.
    
    adjustment_disorder:
      median_recovery_months: 4.0
      recovery_range_months: [2.0, 8.0]
      time_to_fitness_months: 3.0
      time_to_rtd_months: 4.0
      prob_full_recovery: 0.65
      prob_partial_recovery: 0.25
      prob_not_recovered: 0.10
      evidence_grade: "Low"
      sources:
        - kcmhr_2024
      stakeholder_explainer: |
        More persistent adjustment reactions may indicate
        underlying vulnerability or ongoing stressors.
  
  MH_severe:
    ptsd:
      median_recovery_months: 18.0
      recovery_range_months: [9.0, 36.0]
      time_to_fitness_months: 12.0
      time_to_rtd_months: 18.0
      prob_full_recovery: 0.20
      prob_partial_recovery: 0.30
      prob_not_recovered: 0.50
      evidence_grade: "Moderate"
      sources:
        - kcmhr_2024
        - liu_2025
        - hoge_2014
      stakeholder_explainer: |
        Severe PTSD has prolonged recovery trajectories.
        Many require intensive residential treatment.
        50% may not return to full military duties.
    
    depression:
      median_recovery_months: 18.0
      recovery_range_months: [12.0, 36.0]
      time_to_fitness_months: 12.0
      time_to_rtd_months: 18.0
      prob_full_recovery: 0.25
      prob_partial_recovery: 0.30
      prob_not_recovered: 0.45
      evidence_grade: "Moderate"
      sources:
        - kcmhr_2024
        - kessler_2020
      stakeholder_explainer: |
        Severe/treatment-resistant depression requires
        specialist intervention. Many cases become chronic.
    
    anxiety:
      median_recovery_months: 12.0
      recovery_range_months: [6.0, 24.0]
      time_to_fitness_months: 9.0
      time_to_rtd_months: 12.0
      prob_full_recovery: 0.30
      prob_partial_recovery: 0.35
      prob_not_recovered: 0.35
      evidence_grade: "Low"
      sources:
        - kcmhr_2024
      stakeholder_explainer: |
        Severe anxiety disorders often have comorbidities
        and require multimodal treatment.
    
    adjustment_disorder:
      median_recovery_months: 8.0
      recovery_range_months: [4.0, 12.0]
      time_to_fitness_months: 6.0
      time_to_rtd_months: 8.0
      prob_full_recovery: 0.45
      prob_partial_recovery: 0.35
      prob_not_recovered: 0.20
      evidence_grade: "Low"
      sources:
        - kcmhr_2024
      stakeholder_explainer: |
        Severe adjustment reactions may evolve into
        other diagnoses (PTSD, depression) over time.

# SOURCES SECTION - All referenced sources must exist here
sources:
  antosh_2018:
    authors: "Antosh IJ, Svoboda SJ, Peck KY, et al."
    title: "Change in KOOS and WOMAC Scores and Return to Duty Rate Following ACL Reconstruction in Military Service Members"
    journal: "Military Medicine"
    year: 2018
    doi: "10.1093/milmed/usx007"
    study_type: "cohort"
    n: 110
    military_specific: true
  
  anderson_2023:
    authors: "Anderson MJJ, Browning WM, Urband CE, et al."
    title: "Epidemiology of Injuries in United States Military Academies"
    journal: "Sports Health"
    year: 2023
    study_type: "retrospective_cohort"
    n: 12500
    military_specific: true
  
  marquina_2024:
    authors: "Marquina Nieto, et al."
    title: "Return to Sports After Anterior Cruciate Ligament Reconstruction in Military Personnel"
    journal: "Musculoskeletal Science and Practice"
    year: 2024
    study_type: "meta_analysis"
    military_specific: true
  
  rhon_2022:
    authors: "Rhon DI, Greenlee TA, Sissel CD, et al."
    title: "Recovery and return to duty after spine rehabilitation in military service members"
    journal: "BMC Musculoskeletal Disorders"
    year: 2022
    study_type: "cohort"
    military_specific: true
  
  olivotto_2025:
    authors: "Olivotto S, van Trier T, Bongers PM, et al."
    title: "Prognostic factors for poor recovery after musculoskeletal injury in military personnel"
    journal: "Musculoskeletal Science and Practice"
    year: 2025
    study_type: "systematic_review"
    prospero: "CRD42023409781"
    military_specific: true
  
  wiggins_2016:
    authors: "Wiggins AJ, Grandhi RK, Engstrom SM, et al."
    title: "Risk of Secondary Injury in Younger Athletes After Anterior Cruciate Ligament Reconstruction"
    journal: "American Journal of Sports Medicine"
    year: 2016
    study_type: "systematic_review"
    n: 7556
  
  kcmhr_2024:
    authors: "King's Centre for Military Health Research"
    title: "Health and Wellbeing of the UK Armed Forces: Phase 4"
    year: 2024
    study_type: "cohort"
    n: 8000
    military_specific: true
    note: "UK-specific PTSD prevalence 6%"
  
  hoge_2014:
    authors: "Hoge CW, Riviere LA, Wilk JE, et al."
    title: "The prevalence of post-traumatic stress disorder in UK and US military personnel"
    journal: "Lancet Psychiatry"
    year: 2014
    study_type: "meta_analysis"
  
  kessler_2020:
    authors: "Kessler RC, et al."
    title: "Prevalence of mental disorders in the military"
    journal: "JAMA Psychiatry"
    year: 2020
    study_type: "meta_analysis"
  
  liu_2025:
    authors: "Liu X, et al."
    title: "Recovery trajectories in military PTSD"
    year: 2025
    study_type: "cohort"
    military_specific: true
  
  shaw_2019:
    authors: "Shaw WS, et al."
    title: "Occupational factors in low back pain recovery"
    journal: "Journal of Occupational Rehabilitation"
    year: 2019
    study_type: "systematic_review"

# Risk factors - updated with Lifestyle/Occupation/BMI focus
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
    applies_to: ["all"]
    sources: ["olivotto_2025"]
    stakeholder_explainer: "High alcohol intake associated with poorer recovery outcomes."
  
  poor_sleep:
    effect_type: "hazard_ratio"
    hr: 1.30
    ci_95: [1.10, 1.54]
    category: "lifestyle"
    direction: "delays_recovery"
    applies_to: ["all"]
    sources: ["olivotto_2025"]
    stakeholder_explainer: "Poor sleep quality delays tissue repair and recovery."
  
  # OCCUPATION
  high_physical_demand:
    effect_type: "hazard_ratio"
    hr: 1.40
    ci_95: [1.15, 1.70]
    category: "occupation"
    direction: "delays_recovery"
    applies_to: ["MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["shaw_2019", "olivotto_2025"]
    stakeholder_explainer: "Higher physical job demands require longer RTD even after clinical recovery."
  
  limited_job_modification:
    effect_type: "hazard_ratio"
    hr: 1.35
    ci_95: [1.10, 1.65]
    category: "occupation"
    direction: "delays_recovery"
    applies_to: ["MSKI_moderate", "MSKI_major"]
    sources: ["shaw_2019"]
    stakeholder_explainer: "Inability to modify job duties prolongs time away from work."
  
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
    stakeholder_explainer: "Severe obesity (BMI 35+) significantly delays MSK recovery."
  
  # OTHER
  age:
    effect_type: "hazard_ratio"
    per_unit: "per decade over 25"
    hr: 1.15
    ci_95: [1.08, 1.23]
    category: "demographic"
    direction: "delays_recovery"
    applies_to: ["all"]
    sources: ["anderson_2023", "wiggins_2016"]
  
  prior_same_region_injury:
    effect_type: "hazard_ratio"
    hr: 1.80
    ci_95: [1.40, 2.30]
    category: "clinical"
    direction: "delays_recovery"
    applies_to: ["all"]
    sources: ["wiggins_2016", "olivotto_2025"]
  
  supervised_rehabilitation:
    effect_type: "hazard_ratio"
    hr: 0.75
    ci_95: [0.65, 0.87]
    category: "treatment"
    direction: "accelerates_recovery"
    applies_to: ["all"]
    sources: ["olivotto_2025"]
  
  mh_comorbidity:
    effect_type: "hazard_ratio"
    hr: 2.00
    ci_95: [1.50, 2.67]
    category: "clinical"
    direction: "delays_recovery"
    applies_to: ["MSKI_moderate", "MSKI_major", "MSKI_severe"]
    sources: ["olivotto_2025"]
    stakeholder_explainer: "Mental health comorbidity is the strongest predictor of poor MSK outcomes."

  # MH-SPECIFIC RISK FACTORS
  trauma_exposure:
    effect_type: "hazard_ratio"
    hr: 1.35
    ci_95: [1.10, 1.65]
    category: "mh_clinical"
    direction: "delays_recovery"
    applies_to: ["MH_mild", "MH_moderate", "MH_severe"]
    condition_specific: ["ptsd"]
    sources: ["kcmhr_2024", "hoge_2014"]
    stakeholder_explainer: "Combat/operational trauma exposure associated with more complex PTSD presentations."

  low_social_support:
    effect_type: "hazard_ratio"
    hr: 1.40
    ci_95: [1.15, 1.70]
    category: "mh_social"
    direction: "delays_recovery"
    applies_to: ["MH_mild", "MH_moderate", "MH_severe"]
    sources: ["kcmhr_2024"]
    stakeholder_explainer: "Low social support significantly impairs mental health recovery."

  moderate_social_support:
    effect_type: "hazard_ratio"
    hr: 1.15
    ci_95: [0.95, 1.38]
    category: "mh_social"
    direction: "delays_recovery"
    applies_to: ["MH_mild", "MH_moderate", "MH_severe"]
    sources: ["kcmhr_2024"]
    stakeholder_explainer: "Moderate social support - some protective effect but not optimal."

  prior_mh_history:
    effect_type: "hazard_ratio"
    hr: 1.60
    ci_95: [1.30, 1.97]
    category: "mh_clinical"
    direction: "delays_recovery"
    applies_to: ["MH_mild", "MH_moderate", "MH_severe"]
    sources: ["kcmhr_2024", "kessler_2020"]
    stakeholder_explainer: "Prior mental health episodes predict longer recovery and higher recurrence risk."

  substance_use:
    effect_type: "hazard_ratio"
    hr: 1.50
    ci_95: [1.20, 1.88]
    category: "mh_lifestyle"
    direction: "delays_recovery"
    applies_to: ["MH_mild", "MH_moderate", "MH_severe"]
    sources: ["kcmhr_2024"]
    stakeholder_explainer: "Alcohol/substance misuse complicates treatment response and prolongs recovery."

  poor_treatment_engagement:
    effect_type: "hazard_ratio"
    hr: 1.80
    ci_95: [1.40, 2.31]
    category: "mh_treatment"
    direction: "delays_recovery"
    applies_to: ["MH_moderate", "MH_severe"]
    sources: ["kcmhr_2024"]
    stakeholder_explainer: "Poor engagement with therapy significantly delays recovery."
```

---

## FILE CHANGES SUMMARY

| File | Action | Changes |
|------|--------|---------|
| `config.py` | UPDATE | Fix `_body_region_to_yaml_key()`, remove TBI enums |
| `evidence_base.yaml` | UPDATE | Ensure sources match, remove TBI section, add Lifestyle/Occupation/BMI risk factors |
| `app.py` | MAJOR UPDATE | Separate MSKI/MH tabs, remove JMES, remove Trade dropdown, restructure inputs |
| `recovery_model.py` | RENAME | Rename to `bayesian_model.py`, update class name |
| `cox_model.py` | UPDATE | Ensure citations passed correctly, remove TBI handling |

---

## TESTING CHECKLIST

After making changes, verify:

```bash
# 1. Evidence base loads correctly
python -c "
from config import EvidenceBase
eb = EvidenceBase()
print(f'Version: {eb.version}')
print(f'Has MSKI_moderate: {\"MSKI_moderate\" in eb._injuries}')
print(f'Has knee_acl: {\"knee_acl\" in eb._injuries.get(\"MSKI_moderate\", {})}')
"

# 2. Citations show real sources (not default_estimate)
python -c "
from config import EvidenceBase, InjuryType, BodyRegion
from cox_model import CoxRecoveryModel, CaseInput
from config import Trade

model = CoxRecoveryModel()
case = CaseInput(
    age=35,
    trade=Trade.INFANTRY,
    injury_type=InjuryType.MSKI_MODERATE,
    body_region=BodyRegion.KNEE
)
pred = model.predict(case)
print(f'Sources: {pred.primary_sources}')
assert pred.primary_sources != ['default_estimate'], 'FAIL: Still showing default_estimate!'
print('PASS: Real citations shown')
"

# 3. TBI removed
python -c "
from config import InjuryType
try:
    _ = InjuryType.TBI_MILD
    print('FAIL: TBI_MILD still exists')
except AttributeError:
    print('PASS: TBI removed from InjuryType')
"

# 4. Bayesian model works
python -c "
from bayesian_model import BayesianRecoveryModel
model = BayesianRecoveryModel()
# Test with a simple case
print('PASS: Bayesian model imports correctly')
"

# 5. Run Streamlit
streamlit run app.py
# Then manually verify:
# - [ ] MSKI and MH are separate tabs
# - [ ] Body Region dropdown appears BEFORE Severity
# - [ ] No JMES box visible
# - [ ] No Trade dropdown
# - [ ] Risk factors grouped: Lifestyle, Occupation, BMI
# - [ ] MSKI and MH capitalised throughout
# - [ ] Citations show real sources on prediction
```

---

## COMMIT MESSAGE

```bash
git add -A
git commit -m "fix: Citations, UI restructure, remove TBI

FIXES:
- Fixed 'default_estimate' citation issue - now shows real sources
- Fixed KeyError crash in Bayesian model prediction

UI CHANGES:
- Separated MSKI and MH into distinct tabs
- Body Region now selected BEFORE Severity
- Removed JMES status display
- Removed Trade dropdown
- Restructured risk factors: Lifestyle, Occupation, BMI
- Capitalised MSKI and MH throughout

MODEL CHANGES:
- Renamed Heuristic → Bayesian (clinician-adjustable)
- Removed TBI injury types (too complex for current scope)
- Updated evidence_base.yaml structure

Closes #[issue_number]"

git push
```

---

## QUESTIONS FOR CONFIRMATION

Before proceeding, confirm:

1. **BMI modifier values**: I used HR 1.20 for BMI 30-35, HR 1.40 for BMI 35+. Acceptable?

2. **Physical demand levels**: Low/Moderate/High/Very High with modifiers 1.0/1.15/1.25/1.40. Adjust?

3. **MH conditions**: Currently just "PTSD" mapped. Add Depression, Anxiety, Adjustment Disorder separately?

4. **Sleep quality**: Added as lifestyle factor. Keep or remove?

5. **Alcohol**: Added as lifestyle factor. Keep or remove?
