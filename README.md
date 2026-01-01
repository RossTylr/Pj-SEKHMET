<p align="center">
  <img src="assets/images/banner.png" alt="SEKHMET Recovery Predictor" width="100%">
</p>

# SEKHMET Recovery Predictor

**Evidence-based MSKI recovery prediction for Defence workforce planning.**

Clinical decision support tool using Cox proportional hazards modelling calibrated to peer-reviewed military and civilian literature.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pj-sekhmet-ckltbh5thn6walldqpr2pt.streamlit.app)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RossTylr/Pj-SEKHMET)

---

## Live App

**[Open SEKHMET Recovery Predictor](https://pj-sekhmet-ckltbh5thn6walldqpr2pt.streamlit.app)** - No installation required!

---

## Features

### Three Prediction Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **Cox PH (Evidence-based)** | Cox proportional hazards with Weibull baseline, calibrated to peer-reviewed sources | Clinical decision support, research |
| **Bayesian (Clinician-adjustable)** | Adjustable parameters for local calibration | Site-specific tuning |
| **XGBoost (ML/SHAP)** | Machine learning with SHAP explainability | Research, model comparison |

### Individual Prediction
- Input case details (injury type, body region, severity, risk factors)
- **Traffic light summary** (RTD likelihood at 3/6/12 months)
- Recovery timeline with **survival curves and 90% CI**
- Hazard ratio contributions
- **Model agreement comparison** (Cox vs XGBoost)
- **Comparator benchmark** (your case vs typical 30yo)

### Cohort Planning
- Forecast recovery timelines across a team
- Gantt-style availability planning
- Band distribution analysis

### References Tab
- **Full reference list** with DOI links
- **Parameter-to-source mapping** table
- **Evidence summary** statistics and charts
- **BibTeX and text export** for citation managers

---

## Quick Start

### Option 1: Use the Live App (Recommended)

Click the Streamlit badge above or visit [pj-sekhmet-ckltbh5thn6walldqpr2pt.streamlit.app](https://pj-sekhmet-ckltbh5thn6walldqpr2pt.streamlit.app)

### Option 2: Run in GitHub Codespaces

Click the Codespaces badge to open a fully-configured development environment. The Streamlit app will automatically launch on port 8501.

### Option 3: Run Locally

```bash
cd src/predictor
pip install -r ../../requirements.txt
streamlit run app.py
```

---

## Evidence Base (v2.0.0)

The Cox model is calibrated to clinical literature including:
- **9 peer-reviewed sources** from military and civilian populations
- **6 military-specific studies** (UK and US)
- **2,235 total sample size** across studies
- Documented risk factors with hazard ratios

### Key Sources

| Source | Year | Focus | Military |
|--------|------|-------|----------|
| Olivotto et al. | 2025 | MSKI prognostic factors | No |
| Marquina et al. | 2024 | ACL reconstruction meta-analysis | No |
| KCMHR Phase 4 | 2024 | UK military mental health | Yes |
| Anderson et al. | 2023 | Military academy epidemiology | Yes |
| Rhon et al. | 2022 | Spine rehabilitation | Yes |
| Antosh et al. | 2018 | ACL RTD in military | Yes |
| Wiggins et al. | 2016 | ACL reinjury rates | No |
| Hoge et al. | 2014 | Military PTSD | Yes |
| Shaw et al. | 2019 | Occupational LBP factors | No |

Full citations available in the **References tab** of the app.

---

## Risk Factors (Hazard Ratios)

| Factor | HR | Effect | Source |
|--------|-----|--------|--------|
| Age (per decade >25) | 1.15 | Delays recovery | Anderson 2023 |
| Prior same-region injury | 1.80 | Delays recovery | Wiggins 2016 |
| Smoking | 1.43 | Delays recovery | Anderson 2023 |
| BMI >= 30 | 1.20 | Delays recovery | Olivotto 2025 |
| OH Risk High | 1.30 | Delays recovery | Shaw 2019 |
| Supervised rehabilitation | 0.75 | Accelerates recovery | Olivotto 2025 |

---

## Injury Types (MSKI)

| Type | Median Recovery | Evidence Grade |
|------|-----------------|----------------|
| MSKI minor | 1-3 months | Moderate |
| MSKI moderate | 3-9 months | Moderate |
| MSKI major | 6-12 months | Moderate |
| MSKI severe | 12-24+ months | Low |

---

## Recovery Bands

| Band | Range | Workforce Planning |
|------|-------|-----|
| **Fast** | 0-3 months | Short-term cover |
| **Medium** | 3-6 months | Medium-term adjustment |
| **Slow** | 6-12 months | Long-term planning |
| **Complex** | 12+ months | Permanent replacement |

---

## Project Structure

```
Pj-SEKHMET/
├── src/
│   └── predictor/
│       ├── app.py              # Streamlit UI (4 tabs)
│       ├── config.py           # Enums, EvidenceBase loader
│       ├── cox_model.py        # Cox PH survival model
│       ├── bayesian_model.py   # Bayesian adjustable model
│       ├── xgb_model.py        # XGBoost with SHAP
│       └── evidence_base.yaml  # Clinical parameters
├── .devcontainer/              # GitHub Codespaces config
├── requirements.txt
└── README.md
```

---

## Roadmap

- [x] Core prediction model
- [x] Streamlit UI with configurable bands
- [x] Cox PH model with evidence base
- [x] **V2: XGBoost model with SHAP explainability**
- [x] **V2: Occupational Health risk factor (replaces Trade)**
- [x] **V3: Traffic light summary**
- [x] **V3: Survival curve with CI shading**
- [x] **V3: Model agreement indicator**
- [x] **V3: Comparator benchmark**
- [x] **V4: References tab with full citations**
- [x] **V4: BibTeX/text export**
- [x] GitHub Codespaces support
- [ ] CSV cohort upload
- [ ] PDF report export
- [ ] Real data validation

---

## License

MIT

---

*SEKHMET: Egyptian goddess of healing and war*
