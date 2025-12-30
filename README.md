# SEKHMET Recovery Predictor

**Predict recovery trajectories for injured service personnel.**

Clinical decision support + workforce planning tool with evidence-based Cox proportional hazards modelling.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pj-sekhmet-ckltbh5thn6walldqpr2pt.streamlit.app)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RossTylr/Pj-SEKHMET)

## Live App

**[Open the SEKHMET Recovery Predictor](https://pj-sekhmet-ckltbh5thn6walldqpr2pt.streamlit.app)** - No installation required!

## Quick Start

### Option 1: Use the Live App (Recommended)

Click the Streamlit badge above or visit [pj-sekhmet-ckltbh5thn6walldqpr2pt.streamlit.app](https://pj-sekhmet-ckltbh5thn6walldqpr2pt.streamlit.app) to use the app directly in your browser.

### Option 2: Run in GitHub Codespaces

Click the Codespaces badge to open a fully-configured development environment in your browser. The Streamlit app will automatically launch on port 8501.

### Option 3: Run Locally

```bash
cd src/predictor
pip install -r ../../requirements.txt
streamlit run app.py
```

## Features

### Two Prediction Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **Cox PH (Evidence-based)** | Cox proportional hazards with Weibull baseline, calibrated to 22 peer-reviewed sources | Clinical decision support, research |
| **Heuristic (Legacy)** | Rule-based multiplier model | Quick estimates, configurable parameters |

### Individual Prediction
- Input case details (injury type, body region, severity, risk factors)
- Get recovery timeline with survival curves
- View hazard ratio contributions
- JMES outcome probabilities with citations

### Cohort Planning
- Forecast recovery timelines across a team
- Gantt-style availability planning
- Band distribution analysis

### Evidence Base (v1.1.0)
The Cox model is calibrated to clinical literature including:
- **22 peer-reviewed sources** from military and civilian populations
- UK-specific data from KCMHR Phase 4
- mTBI, PTSD, ACL, and lower back parameters
- Documented risk factors with hazard ratios and confidence intervals

## Project Structure

```
Pj-SEKHMET/
├── src/
│   └── predictor/
│       ├── config.py           # Enums, EvidenceBase loader
│       ├── cox_model.py        # Cox PH survival model
│       ├── recovery_model.py   # Legacy heuristic model
│       ├── evidence_base.yaml  # Clinical parameters (22 sources)
│       └── app.py              # Streamlit UI
├── .devcontainer/              # GitHub Codespaces config
├── requirements.txt
└── README.md
```

## Trade Categories

Personnel trades are grouped by physical demands:

| Category | Trades | RTD Modifier |
|----------|--------|--------------|
| **Combat** | Infantry, Royal Marines, Para, Armour, Artillery, Combat Engineer | 1.30x |
| **Combat Support** | Signals, Intelligence, REME, Medic, Military Police | 1.15x |
| **Combat Service Support** | Logistics, AGC, Dental, Veterinary, Chaplain | 1.00x (baseline) |

## Injury Types

| Type | Median Recovery | Evidence Grade |
|------|-----------------|----------------|
| MSKI minor | 1-3 months | Low |
| MSKI moderate | 3-9 months | Moderate |
| MSKI major | 6-12 months | Moderate |
| MSKI severe | 12-24+ months | Low |
| MH mild | 2-4 months | Low |
| MH moderate | 6-18 months | Moderate |
| MH severe | 9-18+ months | Moderate |
| TBI mild (mTBI) | 0.5-3 months | Moderate |
| TBI moderate | 3-18 months | Low |

## Risk Factors (Hazard Ratios)

| Factor | HR | Effect |
|--------|-----|--------|
| Age (per decade >25) | 1.15 | Delays recovery |
| Prior same-region injury | 1.80 | Delays recovery |
| Smoking | 1.43 | Delays recovery |
| Multiple TBI (≥3) | 1.80 | Delays recovery |
| MH comorbidity | 2.00 | Delays recovery |
| Supervised rehabilitation | 0.75 | Accelerates recovery |

## Recovery Bands

| Band | Range | Use |
|------|-------|-----|
| Fast | 0-3 months | Short-term cover |
| Medium | 3-6 months | Medium-term adjustment |
| Slow | 6-12 months | Long-term planning |
| Complex | 12+ months | Permanent replacement |

## Key Evidence Sources

- Antosh IJ et al. (2018) - ACL RTD in military
- KCMHR Phase 4 (2024) - UK military mental health
- Olivotto S et al. (2025) - MSKI prognostic factors
- TBI Center of Excellence (2023) - mTBI outcomes

Full citations available in `src/predictor/evidence_base.yaml`.

## Roadmap

- [x] Core prediction model
- [x] Streamlit UI with configurable bands
- [x] **Cox PH model with evidence base v1.1.0**
- [x] Trade categories (Combat/Support/CSS)
- [x] TBI injury types and mTBI parameters
- [x] Citation display in UI
- [x] GitHub Codespaces support
- [ ] CSV cohort upload
- [ ] PDF report export
- [ ] Real data validation

---

*SEKHMET: Egyptian goddess of healing and war*
