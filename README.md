# SEKHMET Recovery Predictor

**Predict recovery trajectories for injured service personnel.**

Clinical decision support + workforce planning tool.

## Quick Start

```bash
cd src/predictor
pip install -r ../../requirements.txt
streamlit run app.py
```

## Features

### 🧑‍⚕️ Individual Prediction
- Input case details (injury type, body region, severity, etc.)
- Get recovery timeline with probability curves
- View contributing factors
- JMES outcome probabilities

### 📊 Cohort Planning
- Forecast recovery timelines across a team
- Gantt-style availability planning
- Band distribution analysis

### ⚙️ Configurable Parameters
- **Recovery Bands**: Define Fast/Medium/Slow/Complex thresholds
- **Injury Profiles**: Base recovery times, recurrence risk, JMES impact
- **Body Region Modifiers**: Adjust for anatomical complexity
- **Age Modifiers**: Account for age-related recovery differences
- **Trade Physical Demand**: Role-specific adjustments

## Project Structure

```
jmes_mvp/
├── src/
│   ├── predictor/           # ← ACTIVE
│   │   ├── config.py        # Configurable parameters
│   │   ├── recovery_model.py # Core prediction logic
│   │   └── app.py           # Streamlit UI
│   └── synthetic/           # ← PARKED (complete)
│       └── __init__.py      # Synthetic data generator
├── configs/
├── data/
└── requirements.txt
```

## Injury Types

| Type | Base Recovery | JMES Impact |
|------|---------------|-------------|
| MSKI_minor | 1-3 months | Rarely MLD |
| MSKI_moderate | 3-6 months | Often MLD |
| MSKI_major | 6-12 months | MLD/MND |
| MSKI_severe | 12-24+ months | Usually MND |
| MH_mild | 2-4 months | Rarely MLD |
| MH_moderate | 4-9 months | Often MLD |
| MH_severe | 9-18+ months | MLD/MND |

## Recovery Bands (Configurable)

| Band | Default Range | Colour |
|------|---------------|--------|
| Fast | 0-3 months | 🟢 |
| Medium | 3-6 months | 🟡 |
| Slow | 6-12 months | 🟠 |
| Complex | 12+ months | 🔴 |

## Roadmap

- [x] Core prediction model
- [x] Streamlit UI with configurable bands
- [x] Injury type integration
- [x] Body region modifiers
- [ ] CSV cohort upload
- [ ] PDF report export
- [ ] ML model integration (Cox, XGBoost)
- [ ] Real data validation

## Synthetic Data (Parked)

The synthetic data generator is complete and validated:
- 150K personnel / 10 years
- Ethnicity, gender, trade contingency modelling
- Multiple injury tracking

Will be used for model training ahead of real data validation.

---

*SEKHMET: Egyptian goddess of healing and war*
