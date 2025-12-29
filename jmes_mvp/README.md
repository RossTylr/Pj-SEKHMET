<div align="center">
  <img src="data/logo/PJ-SEKHMET.jpeg" alt="PJ SEKHMET Logo" width="300"/>
</div>

# JMES Synthetic Workforce Simulation

A synthetic Defence workforce data generator for JMES (Joint Medical Employment Standard) prediction modeling. Simulates 100,000 tri-service personnel over 5-10 years with realistic turnover, injuries, and medical status transitions.

## Features

- Generates 100,000 synthetic personnel records
- Simulates 60-120 months of longitudinal data
- Models realistic turnover (10% annual / 0.88% monthly)
- Tracks JMES status transitions (MFD -> MLD -> MND)
- Includes injuries, deployments, training, and pregnancy tracking
- Interactive Streamlit dashboard for data exploration
- Configurable via YAML

## Project Structure

```
jmes_mvp/
├── configs/
│   └── simulation_config.yaml  # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── models.py              # Data models and schemas
│   ├── generator.py           # Simulation engine
│   └── app.py                 # Streamlit dashboard
├── tests/
├── data/                      # Generated data output
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Streamlit Dashboard

```bash
streamlit run src/app.py
```

Then:
1. Configure population size and simulation duration in the sidebar
2. Click "Run Simulation"
3. Explore data across Overview, JMES Monitoring, Turnover, Injuries, and Export tabs

### Run Programmatically

```python
from src.generator import SyntheticDataGenerator

gen = SyntheticDataGenerator('configs/simulation_config.yaml', seed=42)
master_df, monthly_df = gen.run_simulation()

# Save to CSV
master_df.to_csv('data/master.csv', index=False)
monthly_df.to_csv('data/monthly.csv', index=False)
```

## Data Schema

### Master Table (Personnel)
One row per synthetic individual:
- person_id, service_branch, gender, age_start, trade
- rank_band, length_of_service_start, engagement_type
- baseline_jmes, unit_env_type, initial_deployability

### Monthly Table (Person-Month)
One row per individual per month:
- person_id, month, age, jmes_current, jmes_event_this_month
- injury_type, injury_severity_score
- deployment_status, training_phase, pregnancy_status
- sick_days, outflow_flag, outflow_reason, inflow_flag
- survival_time, censor_flag

## Configuration

Edit `configs/simulation_config.yaml` to adjust:
- Population size and duration
- Service mix (Army/RN/RAF ratios)
- JMES transition probabilities
- Injury rates and multipliers
- Turnover rates
- Deployment and training frequencies

## Testing

Run the final integration test:
```bash
python3 final_integration_test.py
```

Expected output:
- ALL CHECKS PASSED - MVP READY FOR ITERATION
- Master table: ~1,118 rows (for 1000 initial + inflows)
- Monthly table: ~12,118 rows (for 12-month simulation)

## Validation Checks

The simulation includes Chain-of-Verification checks for:
- Service mix (Army: 60%, RN: 20%, RAF: 20%)
- Gender ratio (~17.5% female)
- Age distribution (mean ~32.5 years)
- JMES baseline (≥80% MFD)
- Population stability (within ±5%)
- Annual turnover (~10%)

## Next Steps

After MVP validation, iterate on:
1. Tune turnover rate if >12% or <8%
2. Add injury-JMES correlation
3. Implement Cox survival analysis prep
4. Scale test to 100K/60mo
5. Add scenario builder UI

## Technical Notes

- Uses NumPy for random generation with reproducible seeds
- Implements index-based Enum selection (avoids numpy.str_ issues)
- Handles LoS edge cases with bounds checking
- Monthly simulation with state tracking per individual
- Population maintained via balanced inflow/outflow

## Requirements

- Python 3.9+
- numpy>=1.24.0
- pandas>=2.0.0
- pyyaml>=6.0
- streamlit>=1.28.0
- plotly>=5.18.0
- pytest>=7.4.0
