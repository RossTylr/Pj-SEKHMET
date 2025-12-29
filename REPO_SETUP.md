# JMES Synthetic Workforce - Repository Setup Guide

## 📦 Quick Start: Add to Existing Repo

### Option A: Download and Extract ZIP

1. Download `jmes_mvp_enhanced.zip` from the chat
2. Extract to your project location:
```bash
unzip jmes_mvp_enhanced.zip -d /path/to/your/project/jmes_mvp
```

### Option B: Copy Files Manually

Create this folder structure in your repo:

```
your-repo/
└── jmes_mvp/
    ├── configs/
    │   ├── simulation_config.yaml      # Basic config
    │   └── enhanced_config.yaml        # Full contingency config
    ├── src/
    │   ├── __init__.py
    │   ├── models.py                   # Basic models
    │   ├── enhanced_models.py          # Extended models (ethnicity, etc.)
    │   ├── generator.py                # Basic generator
    │   ├── enhanced_generator.py       # Full contingency generator
    │   ├── validation.py               # Chain-of-Verification
    │   └── app.py                      # Streamlit dashboard
    ├── tests/
    │   └── test_generator.py
    ├── data/                           # Output directory
    ├── requirements.txt
    ├── README.md
    ├── CURSOR_INSTRUCTIONS.md
    └── BUILD_SPEC.json
```

---

## 🚀 Installation & First Run

### Step 1: Install Dependencies

```bash
cd jmes_mvp
pip install -r requirements.txt
```

Or install directly:
```bash
pip install numpy pandas pyyaml streamlit plotly pytest
```

### Step 2: Test the Generator

```bash
cd src

# Quick test (5K personnel, 24 months) - ~10 seconds
python -c "
from enhanced_generator import EnhancedSyntheticGenerator
gen = EnhancedSyntheticGenerator('../configs/enhanced_config.yaml', seed=42)
gen.config['population']['initial_size'] = 5000
gen.config['population']['simulation_months'] = 24
master, monthly, injuries = gen.run_simulation()
print(f'Master: {master.shape}, Monthly: {monthly.shape}, Injuries: {injuries.shape}')
"
```

### Step 3: Run Streamlit Dashboard

```bash
cd src
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📊 Generate Full Dataset (150K / 10 Years)

### Python Script

```python
from enhanced_generator import EnhancedSyntheticGenerator
import pandas as pd

# Initialize with default config (150K, 120 months)
gen = EnhancedSyntheticGenerator('configs/enhanced_config.yaml', seed=2024)

# Run simulation
master_df, monthly_df, injury_df = gen.run_simulation()

# Save to parquet (recommended for large datasets)
master_df.to_parquet('../data/master.parquet', index=False)
monthly_df.to_parquet('../data/monthly.parquet', index=False)
injury_df.to_parquet('../data/injuries.parquet', index=False)

print("Done!")
```

### Expected Output Sizes

| Dataset | Rows | Columns | File Size |
|---------|------|---------|-----------|
| Master | ~250,000 | 24 | ~50 MB |
| Monthly | ~18,000,000 | 36 | ~2 GB |
| Injuries | ~450,000 | 5 | ~30 MB |

### Runtime Estimates

| Scale | Time |
|-------|------|
| 5K / 24mo | ~10 sec |
| 20K / 60mo | ~1 min |
| 50K / 120mo | ~10 min |
| 150K / 120mo | ~45 min |

---

## 🔧 Configuration Options

Edit `configs/enhanced_config.yaml` to adjust:

### Population
```yaml
population:
  initial_size: 150000    # Starting cohort
  simulation_months: 120  # 10 years
```

### Turnover
```yaml
turnover:
  annual_rate: 0.10       # 10% per year
  monthly_rate: 0.00877   # Derived
```

### Service Mix
```yaml
service_mix:
  Army: 0.58
  RN: 0.22
  RAF: 0.20
```

### JMES Transition Rates
```yaml
jmes:
  base_transitions:
    MFD_to_MLD: 0.007     # Monthly probability
    MFD_to_MND: 0.0008
    MLD_to_MND: 0.012
    MLD_to_MFD: 0.022     # Recovery rate
```

---

## 🧪 Verification Checklist

After generating data, verify these distributions:

```python
import pandas as pd

master = pd.read_parquet('data/master.parquet')
monthly = pd.read_parquet('data/monthly.parquet')

# Check service mix
print("Service:", master['service_branch'].value_counts(normalize=True))
# Expected: Army ~58%, RN ~22%, RAF ~20%

# Check gender
print("Female:", (master['gender'] == 'Female').mean())
# Expected: ~16%

# Check ethnicity
print("Ethnicity:", master['ethnicity'].value_counts(normalize=True).head(5))
# Expected: White_British ~82%

# Check JMES at end
final_month = monthly['month'].max()
final = monthly[monthly['month'] == final_month]
print("JMES:", final['jmes_current'].value_counts(normalize=True))
# Expected: MFD 75-80%, MLD 13-17%, MND 5-10%

# Check turnover
annual_outflow = monthly['outflow_flag'].sum() / (final_month / 12)
print(f"Annual outflow: {annual_outflow:,.0f}")
# Expected: ~10% of population per year
```

---

## 📁 Git Commands

### Add to Existing Repo

```bash
# Navigate to your repo
cd /path/to/your-repo

# Create directory and copy files
mkdir -p jmes_mvp
# (copy or extract files here)

# Add to git
git add jmes_mvp/
git commit -m "Add JMES synthetic workforce generator

- 150K personnel, 10-year longitudinal simulation
- Ethnicity, gender, trade contingency modelling
- Multiple injury tracking with body region
- Pregnancy lifecycle modelling
- JMES deterioration with Cox survival support"

git push
```

### Create New Repo

```bash
# Create new directory
mkdir jmes-workforce-sim
cd jmes-workforce-sim

# Initialize git
git init

# Extract files
unzip /path/to/jmes_mvp_enhanced.zip -d .

# Add .gitignore
echo "*.pyc
__pycache__/
data/*.parquet
data/*.csv
.env
.venv/
*.egg-info/" > .gitignore

# Initial commit
git add .
git commit -m "Initial commit: JMES synthetic workforce generator"

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/jmes-workforce-sim.git
git push -u origin main
```

---

## 🛠️ Development Workflow

### Modify Generator

1. Edit `src/enhanced_generator.py`
2. Test with small scale:
```bash
python -c "from enhanced_generator import ...; # test code"
```
3. Run full validation

### Add New Contingency

1. Add to `enhanced_config.yaml`
2. Add Enum to `enhanced_models.py`
3. Add simulation logic to `enhanced_generator.py`
4. Update verification checks

### Spiral Development Iterations

| Phase | Focus |
|-------|-------|
| MVP | Core generation, basic UI |
| Enhanced | Ethnicity, multiple injuries, pregnancy |
| Modelling | Cox/logistic model preparation |
| Scenarios | Policy lever testing, counterfactuals |

---

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'enhanced_models'"

```bash
# Run from src directory
cd jmes_mvp/src
python your_script.py

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/jmes_mvp/src"
```

### Memory Error on Large Dataset

```python
# Process in chunks
gen.config['population']['chunk_size'] = 10000

# Use parquet instead of CSV (better compression)
df.to_parquet('output.parquet')

# Clear memory periodically
import gc
gc.collect()
```

### Streamlit "Address already in use"

```bash
# Kill existing process
lsof -i :8501 | grep streamlit | awk '{print $2}' | xargs kill

# Or use different port
streamlit run app.py --server.port 8502
```

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Quick test | `python -c "from enhanced_generator import ..."` |
| Run dashboard | `streamlit run app.py` |
| Run tests | `pytest tests/ -v` |
| Generate full data | See Python script above |

---

**You're all set!** 🎉
