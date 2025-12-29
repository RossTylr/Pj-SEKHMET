import sys
sys.path.insert(0, 'src')
from generator import SyntheticDataGenerator

gen = SyntheticDataGenerator('configs/simulation_config.yaml', seed=42)
gen.config['population']['initial_size'] = 100
gen.config['population']['simulation_months'] = 6

master_df, month_df = gen.run_simulation()

assert len(master_df) >= 100, "Master table too small"
assert len(month_df) >= 500, "Monthly table too small"
assert 'person_id' in master_df.columns
assert 'jmes_current' in month_df.columns

print(f"Step 4: Generator verified")
print(f"   Master: {len(master_df)} rows")
print(f"   Monthly: {len(month_df)} rows")
