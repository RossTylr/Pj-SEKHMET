"""
FINAL INTEGRATION TEST
Run from project root: python3 final_integration_test.py
"""

import sys
sys.path.insert(0, 'src')

from generator import SyntheticDataGenerator

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
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {check}")

# 3. Validate temporal consistency
print("\n[3/4] Validating temporal consistency...")
months = sorted(month_df['month'].unique())
assert months == list(range(1, 13)), "Month sequence broken"
print("  PASS Month sequence 1-12")

# 4. Validate business rules
print("\n[4/4] Validating business rules...")
assert (master_df['age_start'] >= 18).all(), "Age < 18 found"
assert (master_df['length_of_service_start'] <= master_df['age_start'] - 18).all(), "LoS > age-18"
print("  PASS Age >= 18")
print("  PASS LoS <= age - 18")

# Summary
print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print(f"  Master table: {len(master_df):,} rows, {len(master_df.columns)} cols")
print(f"  Monthly table: {len(month_df):,} rows, {len(month_df.columns)} cols")
print("=" * 60)

all_passed = all(checks.values())
if all_passed:
    print("\nALL CHECKS PASSED - MVP READY FOR ITERATION")
else:
    print("\nSOME CHECKS FAILED - REVIEW AND FIX")
