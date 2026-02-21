#!/usr/bin/env python3
import pandas as pd
import os
import sys
sys.path.append('.')
import config
import joblib

df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'final_dataset.csv'))

print("=" * 70)
print("DATA QUALITY ANALYSIS - ROOT CAUSE OF PREDICTION ERRORS")
print("=" * 70)

# 1. Zero values problem
print("\n1. MISSING DATA CAUSING PREDICTION FAILURES:")
zero_rev = df[df['revenue'] == 0]
zero_bud = df[df['budget'] == 0]
print(f"   • Movies with ZERO revenue: {len(zero_rev)} (can't calculate error %)")
print(f"   • Movies with ZERO budget: {len(zero_bud)} (unreliable predictions)")
print(f"   • Movies with BOTH zero: {len(df[(df['budget'] == 0) & (df['revenue'] == 0)])}")

# 2. Check what the validation endpoint does with zero revenues
print("\n2. ERROR IN VALIDATION ENDPOINT:")
print("   In app.py line 453: error_pct = abs(pred - row['revenue']) / row['revenue'] * 100")
print(f"   ⚠️  Division by ZERO when revenue = 0!")
print(f"   This causes: ZeroDivisionError or infinite errors")
print(f"   Affected movies: {len(zero_rev)}")

# 3. Movies with revenue but no budget
rev_no_bud = df[(df['revenue'] > 0) & (df['budget'] == 0)]
print(f"\n3. BUDGET DATA MISSING (but revenue exists):")
print(f"   • Movies: {len(rev_no_bud)}")
if len(rev_no_bud) > 0:
    print(f"   Examples: {rev_no_bud['title'].head(3).tolist()}")

# 4. Check actual prediction accuracy on valid movies only
reg_model = joblib.load(os.path.join(config.MODELS_DIR, "best_regression_model.pkl"))
valid_df = df[(df['revenue'] > 1000) & (df['budget'] > 1000)].copy()
print(f"\n4. VALID MOVIES (revenue > $1k AND budget > $1k):")
print(f"   • Total: {len(valid_df)} out of {len(df)}")
print(f"   • These should have reasonable error rates")

# 5. Check highest error movies
print(f"\n5. ANALYZING HIGH-ERROR MOVIES:")
# Find movies with actual data we can test
testable = df[(df['revenue'] > 0) & (df['budget'] > 0)].copy()
print(f"   Total testable movies: {len(testable)}")

# 6. Budget distribution issue
print(f"\n6. BUDGET DATA DISTRIBUTION:")
budget_stats = df[df['budget'] > 0]['budget'].describe()
print(f"   Movies with budget > 0: {(df['budget'] > 0).sum()}")
print(f"   Mean: ${budget_stats['mean']/1e6:.1f}M")
print(f"   Median: ${budget_stats['50%']/1e6:.1f}M")

print("\n" + "=" * 70)
print("SOLUTIONS:")
print("=" * 70)
print("1. FILTER validation to only movies with revenue > 1000")
print("2. HANDLE division by zero in error calculation")
print("3. IMPUTE missing budget values OR exclude from validation")
print("=" * 70)
