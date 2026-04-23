import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Load combined grid results
files = glob.glob('results/combined/all_folds_grid_results_*.csv')
latest = sorted(files)[-1]
df = pd.read_csv(latest)

# Filter for baseline and optimized
base = df[df['type'] == 'baseline'].copy()
opt = df[df['type'] == 'optimized'].copy()

# Merge on method, K, TopN, fold
merged = pd.merge(opt, base, on=['fold', 'method', 'K', 'TopN'], suffixes=('_opt', '_base'))

# Calculate improvements (Positive = Optimized is better)
merged['improvement_val'] = merged['validation_rmse_base'] - merged['validation_rmse_opt']
merged['improvement_test'] = merged['test_RMSE_base'] - merged['test_RMSE_opt']

# Drop NaNs
valid_data = merged.dropna(subset=['improvement_val', 'improvement_test'])

# 1. Scatter Plot: Val vs Test Improvement
plt.figure(figsize=(8, 6))

# Plot each method with a different color
methods = valid_data['method'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))

for i, method in enumerate(methods):
    subset = valid_data[valid_data['method'] == method]
    plt.scatter(
        subset['improvement_val'],
        subset['improvement_test'],
        label=method,
        alpha=0.7,
        s=40,
        color=colors[i]
    )

# Reference lines
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)

# y=x line
min_val = min(valid_data['improvement_val'].min(), valid_data['improvement_test'].min())
max_val = max(valid_data['improvement_val'].max(), valid_data['improvement_test'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle=':', alpha=0.8, label='y=x (Ideal)')

plt.title('Validation vs Test RMSE Improvement\n(Positive = Optimized is better)')
plt.xlabel('Validation RMSE Improvement')
plt.ylabel('Test RMSE Improvement')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()

os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/val_vs_test_scatter.pdf', bbox_inches='tight')
plt.close()

# 2. Concrete Table Data for ACOS vs PCC
print("--- Concrete Example: ACOS vs PCC ---")
example_methods = ['acos', 'pcc']
summary = valid_data[valid_data['method'].isin(example_methods)].groupby('method')[['improvement_val', 'improvement_test']].mean()
print(summary)
