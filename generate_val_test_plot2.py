import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Load combined grid results
files = glob.glob('results/combined/all_folds_grid_results_*.csv')
latest = sorted(files)[-1]
df = pd.read_csv(latest)

base = df[df['type'] == 'baseline'].copy()
opt = df[df['type'] == 'optimized'].copy()

# Load history files to get validation_rmse for baseline
history_data = []
for fold in [8, 9, 10]:
    fold_dir = f'results/fold_{fold:02d}'
    hist_files = glob.glob(f'{fold_dir}/alpha_optimization_history_*.csv')
    if hist_files:
        latest_hist = sorted(hist_files)[-1]
        h_df = pd.read_csv(latest_hist)
        h_df['fold'] = fold
        base_candidates = h_df[np.isclose(h_df['alpha'], 1.0, atol=1e-5)].copy()
        base_candidates = base_candidates.drop_duplicates(subset=['fold', 'method', 'K', 'TopN'])
        history_data.append(base_candidates)

if history_data:
    history_df = pd.concat(history_data, ignore_index=True)
    history_lookup = history_df.rename(columns={'rmse': 'validation_rmse_hist'})
    history_lookup = history_lookup[['fold', 'method', 'K', 'TopN', 'validation_rmse_hist']]
    
    base = pd.merge(base, history_lookup, on=['fold', 'method', 'K', 'TopN'], how='left')
    base['validation_rmse'] = base['validation_rmse_hist']

merged = pd.merge(opt, base, on=['fold', 'method', 'K', 'TopN'], suffixes=('_opt', '_base'))

merged['improvement_val'] = merged['validation_rmse_base'] - merged['validation_rmse_opt']
merged['improvement_test'] = merged['test_RMSE_base'] - merged['test_RMSE_opt']

valid_data = merged.dropna(subset=['improvement_val', 'improvement_test'])

plt.figure(figsize=(7, 5))

# Plot
plt.scatter(
    valid_data['improvement_val'],
    valid_data['improvement_test'],
    alpha=0.5,
    s=30,
    c='blue',
    edgecolors='none'
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
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/val_vs_test_scatter.pdf', bbox_inches='tight')
plt.close()

print("Plot saved to results/figures/val_vs_test_scatter.pdf")
