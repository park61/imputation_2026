import pandas as pd
df = pd.read_csv('results/combined/all_folds_grid_results_20260129_063659.csv')
print("Validation RMSE by TopN:")
print(df.groupby('TopN')['validation_rmse'].mean())
print("\nValidation RMSE by K:")
print(df.groupby('K')['validation_rmse'].mean())
