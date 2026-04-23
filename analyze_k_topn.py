import pandas as pd
import numpy as np
import glob

files = glob.glob('results/combined/all_folds_grid_results_*.csv')
if not files:
    print('No combined files found')
else:
    latest = sorted(files)[-1]
    print(f"Using file: {latest}")
    df = pd.read_csv(latest)
    
    base = df[df['type'] == 'baseline'].copy()
    opt = df[df['type'] == 'optimized'].copy()
    
    merged = pd.merge(base, opt, on=['fold', 'method', 'K', 'TopN'], suffixes=('_base', '_opt'))
    
    k_agg = merged.groupby('K')[['test_RMSE_base', 'test_RMSE_opt']].mean()
    k_agg['diff'] = k_agg['test_RMSE_opt'] - k_agg['test_RMSE_base']
    print('--- K Analysis ---')
    print(k_agg)
    
    topn_agg = merged.groupby('TopN')[['test_RMSE_base', 'test_RMSE_opt']].mean()
    topn_agg['diff'] = topn_agg['test_RMSE_opt'] - topn_agg['test_RMSE_base']
    print('\n--- TopN Analysis ---')
    print(topn_agg)
