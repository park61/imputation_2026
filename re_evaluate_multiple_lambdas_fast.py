import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ==================================================================================
# CONFIGURATION
# ==================================================================================
LAMBDA_VALUES = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
FOLDS_TO_PROCESS = [1]  # V2: Only fold 01 for quick validation
RELEVANCE_THRESHOLD = 4.0
DATA_DIR = "data/movielenz_data"
RESULTS_DIR = "results"

print(f"========================================================")
print(f"MULTI-LAMBDA SMART RE-EVALUATION V2 (WITH VALIDATION)")
print(f"Target Lambdas: {LAMBDA_VALUES}")
print(f"Target Folds: {FOLDS_TO_PROCESS}")
print(f"V2 Changes: Now saves VALIDATION + TEST performance")
print(f"========================================================\n")

# ==================================================================================
# HELPER FUNCTIONS 
# ==================================================================================
def _load_matrix_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    lower = [str(c).lower() for c in df.columns]
    if {"item", "user", "rating"}.issubset(lower):
        def col(name):
            for c in df.columns:
                if str(c).lower() == name: return c
            raise KeyError(name)
        i, u, r = col("item"), col("user"), col("rating")
        mat = df.pivot(index=i, columns=u, values=r)
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat.index = mat.index.astype(int)
        mat.columns = mat.columns.astype(int)
        mat = mat.sort_index().sort_index(axis=1)
        return mat

    first_col = df.columns[0]
    if str(first_col).lower() in ("item", "items", "item_id", "id", "index", "unnamed: 0"):
        df = df.set_index(first_col)
    df.index.name = None
    df.columns.name = None
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    df = df.sort_index().sort_index(axis=1)
    return df

def _combine_single_similarity(S, alpha, clip_neg=True):
    S = np.asarray(S, dtype=float)
    S_eff = np.clip(S, 0.0, None) if clip_neg else S
    S_eff = np.power(S_eff, float(alpha))
    np.fill_diagonal(S_eff, 0.0)
    return S_eff

def _knn_predict_removed_with_S(X, XX, S, K=20, include_negative=False):
    X = np.asarray(X, dtype=float)
    XX = np.asarray(XX, dtype=float)
    S = np.asarray(S, dtype=float)
    n_items, n_users = XX.shape
    K_cap = min(int(K), max(0, n_users - 1))
    
    removed = (~np.isnan(X)) & (np.isnan(XX))
    if not np.any(removed): return {}, np.nan

    rated_mask = ~np.isnan(XX)
    user_means = np.nanmean(XX, axis=0)
    item_means = np.nanmean(XX, axis=1)
    global_mean = float(np.nanmean(XX)) if np.isfinite(np.nanmean(XX)) else 0.0

    preds = {}
    rows, cols = np.where(removed)
    
    for i, j in zip(rows, cols):
        cand = np.where(rated_mask[i])[0]
        cand = cand[cand != j]
        
        pred = global_mean
        if cand.size > 0 and K_cap > 0:
            sims = S[j, cand]
            if not include_negative:
                keep = np.where(sims > 0.0)[0]
                cand = cand[keep]
                sims = sims[keep]
            
            if cand.size > 0:
                K_eff = min(K_cap, cand.size)
                top = np.argsort(-sims)[:K_eff]
                nbrs = cand[top]
                w = sims[top]
                w_sum = np.sum(w)
                if np.isfinite(w_sum) and not np.isclose(w_sum, 0.0):
                    w = w / w_sum
                    pred = float(np.dot(w, XX[i, nbrs]))
                elif np.isfinite(user_means[j]): pred = float(user_means[j])
                elif np.isfinite(item_means[i]): pred = float(item_means[i])
            elif np.isfinite(user_means[j]): pred = float(user_means[j])
            elif np.isfinite(item_means[i]): pred = float(item_means[i])
        elif np.isfinite(user_means[j]): pred = float(user_means[j])
        elif np.isfinite(item_means[i]): pred = float(item_means[i])
            
        preds[(i, j)] = pred

    return preds, 0.0

def rmse_mad_on_test(pred_df, test_df):
    P = pred_df.values
    T = test_df.values
    mask = ~np.isnan(T)
    if not mask.any(): return np.nan, np.nan
    diff = P[mask] - T[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    mad = float(np.mean(np.abs(diff)))
    return rmse, mad

def precision_recall_at_n(pred, train, test, N, relevance_threshold=4.0):
    precisions, recalls = [], []
    for u_idx in range(pred.shape[1]):
        train_col = train.iloc[:, u_idx]
        test_col = test.iloc[:, u_idx]
        pred_col = pred.iloc[:, u_idx]
        
        cand_mask = train_col.isna() & ~test_col.isna()
        if not cand_mask.any(): continue
        
        relevant_mask = (test_col >= relevance_threshold) & cand_mask
        n_rel = int(relevant_mask.sum())
        
        scores = pred_col[cand_mask]
        topN_items = scores.sort_values(ascending=False).head(N).index
        n_hit = int(relevant_mask.loc[topN_items].sum()) if len(topN_items) else 0
        
        prec_u = n_hit / min(N, int(cand_mask.sum())) if int(cand_mask.sum()) > 0 else np.nan
        rec_u = n_hit / n_rel if n_rel > 0 else np.nan
        
        if not np.isnan(prec_u): precisions.append(prec_u)
        if not np.isnan(rec_u): recalls.append(rec_u)
    
    return (float(np.mean(precisions)) if precisions else np.nan, 
            float(np.mean(recalls)) if recalls else np.nan)

# ==================================================================================
# MAIN PROCESS
# ==================================================================================

for fold_num in FOLDS_TO_PROCESS:
    fold_id = f"fold_{fold_num:02d}"
    print(f"\n[{fold_id.upper()}] Loading Data...")
    
    fold_start = time.time()
    
    # Paths
    fold_data_dir = os.path.join(DATA_DIR, fold_id)
    fold_results_dir = os.path.join(RESULTS_DIR, fold_id)
    train_path = os.path.join(fold_data_dir, "train.csv")
    test_path = os.path.join(fold_data_dir, "test.csv")
    
    # 1. Load Data/History
    files = [f for f in os.listdir(fold_results_dir) 
             if f.startswith('alpha_optimization_history_') and f.endswith('.csv')]
    if not files: continue
    
    alpha_file = os.path.join(fold_results_dir, sorted(files)[-1])
    df_history = pd.read_csv(alpha_file)
    XX_train = _load_matrix_csv(train_path)
    XX_test = _load_matrix_csv(test_path)
    
    # 2. Determine Optimal Alpha for ALL Lambdas first (to remove duplicates)
    print(f"  Selecting optimal alphas for {len(LAMBDA_VALUES)} lambda values...")
    
    # Structure to hold selection: lambda_map[lambda] = dataframe of best configs
    lambda_selections = {}
    
    # Collect ALL unique (Method, K, Alpha) needed across ALL lambdas
    # We will compute predictions for these unique combinations ONLY.
    needed_evaluations = set() 
    
    for lam in LAMBDA_VALUES:
        df_history['new_penalty'] = lam * abs(df_history['alpha'] - 1.0)
        df_history['new_score'] = df_history['mse'] + df_history['new_penalty']
        
        # Best Alphas for this lambda
        best_idx = df_history.groupby(['method', 'K', 'TopN'])['new_score'].idxmin()
        best_df = df_history.loc[best_idx].copy()
        lambda_selections[lam] = best_df
        
        # Add to needed evals unique set: (Method, K, Alpha)
        # Note: TopN doesn't affect prediction, only metric calculation.
        # Prediction depends on (Method, K, Alpha).
        for _, row in best_df[['method', 'K', 'alpha']].iterrows():
            needed_evaluations.add((row['method'], row['K'], row['alpha']))
            
    # Convert to list for iteration
    tasks = sorted(list(needed_evaluations))
    print(f"  Optimization: Reduced {len(LAMBDA_VALUES) * len(tasks)} potential calls to {len(tasks)} unique predictions.")
    
    # 3. Batch Evaluation (Predict once, cache results)
    evaluation_cache = {} # Key: (Method, K, Alpha), Value: {pred_df, rmse, mad}
    
    print(f"  Executing {len(tasks)} unique predictions...")
    
    loaded_sims = {}
    
    for i, (method, k, alpha) in enumerate(tasks):
        if i % 10 == 0: print(f"    Progress: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%)", end='\r')
        
        # Load Sim
        if method not in loaded_sims:
            sim_path = os.path.join(fold_data_dir, f"sim_{method}.npy")
            if not os.path.exists(sim_path): continue
            loaded_sims[method] = np.load(sim_path)
        
        S_user = loaded_sims[method]
        S_alpha = _combine_single_similarity(S_user, alpha, clip_neg=True)
        
        preds_dict, _ = _knn_predict_removed_with_S(
            X=XX_test.values, XX=XX_train.values, S=S_alpha, K=k
        )
        
        # Create Pred DF
        pred_df = XX_test.copy()
        pred_df[:] = np.nan
        for (item_idx, user_idx), r in preds_dict.items():
            pred_df.iloc[item_idx, user_idx] = r
            
        rmse, mad = rmse_mad_on_test(pred_df, XX_test)
        
        evaluation_cache[(method, k, alpha)] = {
            'pred_df': pred_df,
            'rmse': rmse,
            'mad': mad
        }
        
    print(f"\n  Predictions complete. Assembling results...")
    
    # 4. Assemble Results for each Lambda (V2: WITH VALIDATION METRICS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for lam in LAMBDA_VALUES:
        best_df = lambda_selections[lam]
        results = []
        
        # Group by (Method, K, Alpha) to reuse cached predictions
        grouped = best_df.groupby(['method', 'K', 'alpha'])
        
        for (method, k, alpha), group in grouped:
            if (method, k, alpha) not in evaluation_cache: continue
            
            cached = evaluation_cache[(method, k, alpha)]
            test_rmse = cached['rmse']
            test_mad = cached['mad']
            
            # V2: Extract VALIDATION metrics from history for each TopN
            for topn in group['TopN'].unique():
                # Find corresponding row in history
                hist_match = df_history[
                    (df_history['method'] == method) &
                    (df_history['K'] == k) &
                    (df_history['TopN'] == topn) &
                    (df_history['alpha'] == alpha)
                ]
                
                if hist_match.empty:
                    print(f"⚠️  Warning: No history found for {method}, K={k}, TopN={topn}, α={alpha}")
                    continue
                
                hist_row = hist_match.iloc[0]
                reg_penalty = lam * abs(alpha - 1.0)
                reg_score = hist_row['mse'] + reg_penalty
                
                results.append({
                    'fold': fold_num,
                    'method': method,
                    'alpha': alpha,
                    'type': 'optimized',
                    'K': k,
                    'TopN': topn,
                    # V2: VALIDATION metrics (alpha selection basis)
                    'validation_mse': hist_row['mse'],
                    'validation_rmse': hist_row['rmse'],
                    'validation_precision': hist_row['precision'],
                    'validation_recall': hist_row['recall'],
                    'regularization_penalty': reg_penalty,
                    'regularized_score': reg_score,
                    # V2: TEST metrics (generalization evaluation)
                    'test_RMSE': test_rmse,
                    'test_MAD': test_mad,
                    'lambda': lam
                })
        
        # V2: Save with v2 suffix
        if results:
            output_path = os.path.join(fold_results_dir, f"grid_search_results_reeval_lambda_{lam}_v2_{timestamp}.csv")
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"    ✓ Saved lambda={lam} ({len(results)} records)")

    print(f"  ✅ Fold {fold_num} Done in {time.time()-fold_start:.1f}s")

print(f"\n========================================================")
print(f"ALL JOBS COMPLETE (V2)")
print(f"========================================================")
print(f"✅ Re-evaluation complete with VALIDATION + TEST metrics")
print(f"📁 Results saved with '_v2_' suffix")
