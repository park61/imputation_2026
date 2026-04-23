import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ==================================================================================
# CONFIGURATION
# ==================================================================================
REGULARIZATION_LAMBDA = 0.002
FOLDS_TO_PROCESS = [1, 2, 3, 4, 5]
RELEVANCE_THRESHOLD = 4.0
DATA_DIR = "data/movielenz_data"
RESULTS_DIR = "results"

print(f"========================================================")
print(f"SMART RE-EVALUATION START")
print(f"Target Lambda: {REGULARIZATION_LAMBDA}")
print(f"Target Folds: {FOLDS_TO_PROCESS}")
print(f"========================================================\n")

# ==================================================================================
# HELPER FUNCTIONS (Copied from notebook)
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

def _knn_predict_removed_with_S(X, XX, S, K=20, include_negative=False, fallback="user_mean"):
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

    return preds, 0.0 # MSE not needed here

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
    print(f"\n[{fold_id.upper()}] Processing...")
    
    # 1. Paths
    fold_data_dir = os.path.join(DATA_DIR, fold_id)
    fold_results_dir = os.path.join(RESULTS_DIR, fold_id)
    
    train_path = os.path.join(fold_data_dir, "train.csv")
    test_path = os.path.join(fold_data_dir, "test.csv")
    
    # Find latest alpha history
    files = [f for f in os.listdir(fold_results_dir) 
             if f.startswith('alpha_optimization_history_') and f.endswith('.csv')]
    if not files:
        print(f"Skipping {fold_id}: No alpha history found")
        continue
    alpha_file = os.path.join(fold_results_dir, sorted(files)[-1])
    
    # 2. Load Data
    print(f"  Loading history: {os.path.basename(alpha_file)}")
    df_history = pd.read_csv(alpha_file)
    
    print(f"  Loading matrices test/train...")
    XX_train = _load_matrix_csv(train_path)
    XX_test = _load_matrix_csv(test_path)
    
    # 3. Determine Optimal Alpha for New Lambda
    print(f"  Applying lambda={REGULARIZATION_LAMBDA} to selection...")
    df_history['new_penalty'] = REGULARIZATION_LAMBDA * abs(df_history['alpha'] - 1.0)
    df_history['new_score'] = df_history['mse'] + df_history['new_penalty']
    
    # Filter for Best Alpha
    best_indices = df_history.groupby(['method', 'K', 'TopN'])['new_score'].idxmin()
    best_configs = df_history.loc[best_indices].copy()
    
    # 4. Evaluate on Test (Smart way: Group by Method-Alpha to minimize matrix calcs)
    # We need to run prediction for each Unique (Method, Alpha) pair
    # Note: K is used in prediction too. So unique (Method, K, Alpha)
    
    unique_evals = best_configs[['method', 'K', 'alpha']].drop_duplicates()
    print(f"  Evaluations needed: {len(unique_evals)} unique (Method, K, Alpha) combinations")
    
    results = []
    completed = 0
    total = len(unique_evals)
    
    # Cache for loaded similarity matrices
    loaded_sims = {} 
    
    for _, row in unique_evals.iterrows():
        method, k, alpha = row['method'], row['K'], row['alpha']
        
        # Load Sim
        if method not in loaded_sims:
            sim_path = os.path.join(fold_data_dir, f"sim_{method}.npy")
            if not os.path.exists(sim_path):
                print(f"    Warning: Sim file for {method} not found")
                continue
            loaded_sims[method] = np.load(sim_path)
            
        S_user = loaded_sims[method]
        
        # Predict
        S_alpha = _combine_single_similarity(S_user, alpha, clip_neg=True)
        preds_dict, _ = _knn_predict_removed_with_S(
            X=XX_test.values,
            XX=XX_train.values,
            S=S_alpha,
            K=k,
            include_negative=False
        )
        
        # Determine RMSE/MAD
        pred_df = XX_test.copy()
        pred_df[:] = np.nan
        for (item_idx, user_idx), r in preds_dict.items():
            pred_df.iloc[item_idx, user_idx] = r
            
        rmse, mad = rmse_mad_on_test(pred_df, XX_test)
        
        # Calculate Precision/Recall for associated TopNs
        # Get all TopNs that use this (Method, K, Alpha) optimization result
        target_topns = best_configs[
            (best_configs['method'] == method) & 
            (best_configs['K'] == k) & 
            (best_configs['alpha'] == alpha)
        ]['TopN'].unique()
        
        for topn in target_topns:
            prec, rec = precision_recall_at_n(pred_df, XX_train, XX_test, topn, RELEVANCE_THRESHOLD)
            
            results.append({
                'fold': fold_num,
                'method': method,
                'alpha': alpha,
                'type': 'optimized',
                'K': k,
                'TopN': topn,
                'RMSE': rmse,
                'MAD': mad,
                'Precision': prec,
                'Recall': rec,
                'lambda': REGULARIZATION_LAMBDA # Add explicit lambda info
            })
            
            # Also add baseline (alpha=1.0) results if logic allows
            # But usually we just want to update the optimized part
            # For full compatibility, we might need baseline rows too.
            # Let's check how many baselines we need. 
            # Actually, baseline doesn't change with lambda, but to have a complete file,
            # we should compute it or copy it. Copying from previous results is safer/faster if available.
            # But here we will just Compute Optimized part for simplicity and correctness.
            # Users can merge with baseline later or we can compute baseline once per (Method, K).
            
        completed += 1
        if completed % 10 == 0:
            print(f"    Progress: {completed}/{total} ({completed/total*100:.1f}%)", end='\r')

    # Also compute Baseline (Alpha=1.0) for completeness
    # Group by (Method, K)
    unique_baselines = best_configs[['method', 'K']].drop_duplicates()
    print(f"\n  Computing Baselines (Alpha=1.0)... {len(unique_baselines)} combinations")
    
    for _, row in unique_baselines.iterrows():
        method, k = row['method'], row['K']
        if method not in loaded_sims: continue
        
        S_user = loaded_sims[method]
        S_one = _combine_single_similarity(S_user, 1.0, clip_neg=True)
        preds_dict, _ = _knn_predict_removed_with_S(XX_test.values, XX_train.values, S_one, K=k)
        
        pred_df = XX_test.copy()
        pred_df[:] = np.nan
        for (item_idx, user_idx), r in preds_dict.items():
            pred_df.iloc[item_idx, user_idx] = r
            
        rmse, mad = rmse_mad_on_test(pred_df, XX_test)
        
        # For all TopNs associated with this method/K
        relevant_topns = best_configs[(best_configs['method'] == method) & (best_configs['K'] == k)]['TopN'].unique()
        
        for topn in relevant_topns:
            prec, rec = precision_recall_at_n(pred_df, XX_train, XX_test, topn, RELEVANCE_THRESHOLD)
            results.append({
                'fold': fold_num,
                'method': method,
                'alpha': 1.0,
                'type': 'baseline',
                'K': k,
                'TopN': topn,
                'RMSE': rmse,
                'MAD': mad,
                'Precision': prec,
                'Recall': rec,
                'lambda': REGULARIZATION_LAMBDA
            })
            
    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(fold_results_dir, f"grid_search_results_reeval_lambda_{REGULARIZATION_LAMBDA}_{timestamp}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n  ✅ Saved: {os.path.basename(output_path)}")
    print(f"     Records: {len(results)}")

print(f"\n========================================================")
print(f"ALL JOBS COMPLETE")
print(f"========================================================")
