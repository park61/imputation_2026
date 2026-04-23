# -*- coding: utf-8 -*-
"""
main_experiment_v6_inner_sim.py
----------------------------------------------------------------------
v5 실험과 동일한 파이프라인이나, 유사도를 sim_inner_{method}.npy (train_inner
기반)에서 로딩. 이로써 Phase 1 alpha 최적화 시 validation 데이터 누수를 제거.

결과는 results/inner_sim/fold_XX/ 에 저장.

실행:
    python main_experiment_v6_inner_sim.py
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

# ── 설정 ──────────────────────────────────────────────────────────────────────
FOLDS_TO_RUN = list(range(2, 7))

K_RANGE    = range(10, 101, 10)
TOPN_RANGE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

RELEVANCE_THRESHOLD   = 4.0
ALPHA_COARSE_GRID     = np.arange(0.0, 10.5, 0.5)
ALPHA_FINE_RADIUS     = 0.5
ALPHA_FINE_STEP       = 0.05
REGULARIZATION_LAMBDA = 0.002

METHODS = [
    "acos","ami","ari","chebyshev","cosine","cpcc","euclidean",
    "ipwr","itr","jaccard","jmsd","kendall_tau_b","manhattan",
    "msd","pcc","spcc","src",
]

DATA_ROOT   = os.path.join(os.path.dirname(__file__), "data", "movielenz_data")
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results", "inner_sim")

# ── 헬퍼 함수들 ──────────────────────────────────────────────────────────────
def _load_matrix_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    lower = [str(c).lower() for c in df.columns]
    if {"item", "user", "rating"}.issubset(lower):
        def col(name):
            for c in df.columns:
                if str(c).lower() == name: return c
        mat = df.pivot(index=col("item"), columns=col("user"), values=col("rating"))
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat.index   = mat.index.astype(int)
        mat.columns = mat.columns.astype(int)
        return mat.sort_index().sort_index(axis=1)
    first_col = df.columns[0]
    if str(first_col).lower() in ("item","items","item_id","id","index","unnamed: 0"):
        df = df.set_index(first_col)
    df.index.name = None; df.columns.name = None
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index   = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.sort_index().sort_index(axis=1)


def _combine_single_similarity(S, alpha, clip_neg=True):
    S = np.asarray(S, dtype=float)
    S_eff = np.clip(S, 0.0, None) if clip_neg else S
    S_eff = np.power(S_eff, float(alpha))
    np.fill_diagonal(S_eff, 0.0)
    return S_eff


def _knn_predict_removed_with_S(X, XX, S, K=20, include_negative=False, fallback="user_mean"):
    X  = np.asarray(X,  dtype=float)
    XX = np.asarray(XX, dtype=float)
    S  = np.asarray(S,  dtype=float)
    n_items, n_users = XX.shape
    K_cap = min(int(K), max(0, n_users - 1))
    removed = (~np.isnan(X)) & (np.isnan(XX))
    if not np.any(removed):
        raise ValueError("No removed entries to evaluate")
    rated_mask  = ~np.isnan(XX)
    user_means  = np.nanmean(XX, axis=0)
    item_means  = np.nanmean(XX, axis=1)
    global_mean = float(np.nanmean(XX)) if np.isfinite(np.nanmean(XX)) else 0.0
    preds = {}; se_sum = 0.0; cnt = 0
    rows, cols = np.where(removed)
    for i, j in zip(rows, cols):
        cand = np.where(rated_mask[i])[0]
        cand = cand[cand != j]
        if cand.size == 0 or K_cap == 0:
            pred = float(user_means[j]) if np.isfinite(user_means[j]) else (
                   float(item_means[i]) if np.isfinite(item_means[i]) else global_mean)
        else:
            sims = S[j, cand]
            if not include_negative:
                keep = np.where(sims > 0.0)[0]
                cand = cand[keep]; sims = sims[keep]
            if cand.size == 0:
                pred = float(user_means[j]) if np.isfinite(user_means[j]) else (
                       float(item_means[i]) if np.isfinite(item_means[i]) else global_mean)
            else:
                K_eff = min(K_cap, cand.size)
                top   = np.argsort(-sims)[:K_eff]
                nbrs  = cand[top]; w = sims[top]
                w_sum = np.sum(w)
                if not np.isfinite(w_sum) or np.isclose(w_sum, 0.0):
                    pred = float(user_means[j]) if np.isfinite(user_means[j]) else global_mean
                else:
                    w    = w / w_sum
                    pred = float(np.dot(w, XX[i, nbrs]))
        preds[(i, j)] = pred
        se_sum += (pred - float(X[i, j])) ** 2; cnt += 1
    mse = se_sum / cnt if cnt > 0 else np.nan
    return preds, mse


def rmse_mad_on_test(pred: pd.DataFrame, test: pd.DataFrame) -> Tuple[float, float]:
    P = pred.values; T = test.values
    mask = ~np.isnan(T)
    if not mask.any(): return np.nan, np.nan
    diff = P[mask] - T[mask]
    return float(np.sqrt(np.mean(diff**2))), float(np.mean(np.abs(diff)))


def precision_recall_at_n(pred, train, test, N, relevance_threshold=4.0):
    precisions, recalls = [], []
    for u_idx in range(pred.shape[1]):
        train_col = train.iloc[:, u_idx]
        test_col  = test.iloc[:,  u_idx]
        pred_col  = pred.iloc[:,  u_idx]
        cand_mask    = train_col.isna() & ~test_col.isna()
        if not cand_mask.any(): continue
        relevant_mask = (test_col >= relevance_threshold) & cand_mask
        n_rel  = int(relevant_mask.sum())
        scores = pred_col[cand_mask]
        topN   = scores.sort_values(ascending=False).head(N).index
        n_hit  = int(relevant_mask.loc[topN].sum()) if len(topN) else 0
        prec_u = n_hit / min(N, int(cand_mask.sum())) if int(cand_mask.sum()) > 0 else np.nan
        rec_u  = n_hit / n_rel if n_rel > 0 else np.nan
        if not np.isnan(prec_u): precisions.append(prec_u)
        if not np.isnan(rec_u):  recalls.append(rec_u)
    return (float(np.mean(precisions)) if precisions else np.nan,
            float(np.mean(recalls))    if recalls    else np.nan)


# ── 메인 실험 ─────────────────────────────────────────────────────────────────
print("=" * 80)
print("  main_experiment_v6_inner_sim.py")
print("  Sim source: sim_inner_{method}.npy  (train_inner, no leakage)")
print("  Start: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 80 + "\n")

os.makedirs(os.path.join(RESULTS_ROOT, "combined"), exist_ok=True)

all_folds_results       = []
all_folds_alpha_history = []
global_start            = time.time()

for fold_num in FOLDS_TO_RUN:
    fold_id   = f"fold_{fold_num:02d}"
    fold_dir  = os.path.join(DATA_ROOT, fold_id)
    res_dir   = os.path.join(RESULTS_ROOT, fold_id)
    os.makedirs(res_dir, exist_ok=True)

    TRAIN_INNER_CSV = os.path.join(fold_dir, "train_inner.csv")
    VALIDATION_CSV  = os.path.join(fold_dir, "validation.csv")
    TRAIN_CSV       = os.path.join(fold_dir, "train.csv")
    TEST_CSV        = os.path.join(fold_dir, "test.csv")

    # 이미 완료된 fold 건너뜀
    existing = [f for f in os.listdir(res_dir)
                if f.startswith("grid_search_results_") and f.endswith(".csv")]
    if existing:
        print(f"[{fold_id}] Already done ({existing[-1]}) - skipping")
        df_ex = pd.read_csv(os.path.join(res_dir, sorted(existing)[-1]))
        df_ex['fold'] = fold_num
        all_folds_results.append(df_ex)
        continue

    for f in [TRAIN_INNER_CSV, VALIDATION_CSV, TRAIN_CSV, TEST_CSV]:
        if not os.path.exists(f):
            print(f"[{fold_id}] WARNING: {f} not found - skipping"); break
    else:
        pass

    print("\n" + "=" * 80)
    print(f"  FOLD {fold_num:02d}")
    print("=" * 80)

    XX_train_inner = _load_matrix_csv(TRAIN_INNER_CSV)
    XX_validation  = _load_matrix_csv(VALIDATION_CSV)
    XX_train_full  = _load_matrix_csv(TRAIN_CSV)
    XX_test        = _load_matrix_csv(TEST_CSV)

    print(f"  train_inner: {XX_train_inner.shape}  validation: {XX_validation.shape}")
    print(f"  train_full : {XX_train_full.shape}   test      : {XX_test.shape}")

    all_results   = []
    ALPHA_HISTORY = []
    fold_start    = time.time()
    completed     = 0
    total_exp     = len(METHODS) * len(K_RANGE) * len(TOPN_RANGE)

    for method in METHODS:
        sim_file = os.path.join(fold_dir, f"sim_inner_{method}.npy")
        if not os.path.exists(sim_file):
            print(f"  WARNING: {method}: sim_inner file not found - skipping (run precompute first)")
            continue

        S_user = np.load(sim_file)
        print(f"\n  [{method.upper()}]", flush=True)

        for k in K_RANGE:
            print(f"    K={k}: alpha 최적화...", end=" ", flush=True)

            # ── Phase 1: Coarse search ────────────────────────────────────────
            best_coarse_alpha = None
            best_coarse_score = np.inf

            for alpha in ALPHA_COARSE_GRID:
                S_alpha = _combine_single_similarity(S_user, alpha)
                preds_dict, mse = _knn_predict_removed_with_S(
                    X=XX_validation.values, XX=XX_train_inner.values,
                    S=S_alpha, K=k, include_negative=False, fallback="user_mean")
                reg_penalty = REGULARIZATION_LAMBDA * abs(alpha - 1.0)
                reg_score   = mse + reg_penalty
                pred_df = XX_validation.copy(); pred_df[:] = np.nan
                for (ii, jj), v in preds_dict.items():
                    pred_df.iloc[ii, jj] = v
                for topn in TOPN_RANGE:
                    prec, rec = precision_recall_at_n(
                        pred_df, XX_train_inner, XX_validation, topn, RELEVANCE_THRESHOLD)
                    ALPHA_HISTORY.append({
                        'fold': fold_num, 'method': method, 'K': k, 'TopN': topn,
                        'phase': 'coarse', 'alpha': alpha, 'mse': mse,
                        'rmse': np.sqrt(mse), 'regularization_penalty': reg_penalty,
                        'regularized_score': reg_score, 'precision': prec, 'recall': rec})
                if reg_score < best_coarse_score:
                    best_coarse_score = reg_score; best_coarse_alpha = alpha

            # ── Phase 1: Fine search ──────────────────────────────────────────
            fine_start_a = max(0.0, best_coarse_alpha - ALPHA_FINE_RADIUS)
            fine_end_a   = min(3.0, best_coarse_alpha + ALPHA_FINE_RADIUS)
            fine_grid    = np.arange(fine_start_a, fine_end_a + ALPHA_FINE_STEP/2, ALPHA_FINE_STEP)

            best_fine_alpha = best_coarse_alpha
            best_fine_score = best_coarse_score

            for alpha in fine_grid:
                S_alpha = _combine_single_similarity(S_user, alpha)
                preds_dict, mse = _knn_predict_removed_with_S(
                    X=XX_validation.values, XX=XX_train_inner.values,
                    S=S_alpha, K=k, include_negative=False, fallback="user_mean")
                reg_penalty = REGULARIZATION_LAMBDA * abs(alpha - 1.0)
                reg_score   = mse + reg_penalty
                pred_df = XX_validation.copy(); pred_df[:] = np.nan
                for (ii, jj), v in preds_dict.items():
                    pred_df.iloc[ii, jj] = v
                for topn in TOPN_RANGE:
                    prec, rec = precision_recall_at_n(
                        pred_df, XX_train_inner, XX_validation, topn, RELEVANCE_THRESHOLD)
                    ALPHA_HISTORY.append({
                        'fold': fold_num, 'method': method, 'K': k, 'TopN': topn,
                        'phase': 'fine', 'alpha': alpha, 'mse': mse,
                        'rmse': np.sqrt(mse), 'regularization_penalty': reg_penalty,
                        'regularized_score': reg_score, 'precision': prec, 'recall': rec})
                if reg_score < best_fine_score:
                    best_fine_score = reg_score; best_fine_alpha = alpha

            optimal_alpha = best_fine_alpha
            print(f"α*={optimal_alpha:.3f}", end="  ", flush=True)

            # ── Phase 2: Test evaluation ──────────────────────────────────────
            # Optimized
            S_opt = _combine_single_similarity(S_user, optimal_alpha)
            preds_opt, _ = _knn_predict_removed_with_S(
                X=XX_test.values, XX=XX_train_full.values,
                S=S_opt, K=k, include_negative=False, fallback="user_mean")
            pred_df_opt = XX_test.copy(); pred_df_opt[:] = np.nan
            for (ii, jj), v in preds_opt.items():
                pred_df_opt.iloc[ii, jj] = v
            test_rmse_opt, test_mad_opt = rmse_mad_on_test(pred_df_opt, XX_test)

            # Baseline (α=1)
            S_base = _combine_single_similarity(S_user, 1.0)
            preds_base, _ = _knn_predict_removed_with_S(
                X=XX_test.values, XX=XX_train_full.values,
                S=S_base, K=k, include_negative=False, fallback="user_mean")
            pred_df_base = XX_test.copy(); pred_df_base[:] = np.nan
            for (ii, jj), v in preds_base.items():
                pred_df_base.iloc[ii, jj] = v
            test_rmse_base, test_mad_base = rmse_mad_on_test(pred_df_base, XX_test)

            # ── Val metrics lookup ────────────────────────────────────────────
            val_metrics_for_alpha = {}
            for topn in TOPN_RANGE:
                hist_match = [h for h in ALPHA_HISTORY
                              if h['method'] == method and h['K'] == k
                              and h['TopN'] == topn and np.isclose(h['alpha'], optimal_alpha, atol=1e-6)]
                if hist_match:
                    val_metrics_for_alpha[topn] = hist_match[-1]

            # ── Store results ─────────────────────────────────────────────────
            for topn in TOPN_RANGE:
                test_prec_opt,  test_rec_opt  = precision_recall_at_n(
                    pred_df_opt,  XX_train_full, XX_test, topn, RELEVANCE_THRESHOLD)
                test_prec_base, test_rec_base = precision_recall_at_n(
                    pred_df_base, XX_train_full, XX_test, topn, RELEVANCE_THRESHOLD)
                vm = val_metrics_for_alpha.get(topn, {})
                all_results.append({
                    'fold': fold_num, 'method': method, 'alpha': optimal_alpha,
                    'type': 'optimized', 'K': k, 'TopN': topn,
                    'validation_mse': vm.get('mse', np.nan),
                    'validation_rmse': vm.get('rmse', np.nan),
                    'validation_precision': vm.get('precision', np.nan),
                    'validation_recall': vm.get('recall', np.nan),
                    'regularization_penalty': vm.get('regularization_penalty', np.nan),
                    'regularized_score': vm.get('regularized_score', np.nan),
                    'test_RMSE': test_rmse_opt, 'test_MAD': test_mad_opt,
                    'test_Precision': test_prec_opt, 'test_Recall': test_rec_opt,
                    'RMSE': test_rmse_opt, 'MAD': test_mad_opt,
                    'Precision': test_prec_opt, 'Recall': test_rec_opt,
                })
                all_results.append({
                    'fold': fold_num, 'method': method, 'alpha': 1.0,
                    'type': 'baseline', 'K': k, 'TopN': topn,
                    'validation_mse': np.nan, 'validation_rmse': np.nan,
                    'validation_precision': np.nan, 'validation_recall': np.nan,
                    'regularization_penalty': 0.0, 'regularized_score': np.nan,
                    'test_RMSE': test_rmse_base, 'test_MAD': test_mad_base,
                    'test_Precision': test_prec_base, 'test_Recall': test_rec_base,
                    'RMSE': test_rmse_base, 'MAD': test_mad_base,
                    'Precision': test_prec_base, 'Recall': test_rec_base,
                })
                completed += 1
            print(f"RMSE opt={test_rmse_opt:.4f} base={test_rmse_base:.4f}")

    # ── Fold 결과 저장 ────────────────────────────────────────────────────────
    if all_results:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_csv = os.path.join(res_dir, f"grid_search_results_{ts}.csv")
        alp_csv = os.path.join(res_dir, f"alpha_optimization_history_{ts}.csv")
        pd.DataFrame(all_results).to_csv(res_csv, index=False)
        pd.DataFrame(ALPHA_HISTORY).to_csv(alp_csv, index=False)
        print(f"\n  [SAVED] [{fold_id}] {res_csv}")
        print(f"  [SAVED] [{fold_id}] {alp_csv}")
        df_fold = pd.DataFrame(all_results)
        df_fold['fold'] = fold_num
        all_folds_results.append(df_fold)
        all_folds_alpha_history.append(pd.DataFrame(ALPHA_HISTORY))
    elapsed = time.time() - fold_start
    print(f"  [TIME] [{fold_id}] elapsed: {elapsed/60:.1f} min")

# ── 전체 결합 저장 ────────────────────────────────────────────────────────────
if all_folds_results:
    combined_dir = os.path.join(RESULTS_ROOT, "combined")
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_combined = pd.concat(all_folds_results, ignore_index=True)
    df_combined.to_csv(os.path.join(combined_dir, f"all_folds_grid_results_{ts}.csv"), index=False)
    if all_folds_alpha_history:
        df_hist = pd.concat(all_folds_alpha_history, ignore_index=True)
        df_hist.to_csv(os.path.join(combined_dir, f"all_folds_alpha_history_{ts}.csv"), index=False)
    print(f"\n[DONE] All results saved: results/inner_sim/combined/")

print("\n" + "=" * 80)
print(f"  DONE  Total elapsed: {(time.time()-global_start)/60:.1f} min")
print("  End: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 80)
