# -*- coding: utf-8 -*-
"""
analyze_overfitting_lambda0.py
=============================================================
과적합에 대한 실험 정리

[Phase 1] λ=0 조건에서 fold별 α 선택 분포 분석
  - 각 (fold, method, K)에서 λ=0으로 선택된 α 확인
  - fold 간 α 분산이 크면 → 과적합 신호
  - 테스트 성능 분석 (현재 lambda=0.002 vs lambda=0)

[Phase 2] 새 페날티: λ · |avg_α − α| 테스트
  - Phase 1에서 구한 avg_α (LOO: 다른 9개 fold 평균)를 페날티 중심으로
  - 동일 λ=0.002, 중심만 α=1 → avg_α 로 교체
  - 테스트 성능 비교: baseline / current(λ=0.002,α₀=1) / λ=0 / new(λ=0.002,α₀=avg_α)

CONFIGURATION:
  PHASE1_ONLY = True   → 예측 없이 α 분포 분석만 (빠름)
  PHASE1_ONLY = False  → 예측까지 포함한 전체 분석 (느림, ~fold당 수십분)
=============================================================
"""
import os, glob, time
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PHASE1_ONLY     = False          # True: α분포만, False: 예측까지 포함
LAMBDA_CENTERED = 0.002          # 새 페날티에 사용할 λ (avg_α 중심)
LAMBDA_ORIGINAL = 0.002          # 기존 페날티 λ
RELEVANCE_THRESHOLD = 4.0
K_RANGE    = range(10, 101, 10)
TOPN_RANGE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
METHODS    = [
    "acos","ami","ari","chebyshev","cosine","cpcc","euclidean",
    "ipwr","itr","jaccard","jmsd","kendall_tau_b","manhattan",
    "msd","pcc","spcc","src",
]

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE, "data", "movielenz_data")
RES_ROOT  = os.path.join(BASE, "results", "inner_sim")
ALPHA_COARSE_GRID = np.arange(0.0, 10.5, 0.5)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: 최신 파일 선택
# ─────────────────────────────────────────────────────────────────────────────
def _latest(fold_num: int, prefix: str) -> str:
    d = os.path.join(RES_ROOT, f"fold_{fold_num:02d}")
    files = sorted(glob.glob(os.path.join(d, f"{prefix}*.csv")))
    if not files:
        raise FileNotFoundError(f"No file matching {prefix}* in {d}")
    return files[-1]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: 행렬 로드
# ─────────────────────────────────────────────────────────────────────────────
def _load_matrix(path: str) -> pd.DataFrame:
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
    first = df.columns[0]
    if str(first).lower() in ("item","items","item_id","id","index","unnamed: 0"):
        df = df.set_index(first)
    df.index.name = None; df.columns.name = None
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index   = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.sort_index().sort_index(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: KNN 예측 (test 전용)
# ─────────────────────────────────────────────────────────────────────────────
def _predict_test(XX_train_np, XX_test_np, S_alpha, K):
    """XX_train 기반 예측, XX_test의 실제값과 비교"""
    n_items, n_users = XX_train_np.shape
    K_cap = min(int(K), max(0, n_users - 1))
    removed = ~np.isnan(XX_test_np)          # test에 값이 있는 위치
    rated   = ~np.isnan(XX_train_np)         # train에서 이미 평점 있는 위치
    user_means  = np.nanmean(XX_train_np, axis=0)
    item_means  = np.nanmean(XX_train_np, axis=1)
    global_mean = float(np.nanmean(XX_train_np)) if np.isfinite(np.nanmean(XX_train_np)) else 3.0

    pred_mat = np.full_like(XX_test_np, np.nan)
    rows, cols = np.where(removed)
    for i, j in zip(rows, cols):
        cand = np.where(rated[i])[0]
        if cand.size == 0 or K_cap == 0:
            pred = (float(user_means[j]) if np.isfinite(user_means[j])
                    else float(item_means[i]) if np.isfinite(item_means[i])
                    else global_mean)
        else:
            sims = S_alpha[j, cand]
            keep = np.where(sims > 0.0)[0]
            cand2, sims2 = cand[keep], sims[keep]
            if cand2.size == 0:
                pred = (float(user_means[j]) if np.isfinite(user_means[j])
                        else float(item_means[i]) if np.isfinite(item_means[i])
                        else global_mean)
            else:
                K_eff = min(K_cap, cand2.size)
                top   = np.argsort(-sims2)[:K_eff]
                w     = sims2[top]
                ws    = np.sum(w)
                if not np.isfinite(ws) or np.isclose(ws, 0.0):
                    pred = float(user_means[j]) if np.isfinite(user_means[j]) else global_mean
                else:
                    pred = float(np.dot(w / ws, XX_train_np[i, cand2[top]]))
        pred_mat[i, j] = pred
    return pred_mat

def _rmse_mad(pred_mat, test_np):
    mask = ~np.isnan(test_np)
    if not mask.any(): return np.nan, np.nan
    d = pred_mat[mask] - test_np[mask]
    return float(np.sqrt(np.mean(d**2))), float(np.mean(np.abs(d)))

def _precision_recall(pred_mat, train_np, test_np, N, thr=4.0):
    precs, recs = [], []
    for j in range(pred_mat.shape[1]):
        cand = np.where(np.isnan(train_np[:, j]) & ~np.isnan(test_np[:, j]))[0]
        if len(cand) == 0: continue
        n_rel = int(np.sum(test_np[cand, j] >= thr))
        if len(cand) == 0: continue
        scores = pred_mat[cand, j]
        valid  = ~np.isnan(scores)
        if not valid.any(): continue
        top_idx = cand[np.argsort(-scores[valid])[:N]]
        n_hit   = int(np.sum(test_np[top_idx, j] >= thr))
        precs.append(n_hit / min(N, int(valid.sum())))
        if n_rel > 0: recs.append(n_hit / n_rel)
    return (float(np.mean(precs)) if precs else np.nan,
            float(np.mean(recs))  if recs  else np.nan)

def _sim_power(S, alpha):
    S2 = np.clip(np.asarray(S, dtype=float), 0.0, None)
    S2 = np.power(S2, float(alpha))
    np.fill_diagonal(S2, 0.0)
    return S2

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: α 분포 분석
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  analyze_overfitting_lambda0.py")
print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\n" + "═" * 80)
print("  PHASE 1: λ=0 조건에서 fold별 α 선택 분포 분석")
print("═" * 80)

# 각 fold의 history 로드 (coarse grid만 사용)
fold_alpha_lam0 = {}   # fold_num → DataFrame[(method, K)] → best_alpha

all_hist_rows = []
for fold_num in range(1, 11):
    path = _latest(fold_num, "alpha_optimization_history_")
    df   = pd.read_csv(path)
    # coarse grid에서 λ=0 best α: mse 기준 argmin, (method, K) 단위 (MSE는 TopN 무관)
    coarse = df[df['phase'] == 'coarse'].drop_duplicates(['method', 'K', 'alpha'])
    best   = coarse.loc[coarse.groupby(['method', 'K'])['mse'].idxmin()].copy()
    best['fold'] = fold_num
    fold_alpha_lam0[fold_num] = best[['fold','method','K','alpha','mse']].reset_index(drop=True)
    all_hist_rows.append(best)
    print(f"  fold_{fold_num:02d}: loaded {path.split('/')[-1]}")

df_lam0 = pd.concat(all_hist_rows, ignore_index=True)
df_lam0.rename(columns={'alpha': 'alpha_lam0', 'mse': 'val_mse_lam0'}, inplace=True)

print("\n─── [1-A] 메서드별 λ=0 선택된 α 통계 (10-fold 평균) ───────────────────────")
stats = df_lam0.groupby('method')['alpha_lam0'].agg(
    mean='mean', std='std', min='min', max='max',
    pct25=lambda x: x.quantile(0.25), median='median', pct75=lambda x: x.quantile(0.75)
).round(3)
stats['range'] = stats['max'] - stats['min']
stats = stats.sort_values('std', ascending=False)
print(f"\n{'Method':<18} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'Range':>6} {'Median':>6}")
print("-" * 60)
for m, row in stats.iterrows():
    flag = " ◀ HIGH VAR" if row['std'] > 1.0 else (" ◀ MED VAR" if row['std'] > 0.5 else "")
    print(f"  {m:<16} {row['mean']:>6.3f} {row['std']:>6.3f} {row['min']:>6.2f} {row['max']:>6.2f} "
          f"{row['range']:>6.2f} {row['median']:>6.3f}{flag}")

print("\n─── [1-B] K별 α 분산 (method 평균) ───────────────────────────────────────")
k_stats = df_lam0.groupby('K')['alpha_lam0'].agg(mean='mean', std='std').round(3)
print(f"\n{'K':>5} {'Mean α':>8} {'Std α':>8}")
print("-" * 25)
for k, row in k_stats.iterrows():
    print(f"  {k:>3}   {row['mean']:>8.3f}  {row['std']:>8.3f}")

print("\n─── [1-C] fold × method α 선택 행렬 (K=50 기준) ────────────────────────────")
df_k50 = df_lam0[df_lam0['K'] == 50].pivot(index='fold', columns='method', values='alpha_lam0')
# 열 정렬: std 내림차순
col_order = stats.index.tolist()
df_k50 = df_k50[[c for c in col_order if c in df_k50.columns]]
header = f"{'Fold':>5}  " + "  ".join(f"{m[:9]:>9}" for m in df_k50.columns)
print(header)
print("-" * len(header))
for fold, row in df_k50.iterrows():
    vals = "  ".join(f"{v:>9.2f}" for v in row.values)
    print(f"  {fold:>3}    {vals}")
row_std = df_k50.std()
print("-" * len(header))
print(f"  {'Std':>3}    " + "  ".join(f"{v:>9.3f}" for v in row_std.values))

print("\n─── [1-D] λ=0 vs λ=0.002 평균 α 비교 (method별) ───────────────────────────")
# λ=0.002 결과 로드 (기존 grid_search_results)
def _latest_grid(fold_num):
    d = os.path.join(RES_ROOT, f"fold_{fold_num:02d}")
    files = sorted(glob.glob(os.path.join(d, "grid_search_results_[0-9]*.csv")))
    return files[-1] if files else None

rows_lam002 = []
for fn in range(1, 11):
    p = _latest_grid(fn)
    if p is None: continue
    df = pd.read_csv(p)
    opt = df[df['type'] == 'optimized']
    # alpha는 (method, K) 단위로 동일 (TopN 무관)
    best = opt.drop_duplicates(['method', 'K'])[['method','K','alpha']].copy()
    best['fold'] = fn
    rows_lam002.append(best)

df_lam002 = pd.concat(rows_lam002, ignore_index=True)
df_lam002.rename(columns={'alpha': 'alpha_lam002'}, inplace=True)

merged = pd.merge(
    df_lam0.groupby('method')['alpha_lam0'].mean().reset_index(),
    df_lam002.groupby('method')['alpha_lam002'].mean().reset_index(),
    on='method'
)
merged['delta'] = merged['alpha_lam0'] - merged['alpha_lam002']
merged = merged.sort_values('delta', ascending=False)
print(f"\n{'Method':<18} {'λ=0 avg α':>10} {'λ=0.002 avg α':>14} {'Δ':>8}")
print("-" * 55)
for _, row in merged.iterrows():
    flag = " ◀ OVERFIT" if abs(row['delta']) > 0.5 else ""
    print(f"  {row['method']:<16} {row['alpha_lam0']:>10.3f} {row['alpha_lam002']:>14.3f} "
          f"{row['delta']:>8.3f}{flag}")

if PHASE1_ONLY:
    print("\n[PHASE1_ONLY=True] Phase 2 건너뜀.")
    import sys; sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# LOO avg_α 계산 (per method, K)
# fold i에 대해: avg_α_{m,K}^{-i} = mean(α_{j,m,K} for j≠i)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 80)
print("  PHASE 2: 테스트 성능 재평가")
print(f"  조건 1: λ=0     (페날티 없음, 순수 MSE 최소화)")
print(f"  조건 2: λ={LAMBDA_CENTERED}  (avg_α LOO 중심 페날티)")
print("═" * 80)

# (fold, method, K, alpha_lam0) 테이블
df_lam0_mk = df_lam0[['fold','method','K','alpha_lam0']].copy()

# LOO avg_α
lam0_ref = df_lam0_mk.copy()
loo_list  = []
for fold_num in range(1, 11):
    other = lam0_ref[lam0_ref['fold'] != fold_num]
    avg   = other.groupby(['method','K'])['alpha_lam0'].mean().reset_index()
    avg.rename(columns={'alpha_lam0': 'avg_alpha_loo'}, inplace=True)
    avg['fold'] = fold_num
    loo_list.append(avg)
df_loo = pd.concat(loo_list, ignore_index=True)

# LOO avg_α 요약 출력
print("\n─── [2-A] LOO avg_α 요약 (method별, K 평균) ─────────────────────────────")
loo_method = df_loo.groupby('method')['avg_alpha_loo'].agg(['mean','std']).round(3)
print(f"\n{'Method':<18} {'avg_α (LOO mean)':>16} {'std across K':>14}")
print("-" * 53)
for m, row in loo_method.sort_values('mean', ascending=False).iterrows():
    print(f"  {m:<16} {row['mean']:>16.3f} {row['std']:>14.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: 각 fold에서 예측 실행
# ─────────────────────────────────────────────────────────────────────────────
# history에서 최적 α 재선택: 새 페날티 λ*|α - avg_α_loo|
# coarse grid + 해당 fold의 mse
all_fold_results = []

for fold_num in range(1, 11):
    fold_start = time.time()
    fold_id    = f"fold_{fold_num:02d}"
    print(f"\n{'='*60}")
    print(f"  [{fold_id}] 예측 실행 중...")

    fold_dir   = os.path.join(DATA_ROOT, fold_id)
    train_path = os.path.join(fold_dir, "train.csv")
    test_path  = os.path.join(fold_dir, "test.csv")

    XX_train = _load_matrix(train_path).values.astype(float)
    XX_test  = _load_matrix(test_path).values.astype(float)

    # 이 fold의 history (coarse only)
    path_hist = _latest(fold_num, "alpha_optimization_history_")
    df_hist_f = pd.read_csv(path_hist)
    coarse_f  = df_hist_f[df_hist_f['phase'] == 'coarse'].drop_duplicates(['method','K','alpha'])

    # LOO avg_α for this fold
    loo_f = df_loo[df_loo['fold'] == fold_num][['method','K','avg_alpha_loo']]

    # 기존 λ=0.002 selected α (from grid_search_results)
    path_grid = _latest_grid(fold_num)
    df_grid_f = pd.read_csv(path_grid) if path_grid else None

    for method in METHODS:
        sim_path = os.path.join(fold_dir, f"sim_inner_{method}.npy")
        if not os.path.exists(sim_path):
            print(f"  WARNING: {sim_path} not found, skip {method}")
            continue
        S_raw = np.load(sim_path)

        for k in K_RANGE:
            # ── 1. λ=0 최적 α (coarse) ──────────────────────────────────────
            coarse_mk = coarse_f[(coarse_f['method'] == method) & (coarse_f['K'] == k)]
            if coarse_mk.empty: continue
            alpha_lam0_val = float(coarse_mk.loc[coarse_mk['mse'].idxmin(), 'alpha'])

            # ── 2. 새 페날티: λ·|α - avg_α_loo| ────────────────────────────
            loo_row = loo_f[(loo_f['method'] == method) & (loo_f['K'] == k)]
            avg_alpha_loo = float(loo_row['avg_alpha_loo'].values[0]) if not loo_row.empty else 1.0
            coarse_mk = coarse_mk.copy()
            coarse_mk['new_penalty'] = LAMBDA_CENTERED * np.abs(coarse_mk['alpha'] - avg_alpha_loo)
            coarse_mk['new_score']   = coarse_mk['mse'] + coarse_mk['new_penalty']
            alpha_new_val = float(coarse_mk.loc[coarse_mk['new_score'].idxmin(), 'alpha'])

            # ── 3. 기존 λ=0.002 α (grid_search_results에서) ─────────────────
            if df_grid_f is not None:
                mask_opt = ((df_grid_f['method'] == method) & (df_grid_f['K'] == k) &
                            (df_grid_f['type'] == 'optimized'))
                alpha_lam002_val = (float(df_grid_f[mask_opt]['alpha'].values[0])
                                    if mask_opt.any() else 1.0)
            else:
                alpha_lam002_val = 1.0

            # ── 4. 필요한 고유 (alpha) 값만 예측 ────────────────────────────
            alphas_needed = {alpha_lam0_val, alpha_new_val, alpha_lam002_val, 1.0}
            pred_cache = {}
            for a in alphas_needed:
                S_a = _sim_power(S_raw, a)
                pred_cache[a] = _predict_test(XX_train, XX_test, S_a, k)

            # ── 5. 각 TopN에 대해 메트릭 계산 ───────────────────────────────
            for topn in TOPN_RANGE:
                for cond_name, a_val in [
                    ("baseline",   1.0),
                    ("lam002_a1",  alpha_lam002_val),
                    ("lam0",       alpha_lam0_val),
                    ("lam_avgalpha", alpha_new_val),
                ]:
                    pm = pred_cache[a_val]
                    rmse, mad = _rmse_mad(pm, XX_test)
                    prec, rec = _precision_recall(pm, XX_train, XX_test, topn, RELEVANCE_THRESHOLD)
                    all_fold_results.append({
                        'fold': fold_num, 'method': method, 'K': k, 'TopN': topn,
                        'condition': cond_name,
                        'alpha': a_val,
                        'avg_alpha_loo': avg_alpha_loo,
                        'test_RMSE': rmse, 'test_MAD': mad,
                        'test_Precision': prec, 'test_Recall': rec,
                    })

    elapsed = time.time() - fold_start
    print(f"  [{fold_id}] Done. elapsed: {elapsed/60:.1f} min")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 결과 정리 및 출력
# ─────────────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_fold_results)

# 저장
ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
out_p = os.path.join(BASE, f"overfitting_analysis_results_{ts}.csv")
df_res.to_csv(out_p, index=False)
print(f"\n[SAVED] {out_p}")

# ── 전체 요약 (모든 fold, K, TopN 평균) ──────────────────────────────────────
print("\n" + "═" * 80)
print("  PHASE 2 전체 요약")
print("═" * 80)

# 조건 이름
COND_LABELS = {
    "baseline":     "Baseline (α=1)",
    "lam002_a1":    "λ=0.002/α₀=1   (현행)",
    "lam0":         "λ=0   (과적합)",
    "lam_avgalpha": f"λ={LAMBDA_CENTERED}/α₀=avg_α (신규)",
}

print("\n─── [2-B] 조건별 전체 평균 성능 ────────────────────────────────────────────")
overall = (df_res.groupby('condition')[['test_RMSE','test_MAD','test_Precision','test_Recall']]
           .mean().round(5))
print(f"\n{'Condition':<30} {'RMSE':>8} {'MAD':>8} {'Precision':>10} {'Recall':>8}")
print("-" * 68)
for cond in ["baseline","lam002_a1","lam0","lam_avgalpha"]:
    if cond not in overall.index: continue
    row  = overall.loc[cond]
    rmse_base = overall.loc['baseline', 'test_RMSE']
    delta = (row['test_RMSE'] - rmse_base) / rmse_base * 100
    label = COND_LABELS[cond]
    print(f"  {label:<28} {row['test_RMSE']:>8.5f} {row['test_MAD']:>8.5f} "
          f"{row['test_Precision']:>10.5f} {row['test_Recall']:>8.5f}  ({delta:+.2f}%)")

print("\n─── [2-C] fold별 RMSE 비교 ─────────────────────────────────────────────────")
fold_rmse = df_res.groupby(['fold','condition'])['test_RMSE'].mean().unstack('condition')
print(f"\n{'Fold':>5}  {'Baseline':>10} {'λ=0.002/α=1':>12} {'λ=0':>10} {'λ/avg_α':>10}  {'Δ(λ=0)%':>9} {'Δ(avg_α)%':>10}")
print("-" * 75)
for fold in range(1, 11):
    if fold not in fold_rmse.index: continue
    r   = fold_rmse.loc[fold]
    d0  = (r.get('lam0', np.nan)          - r.get('baseline', np.nan)) / r.get('baseline', np.nan) * 100
    d_a = (r.get('lam_avgalpha', np.nan)  - r.get('baseline', np.nan)) / r.get('baseline', np.nan) * 100
    print(f"  {fold:>3}    {r.get('baseline',np.nan):>10.5f} {r.get('lam002_a1',np.nan):>12.5f} "
          f"{r.get('lam0',np.nan):>10.5f} {r.get('lam_avgalpha',np.nan):>10.5f}  "
          f"{d0:>9.2f}% {d_a:>10.2f}%")

avg_row = fold_rmse.mean()
d0_avg  = (avg_row.get('lam0', np.nan) - avg_row.get('baseline', np.nan)) / avg_row.get('baseline', np.nan) * 100
da_avg  = (avg_row.get('lam_avgalpha', np.nan) - avg_row.get('baseline', np.nan)) / avg_row.get('baseline', np.nan) * 100
print("-" * 75)
print(f"  {'AVG':>3}    {avg_row.get('baseline',np.nan):>10.5f} {avg_row.get('lam002_a1',np.nan):>12.5f} "
      f"{avg_row.get('lam0',np.nan):>10.5f} {avg_row.get('lam_avgalpha',np.nan):>10.5f}  "
      f"{d0_avg:>9.2f}% {da_avg:>10.2f}%")

print("\n─── [2-D] method별 RMSE 비교 ────────────────────────────────────────────────")
meth_rmse = df_res.groupby(['method','condition'])['test_RMSE'].mean().unstack('condition').round(5)
print(f"\n{'Method':<18} {'Baseline':>10} {'λ=0.002/α=1':>12} {'λ=0':>10} {'λ/avg_α':>10}  "
      f"{'Δ(λ=0.002)%':>12} {'Δ(λ=0)%':>9} {'Δ(avg_α)%':>10}")
print("-" * 90)
for m in sorted(meth_rmse.index):
    r   = meth_rmse.loc[m]
    d002 = (r.get('lam002_a1',np.nan)    - r.get('baseline',np.nan)) / r.get('baseline',np.nan) * 100
    d0   = (r.get('lam0',np.nan)          - r.get('baseline',np.nan)) / r.get('baseline',np.nan) * 100
    da   = (r.get('lam_avgalpha',np.nan)  - r.get('baseline',np.nan)) / r.get('baseline',np.nan) * 100
    print(f"  {m:<16} {r.get('baseline',np.nan):>10.5f} {r.get('lam002_a1',np.nan):>12.5f} "
          f"{r.get('lam0',np.nan):>10.5f} {r.get('lam_avgalpha',np.nan):>10.5f}  "
          f"{d002:>12.2f}% {d0:>9.2f}% {da:>10.2f}%")

print("\n─── [2-E] α 선택값 비교 (method별, K 평균) ─────────────────────────────────")
alpha_cmp = df_res.drop_duplicates(['fold','method','K','condition'])
alpha_cmp = alpha_cmp.groupby(['method','condition'])['alpha'].mean().unstack('condition').round(3)
print(f"\n{'Method':<18} {'Baseline':>10} {'λ=0.002/α=1':>12} {'λ=0':>10} {'λ/avg_α':>10}")
print("-" * 55)
for m in sorted(alpha_cmp.index):
    r = alpha_cmp.loc[m]
    print(f"  {m:<16} {r.get('baseline',np.nan):>10.3f} {r.get('lam002_a1',np.nan):>12.3f} "
          f"{r.get('lam0',np.nan):>10.3f} {r.get('lam_avgalpha',np.nan):>10.3f}")

print("\n─── [2-F] 과적합 지표 요약 ──────────────────────────────────────────────────")
# λ=0에서 각 method별 fold간 alpha std
alpha_std_lam0 = (df_res[df_res['condition']=='lam0']
                  .drop_duplicates(['fold','method','K'])
                  .groupby('method')['alpha']
                  .std().round(3).sort_values(ascending=False))
print("\n  [λ=0 fold간 α 표준편차 — 높을수록 과적합 심각]")
print(f"\n{'Method':<18} {'α Std (λ=0)':>12}  {'해석':>10}")
print("-" * 45)
for m, std_val in alpha_std_lam0.items():
    if std_val > 2.0:    label = "심각 ◀◀"
    elif std_val > 1.0:  label = "높음  ◀"
    elif std_val > 0.5:  label = "보통"
    else:                label = "낮음"
    print(f"  {m:<16} {std_val:>12.3f}  {label}")

print("\n" + "=" * 80)
print(f"  DONE  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
