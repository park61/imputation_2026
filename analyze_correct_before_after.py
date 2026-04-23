"""
수정된 Before/After 비교:
  BUGGY λ=0 (archive _lambda_0.0_v1_* 파일들) vs FIXED λ=0 (새 실험)
phase5_analysis.py 의 오류 수정 버전
"""
import pandas as pd
import numpy as np
import glob
import os

ARCHIVE_DIR = "results/archive_buggy_20260415"
INNER_SIM_DIR = "results/inner_sim"
BUGGY_METHODS = ["msd", "jmsd", "acos"]

# ── 1. 버기 λ=0 데이터 로드 (아카이브에서 _lambda_0.0_v1_ 파일) ───────────────
lam0_files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "fold_*",
                                            "grid_search_results_inner_lambda_0.0_v1_*.csv")))
print(f"버기 λ=0 파일 수: {len(lam0_files)}")

buggy_dfs = []
for f in lam0_files:
    df = pd.read_csv(f)
    fold_num = int(os.path.basename(os.path.dirname(f)).split("_")[1])
    df["fold"] = fold_num
    buggy_dfs.append(df)
buggy_df = pd.concat(buggy_dfs, ignore_index=True)
print(f"버기 λ=0 전체 rows: {len(buggy_df)}, folds: {sorted(buggy_df['fold'].unique())}")
print(f"methods: {sorted(buggy_df['method'].unique())}")
print()

# ── 2. 수정된 λ=0 데이터 로드 ────────────────────────────────────────────────
fixed_rows = []
for fold_num in range(1, 11):
    fold_dir = f"{INNER_SIM_DIR}/fold_{fold_num:02d}"
    files = sorted(glob.glob(os.path.join(fold_dir, "grid_search_results_202604*.csv")))
    files = [f for f in files if "_lambda_" not in os.path.basename(f)]
    if not files:
        print(f"WARNING: fold_{fold_num:02d} fixed 파일 없음")
        continue
    df = pd.read_csv(files[-1])
    df["fold"] = fold_num
    fixed_rows.append(df)
fixed_df = pd.concat(fixed_rows, ignore_index=True)
print(f"수정된 λ=0 전체 rows: {len(fixed_df)}, folds: {sorted(fixed_df['fold'].unique())}")
print()

# ── 3. 비교 함수 ───────────────────────────────────────────────────────────────
def method_K_summary(df, method, type_filter="optimized"):
    sub = df[(df['method'] == method) & (df['type'] == type_filter)]
    return sub.groupby(['fold', 'K']).agg(
        alpha=('alpha', 'mean'),
        val_rmse=('validation_rmse', 'mean'),
        test_rmse=('test_RMSE', 'mean')
    ).reset_index()

print("=" * 70)
print("  올바른 Before/After 비교: 버기 λ=0 vs 수정된 λ=0 (10-fold 평균)")
print("=" * 70)

for method in BUGGY_METHODS:
    buggy_s = method_K_summary(buggy_df, method)
    fixed_s = method_K_summary(fixed_df, method)

    # 10-fold 평균
    buggy_K = buggy_s.groupby('K').agg(alpha_buggy=('alpha', 'mean'),
                                         rmse_buggy=('test_rmse', 'mean'))
    fixed_K = fixed_s.groupby('K').agg(alpha_fixed=('alpha', 'mean'),
                                         rmse_fixed=('test_rmse', 'mean'))
    comp = buggy_K.join(fixed_K)
    comp['delta_rmse'] = comp['rmse_fixed'] - comp['rmse_buggy']

    print(f"\n  [{method.upper()}]")
    print(f"  {'K':>6}  {'α*(buggy)':>10}  {'α*(fixed)':>10}  {'RMSE_buggy':>12}  {'RMSE_fixed':>12}  {'Δ RMSE':>10}")
    print("  " + "-" * 68)
    for k, row in comp.iterrows():
        delta_str = f"{row['delta_rmse']:+.4f}"
        marker = " ↓" if row['delta_rmse'] < 0 else " ↑"
        print(f"  {k:>6d}  {row['alpha_buggy']:>10.3f}  {row['alpha_fixed']:>10.3f}  "
              f"{row['rmse_buggy']:>12.4f}  {row['rmse_fixed']:>12.4f}  {delta_str}{marker}")

print()
print("=" * 70)
print("  추가: phase5에서 사용한 잘못된 비교 (버기 λ=0.002 vs 수정된 λ=0)")
print("=" * 70)

# 아카이브에서 lambda=0.002가 아닌 기본 파일 (March 20260311 원본)
orig_files = [f for f in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "fold_*", "grid_search_results_2*.csv")))
              if "_lambda_" not in os.path.basename(f)]

# fold당 1개만 사용 (중복 제거: fold_01만 1개, fold_02~05는 3개씩)
orig_dfs = {}
for f in orig_files:
    fold_num = int(os.path.basename(os.path.dirname(f)).split("_")[1])
    if fold_num not in orig_dfs:
        df = pd.read_csv(f)
        df["fold"] = fold_num
        orig_dfs[fold_num] = df
orig_df = pd.concat(list(orig_dfs.values()), ignore_index=True)
print(f"\n원본 아카이브(λ≠0) folds: {sorted(orig_df['fold'].unique())}")

for method in BUGGY_METHODS:
    orig_s = method_K_summary(orig_df, method)
    fixed_s = method_K_summary(fixed_df, method)
    orig_K = orig_s.groupby('K').agg(alpha_orig=('alpha', 'mean'), rmse_orig=('test_rmse', 'mean'))
    fixed_K = fixed_s.groupby('K').agg(alpha_fixed=('alpha', 'mean'), rmse_fixed=('test_rmse', 'mean'))
    comp = orig_K.join(fixed_K)
    comp['delta'] = comp['rmse_fixed'] - comp['rmse_orig']
    print(f"\n  [{method.upper()}] phase5 비교 (λ=0.002 buggy vs λ=0 fixed):")
    print(f"  {'K':>5}  {'α*_orig(λ=0.002)':>17}  {'α*_fixed(λ=0)':>14}  {'RMSE_orig':>10}  {'RMSE_fixed':>10}  {'Δ':>8}")
    print("  " + "-" * 75)
    for k, row in comp.iterrows():
        print(f"  {k:5d}  {row['alpha_orig']:>17.3f}  {row['alpha_fixed']:>14.3f}  "
              f"{row['rmse_orig']:>10.4f}  {row['rmse_fixed']:>10.4f}  {row['delta']:>+8.4f}")

print("\n" + "=" * 70)
print("  최종 정리: 두 비교의 차이")
print("=" * 70)
print("\n [올바른 비교] 버기 λ=0 vs 수정된 λ=0: 순수 버그 수정 효과")
print(" [phase5 비교] 버기 λ=0.002 vs 수정된 λ=0: 버그 수정 + lambda 변경 효과 혼재")
print()

# 순수 lambda 효과 추정 (버기 λ=0.002 vs 버기 λ=0)
print("  [참고] 버기 λ=0.002 vs 버기 λ=0 (lambda 효과만):")
for method in BUGGY_METHODS:
    buggy_lam0 = method_K_summary(buggy_df, method)
    orig_s = method_K_summary(orig_df, method)
    b0 = buggy_lam0.groupby('K').agg(r_lam0=('test_rmse', 'mean'))
    bo2 = orig_s.groupby('K').agg(r_lam002=('test_rmse', 'mean'))
    comp = bo2.join(b0)
    comp['delta_lam'] = comp['r_lam0'] - comp['r_lam002']
    k_avg_delta = comp['delta_lam'].mean()
    print(f"  {method}: K평균 Δ={k_avg_delta:+.4f} (λ=0 → {'-' if k_avg_delta < 0 else '+'}개선)")
