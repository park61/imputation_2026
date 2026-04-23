"""
test_summary_20260417.tex 데이터 검증 스크립트
- MSD/JMSD/ACOS before/after 표 수치 확인
- 17개 메서드 순위 수치 확인
- Method×Fold α* 표 수치 확인
"""
import os
import pandas as pd
import numpy as np
import glob

BASE = r"d:\Dropbox\python_workspace(백업)\imputation_project"
INNER = os.path.join(BASE, "results", "inner_sim")
ARCHIVE = os.path.join(BASE, "results", "archive_buggy_20260415")

# ─── 1. Fixed 데이터 로드 ─────────────────────────────────────────────────────
print("="*70)
print("1. FIXED 데이터 로드")
print("="*70)

# acos/jmsd/msd fixed (전체 fold 통합)
fixed_acos_path = os.path.join(INNER, "combined", "all_folds_grid_results_20260416_004853.csv")
df_fixed_3 = pd.read_csv(fixed_acos_path)
print(f"Fixed (acos/jmsd/msd): {df_fixed_3.shape}, methods: {sorted(df_fixed_3['method'].unique())}")
print(f"  columns: {list(df_fixed_3.columns)}")

# 14개 메서드 fixed - fold 1
f01 = pd.read_csv(os.path.join(INNER, "fold_01", "grid_search_results_inner_lambda_0.0_v1_20260408_082727.csv"))
print(f"  fold_01 fixed (14개): {f01.shape}, methods: {sorted(f01['method'].unique())[:5]}...")

# fold 2~6
f26 = pd.read_csv(os.path.join(INNER, "combined", "all_folds_grid_results_20260415_145918.csv"))
print(f"  fold2-6 fixed (14개): {f26.shape}, methods: {sorted(f26['method'].unique())[:5]}...")

# fold 7~10
fold_files = {
    7: os.path.join(INNER, "fold_07", "grid_search_results_inner_lambda_0.0_v1_20260408_084959.csv"),
    8: os.path.join(INNER, "fold_08", "grid_search_results_inner_lambda_0.0_v1_20260408_090701.csv"),
    9: os.path.join(INNER, "fold_09", "grid_search_results_inner_lambda_0.0_v1_20260408_092409.csv"),
    10: os.path.join(INNER, "fold_10", "grid_search_results_inner_lambda_0.0_v1_20260408_094122.csv"),
}
f710_parts = []
for fnum, fpath in fold_files.items():
    df_tmp = pd.read_csv(fpath)
    f710_parts.append(df_tmp)
f710 = pd.concat(f710_parts, ignore_index=True)
print(f"  fold7-10 fixed (14개): {f710.shape}")

# 14개 통합
df_fixed_14 = pd.concat([f01, f26, f710], ignore_index=True)
print(f"  14개 메서드 총합: {df_fixed_14.shape}, folds: {sorted(df_fixed_14['fold'].unique())}")

# 전체 fixed 통합
df_fixed_all = pd.concat([df_fixed_3, df_fixed_14], ignore_index=True)
print(f"  전체 fixed: {df_fixed_all.shape}")

# 컬럼 정규화 (lambda 열 있으면 제거)
if 'lambda' in df_fixed_all.columns:
    df_fixed_all = df_fixed_all.drop(columns=['lambda'])

# ─── 2. Buggy 데이터 로드 ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("2. BUGGY 데이터 로드 (archive_buggy_20260415)")
print("="*70)

buggy_parts = []
for fold_dir in sorted(glob.glob(os.path.join(ARCHIVE, "fold_*"))):
    csvs = sorted(glob.glob(os.path.join(fold_dir, "*.csv")))
    if csvs:
        df_tmp = pd.read_csv(csvs[0])
        buggy_parts.append(df_tmp)
        print(f"  {os.path.basename(fold_dir)}: {df_tmp.shape}, methods: {sorted(df_tmp['method'].unique())[:3]}...")

df_buggy = pd.concat(buggy_parts, ignore_index=True)
print(f"Buggy total: {df_buggy.shape}, folds: {sorted(df_buggy['fold'].unique())}")
print(f"Buggy methods: {sorted(df_buggy['method'].unique())}")

# ─── 3. MSD Before/After 표 검증 ─────────────────────────────────────────────
print("\n" + "="*70)
print("3. MSD Before/After 표 검증 (슬라이드 2)")
print("="*70)

# Fixed MSD: K별 alpha* 평균 (10-fold × all TopN)
msd_fixed = df_fixed_3[df_fixed_3['method'] == 'msd'].copy()
# optimized only
msd_fixed_opt = msd_fixed[msd_fixed['type'] == 'optimized']
print(f"MSD Fixed optimized rows: {len(msd_fixed_opt)}")
print(f"  folds: {sorted(msd_fixed_opt['fold'].unique())}")

# alpha* per K: 10-fold × all TopN 평균
msd_fixed_k = msd_fixed_opt.groupby('K')['alpha'].mean().sort_index()
print("\nMSD Fixed α* (10-fold, all TopN 평균):")
print(msd_fixed_k.to_string())

# Fixed MSD RMSE per K
msd_fixed_rmse = msd_fixed_opt.groupby('K')['test_RMSE'].mean().sort_index()
print("\nMSD Fixed RMSE (10-fold, all TopN 평균):")
print(msd_fixed_rmse.round(4).to_string())

# Buggy MSD
msd_buggy = df_buggy[df_buggy['method'] == 'msd'].copy()
if 'type' in msd_buggy.columns:
    msd_buggy_opt = msd_buggy[msd_buggy['type'] == 'optimized']
else:
    # alpha* = alpha that minimizes validation RMSE per (fold, K, TopN)
    idx = msd_buggy.groupby(['fold','K','TopN'])['validation_rmse'].idxmin()
    msd_buggy_opt = msd_buggy.loc[idx]

print(f"\nMSD Buggy optimized rows: {len(msd_buggy_opt)}")
msd_buggy_k = msd_buggy_opt.groupby('K')['alpha'].mean().sort_index()
print("\nMSD Buggy α* (10-fold, all TopN 평균):")
print(msd_buggy_k.to_string())

msd_buggy_rmse_col = 'test_RMSE' if 'test_RMSE' in msd_buggy_opt.columns else 'RMSE'
msd_buggy_rmse = msd_buggy_opt.groupby('K')[msd_buggy_rmse_col].mean().sort_index()
print("\nMSD Buggy RMSE (10-fold, all TopN 평균):")
print(msd_buggy_rmse.round(4).to_string())

# ─── 4. JMSD Before/After 표 검증 ────────────────────────────────────────────
print("\n" + "="*70)
print("4. JMSD Before/After 표 검증 (슬라이드 3)")
print("="*70)

jmsd_fixed = df_fixed_3[df_fixed_3['method'] == 'jmsd']
jmsd_fixed_opt = jmsd_fixed[jmsd_fixed['type'] == 'optimized']
jmsd_fixed_k = jmsd_fixed_opt.groupby('K')['alpha'].mean().sort_index()
jmsd_fixed_rmse = jmsd_fixed_opt.groupby('K')['test_RMSE'].mean().sort_index()
print("JMSD Fixed α* per K:")
print(jmsd_fixed_k.round(3).to_string())
print("JMSD Fixed RMSE per K:")
print(jmsd_fixed_rmse.round(4).to_string())

jmsd_buggy = df_buggy[df_buggy['method'] == 'jmsd']
if 'type' in jmsd_buggy.columns:
    jmsd_buggy_opt = jmsd_buggy[jmsd_buggy['type'] == 'optimized']
else:
    idx = jmsd_buggy.groupby(['fold','K','TopN'])['validation_rmse'].idxmin()
    jmsd_buggy_opt = jmsd_buggy.loc[idx]
jmsd_buggy_k = jmsd_buggy_opt.groupby('K')['alpha'].mean().sort_index()
jmsd_buggy_rmse_col = 'test_RMSE' if 'test_RMSE' in jmsd_buggy_opt.columns else 'RMSE'
jmsd_buggy_rmse = jmsd_buggy_opt.groupby('K')[jmsd_buggy_rmse_col].mean().sort_index()
print("JMSD Buggy α* per K:")
print(jmsd_buggy_k.round(3).to_string())
print("JMSD Buggy RMSE per K:")
print(jmsd_buggy_rmse.round(4).to_string())

# ─── 5. ACOS Before/After 표 검증 ────────────────────────────────────────────
print("\n" + "="*70)
print("5. ACOS Before/After 표 검증 (슬라이드 4)")
print("="*70)

acos_fixed = df_fixed_3[df_fixed_3['method'] == 'acos']
acos_fixed_opt = acos_fixed[acos_fixed['type'] == 'optimized']
acos_fixed_k = acos_fixed_opt.groupby('K')['alpha'].mean().sort_index()
acos_fixed_rmse = acos_fixed_opt.groupby('K')['test_RMSE'].mean().sort_index()
print("ACOS Fixed α* per K:")
print(acos_fixed_k.round(3).to_string())
print("ACOS Fixed RMSE per K:")
print(acos_fixed_rmse.round(4).to_string())

acos_buggy = df_buggy[df_buggy['method'] == 'acos']
if 'type' in acos_buggy.columns:
    acos_buggy_opt = acos_buggy[acos_buggy['type'] == 'optimized']
else:
    idx = acos_buggy.groupby(['fold','K','TopN'])['validation_rmse'].idxmin()
    acos_buggy_opt = acos_buggy.loc[idx]
acos_buggy_k = acos_buggy_opt.groupby('K')['alpha'].mean().sort_index()
acos_buggy_rmse_col = 'test_RMSE' if 'test_RMSE' in acos_buggy_opt.columns else 'RMSE'
acos_buggy_rmse = acos_buggy_opt.groupby('K')[acos_buggy_rmse_col].mean().sort_index()
print("ACOS Buggy α* per K:")
print(acos_buggy_k.round(3).to_string())
print("ACOS Buggy RMSE per K:")
print(acos_buggy_rmse.round(4).to_string())

# ─── 6. 17개 메서드 순위 검증 ────────────────────────────────────────────────
print("\n" + "="*70)
print("6. 17개 메서드 순위 검증 (All-K 평균, Fixed)")
print("="*70)

# fixed optimized only
if 'type' in df_fixed_all.columns:
    df_fixed_opt = df_fixed_all[df_fixed_all['type'] == 'optimized']
else:
    df_fixed_opt = df_fixed_all

# All-K 평균
method_rmse_fixed = df_fixed_opt.groupby('method')['test_RMSE'].mean().sort_values()
print("Fixed 17개 메서드 All-K 평균 RMSE:")
for rank, (method, rmse) in enumerate(method_rmse_fixed.items(), 1):
    print(f"  {rank:2d}. {method:<20s} {rmse:.6f}")

# Buggy ranking
if 'type' in df_buggy.columns:
    df_buggy_opt = df_buggy[df_buggy['type'] == 'optimized']
else:
    # Need to find α* for each (fold, K, TopN) for each method
    idx = df_buggy.groupby(['fold','K','TopN','method'])['validation_rmse'].idxmin()
    df_buggy_opt = df_buggy.loc[idx]

rmse_col_buggy = 'test_RMSE' if 'test_RMSE' in df_buggy_opt.columns else 'RMSE'
method_rmse_buggy = df_buggy_opt.groupby('method')[rmse_col_buggy].mean().sort_values()
print("\nBuggy 메서드 All-K 평균 RMSE (from archive_buggy, inner_sim):")
for rank, (method, rmse) in enumerate(method_rmse_buggy.items(), 1):
    print(f"  {rank:2d}. {method:<20s} {rmse:.6f}")

# ─── 7. Method×Fold α* 표 검증 (K=10, TopN=5) ────────────────────────────────
print("\n" + "="*70)
print("7. Method×Fold α* 표 검증 (K=10, TopN=5, Fixed)")
print("="*70)

# Fixed: K=10, TopN=5, optimized
k10_topn5 = df_fixed_opt[(df_fixed_opt['K']==10) & (df_fixed_opt['TopN']==5)]
print("Fixed K=10, TopN=5 α* per method per fold:")
pivot = k10_topn5.pivot_table(index='method', columns='fold', values='alpha', aggfunc='mean')
print(pivot.round(2).to_string())

# Buggy: K=10, TopN=5
k10_topn5_b = df_buggy_opt[(df_buggy_opt['K']==10) & (df_buggy_opt['TopN']==5)]
print("\nBuggy K=10, TopN=5 α* per method per fold:")
pivot_b = k10_topn5_b.pivot_table(index='method', columns='fold', values='alpha', aggfunc='mean')
print(pivot_b.round(2).to_string())

print("\n>>> 스크립트 완료")
