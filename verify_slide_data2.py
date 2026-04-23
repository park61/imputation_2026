"""
올바른 파일 명시적 지정으로 데이터 검증
"""
import os
import pandas as pd
import numpy as np

BASE = r"d:\Dropbox\python_workspace(백업)\imputation_project"
INNER = os.path.join(BASE, "results", "inner_sim")
ARCHIVE = os.path.join(BASE, "results", "archive_buggy_20260415")

# ─── Fixed 데이터 로드 ────────────────────────────────────────────────────────
# acos/jmsd/msd fixed
df_fixed_3 = pd.read_csv(os.path.join(INNER, "combined", "all_folds_grid_results_20260416_004853.csv"))

# 14개 메서드 fixed
parts14 = []
parts14.append(pd.read_csv(os.path.join(INNER, "fold_01", "grid_search_results_inner_lambda_0.0_v1_20260408_082727.csv")))
parts14.append(pd.read_csv(os.path.join(INNER, "combined", "all_folds_grid_results_20260415_145918.csv")))
for fold_n, fname in [(7,"fold_07"), (8,"fold_08"), (9,"fold_09"), (10,"fold_10")]:
    base = f"grid_search_results_inner_lambda_0.0_v1_20260408_08"
    fold_map = {7:"4959", 8:"0701", 9:"2409", 10:"4122"}
    f = os.path.join(INNER, fname, f"grid_search_results_inner_lambda_0.0_v1_20260408_0{fold_map[fold_n][0]}{fold_map[fold_n][1:]}.csv")
    parts14.append(pd.read_csv(f))

df_fixed_14 = pd.concat(parts14, ignore_index=True)
if 'lambda' in df_fixed_14.columns:
    df_fixed_14 = df_fixed_14.drop(columns=['lambda'])

df_fixed_all = pd.concat([df_fixed_3, df_fixed_14], ignore_index=True)
df_fixed_opt = df_fixed_all[df_fixed_all['type'] == 'optimized']
print(f"Fixed total optimized: {len(df_fixed_opt)}, methods: {len(df_fixed_opt['method'].unique())}")

# ─── Buggy 데이터 로드 (올바른 λ=0 파일만) ───────────────────────────────────
buggy_parts = []
BUGGY_FILENAMES = {
    1: "grid_search_results_20260311_132613.csv",
    2: "grid_search_results_20260311_132613.csv",
    3: "grid_search_results_20260311_132613.csv",
    4: "grid_search_results_20260311_132613.csv",
    5: "grid_search_results_20260311_132613.csv",
    6: "grid_search_results_20260311_132613.csv",
    7: "grid_search_results_20260311_132613.csv",
    8: "grid_search_results_20260311_132613.csv",
    9: "grid_search_results_20260311_132613.csv",
    10: "grid_search_results_20260311_132613.csv",
}

# 실제로 어떤 파일들이 있는지 확인
import glob
for fold_n in range(1, 11):
    fold_dir = os.path.join(ARCHIVE, f"fold_{fold_n:02d}")
    csvs = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(fold_dir, "grid_search_results_20260311*.csv")))]
    print(f"fold_{fold_n:02d}: {csvs}")

print()

# λ=0 인 버기 파일: grid_search_results_20260311_132613.csv (fold_01에서 확인됨)
# 다른 fold는 다른 파일명을 가질 수 있음
for fold_n in range(1, 11):
    fold_dir = os.path.join(ARCHIVE, f"fold_{fold_n:02d}")
    # λ=0 파일 찾기: lambda가 없거나 lambda_0.0인 파일
    # grid_search_results_20260311_*.csv (non-lambda 파일)
    candidates = sorted(glob.glob(os.path.join(fold_dir, "grid_search_results_2026031[12]_*.csv")))
    if candidates:
        df_tmp = pd.read_csv(candidates[0])
        # lambda 열이 있으면 λ=0만 필터
        if 'lambda' in df_tmp.columns:
            df_tmp = df_tmp[df_tmp['lambda'] == 0.0]
        buggy_parts.append(df_tmp)
        print(f"fold_{fold_n:02d}: loaded {os.path.basename(candidates[0])}, shape={df_tmp.shape}, methods={sorted(df_tmp['method'].unique())[:3]}...")
    else:
        print(f"fold_{fold_n:02d}: NO FILE FOUND")

df_buggy = pd.concat(buggy_parts, ignore_index=True)
df_buggy_opt = df_buggy[df_buggy['type'] == 'optimized']
print(f"\nBuggy total optimized: {len(df_buggy_opt)}")
print(f"Buggy methods: {sorted(df_buggy_opt['method'].unique())}")

# ─── MSD Before/After 검증 ───────────────────────────────────────────────────
print("\n" + "="*60)
print("MSD Before/After 검증")
print("="*60)

msd_fixed_opt = df_fixed_opt[df_fixed_opt['method']=='msd']
msd_buggy_opt = df_buggy_opt[df_buggy_opt['method']=='msd']

print(f"MSD Fixed opt rows: {len(msd_fixed_opt)}, MSD Buggy opt rows: {len(msd_buggy_opt)}")

msd_f_k = msd_fixed_opt.groupby('K').agg(alpha=('alpha','mean'), RMSE=('test_RMSE','mean')).round(4)
print("\nMSD Fixed per K (10-fold, all TopN avg):")
print(msd_f_k.to_string())

if len(msd_buggy_opt) > 0:
    rmse_col = 'test_RMSE' if 'test_RMSE' in msd_buggy_opt.columns else 'RMSE'
    msd_b_k = msd_buggy_opt.groupby('K').agg(alpha=('alpha','mean'), RMSE=(rmse_col,'mean')).round(4)
    print("\nMSD Buggy per K (10-fold, all TopN avg):")
    print(msd_b_k.to_string())

    print("\nDelta RMSE (Fixed - Buggy):")
    delta = msd_f_k['RMSE'] - msd_b_k['RMSE']
    print(delta.round(4).to_string())
else:
    print("WARNING: No MSD buggy optimized rows found!")

# ─── JMSD Before/After 검증 ──────────────────────────────────────────────────
print("\n" + "="*60)
print("JMSD Before/After 검증")
print("="*60)

jmsd_fixed_opt = df_fixed_opt[df_fixed_opt['method']=='jmsd']
jmsd_buggy_opt = df_buggy_opt[df_buggy_opt['method']=='jmsd']

jmsd_f_k = jmsd_fixed_opt.groupby('K').agg(alpha=('alpha','mean'), RMSE=('test_RMSE','mean')).round(4)
print("JMSD Fixed per K:")
print(jmsd_f_k.to_string())

if len(jmsd_buggy_opt) > 0:
    rmse_col = 'test_RMSE' if 'test_RMSE' in jmsd_buggy_opt.columns else 'RMSE'
    jmsd_b_k = jmsd_buggy_opt.groupby('K').agg(alpha=('alpha','mean'), RMSE=(rmse_col,'mean')).round(4)
    print("JMSD Buggy per K:")
    print(jmsd_b_k.to_string())
    delta = jmsd_f_k['RMSE'] - jmsd_b_k['RMSE']
    print("Delta RMSE:")
    print(delta.round(4).to_string())

# ─── ACOS Before/After 검증 ──────────────────────────────────────────────────
print("\n" + "="*60)
print("ACOS Before/After 검증")
print("="*60)

acos_fixed_opt = df_fixed_opt[df_fixed_opt['method']=='acos']
acos_buggy_opt = df_buggy_opt[df_buggy_opt['method']=='acos']

acos_f_k = acos_fixed_opt.groupby('K').agg(alpha=('alpha','mean'), RMSE=('test_RMSE','mean')).round(4)
print("ACOS Fixed per K:")
print(acos_f_k.to_string())

if len(acos_buggy_opt) > 0:
    rmse_col = 'test_RMSE' if 'test_RMSE' in acos_buggy_opt.columns else 'RMSE'
    acos_b_k = acos_buggy_opt.groupby('K').agg(alpha=('alpha','mean'), RMSE=(rmse_col,'mean')).round(4)
    print("ACOS Buggy per K:")
    print(acos_b_k.to_string())
    delta = acos_f_k['RMSE'] - acos_b_k['RMSE']
    print("Delta RMSE:")
    print(delta.round(4).to_string())

# ─── 17개 메서드 순위 검증 ───────────────────────────────────────────────────
print("\n" + "="*60)
print("17개 메서드 순위 (Fixed, All-K 평균)")
print("="*60)
ranking_fixed = df_fixed_opt.groupby('method')['test_RMSE'].mean().sort_values()
for rank, (m, v) in enumerate(ranking_fixed.items(), 1):
    print(f"  {rank:2d}. {m:<22s} {v:.6f}")

print("\n17개 메서드 순위 (Buggy, All-K 평균)")
rmse_col = 'test_RMSE' if 'test_RMSE' in df_buggy_opt.columns else 'RMSE'
ranking_buggy = df_buggy_opt.groupby('method')[rmse_col].mean().sort_values()
for rank, (m, v) in enumerate(ranking_buggy.items(), 1):
    print(f"  {rank:2d}. {m:<22s} {v:.6f}")

# ─── Method×Fold α* (K=10, TopN=5) 검증 ─────────────────────────────────────
print("\n" + "="*60)
print("Method×Fold α* (K=10, TopN=5, Fixed)")
print("="*60)
k10t5_fixed = df_fixed_opt[(df_fixed_opt['K']==10) & (df_fixed_opt['TopN']==5)]
pivot_fixed = k10t5_fixed.pivot_table(index='method', columns='fold', values='alpha', aggfunc='mean')
print(pivot_fixed.round(2).to_string())

print("\nMethod×Fold α* (K=10, TopN=5, Buggy)")
k10t5_buggy = df_buggy_opt[(df_buggy_opt['K']==10) & (df_buggy_opt['TopN']==5)]
rmse_col = 'test_RMSE' if 'test_RMSE' in k10t5_buggy.columns else 'RMSE'
pivot_buggy = k10t5_buggy.pivot_table(index='method', columns='fold', values='alpha', aggfunc='mean')
print(pivot_buggy.round(2).to_string())
