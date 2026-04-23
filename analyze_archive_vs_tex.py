"""
버기 아카이브 데이터와 이전 tex(test_summary_20260415) 보고 수치 비교 분석
핵심 질문: phase5가 비교한 "buggy" 데이터는 실제로 어떤 실험 결과인가?
"""
import pandas as pd
import numpy as np
import glob
import os

ARCHIVE_DIR = "results/archive_buggy_20260415"
INNER_SIM_DIR = "results/inner_sim"

# ── 1. 아카이브 데이터 구조 확인 ────────────────────────────────────────────
print("=== 아카이브 파일 목록 ===")
arch_files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "fold_*", "*.csv")))
for f in arch_files:
    df = pd.read_csv(f)
    methods = sorted(df['method'].unique())
    alphas = sorted(df['alpha'].unique()) if 'alpha' in df.columns else []
    n = len(df)
    print(f"  {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}: "
          f"rows={n}, methods={methods[:5]}{'...' if len(methods)>5 else ''}")

print()

# ── 2. 아카이브에서 MSD만 추출 ────────────────────────────────────────────────
arch_all = []
for f in arch_files:
    df = pd.read_csv(f)
    fold_num = int(os.path.basename(os.path.dirname(f)).split("_")[1])
    df["fold"] = fold_num
    arch_all.append(df)
arch_df = pd.concat(arch_all, ignore_index=True)

print(f"아카이브 전체 rows: {len(arch_df)}")
print(f"아카이브 methods: {sorted(arch_df['method'].unique())}")
print(f"아카이브 alpha 범위: {arch_df['alpha'].min()} ~ {arch_df['alpha'].max()}")

# regularization_lambda 컬럼이 있으면 확인
if 'regularization_lambda' in arch_df.columns or 'lambda' in arch_df.columns:
    lam_col = 'regularization_lambda' if 'regularization_lambda' in arch_df.columns else 'lambda'
    print(f"아카이브 lambda 값: {sorted(arch_df[lam_col].unique())}")
if 'regularization_penalty' in arch_df.columns:
    print(f"regularization_penalty 샘플: {arch_df['regularization_penalty'].unique()[:5]}")
print()

# ── 3. 아카이브 MSD: fold × K 별 최적 alpha ─────────────────────────────────
msd_arch = arch_df[(arch_df['method'] == 'msd') & (arch_df['type'] == 'optimized')].copy()
print(f"아카이브 MSD optimized rows: {len(msd_arch)}")
if len(msd_arch) > 0:
    pivot_arch = msd_arch.groupby(['fold', 'K'])['alpha'].mean().unstack('K')
    print("=== 아카이브 MSD 최적 alpha: fold × K ===")
    print(pivot_arch.round(2).to_string())
    at_10 = (msd_arch['alpha'] >= 10.0).sum()
    print(f"\nalpha=10.0 비율: {at_10}/{len(msd_arch)} = {at_10/len(msd_arch)*100:.1f}%")

print()

# ── 4. 이전 tex에서 보고된 값과 비교 ------------------------------------------
# tex(20260415)에서 lambda=0 MSD alpha 표 (lines ~1790):
# K=10→9.950, K=20-100→10.000
# 이 값들은 FIXED 실험 결과인가, OLD 실험 결과인가?
print("=== tex(20260415) 보고 값 vs 아카이브 값 비교 ===")
print("tex 보고 [lambda=0, MSD alpha (all fold avg)]:")
tex_vals = {10: 9.950, 20: 10.0, 30: 10.0, 40: 10.0, 50: 10.0, 60: 10.0, 70: 10.0, 80: 10.0, 90: 10.0, 100: 10.0}
if len(msd_arch) > 0:
    archive_K_avg = msd_arch.groupby('K')['alpha'].mean()
    for k in sorted(tex_vals.keys()):
        v_tex = tex_vals.get(k, "N/A")
        v_arch = archive_K_avg.get(k, "N/A")
        if isinstance(v_arch, float):
            print(f"  K={k:3d}: tex={v_tex:.3f}, archive={v_arch:.3f}, diff={v_tex-v_arch:.3f}")
        else:
            print(f"  K={k:3d}: tex={v_tex:.3f}, archive=N/A")

print()

# ── 5. FIXED 결과와 tex 비교 ─────────────────────────────────────────────────
fixed_rows = []
for fold_num in range(1, 11):
    fold_dir = f"{INNER_SIM_DIR}/fold_{fold_num:02d}"
    files = sorted(glob.glob(os.path.join(fold_dir, "grid_search_results_202604*.csv")))
    files = [f for f in files if "_lambda_" not in os.path.basename(f)]
    if not files:
        continue
    df = pd.read_csv(files[-1])
    df["fold"] = fold_num
    fixed_rows.append(df)
fixed_df = pd.concat(fixed_rows, ignore_index=True)

msd_fixed = fixed_df[(fixed_df['method'] == 'msd') & (fixed_df['type'] == 'optimized')].copy()
fixed_K_avg = msd_fixed.groupby('K')['alpha'].mean()

print("=== FIXED 결과 vs tex(20260415) 보고 값 비교 ===")
print("주의: tex(20260415) lambda=0 분석은 어떤 실험 기준?")
for k in sorted(tex_vals.keys()):
    v_tex = tex_vals.get(k, 0.0)
    v_fixed = fixed_K_avg.get(k, float('nan'))
    print(f"  K={k:3d}: tex={v_tex:.3f}, FIXED={v_fixed:.3f}")

print()
print("결론 체크:")
print("  tex 값과 FIXED 값이 일치 → tex는 이미 FIXED 결과를 보고했던 것")
print("  tex 값과 ARCHIVE 값이 일치 → tex는 BUGGY 결과를 보고했던 것")
