"""
핵심 검증: phase5_analysis 비교 오류 여부 확인
버기 아카이브에서 어떤 lambda 값을 사용했는가?
lambda=0.002(원래) vs lambda=0(비교대상)이 뒤섞였는가?
"""
import pandas as pd
import numpy as np
import glob
import os

ARCHIVE_DIR = "results/archive_buggy_20260415"

# phase5_analysis.py 와 동일한 방식으로 아카이브 로드
arch_files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "fold_*", "grid_search_results_2*.csv")))
arch_files = [f for f in arch_files if "_lambda_" not in os.path.basename(f)]

print(f"phase5 가 로드하는 archive 파일 수: {len(arch_files)}")
for f in arch_files[:5]:
    print(f"  {os.path.relpath(f)}")
print()

# 첫 번째 파일 내용 자세히 확인
f0 = arch_files[0]
df0 = pd.read_csv(f0)
print(f"=== {os.path.basename(f0)} ===")
print(f"컬럼: {df0.columns.tolist()}")
print(f"rows: {len(df0)}")
print(f"types: {sorted(df0['type'].unique())}")
print(f"methods: {sorted(df0['method'].unique())}")
print(f"K: {sorted(df0['K'].unique())}")

# regularization 관련 컬럼 확인
reg_cols = [c for c in df0.columns if 'reg' in c.lower() or 'lambda' in c.lower() or 'penalty' in c.lower()]
print(f"regularization 관련 컬럼: {reg_cols}")
for col in reg_cols:
    vals = df0[col].unique()
    print(f"  {col}: {vals[:5]}")
print()

# MSD optimized rows 확인
msd_rows = df0[(df0['method'] == 'msd') & (df0['type'] == 'optimized')]
print(f"MSD optimized rows: {len(msd_rows)}")
print(msd_rows[['K', 'TopN', 'alpha'] + reg_cols + ['validation_rmse', 'test_RMSE']].head(10).to_string())
print()

# ── 핵심: 아카이브 내 lambda=0 전용 파일 확인 ──────────────────────────────
lam0_files = glob.glob(os.path.join(ARCHIVE_DIR, "fold_*", "*lambda_0.0*.csv"))
print(f"lambda=0.0 전용 파일 수 (아카이브): {len(lam0_files)}")
for f in lam0_files[:5]:
    print(f"  {os.path.relpath(f)}")
print()

if lam0_files:
    df_lam0 = pd.read_csv(lam0_files[0])
    msd_lam0 = df_lam0[(df_lam0['method'] == 'msd') & (df_lam0['type'] == 'optimized')]
    print(f"lambda=0 파일의 MSD optimized rows: {len(msd_lam0)}")
    if len(msd_lam0) > 0:
        lam0_pivot = msd_lam0.groupby('K')['alpha'].mean()
        print("lambda=0 파일 K별 MSD alpha (fold별 평균):")
        print(lam0_pivot.round(2).to_string())
    print()

# ── tex(20260415) 에서 보고된 lambda=0 msd alpha 값 ─────────────────────────
# "Lambda=0 Alpha 선택: Method × K 평균" 표에서:
# msd K=10→9.950, K=20-100→10.000 (lines 1790+ in tex)
tex_lam0_vals = {10: 9.950, 20: 10.0, 30: 10.0, 40: 10.0, 50: 10.0,
                 60: 10.0, 70: 10.0, 80: 10.0, 90: 10.0, 100: 10.0}

print("=== 세 버전 비교: K별 MSD 최적 alpha ===")
print(f"{'K':>5s}  {'tex_lam0(OLD)':>13s}  {'archive_main(phase5)':>20s}  {'fixed_lam0(NEW)':>15s}")
print("-" * 65)

# phase5 archive (main, no _lambda_)
dfs_arch_main = []
for f in arch_files:
    fold_df = pd.read_csv(f)
    fold_num = int(os.path.basename(os.path.dirname(f)).split("_")[1])
    fold_df["fold"] = fold_num
    dfs_arch_main.append(fold_df)
arch_main_df = pd.concat(dfs_arch_main, ignore_index=True)
msd_arch_main = arch_main_df[(arch_main_df['method'] == 'msd') & (arch_main_df['type'] == 'optimized')]
arch_main_K = msd_arch_main.groupby('K')['alpha'].mean()

# fixed (new runs)
fixed_rows = []
for fold_num in range(1, 11):
    fold_dir = f"results/inner_sim/fold_{fold_num:02d}"
    files = sorted(glob.glob(os.path.join(fold_dir, "grid_search_results_202604*.csv")))
    files = [f for f in files if "_lambda_" not in os.path.basename(f)]
    if not files:
        continue
    df = pd.read_csv(files[-1])
    df["fold"] = fold_num
    fixed_rows.append(df)
fixed_df = pd.concat(fixed_rows, ignore_index=True)
msd_fixed = fixed_df[(fixed_df['method'] == 'msd') & (fixed_df['type'] == 'optimized')]
fixed_K = msd_fixed.groupby('K')['alpha'].mean()

for k in sorted(tex_lam0_vals.keys()):
    t = tex_lam0_vals[k]
    a_main = arch_main_K.get(k, float('nan'))
    f_val = fixed_K.get(k, float('nan'))
    tex_match_arch = abs(t - a_main) < 0.1 if not np.isnan(a_main) else False
    tex_match_fixed = abs(t - f_val) < 0.1 if not np.isnan(f_val) else False
    print(f"  K={k:3d}:  {t:6.3f}  |  {a_main:6.3f}  "
          f"{'← tex일치' if tex_match_arch else '          ':10s}  |  {f_val:6.3f}  "
          f"{'← tex일치' if tex_match_fixed else '':10s}")

print()
print("=== 결론 ===")
# tex 값은 K=10→9.95 / K>=20→10.0
# archive_main 값은?
k10_a = arch_main_K.get(10, float('nan'))
k20_a = arch_main_K.get(20, float('nan'))
k10_f = fixed_K.get(10, float('nan'))
k20_f = fixed_K.get(20, float('nan'))
print(f"K=10: tex=9.95, archive_main={k10_a:.3f}, fixed={k10_f:.3f}")
print(f"K=20: tex=10.0, archive_main={k20_a:.3f}, fixed={k20_f:.3f}")
print()
if abs(9.95 - k10_a) < 0.2:
    print("→ tex 값 ≈ archive_main 값: phase5가 올바른 buggy-lambda0 데이터를 사용함")
elif abs(9.95 - k10_f) < 0.2:
    print("→ tex 값 ≈ fixed 값: tex는 FIXED 결과를 이미 보고했거나 같은 분포")
else:
    print(f"→ tex 값({9.95:.3f}) ≠ archive({k10_a:.3f}) ≠ fixed({k10_f:.3f})")
    print("  → tex는 lambda=0 전용 파일(_lambda_0.0_v1_...) 기반 보고")
    print("  → phase5는 다른 lambda(아마 0.002) 아카이브 기반 비교 → 비교 오류!")
