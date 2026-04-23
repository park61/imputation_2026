"""
MSD alpha 선택 정밀 분석
- 수정 후 fold x K 별 최적 alpha 분포
- alpha = 10.0 (그리드 상한) 고착 여부 확인
- 그리드 상한이 진짜 문제인지, 유사도 분포 특성인지 판별
"""
import pandas as pd
import numpy as np
import glob
import os

INNER_SIM_DIR = "results/inner_sim"

# fold별 fixed 결과 로드
rows = []
for fold_num in range(1, 11):
    fold_dir = f"{INNER_SIM_DIR}/fold_{fold_num:02d}"
    files = sorted(glob.glob(os.path.join(fold_dir, "grid_search_results_202604*.csv")))
    files = [f for f in files if "_lambda_" not in os.path.basename(f)]
    if not files:
        print(f"WARNING: fold_{fold_num:02d} 에 파일 없음!")
        continue
    df = pd.read_csv(files[-1])
    df["fold"] = fold_num
    rows.append(df)

df_all = pd.concat(rows, ignore_index=True)
print(f"총 행수: {len(df_all)}, folds: {sorted(df_all['fold'].unique())}")
print(f"K 목록: {sorted(df_all['K'].unique())}")
print(f"alpha 범위: {df_all['alpha'].min()} ~ {df_all['alpha'].max()}")
print()

# ── 1. MSD: fold × K 별 최적 alpha ──────────────────────────────────────────
msd_opt = df_all[(df_all['method'] == 'msd') & (df_all['type'] == 'optimized')].copy()

# TopN은 RMSE에 무관 → K별로 TopN 평균 (같은 fold, K에서 TopN별로 alpha가 다를 수 있음)
msd_pivot = msd_opt.groupby(['fold', 'K'])['alpha'].mean().unstack(level='K')
print("=== MSD 최적 alpha: fold x K (수정 후) ===")
print(msd_pivot.round(2).to_string())
print()

# ── 2. alpha=10.0 (상한) 비율 ────────────────────────────────────────────────
total_opt = len(msd_opt)
at_ceiling = (msd_opt['alpha'] >= 10.0).sum()
below_ceiling = (msd_opt['alpha'] < 10.0).sum()
GRID_MAX = msd_opt['alpha'].max()
print(f"alpha 그리드 최댓값: {GRID_MAX}")
print(f"alpha = {GRID_MAX} (ceiling): {at_ceiling}/{total_opt} = {at_ceiling/total_opt*100:.1f}%")
print(f"alpha < {GRID_MAX}:           {below_ceiling}/{total_opt} = {below_ceiling/total_opt*100:.1f}%")
print()

below = msd_opt[msd_opt['alpha'] < 10.0][['fold','K','TopN','alpha','validation_rmse','test_RMSE']].sort_values(['K','fold'])
print("alpha < 10.0 인 케이스:")
if len(below) == 0:
    print("  없음!")
else:
    print(below.to_string())
print()

# ── 3. K=10만 분리 분석 (alpha가 10.0이 아닌 케이스) ────────────────────────
print("=== K=10 에서의 fold별 alpha (수정 후) ===")
msd_k10 = msd_opt[msd_opt['K'] == 10].groupby('fold')['alpha'].mean()
print(msd_k10.round(2).to_string())
print(f"K=10 alpha 평균: {msd_k10.mean():.3f}, std: {msd_k10.std():.3f}, max: {msd_k10.max():.1f}")
print()

# ── 4. validation RMSE: alpha 대비 경향 (K=20, fold=1 예시) ──────────────────
# CSV에는 optimized/baseline 2개 row만 있으므로 전체 곡선은 없음
# → 대신 K=20에서 fold별 opt_alpha와 val_rmse 분포 확인
msd_k20 = msd_opt[msd_opt['K'] == 20].groupby('fold')[['alpha','validation_rmse','test_RMSE']].mean()
print("=== K=20, fold별 opt_alpha / val_rmse / test_RMSE ===")
print(msd_k20.round(5).to_string())
print()

# ── 5. JMSD, ACOS: alpha < 10.0 비율 비교 ──────────────────────────────────
for method in ['jmsd', 'acos']:
    sub = df_all[(df_all['method'] == method) & (df_all['type'] == 'optimized')]
    at_c = (sub['alpha'] >= 10.0).sum()
    tot = len(sub)
    print(f"{method}: alpha=10 비율 {at_c}/{tot} = {at_c/tot*100:.1f}%, "
          f"alpha 범위 [{sub['alpha'].min():.3f}, {sub['alpha'].max():.2f}]")

print()

# ── 6. MSD 최종 결론: 진짜 그리드 상한 포화인가? ────────────────────────────
print("=" * 60)
print("결론 분석:")
print(f"  MSD alpha = GRID_MAX({GRID_MAX}) 인 비율: {at_ceiling/total_opt*100:.1f}%")
print(f"  K>=20 에서 count at ceiling: {(msd_opt[msd_opt['K']>=20]['alpha']>=10).sum()} / {len(msd_opt[msd_opt['K']>=20])}")
print(f"  K=10  에서 count at ceiling: {(msd_opt[msd_opt['K']==10]['alpha']>=10).sum()} / {len(msd_opt[msd_opt['K']==10])}")
print()

# MSD K=10 fold별 alpha 분포
k10_alphas = msd_opt[msd_opt['K']==10].groupby('fold')['alpha'].mean()
print(f"  K=10 fold별 alpha: {dict(k10_alphas.round(1))}")
print(f"  K=10 최소 alpha: {k10_alphas.min():.1f}, 최대: {k10_alphas.max():.1f}")
print()

if at_ceiling / total_opt > 0.85:
    print("  --> 즉, K>=20에서 거의 모든 경우 alpha=10(상한) 고착")
    print("      그리드 상한 포화 (True Optimum > 10) 가능성 높음")
    print("  --> 권고: alpha 그리드 상한을 10 -> 20~30으로 확장 후 재실험 필요")
else:
    print("  --> alpha=10 고착이 아님, 유사도 분포 특성에 의한 자연 수렴")
