"""
즉시 검증 스크립트: Alpha 불일치 원인 규명 및 공정한 비교

목표:
1. Grid Search에서 사용한 alpha와 History에서 최적인 alpha의 불일치 원인 파악
2. 올바른 alpha로 Test 재평가
3. 공정한 ALPHA=1 vs 최적 alpha 비교 도출
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

print("=" * 100)
print("🔬 Alpha 불일치 원인 규명 및 공정한 비교")
print("=" * 100)

# ==================================================================================
# Step 1: 데이터 로드 (Fold 1-5)
# ==================================================================================
print("\n📂 Step 1: Grid Search와 Alpha History 로드\n")

# Grid Search 결과 (Test 성능)
grid_results = []
grid_files_loaded = []

for fold_num in range(1, 6):
    fold_id = f"fold_{fold_num:02d}"
    results_dir = f"results/{fold_id}"
    
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) 
                if f.startswith('grid_search_results_') 
                and not 'reeval' in f
                and f.endswith('.csv')]
        if files:
            latest_file = sorted(files)[-1]
            filepath = os.path.join(results_dir, latest_file)
            df = pd.read_csv(filepath)
            grid_results.append(df)
            grid_files_loaded.append((fold_num, latest_file))

df_grid = pd.concat(grid_results, ignore_index=True)
print(f"✅ Grid Search: {len(df_grid):,} records from {len(grid_files_loaded)} folds")
for fold_num, fname in grid_files_loaded:
    print(f"   Fold {fold_num:02d}: {fname}")

# Alpha History 로드 (Validation 성능)
alpha_history = []
history_files_loaded = []

for fold_num in range(1, 6):
    fold_id = f"fold_{fold_num:02d}"
    results_dir = f"results/{fold_id}"
    
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) 
                if f.startswith('alpha_optimization_history_') 
                and f.endswith('.csv')]
        if files:
            latest_file = sorted(files)[-1]
            filepath = os.path.join(results_dir, latest_file)
            df = pd.read_csv(filepath)
            alpha_history.append(df)
            history_files_loaded.append((fold_num, latest_file))

df_history = pd.concat(alpha_history, ignore_index=True)
print(f"✅ Alpha History: {len(df_history):,} records from {len(history_files_loaded)} folds")
for fold_num, fname in history_files_loaded:
    print(f"   Fold {fold_num:02d}: {fname}")

# ==================================================================================
# Step 2: Alpha History 구조 분석
# ==================================================================================
print("\n" + "=" * 100)
print("📊 Step 2: Alpha History 구조 분석")
print("=" * 100)

print(f"\nAlpha History 컬럼: {list(df_history.columns)}")
print(f"\n'phase' 컬럼 값: {df_history['phase'].unique()}")
print(f"\n각 (fold, method, K, TopN) 조합별 alpha 개수:")

alpha_counts = df_history.groupby(['fold', 'method', 'K', 'TopN'])['alpha'].nunique()
print(f"  최소: {alpha_counts.min()}, 최대: {alpha_counts.max()}, 평균: {alpha_counts.mean():.1f}")

# 각 K별로 몇 개의 alpha가 있는지 확인
print(f"\n각 (fold, method, K) 조합별 alpha 개수 (TopN 통합):")
alpha_counts_k = df_history.groupby(['fold', 'method', 'K'])['alpha'].nunique()
print(f"  최소: {alpha_counts_k.min()}, 최대: {alpha_counts_k.max()}, 평균: {alpha_counts_k.mean():.1f}")

# ==================================================================================
# Step 3: Grid Search 구조 분석
# ==================================================================================
print("\n" + "=" * 100)
print("📊 Step 3: Grid Search 구조 분석")
print("=" * 100)

print(f"\nGrid Search 컬럼: {list(df_grid.columns)}")
print(f"\n'type' 컬럼 값: {df_grid['type'].unique()}")

# 각 (fold, method, K) 조합별 몇 개의 TopN이 있는지 확인
topn_counts = df_grid.groupby(['fold', 'method', 'K']).size()
print(f"\n각 (fold, method, K) 조합별 레코드 개수:")
print(f"  Baseline (alpha=1.0): {topn_counts.unique()}")

# ==================================================================================
# Step 4: 핵심 이슈 - Alpha 선택 방식의 차이
# ==================================================================================
print("\n" + "=" * 100)
print("🔍 Step 4: 핵심 이슈 파악")
print("=" * 100)

print("""
발견된 문제:

Alpha History 저장 방식:
  - (fold, method, K, TopN)마다 다양한 alpha 값들의 성능이 저장됨
  - 예: method='cosine', K=30, TopN=5일 때 여러 alpha (0.1, 0.2, ..., 3.0)의 mse 저장
  - 각 TopN마다 독립적으로 최적 alpha가 다를 수 있음

Grid Search 선택 방식:
  - (fold, method, K)에서 "ONE" alpha를 선택하여
  - 모든 TopN에 같은 alpha를 적용
  - 이것이 불일치의 원인!

해결책:
  1. Alpha History에서 각 TopN별로 최적 alpha를 찾기 (현재 상황 = TopN별 최적 alpha)
  2. 또는 Alpha History에서 K 레벨에서 모든 TopN을 통합해서 최적 alpha 찾기
  3. Grid Search와 동일한 방식으로 alpha를 선택해서 Test 재평가
""")

# ==================================================================================
# Step 5: 두 가지 Alpha 선택 전략 비교
# ==================================================================================
print("\n" + "=" * 100)
print("🎯 Step 5: Alpha 선택 전략")
print("=" * 100)

print("""
전략 A) TopN별 독립적 최적화 (현재 Grid Search 방식)
  - 각 (fold, method, K, TopN)마다 MSE를 최소화하는 alpha 선택
  - 최대 유연성, 각 TopN에 맞춘 alpha
  - 문제: Grid Search에서는 TopN별로 다른 alpha 사용 불가능

전략 B) K별 통합 최적화 (권장)
  - 각 (fold, method, K)에서 모든 TopN의 평균 MSE를 최소화하는 alpha 선택
  - Grid Search와 일관성 있음
  - 공정한 비교 가능

즉시 구현: 전략 B 사용
""")

# ==================================================================================
# Step 6: 각 (fold, method, K)에서 최적 alpha 찾기 (전략 B)
# ==================================================================================
print("\n" + "=" * 100)
print("🔧 Step 6: 올바른 Alpha 선택 (TopN 통합)")
print("=" * 100)

# Alpha History에서 각 (fold, method, K)별로 평균 MSE를 계산
print("\n각 (fold, method, K) 조합에서 평균 MSE 계산...", end='')

optimal_alphas = []

for (fold, method, k), group in df_history.groupby(['fold', 'method', 'K']):
    # 각 alpha별로 모든 TopN의 평균 MSE 계산
    alpha_avg_mse = group.groupby('alpha')['mse'].mean()
    
    # 최적 alpha (최소 MSE) 찾기
    best_alpha = alpha_avg_mse.idxmin()
    best_mse = alpha_avg_mse.min()
    
    optimal_alphas.append({
        'fold': fold,
        'method': method,
        'K': k,
        'optimal_alpha': best_alpha,
        'avg_mse': best_mse,
        'n_alphas_tested': len(alpha_avg_mse)
    })

df_optimal = pd.DataFrame(optimal_alphas)
print(f" ✓")

print(f"\n✅ 올바른 Alpha 선택 완료: {len(df_optimal):,} (fold, method, K) 조합")
print(f"\n샘플 (처음 10개):")
print(df_optimal.head(10).to_string(index=False))

# ==================================================================================
# Step 7: Grid Search의 현재 alpha와 비교
# ==================================================================================
print("\n" + "=" * 100)
print("📊 Step 7: Grid Search vs 올바른 Alpha 비교")
print("=" * 100)

# Grid Search에서 현재 사용된 alpha 추출 (각 (fold, method, K) 조합에서 첫 번째 레코드)
current_alphas = df_grid.drop_duplicates(subset=['fold', 'method', 'K', 'type'])[
    ['fold', 'method', 'K', 'type', 'alpha']
].copy()

# baseline과 optimized 분리
baseline_alphas = current_alphas[current_alphas['type'] == 'baseline'][['fold', 'method', 'K', 'alpha']]
optimized_alphas = current_alphas[current_alphas['type'] == 'optimized'][['fold', 'method', 'K', 'alpha']]

baseline_alphas = baseline_alphas.rename(columns={'alpha': 'current_baseline_alpha'})
optimized_alphas = optimized_alphas.rename(columns={'alpha': 'current_optimized_alpha'})

# 올바른 alpha와 병합
comparison = df_optimal.copy()
comparison = comparison.merge(optimized_alphas, on=['fold', 'method', 'K'], how='left')

print(f"\n올바른 Alpha vs 현재 Grid Search의 Optimized Alpha:")
print(f"  - 병합 성공: {comparison['current_optimized_alpha'].notna().sum()} / {len(comparison)}")
print(f"\n불일치 분석:")

comparison['alpha_diff'] = comparison['optimal_alpha'] - comparison['current_optimized_alpha']
mismatch = comparison[comparison['alpha_diff'].abs() > 0.001]

print(f"  - 완전 일치: {(comparison['alpha_diff'] == 0).sum():,}")
print(f"  - 불일치 (차이 > 0.001): {len(mismatch):,}")

if len(mismatch) > 0:
    print(f"\n불일치 샘플:")
    print(mismatch[[
        'fold', 'method', 'K', 'optimal_alpha', 'current_optimized_alpha', 'alpha_diff'
    ]].head(10).to_string(index=False))
    
    print(f"\n차이 통계:")
    print(f"  최대 차이: {comparison['alpha_diff'].abs().max():.4f}")
    print(f"  평균 차이: {comparison['alpha_diff'].abs().mean():.4f}")
    print(f"  중앙값 차이: {comparison['alpha_diff'].abs().median():.4f}")

# ==================================================================================
# Step 8: 최종 권장사항
# ==================================================================================
print("\n" + "=" * 100)
print("💡 최종 권장사항")
print("=" * 100)

print(f"""
발견:
1. ✅ Alpha History에서 (fold, method, K)별로 최적 alpha 추출 가능
2. {'⚠️  Grid Search와 불일치 확인됨' if len(mismatch) > 0 else '✅ Grid Search와 일치함'}

다음 단계:

Option A) 현재 결과 수용 (빠름 - 1시간)
  1. 현재 alpha 값들이 올바르다고 가정
  2. Lambda 정규화를 더 강하게 적용 (λ = 0.05~0.20)
  3. 더 나은 alpha 찾기
  장점: 빠르고 실용적
  단점: 근본 원인 미해결

Option B) Nested CV 실행 (느림 - 30시간+)
  1. Inner CV (3-fold)로 Alpha 선택 과정 자체를 검증
  2. Inner Test vs Outer Test 비교로 일반화 능력 평가
  3. 가장 신뢰할 수 있는 결론
  장점: 과학적이고 신뢰성 높음
  단점: 계산량 많음

🎯 추천: Option A → Option B 순서
  - 먼저 Lambda 범위 확대로 성능 개선 시도 (빠름)
  - 시간이 허락하면 Nested CV 실행 (정확함)
""")

# 결과 저장
result_file = f"results/alpha_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
os.makedirs('results', exist_ok=True)

result_summary = {
    'timestamp': datetime.now().isoformat(),
    'grid_search_records': len(df_grid),
    'alpha_history_records': len(df_history),
    'optimal_alphas_found': len(df_optimal),
    'mismatches': len(mismatch),
    'max_alpha_diff': float(comparison['alpha_diff'].abs().max()) if len(mismatch) > 0 else 0,
    'mean_alpha_diff': float(comparison['alpha_diff'].abs().mean()) if len(mismatch) > 0 else 0,
    'recommendation': 'Lambda regularization increase (0.05-0.20) then Nested CV'
}

import json
with open(result_file, 'w') as f:
    json.dump(result_summary, f, indent=2)

print(f"\n✅ 분석 결과 저장: {result_file}")
