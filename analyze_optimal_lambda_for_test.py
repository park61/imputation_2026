import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

print("=" * 80)
print("최적 Lambda 탐색: 테스트 성능 기준 분석")
print("=" * 80)

# ==================================================================================
# CONFIGURATION
# ==================================================================================
LAMBDA_VALUES = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
FOLDS_TO_ANALYZE = [1, 2, 3, 4, 5]
RESULTS_DIR = "results"
OUTPUT_DIR = "results/lambda_optimization"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================================================
# Step 1: 최신 타임스탬프 결과 파일 수집
# ==================================================================================
print("\n📂 Step 1: 재평가 결과 파일 수집 중...")

all_results = []

for fold_num in FOLDS_TO_ANALYZE:
    fold_id = f"fold_{fold_num:02d}"
    fold_dir = os.path.join(RESULTS_DIR, fold_id)
    
    if not os.path.exists(fold_dir):
        print(f"  ⚠️  {fold_id} 디렉토리 없음")
        continue
    
    # Lambda별로 최신 파일 찾기
    for lam in LAMBDA_VALUES:
        pattern = os.path.join(fold_dir, f"grid_search_results_reeval_lambda_{lam}_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            print(f"  ⚠️  {fold_id}, λ={lam}: 파일 없음")
            continue
        
        # 최신 파일 선택 (타임스탬프 기준)
        latest_file = sorted(files)[-1]
        
        df = pd.read_csv(latest_file)
        df['lambda'] = lam  # 혹시 컬럼이 없을 경우 대비
        all_results.append(df)

if not all_results:
    print("\n❌ 분석할 데이터가 없습니다!")
    exit(1)

df_all = pd.concat(all_results, ignore_index=True)
print(f"✅ 로딩 완료: {len(df_all):,} records from {len(all_results)} files")
print(f"   Folds: {sorted(df_all['fold'].unique())}")
print(f"   Lambdas: {sorted(df_all['lambda'].unique())}")

# ==================================================================================
# Step 2: Lambda별 성능 집계 (Test RMSE 기준)
# ==================================================================================
print("\n🔬 Step 2: Lambda별 테스트 성능 집계...")

# 각 Lambda에 대해 평균 RMSE, MAD 계산
lambda_summary = df_all.groupby('lambda').agg({
    'RMSE': ['mean', 'std', 'min', 'max'],
    'MAD': ['mean', 'std', 'min', 'max'],
    'alpha': ['mean', 'std', 'min', 'max']
}).round(6)

lambda_summary.columns = ['_'.join(col).strip() for col in lambda_summary.columns.values]
lambda_summary = lambda_summary.reset_index()

# Extreme alpha 비율 계산
extreme_alpha_pct = df_all.groupby('lambda').apply(
    lambda x: (x['alpha'] >= 2.0).mean() * 100
).reset_index(name='extreme_alpha_pct')

lambda_summary = lambda_summary.merge(extreme_alpha_pct, on='lambda')

print("\n📊 Lambda별 테스트 성능 요약:")
print("-" * 100)
print(f"{'λ':>8s} {'RMSE_mean':>12s} {'RMSE_std':>12s} {'MAD_mean':>12s} "
      f"{'Alpha_mean':>12s} {'Extreme_α%':>12s}")
print("-" * 100)

for _, row in lambda_summary.iterrows():
    print(f"{row['lambda']:8.4f} {row['RMSE_mean']:12.6f} {row['RMSE_std']:12.6f} "
          f"{row['MAD_mean']:12.6f} {row['alpha_mean']:12.6f} {row['extreme_alpha_pct']:12.2f}")

# ==================================================================================
# Step 3: 최적 Lambda 찾기
# ==================================================================================
print("\n🎯 Step 3: 최적 Lambda 선정...")

# 1. 순수 RMSE 기준 최적값
best_rmse_idx = lambda_summary['RMSE_mean'].idxmin()
best_rmse = lambda_summary.loc[best_rmse_idx]

print(f"\n1️⃣ 최소 Test RMSE 기준:")
print(f"   λ = {best_rmse['lambda']:.4f}")
print(f"   RMSE = {best_rmse['RMSE_mean']:.6f} ± {best_rmse['RMSE_std']:.6f}")
print(f"   MAD = {best_rmse['MAD_mean']:.6f}")
print(f"   Avg Alpha = {best_rmse['alpha_mean']:.3f}")
print(f"   Extreme Alpha % = {best_rmse['extreme_alpha_pct']:.2f}%")

# 2. Elbow Point 기준 (RMSE 증가 < 0.1% 이내에서 가장 큰 λ)
baseline_rmse = lambda_summary.loc[lambda_summary['lambda'] == min(LAMBDA_VALUES), 'RMSE_mean'].values[0]
threshold_rmse = baseline_rmse * 1.001  # 0.1% tolerance

candidates = lambda_summary[lambda_summary['RMSE_mean'] <= threshold_rmse]

if not candidates.empty:
    recommended_idx = candidates['lambda'].idxmax()
    recommended = lambda_summary.loc[recommended_idx]
    
    print(f"\n2️⃣ 추천 Lambda (Elbow Point - RMSE 0.1% 이내, 최대 λ):")
    print(f"   λ = {recommended['lambda']:.4f}")
    print(f"   RMSE = {recommended['RMSE_mean']:.6f} ± {recommended['RMSE_std']:.6f}")
    print(f"   Baseline 대비: {(recommended['RMSE_mean'] - baseline_rmse) / baseline_rmse * 100:+.4f}%")
    print(f"   MAD = {recommended['MAD_mean']:.6f}")
    print(f"   Avg Alpha = {recommended['alpha_mean']:.3f}")
    print(f"   Extreme Alpha % = {recommended['extreme_alpha_pct']:.2f}%")
else:
    recommended = best_rmse
    print(f"\n2️⃣ 추천 Lambda: 최소 RMSE와 동일")

# 3. Alpha 안정성 고려 (Extreme Alpha < 5% & RMSE 최소)
stable_candidates = lambda_summary[lambda_summary['extreme_alpha_pct'] < 5.0]

if not stable_candidates.empty:
    stable_best_idx = stable_candidates['RMSE_mean'].idxmin()
    stable_best = lambda_summary.loc[stable_best_idx]
    
    print(f"\n3️⃣ Alpha 안정성 고려 (Extreme α < 5%):")
    print(f"   λ = {stable_best['lambda']:.4f}")
    print(f"   RMSE = {stable_best['RMSE_mean']:.6f} ± {stable_best['RMSE_std']:.6f}")
    print(f"   MAD = {stable_best['MAD_mean']:.6f}")
    print(f"   Avg Alpha = {stable_best['alpha_mean']:.3f}")
    print(f"   Extreme Alpha % = {stable_best['extreme_alpha_pct']:.2f}%")

# ==================================================================================
# Step 4: 시각화
# ==================================================================================
print("\n📊 Step 4: 시각화 생성 중...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Lambda Impact on Test Performance', fontsize=16, fontweight='bold')

# Plot 1: RMSE vs Lambda
ax1 = axes[0, 0]
ax1.plot(lambda_summary['lambda'], lambda_summary['RMSE_mean'], 'o-', color='tab:blue', linewidth=2)
ax1.fill_between(lambda_summary['lambda'], 
                  lambda_summary['RMSE_mean'] - lambda_summary['RMSE_std'],
                  lambda_summary['RMSE_mean'] + lambda_summary['RMSE_std'],
                  alpha=0.3, color='tab:blue')
ax1.axvline(best_rmse['lambda'], color='red', linestyle='--', alpha=0.7, label=f"Best λ={best_rmse['lambda']:.3f}")
if 'recommended' in locals() and recommended['lambda'] != best_rmse['lambda']:
    ax1.axvline(recommended['lambda'], color='green', linestyle='--', alpha=0.7, label=f"Recommended λ={recommended['lambda']:.3f}")
ax1.set_xlabel('Lambda (λ)', fontsize=11)
ax1.set_ylabel('Test RMSE', fontsize=11)
ax1.set_title('Test RMSE vs Lambda', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: MAD vs Lambda
ax2 = axes[0, 1]
ax2.plot(lambda_summary['lambda'], lambda_summary['MAD_mean'], 'o-', color='tab:orange', linewidth=2)
ax2.fill_between(lambda_summary['lambda'], 
                  lambda_summary['MAD_mean'] - lambda_summary['MAD_std'],
                  lambda_summary['MAD_mean'] + lambda_summary['MAD_std'],
                  alpha=0.3, color='tab:orange')
ax2.set_xlabel('Lambda (λ)', fontsize=11)
ax2.set_ylabel('Test MAD', fontsize=11)
ax2.set_title('Test MAD vs Lambda', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Alpha Mean vs Lambda
ax3 = axes[1, 0]
ax3.plot(lambda_summary['lambda'], lambda_summary['alpha_mean'], 'o-', color='tab:green', linewidth=2)
ax3.fill_between(lambda_summary['lambda'], 
                  lambda_summary['alpha_mean'] - lambda_summary['alpha_std'],
                  lambda_summary['alpha_mean'] + lambda_summary['alpha_std'],
                  alpha=0.3, color='tab:green')
ax3.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Baseline α=1.0')
ax3.set_xlabel('Lambda (λ)', fontsize=11)
ax3.set_ylabel('Mean Alpha', fontsize=11)
ax3.set_title('Alpha vs Lambda', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Extreme Alpha % vs Lambda
ax4 = axes[1, 1]
ax4.plot(lambda_summary['lambda'], lambda_summary['extreme_alpha_pct'], 'o-', color='tab:red', linewidth=2)
ax4.axhline(5.0, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
ax4.set_xlabel('Lambda (λ)', fontsize=11)
ax4.set_ylabel('Extreme Alpha % (α ≥ 2.0)', fontsize=11)
ax4.set_title('Alpha Stability vs Lambda', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()

# 저장
plot_path = os.path.join(OUTPUT_DIR, 'test_performance_vs_lambda.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"   ✅ 그래프 저장: {plot_path}")

# ==================================================================================
# Step 5: 상세 분석 - Method별 Lambda 영향
# ==================================================================================
print("\n🔍 Step 5: Method별 Lambda 영향 분석...")

method_lambda_summary = df_all.groupby(['method', 'lambda']).agg({
    'RMSE': 'mean',
    'MAD': 'mean',
    'alpha': 'mean'
}).reset_index()

# Method별 최적 Lambda
print("\n📋 Method별 최적 Lambda (Test RMSE 기준):")
print("-" * 60)
print(f"{'Method':>10s} {'Best λ':>10s} {'RMSE':>12s} {'Alpha':>10s}")
print("-" * 60)

method_best_lambdas = []
for method in sorted(df_all['method'].unique()):
    method_data = method_lambda_summary[method_lambda_summary['method'] == method]
    best_idx = method_data['RMSE'].idxmin()
    best = method_data.loc[best_idx]
    method_best_lambdas.append(best)
    print(f"{method:>10s} {best['lambda']:10.4f} {best['RMSE']:12.6f} {best['alpha']:10.3f}")

# ==================================================================================
# Step 6: 결과 저장
# ==================================================================================
print("\n💾 Step 6: 결과 저장 중...")

# Lambda 요약
summary_path = os.path.join(OUTPUT_DIR, 'lambda_test_performance_summary.csv')
lambda_summary.to_csv(summary_path, index=False)
print(f"   ✅ Lambda 요약: {summary_path}")

# Method별 Lambda 영향
method_summary_path = os.path.join(OUTPUT_DIR, 'method_lambda_test_performance.csv')
method_lambda_summary.to_csv(method_summary_path, index=False)
print(f"   ✅ Method별 요약: {method_summary_path}")

# 최적 Lambda 권장사항
recommendations = {
    'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'best_rmse_lambda': float(best_rmse['lambda']),
    'best_rmse_value': float(best_rmse['RMSE_mean']),
    'best_rmse_std': float(best_rmse['RMSE_std']),
    'recommended_lambda': float(recommended['lambda']) if 'recommended' in locals() else float(best_rmse['lambda']),
    'recommended_rmse': float(recommended['RMSE_mean']) if 'recommended' in locals() else float(best_rmse['RMSE_mean']),
    'baseline_rmse': float(baseline_rmse),
    'improvement_pct': float((baseline_rmse - best_rmse['RMSE_mean']) / baseline_rmse * 100),
}

import json
rec_path = os.path.join(OUTPUT_DIR, 'optimal_lambda_recommendation.json')
with open(rec_path, 'w') as f:
    json.dump(recommendations, f, indent=2)
print(f"   ✅ 권장사항: {rec_path}")

# ==================================================================================
# Final Summary
# ==================================================================================
print("\n" + "=" * 80)
print("💡 최종 결론")
print("=" * 80)

print(f"\n✅ 테스트 성능 최적화를 위한 Lambda 선택:")
print(f"   - 최소 RMSE Lambda: {best_rmse['lambda']:.4f} (RMSE={best_rmse['RMSE_mean']:.6f})")

if 'recommended' in locals() and recommended['lambda'] != best_rmse['lambda']:
    print(f"   - 추천 Lambda (안정성 고려): {recommended['lambda']:.4f} (RMSE={recommended['RMSE_mean']:.6f})")
    print(f"     → Baseline 대비 RMSE 증가: {(recommended['RMSE_mean'] - baseline_rmse) / baseline_rmse * 100:+.4f}%")

if 'stable_best' in locals():
    print(f"   - Alpha 안정성 우선: {stable_best['lambda']:.4f} (RMSE={stable_best['RMSE_mean']:.6f}, Extreme α={stable_best['extreme_alpha_pct']:.2f}%)")

print(f"\n📈 성능 개선:")
print(f"   - Baseline (λ={min(LAMBDA_VALUES)}): RMSE={baseline_rmse:.6f}")
print(f"   - 최적값 (λ={best_rmse['lambda']:.4f}): RMSE={best_rmse['RMSE_mean']:.6f}")
print(f"   - 개선율: {(baseline_rmse - best_rmse['RMSE_mean']) / baseline_rmse * 100:.4f}%")

print(f"\n📊 모든 결과는 '{OUTPUT_DIR}' 디렉토리에 저장되었습니다.")
print("=" * 80)
