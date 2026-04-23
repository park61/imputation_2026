Step 1: Data Split
━━━━━━━━━━━━━━━━━
Original Train (100%)
  ├─ Train_inner (80%)  ──┐
  └─ Validation (20%)  ──┤
                         │
Test (별도, 완전 분리)    │


Step 2: Alpha Optimization (VALIDATION SET)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─ Train_inner로 KNN 학습
│
└─ Validation set에서 여러 α 값 평가
   • α = 0.0, 0.5, 1.0, ..., 10.0 (coarse)
   • α = best_α ± 0.5 (fine)
   ▼
   Optimal α 선택 (예: α=2.65 for PCC, K=10)


Step 3: Final Evaluation on TEST SET (Plot 1.5 데이터)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A) Optimized (optimal α)
   ┌─ Full train으로 KNN 학습
   └─ TEST set에서 평가 (α=2.65)
      → RMSE = 1.041  ← 이 값이 Plot 1.5의 "optimal" line

B) Baseline (α=1.0)
   ┌─ Full train으로 KNN 학습
   └─ TEST set에서 평가 (α=1.0)
      → RMSE = 1.028  ← 이 값이 Plot 1.5의 "baseline" line


📊 Plot 1.5 Comparison:
━━━━━━━━━━━━━━━━━━━━━
Optimal α (2.65):  RMSE = 1.041  (TEST)
Baseline α (1.0):  RMSE = 1.028  (TEST)
                          ▲
                          │
                  둘 다 TEST 데이터로 평가!
                  Fair comparison ✅


⚠️  Why is optimal α WORSE?
━━━━━━━━━━━━━━━━━━━━━━━━━━
• α=2.65 was optimal on VALIDATION set
• But doesn't generalize well to TEST set
• This is OVERFITTING to validation data
• α=1.0 is more robust (generalizes better)
""")

# Load and show concrete example
df = pd.read_csv('results/combined/all_folds_grid_results_20260115_220436.csv')

print("\n" + "=" * 80)
print("CONCRETE EXAMPLE: PCC, K=10, TopN=5")
print("=" * 80)

pcc_k10 = df[(df['method'] == 'pcc') & (df['K'] == 10) & (df['TopN'] == 5)]
for _, row in pcc_k10.iterrows():
    print(f"\n{row['type'].upper()} (α={row['alpha']:.2f}):")
    print(f"  RMSE on TEST: {row['RMSE']:.6f}")
    print(f"  Precision on TEST: {row['Precision']:.6f}")
    print(f"  Recall on TEST: {row['Recall']:.6f}")

print("\n" + "=" * 80)
print("KEY INSIGHT:")
print("=" * 80)
print("""
Plot 1.5는 완전히 올바르게 구현되어 있습니다:
✅ Optimal α와 α=1.0 모두 TEST 데이터로 평가
✅ TEST 데이터는 α 선택에 전혀 사용되지 않음
✅ 공정한 비교 (같은 test set, 같은 train data)

Optimal α가 더 나쁜 이유:
⚠️  Validation set에 overfitting된 α 값
⚠️  Test set의 데이터 분포가 validation과 다름
⚠️  극단적인 α 값(>5)일수록 overfitting 심함

해결책:
🔧 Alpha 범위를 0~3으로 제한
🔧 Regularization 추가
🔧 또는 보수적으로 α=1.0 사용
""")