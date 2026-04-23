# 📊 Plot별 데이터 소스 분석: VALIDATION vs TEST 구분

## 🎯 분석 목적
이 노트북의 핵심은 **α=1.0 (baseline) vs 최적 α (optimized)의 성능 비교**입니다.  
각 Plot이 어떤 데이터 소스를 사용하는지 분석하여 **공정한 비교**가 이루어지는지 검증합니다.

---

## 📚 데이터 구조 이해

### 실험 흐름
```
전체 데이터 (100%)
├── Train (60%) → 모델 학습
├── Validation (20%) → Alpha 최적화 (df_alpha)
└── Test (20%) → 최종 성능 평가 (df_grid, df_baseline)
```

### 데이터 파일 구조

#### 1. **Alpha History** (`df_alpha`)
- **CSV**: `all_folds_alpha_history_*.csv`
- **데이터셋**: VALIDATION (20%)
- **내용**: Alpha 최적화 과정에서 테스트한 모든 α 값 (0.1~2.0)
- **컬럼**: fold, method, K, alpha, RMSE, regularization_penalty, regularized_score
- **목적**: Alpha 최적화 과정 시각화 (어떤 α가 선택되었는지)
- **⚠️ 중요**: VALIDATION 데이터이므로 TEST 데이터와 직접 비교 불가능

#### 2. **Grid Results** (`df_combined` → `df_grid`, `df_baseline`)
- **CSV**: `all_folds_grid_results_*.csv`
- **데이터셋**: TEST (20%)
- **내용**: 
  - `type='optimized'`: 최적 α로 평가한 TEST 성능 → `df_grid`
  - `type='baseline'`: α=1.0으로 평가한 TEST 성능 → `df_baseline`
- **컬럼**: fold, method, K, TopN, alpha, type, RMSE, Precision, Recall, MAD
- **목적**: 최종 성능 평가 및 α=1.0 vs 최적 α 비교
- **✅ 공정한 비교 가능**: 동일한 TEST 데이터셋 사용

---

## 🔍 Plot별 데이터 소스 분석

### ✅ Plot 0: Alpha Optimization Trajectory (개별 fold)
**데이터**: `df_alpha` (VALIDATION)

**목적**: 
- Alpha 최적화 과정 시각화
- RMSE가 α 값에 따라 어떻게 변하는지 확인
- V3 regularization이 어떻게 α 선택에 영향을 미치는지 확인

**비교 대상**: 없음 (단순 trajectory 시각화)

**공정성 평가**: ✅ **해당 없음**
- 이 plot은 비교가 아닌 최적화 과정 시각화
- VALIDATION 데이터 사용이 정당함 (alpha는 validation으로 선택됨)

**출력**:
- Blue circles: Coarse search (0.0-10.0, step 0.5)
- Red stars with F# labels: 각 fold의 최적 α
- Y축: RMSE or Regularized Score (MSE + λ×|α-1.0|)

---

### ✅ Plot 0.5: Alpha Optimization Trajectory (multi-fold aggregation)
**데이터**: `df_alpha` (VALIDATION)

**목적**: 
- Plot 0의 multi-fold 버전
- 여러 fold 간 alpha trajectory 평균 확인
- Regularization penalty의 영향 분석

**비교 대상**: 없음 (trajectory 시각화)

**공정성 평가**: ✅ **해당 없음**
- Plot 0과 동일한 목적
- VALIDATION 데이터 사용이 정당함

**출력**:
- Mean trajectory ± 1σ (across folds)
- Individual fold lines (optional)

---

### ✅ Plot 1: K vs Performance (Multi-Fold Aggregation)
**데이터**: `df_grid` (TEST, type='optimized')

**목적**: 
- K 값에 따른 성능 변화 확인
- 최적 α를 사용한 TEST 성능
- Multi-fold 통계 (mean, std, 95% CI)

**비교 대상**: 없음 (단일 조건 성능 확인)

**공정성 평가**: ✅ **정당함**
- TEST 데이터 사용
- Optimal α의 최종 성능만 표시
- α=1.0과 비교하지 않음

**출력**:
- Solid line with markers: Mean RMSE across folds
- Shaded area: ±1 std
- Star (★): Optimal K
- Legend shows n_folds for each method

---

### ⚠️ Plot 1.5: Alpha Optimization Impact - K vs Performance
**데이터**: 
- **Baseline (α=1.0)**: `df_baseline` (TEST, type='baseline') ✅
- **Optimized (최적 α)**: `df_grid` (TEST, type='optimized') ✅

**목적**: 
- α=1.0 vs 최적 α의 TEST 성능 비교
- K 값에 따른 성능 차이 확인

**비교 대상**: Baseline vs Optimized (both TEST)

**공정성 평가**: ✅ **공정함** (수정 후)
- ✅ **수정 전 문제**: Alpha History (VALIDATION) vs Grid Results (TEST) 비교 → 불공정
- ✅ **수정 후**: 둘 다 Grid Results (TEST) 사용 → 공정함
- Step 2에서 이미 분리: `df_grid`, `df_baseline`

**코드**:
```python
# ✅ 수정된 코드
df_baseline_agg = df_baseline.groupby(['fold', 'method', 'K']).agg({
    'RMSE': 'mean',
    'alpha': 'first'
}).reset_index()

df_optimized_agg = df_grid.groupby(['fold', 'method', 'K']).agg({
    'RMSE': 'mean',
    'alpha': 'first'
}).reset_index()
```

**출력**:
- Solid line (●): Optimal α (TEST)
- Dashed line (■): α=1.0 (TEST)
- Shaded area: ±1 std for optimal α
- Star (★): Optimal K

---

### ✅ Plot 2: Box Plot - Fold Distribution Comparison
**데이터**: `df_grid` (TEST, type='optimized')

**목적**: 
- Fold 간 RMSE 분포 확인
- Outlier 탐지
- 안정성 평가

**비교 대상**: 없음 (fold 간 분산 확인)

**공정성 평가**: ✅ **정당함**
- TEST 데이터만 사용
- α=1.0과 비교하지 않음

**출력**:
- Box plot for each method
- Red diamond: Mean
- Blue line: Median
- Box: IQR
- Whiskers: 1.5×IQR

---

### ⚠️ Plot 3: Alpha Optimization Impact Analysis
**데이터**:
- **Baseline (α=1.0)**: `df_baseline` (TEST, type='baseline') ✅
- **Optimized (최적 α)**: `df_grid` (TEST, type='optimized') ✅

**목적**: 
- α=1.0 vs 최적 α의 통계적 비교
- t-test, paired comparison
- 개선 정도 정량화

**비교 대상**: Baseline vs Optimized (both TEST)

**공정성 평가**: ✅ **공정함** (수정 후)
- ✅ **수정 전 문제**: Alpha History (VALIDATION) vs Grid Results (TEST) 비교 → 불공정
- ✅ **수정 후**: 둘 다 Grid Results (TEST) 사용 → 공정함

**코드**:
```python
# ✅ 수정된 코드
# Previously used Alpha History (VALIDATION data) - this was incorrect!
# Note: df_baseline and df_grid are already separated in Step 2

df_baseline_agg_plot3 = df_baseline.groupby(['fold', 'method', 'K']).agg({
    'RMSE': 'mean',
    'alpha': 'first'  # Should be 1.0
}).reset_index()

df_optimized_agg_plot3 = df_grid.groupby(['fold', 'method', 'K']).agg({
    'RMSE': 'mean',
    'alpha': 'first'
}).reset_index()

# Merge for comparison
df_comparison = df_baseline_agg_plot3[['fold', 'method', 'K', 'RMSE']].rename(
    columns={'RMSE': 'rmse_alpha1'}
)
df_comparison = df_comparison.merge(
    df_optimized_agg_plot3[['fold', 'method', 'K', 'RMSE', 'alpha']].rename(
        columns={'RMSE': 'rmse_optimal'}
    ),
    on=['fold', 'method', 'K'],
    how='inner'
)
```

**출력**:
- RMSE improvement distribution
- t-test results (p-value)
- Mean improvement ± std
- Number of configs improved/degraded

---

### ⚠️ Plot 4: Interactive Alpha Improvement by K
**데이터**: `df_comparison` (Plot 3에서 생성된 비교 데이터)

**목적**: 
- Method별 α optimization 효과 시각화
- K별 개선 정도 확인

**비교 대상**: Baseline vs Optimized (both TEST, inherited from Plot 3)

**공정성 평가**: ✅ **공정함** (Plot 3 수정으로 자동 수정됨)
- ✅ **수정 전 문제**: Plot 3이 불공정했으므로 Plot 4도 불공정
- ✅ **수정 후**: Plot 3이 공정해졌으므로 Plot 4도 공정함
- `df_comparison`은 Plot 3에서 생성되므로 Plot 3 수정으로 자동 수정

**출력**:
- Bar chart showing RMSE improvement %
- Positive bars: Improvement (optimal α better)
- Negative bars: Degradation (α=1.0 better)
- Interactive method selection

---

### ✅ Plot 5: TopN vs Precision (Multi-Fold Aggregation)
**데이터**: `df_grid` (TEST, type='optimized')

**목적**: 
- TopN에 따른 Precision 변화
- 최적 α 사용한 TEST 성능

**비교 대상**: 없음 (단일 조건 성능)

**공정성 평가**: ✅ **정당함**
- TEST 데이터만 사용
- α=1.0과 비교하지 않음

**출력**:
- Line plot: TopN vs Precision
- Shaded area: ±1 std
- Mean across folds

---

### ✅ Plot 6: TopN vs Recall (Multi-Fold Aggregation)
**데이터**: `df_grid` (TEST, type='optimized')

**목적**: 
- TopN에 따른 Recall 변화
- 최적 α 사용한 TEST 성능

**비교 대상**: 없음 (단일 조건 성능)

**공정성 평가**: ✅ **정당함**
- TEST 데이터만 사용
- α=1.0과 비교하지 않음

**출력**:
- Line plot: TopN vs Recall
- Shaded area: ±1 std
- Mean across folds

---

### ✅ Plot 7: TopN vs Precision - Alpha Comparison
**데이터**:
- **Baseline (α=1.0)**: `df_baseline_topn` (TEST, type='baseline') ✅
- **Optimized (최적 α)**: `df_grid` (TEST, type='optimized') ✅

**목적**: 
- TopN 변화에 따른 Precision 비교
- α=1.0 vs 최적 α

**비교 대상**: Baseline vs Optimized (both TEST)

**공정성 평가**: ✅ **공정함**
- ✅ 둘 다 TEST 데이터 사용
- Step 3에서 `df_baseline_topn = df_baseline.copy()` 설정
- 공정한 비교

**코드**:
```python
# Step 3에서 설정
if 'df_baseline' in locals() and not df_baseline.empty:
    df_baseline_topn = df_baseline.copy()  # TEST data
```

**출력**:
- Solid line (●): Optimal α (TEST)
- Dashed line (■): α=1.0 (TEST)
- Shaded area: ±1 std
- TopN on X-axis

---

### ✅ Plot 8: TopN vs Recall - Alpha Comparison
**데이터**:
- **Baseline (α=1.0)**: `df_baseline_topn` (TEST, type='baseline') ✅
- **Optimized (최적 α)**: `df_grid` (TEST, type='optimized') ✅

**목적**: 
- TopN 변화에 따른 Recall 비교
- α=1.0 vs 최적 α

**비교 대상**: Baseline vs Optimized (both TEST)

**공정성 평가**: ✅ **공정함**
- Plot 7과 동일한 구조
- 둘 다 TEST 데이터 사용

**출력**:
- Solid line (●): Optimal α (TEST)
- Dashed line (■): α=1.0 (TEST)
- Shaded area: ±1 std
- TopN on X-axis

---

### ✅ Table 1: Method Performance Summary
**데이터**: `df_grid` (TEST, type='optimized')

**목적**: 
- Method별 최종 성능 순위
- Complete methods only (모든 fold에 존재)
- Multi-fold 통계

**비교 대상**: 없음 (순위 매기기)

**공정성 평가**: ✅ **정당함**
- TEST 데이터만 사용
- Optimal α의 최종 성능만 표시

**출력**:
- Rank, Method, RMSE±std, Precision@20, Recall@20, CV%
- Sorted by RMSE mean

---

### ✅ KEY INSIGHTS Section
**데이터**:
- `df_grid` (TEST, optimized)
- `df_baseline` (TEST, baseline)
- `df_comparison_complete` (Plot 3에서 생성, TEST vs TEST)

**목적**: 
- 최종 결과 요약
- Best method 추천
- Alpha optimization effectiveness 평가

**비교 대상**: Baseline vs Optimized (both TEST)

**공정성 평가**: ✅ **공정함** (수정 후)
- ✅ **수정 전 문제**: 일부 메트릭이 validation 데이터 기반이었을 가능성
- ✅ **수정 후**: 
  - Plot 3 수정으로 `df_comparison_complete`가 TEST 데이터 기반
  - Alpha optimization effectiveness: 46.67% degradation (TEST 기준)
  - Method-specific recommendations (TEST 기준)

**출력**:
- Best method (lowest RMSE)
- Most stable method (lowest CV)
- Alpha optimization effectiveness
  - Configs improved: 53.33%
  - Configs degraded: 46.67%
  - Mean improvement: -1.21%
- Production recommendations with alpha guidance

---

## 📊 요약 테이블

| Plot | 데이터 소스 | Baseline (α=1.0) | Optimized (최적 α) | 비교 공정성 | 수정 여부 |
|------|------------|------------------|-------------------|------------|----------|
| **Plot 0** | VALIDATION (df_alpha) | - | - | ✅ 해당없음 | - |
| **Plot 0.5** | VALIDATION (df_alpha) | - | - | ✅ 해당없음 | - |
| **Plot 1** | TEST (df_grid) | - | Optimized | ✅ 정당함 | - |
| **Plot 1.5** | TEST (df_baseline, df_grid) | **TEST** | **TEST** | ✅ **공정함** | ✅ **수정됨** |
| **Plot 2** | TEST (df_grid) | - | Optimized | ✅ 정당함 | - |
| **Plot 3** | TEST (df_baseline, df_grid) | **TEST** | **TEST** | ✅ **공정함** | ✅ **수정됨** |
| **Plot 4** | TEST (df_comparison) | **TEST** | **TEST** | ✅ **공정함** | ✅ **자동수정** |
| **Plot 5** | TEST (df_grid) | - | Optimized | ✅ 정당함 | - |
| **Plot 6** | TEST (df_grid) | - | Optimized | ✅ 정당함 | - |
| **Plot 7** | TEST (df_baseline_topn, df_grid) | **TEST** | **TEST** | ✅ 공정함 | - |
| **Plot 8** | TEST (df_baseline_topn, df_grid) | **TEST** | **TEST** | ✅ 공정함 | - |
| **Table 1** | TEST (df_grid) | - | Optimized | ✅ 정당함 | - |
| **KEY INSIGHTS** | TEST (df_grid, df_baseline, df_comparison_complete) | **TEST** | **TEST** | ✅ **공정함** | ✅ **수정됨** |

---

## 🔴 발견된 문제점 (수정 완료)

### ❌ Plot 1.5 (수정 전)
**문제**: Alpha History (VALIDATION) vs Grid Results (TEST) 비교
- Baseline: Alpha History에서 α=1.0 필터링 → VALIDATION 데이터
- Optimized: Grid Results type='optimized' → TEST 데이터
- **불공정**: 다른 데이터셋 비교

**수정**: 둘 다 Grid Results 사용
- Baseline: `df_baseline` (TEST, type='baseline')
- Optimized: `df_grid` (TEST, type='optimized')

### ❌ Plot 3 (수정 전)
**문제**: 동일 (Plot 1.5와 같은 문제)

**수정**: 둘 다 Grid Results 사용

### ❌ Plot 4 (자동 수정)
**문제**: Plot 3의 `df_comparison` 사용 → Plot 3이 불공정하면 Plot 4도 불공정

**수정**: Plot 3 수정으로 자동 해결

---

## ✅ 수정 후 검증 결과

Terminal 검증 스크립트 실행 결과:

```
✅ Plot 1.5 데이터 소스:
   • Baseline (α=1.0):  Grid Results type='baseline' → TEST data
   • Optimized:         Grid Results type='optimized' → TEST data
   → 둘 다 동일한 TEST 데이터셋으로 평가됨

✅ Plot 3 데이터 소스:
   • Baseline (α=1.0):  Grid Results type='baseline' → TEST data
   • Optimized:         Grid Results type='optimized' → TEST data
   → 둘 다 동일한 TEST 데이터셋으로 평가됨

✅ Plot 4 데이터 소스:
   • Plot 3의 df_comparison 사용
   → Plot 3이 수정되었으므로 자동으로 수정됨

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
결론: 모든 성능 비교가 이제 공정하게 TEST 데이터로 수행됩니다!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  Alpha Optimization 성능 악화 분석:
   • Total configs: 30
   • Negative improvements: 14 (46.67%)
   • Positive improvements: 16 (53.33%)

   ⚠️  주의: 46.7%가 baseline보다 나쁨
       일부 메서드/K에서 alpha 최적화가 비효과적
```

---

## 🎯 결론

### ✅ 현재 상태 (수정 후)
1. **모든 비교 Plot (1.5, 3, 4, 7, 8)**은 **TEST 데이터만** 사용
2. **α=1.0 vs 최적 α 비교**가 **공정**하게 수행됨
3. **VALIDATION vs TEST 혼용 문제 완전 해결**

### 📌 데이터 사용 원칙
1. **Alpha 최적화 과정 시각화** (Plot 0, 0.5): VALIDATION 데이터 사용 정당
2. **최종 성능 평가** (Plot 1, 2, 5, 6, Table 1): TEST 데이터 (optimized만)
3. **α=1.0 vs 최적 α 비교** (Plot 1.5, 3, 4, 7, 8): 둘 다 TEST 데이터 사용 **필수**

### 🔑 핵심 발견
- **46.67%의 설정이 α 최적화로 성능 악화**
- **Method-specific α 전략 필요**
- **V3 regularization (λ=0.01) 조정 필요 가능성**

### 📝 추가 작업 필요
- [ ] Notebook 전체 실행하여 출력 검증
- [ ] 남은 fold (5-10) 실행하여 통계적 신뢰도 향상
- [ ] Method-specific regularization penalty 실험
- [ ] Precision/Recall 기반 alpha optimization 시도

---

## 📚 관련 문서
- `DATA_SOURCE_FIX_SUMMARY.md` - Plot 1.5/3/4 수정 상세 내역
- `RESULTS_ANALYSIS_UPDATE.md` - KEY INSIGHTS 섹션 업데이트
- `EXPERIMENT_WORKFLOW_V2.md` - V3 실험 설계 전체 흐름
