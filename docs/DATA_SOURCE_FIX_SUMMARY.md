# 데이터 소스 수정 요약

## 수정일: 2026년 1월 22일

## 문제 발견

시각화 노트북 `visualize_grid_results_v3.ipynb`의 일부 플롯이 **서로 다른 데이터셋을 비교**하는 문제 발견.

### 데이터셋 구조

```
전체 데이터 분할:
├─ 훈련 데이터 (60%): 모델 학습
├─ 검증 데이터 (20%): Alpha 최적화 수행
└─ 테스트 데이터 (20%): 최종 성능 평가
```

### 데이터 파일

1. **Grid Results** (`all_folds_grid_results_*.csv`):
   - **데이터셋**: 테스트 데이터
   - **용도**: 최종 성능 평가
   - **구조**:
     - `type='baseline'`: α=1.0로 테스트 평가 (5,100 rows)
     - `type='optimized'`: 최적 α로 테스트 평가 (5,100 rows)

2. **Alpha History** (`all_folds_alpha_history_*.csv`):
   - **데이터셋**: 검증 데이터
   - **용도**: Alpha 최적화 과정 기록
   - **구조**: 각 alpha 값별 검증 성능 (183,490 rows)

---

## 문제가 있던 플롯

### ❌ Plot 1.5: Alpha Optimization Impact - K vs Performance

**문제점**:
```python
# 잘못된 구현
- Solid line (optimal α): Grid Results → 테스트 데이터 ✅
- Dashed line (α=1.0): Alpha History → 검증 데이터 ❌
```

**구체적 예시** (FOLD 2, PCC, K=20):
- Grid Results (테스트): α=1.0의 RMSE = **1.001972**
- Alpha History (검증): α=1.0의 RMSE = **0.847825**
- **차이**: 0.154148 (약 15.4% 차이!)

**수정**:
```python
# ✅ 수정된 구현
df_baseline = df_grid[df_grid['type'] == 'baseline'].copy()  # 테스트 데이터
df_optimized = df_grid[df_grid['type'] == 'optimized'].copy()  # 테스트 데이터
```

---

### ❌ Plot 3: Alpha Optimization Impact Analysis (통계)

**문제점**:
```python
# 잘못된 구현
alpha_1_history = df_alpha[df_alpha['alpha'] == 1.0]  # 검증 데이터 ❌
df_optimal = df_grid[...]  # 테스트 데이터 ✅

# improvement 계산이 부정확!
rmse_alpha1 = 검증 데이터
rmse_optimal = 테스트 데이터
```

**수정**:
```python
# ✅ 수정된 구현
df_baseline_test = df_grid[df_grid['type'] == 'baseline'].copy()  # 테스트
df_optimized_test = df_grid[df_grid['type'] == 'optimized'].copy()  # 테스트
```

---

### ❌ Plot 4: Interactive Alpha Improvement by K

**문제점**: Plot 3의 잘못된 데이터를 시각화

**수정**: Plot 3 수정으로 자동 해결

---

## 수정 결과

### ✅ 모든 플롯 데이터 소스 검증

| 플롯 | 데이터 소스 | 상태 |
|------|------------|------|
| Plot 0: Method Performance Comparison | Grid Results (테스트) | ✅ 올바름 |
| Plot 0.5: Alpha Optimization Trajectory | Alpha History (검증) | ✅ 올바름* |
| Plot 1: K vs Performance (Multi-Method) | Grid Results (테스트) | ✅ 올바름 |
| **Plot 1.5: Alpha Impact - K vs Performance** | **Grid Results (테스트)** | **✅ 수정됨** |
| Plot 2: K vs Performance (Single Method) | Grid Results (테스트) | ✅ 올바름 |
| **Plot 3: Alpha Impact Analysis** | **Grid Results (테스트)** | **✅ 수정됨** |
| **Plot 4: Alpha Improvement by K** | **Grid Results (테스트)** | **✅ 수정됨** |
| Plot 5: TopN vs Precision (Multi-Fold) | Grid Results (테스트) | ✅ 올바름 |
| Plot 6: TopN vs Precision (Single Method) | Grid Results (테스트) | ✅ 올바름 |

\* Plot 0.5는 최적화 **과정**을 보여주는 것이므로 검증 데이터 사용이 적절함

---

## 핵심 원칙

### ✅ 올바른 데이터 사용 원칙

1. **성능 비교**: 항상 **테스트 데이터**만 사용
   - Baseline vs Optimized
   - Method vs Method
   - K vs K
   
2. **최적화 과정 시각화**: **검증 데이터** 사용 가능
   - Alpha trajectory (Plot 0.5)
   - Coarse/Fine search 과정
   
3. **절대 금지**: 검증과 테스트 혼용 비교
   - ❌ `검증 데이터의 α=1.0` vs `테스트 데이터의 optimal α`
   - ✅ `테스트 데이터의 α=1.0` vs `테스트 데이터의 optimal α`

---

## 검증 결과

### 수정 후 데이터 검증

```
✅ Plot 1.5 데이터 소스:
   • Baseline (α=1.0):  Grid Results type='baseline' → TEST data
   • Optimized:         Grid Results type='optimized' → TEST data
   → 510 configs, 공정한 비교

✅ Plot 3 데이터 소스:
   • Baseline (α=1.0):  Grid Results type='baseline' → TEST data
   • Optimized:         Grid Results type='optimized' → TEST data
   → 510 configs, 공정한 비교
```

### Alpha Optimization 실제 효과 (테스트 데이터 기준)

```
전체 성능:
• Total configs: 510
• Mean improvement: -1.21%
• Median improvement: 0.00%
• Positive improvements: 272 (53.33%)
• Negative improvements: 238 (46.67%)

Top 5 improvements:
  FOLD 4, ACOS, K=10: 5.82% (α=0.05)
  FOLD 3, ACOS, K=10: 5.56% (α=0.05)
  FOLD 2, ACOS, K=10: 4.99% (α=0.05)
  FOLD 4, ACOS, K=20: 3.83% (α=0.05)
  FOLD 3, ACOS, K=20: 3.68% (α=0.10)

Bottom 5 (degradations):
  FOLD 4, EUCLIDEAN, K=80: -9.28% (α=5.00)
  FOLD 4, EUCLIDEAN, K=90: -9.17% (α=5.00)
  FOLD 4, EUCLIDEAN, K=100: -9.07% (α=5.00)
  FOLD 3, EUCLIDEAN, K=80: -8.39% (α=5.00)
  FOLD 3, EUCLIDEAN, K=90: -8.29% (α=5.00)
```

**해석**:
- ACOS 메서드는 alpha 최적화로 최대 5.82% 개선
- EUCLIDEAN 메서드는 높은 K에서 alpha 최적화가 역효과
- 전체적으로 53.33%는 개선, 46.67%는 baseline보다 나쁨
- Regularization 효과가 메서드/K에 따라 다름을 확인

---

## 수정 파일

- `notebooks/visualize_grid_results_v3.ipynb`:
  - Plot 1.5 데이터 준비 섹션 수정
  - Plot 1.5 시각화 함수 수정
  - Plot 1.5 설명 업데이트
  - Plot 3 데이터 준비 섹션 수정
  - Plot 3 제목 업데이트

---

## 교훈

1. **데이터 분할 명확화**: 훈련/검증/테스트의 역할을 명확히 구분
2. **데이터 소스 문서화**: 각 플롯이 어떤 데이터를 사용하는지 명시
3. **공정한 비교**: 비교 대상은 항상 동일한 데이터셋에서 평가
4. **검증 중요성**: 코드 리뷰 시 데이터 소스 일관성 체크 필수

---

## 관련 문서

- `BUG_FIX_SUMMARY.md`: V3 regularization 버그 수정 기록
- `EXPERIMENT_WORKFLOW_V2.md`: 실험 프로세스 문서
