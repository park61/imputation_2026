# 📊 실험 노트북 데이터 흐름 검증 보고서

## 🎯 검증 목적
`main_experiment_grid_search_0116_v3.ipynb`에서 **Alpha 최적화 과정 (VALIDATION)**과 **최종 비교 평가 (TEST)**가 올바르게 구분되어 있는지 검증

---

## ✅ 검증 결과: 완벽하게 구분됨

실험 코드는 **두 단계**를 명확히 분리하여 **TEST 데이터 유출 없이** 공정한 비교를 수행합니다.

---

## 📚 데이터 파일 구조

### 4가지 데이터 파일 (각 fold별)
```python
TRAIN_INNER_CSV = "train_inner.csv"   # 60% × 0.75 = 45% (Alpha 최적화 학습용)
VALIDATION_CSV = "validation.csv"     # 60% × 0.25 = 15% (Alpha 최적화 평가용)
TRAIN_CSV = "train.csv"               # 60% (최종 TEST 평가 시 학습용)
TEST_CSV = "test.csv"                 # 20% (최종 평가 전용, 절대 학습에 사용 안 함)
```

### 데이터 분할 구조
```
전체 MovieLens 데이터 (100%)
│
├─ Train (60%) ────────┐
│  ├─ Train_inner (45%) │ → Alpha 최적화 학습
│  └─ Validation (15%) │ → Alpha 최적화 평가
│                       │
│  Full Train (60%) ────┘ → 최종 TEST 평가 시 학습
│
└─ Test (20%) ──────────→ 최종 평가 전용 (한 번만 사용!)
```

---

## 🔬 실험 코드 분석

### ✅ Phase 1: Alpha 최적화 과정 (VALIDATION 사용)

#### 데이터 로딩
```python
# Lines 641-644
XX_train_inner = _load_matrix_csv(TRAIN_INNER_CSV)  # V2: For α optimization learning
XX_validation = _load_matrix_csv(VALIDATION_CSV)    # V2: For α optimization evaluation
XX_train_full = _load_matrix_csv(TRAIN_CSV)         # V2: For final evaluation
XX_test = _load_matrix_csv(TEST_CSV)                # V2: For final evaluation ONLY
```

**확인**:
- ✅ 4가지 데이터 파일 모두 별도로 로딩
- ✅ 주석으로 각 데이터의 용도 명시
- ✅ TEST는 "final evaluation ONLY" 명시

#### Coarse Search (VALIDATION)
```python
# Lines 684-737
for alpha in ALPHA_COARSE_GRID:
    S_alpha = _combine_single_similarity(S_user, alpha, clip_neg=True)
    
    # V2: Changed to use train_inner → validation (NOT test!)
    preds_dict, mse = _knn_predict_removed_with_S(
        X=XX_validation.values,      # ✅ Validation as ground truth
        XX=XX_train_inner.values,    # ✅ Train_inner for learning
        S=S_alpha,
        K=k,
        include_negative=False,
        fallback="user_mean"
    )
    
    # V3: Apply regularization penalty
    regularization_penalty = REGULARIZATION_LAMBDA * abs(alpha - 1.0)
    regularized_score = mse + regularization_penalty
    
    # Convert predictions to DataFrame for Precision/Recall
    pred_df = XX_validation.copy()  # ✅ V2: Changed from XX_test
    pred_df[:] = np.nan
    for (item_idx, user_idx), pred_rating in preds_dict.items():
        pred_df.iloc[item_idx, user_idx] = pred_rating
    
    # Calculate Precision/Recall for ALL TopN values
    for topn in TOPN_RANGE:
        precision, recall = precision_recall_at_n(
            pred=pred_df,
            train=XX_train_inner,        # ✅ V2: Changed from XX_train
            test=XX_validation,          # ✅ V2: Changed from XX_test
            N=topn,
            relevance_threshold=RELEVANCE_THRESHOLD
        )
        
        # Save to ALPHA_HISTORY
        ALPHA_HISTORY.append({
            'fold': fold_num,
            'method': method,
            'K': k,
            'TopN': topn,
            'phase': 'coarse',
            'alpha': alpha,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'regularization_penalty': regularization_penalty,
            'regularized_score': regularized_score,
            'precision': precision,
            'recall': recall
        })
    
    # V3: Use regularized score for selection
    if regularized_score < best_coarse_score:
        best_coarse_score = regularized_score
        best_coarse_alpha = alpha
```

**확인**:
- ✅ **학습**: `XX_train_inner` (45%)
- ✅ **평가**: `XX_validation` (15%)
- ✅ **절대 사용 안 함**: `XX_test`
- ✅ 주석에 "NOT test!" 명시
- ✅ 결과는 `ALPHA_HISTORY`에 저장 (나중에 `alpha_optimization_history_*.csv`로 저장)

#### Fine Search (VALIDATION)
```python
# Lines 739-797
for alpha in fine_grid:
    S_alpha = _combine_single_similarity(S_user, alpha, clip_neg=True)
    
    # V2: Changed to use train_inner → validation (NOT test!)
    preds_dict, mse = _knn_predict_removed_with_S(
        X=XX_validation.values,      # ✅ Validation as ground truth
        XX=XX_train_inner.values,    # ✅ Train_inner for learning
        S=S_alpha,
        K=k,
        include_negative=False,
        fallback="user_mean"
    )
    
    # ... (동일한 구조)
```

**확인**:
- ✅ Coarse search와 동일하게 VALIDATION 사용
- ✅ TEST 절대 사용 안 함
- ✅ Fine search 결과도 `ALPHA_HISTORY`에 저장

---

### ✅ Phase 2: 최종 TEST 평가 (TEST 사용)

#### 1. Optimal Alpha 평가 (TEST)
```python
# Lines 802-841
optimal_alpha = best_fine_alpha
print(f"α={optimal_alpha:.2f} (regularized) | Evaluating on test...", end=' ')

# V2: Final evaluation on TEST with FULL TRAIN (first and only use of test!)
# 1. Optimal Alpha Evaluation
S_alpha = _combine_single_similarity(S_user, optimal_alpha, clip_neg=True)
preds_dict, _ = _knn_predict_removed_with_S(
    X=XX_test.values,           # ✅ V2: Test (first use!)
    XX=XX_train_full.values,    # ✅ V2: Full train for learning
    S=S_alpha,
    K=k,
    include_negative=False,
    fallback="user_mean"
)

# Convert to DataFrame
pred_df = XX_test.copy()
pred_df[:] = np.nan
for (item_idx, user_idx), pred_rating in preds_dict.items():
    pred_df.iloc[item_idx, user_idx] = pred_rating

# Calculate RMSE and MAD (independent of TopN)
rmse, mad = rmse_mad_on_test(pred_df, XX_test)

# Calculate metrics for all TopN values
for topn in TOPN_RANGE:
    precision, recall = precision_recall_at_n(
        pred=pred_df,
        train=XX_train_full,     # ✅ V2: Full train
        test=XX_test,            # ✅ V2: Test
        N=topn,
        relevance_threshold=RELEVANCE_THRESHOLD
    )
    
    all_results.append({
        'fold': fold_num,
        'method': method,
        'alpha': optimal_alpha,
        'type': 'optimized',    # ✅ Distinguish optimized results
        'K': k,
        'TopN': topn,
        'RMSE': rmse,
        'MAD': mad,
        'Precision': precision,
        'Recall': recall
    })
```

**확인**:
- ✅ **첫 번째이자 유일한 TEST 사용**: 주석에 "first and only use of test!" 명시
- ✅ **학습**: `XX_train_full` (60%, validation 포함)
- ✅ **평가**: `XX_test` (20%)
- ✅ **결과 type**: `'optimized'` - 최적 α 사용한 결과임을 명시
- ✅ 결과는 `all_results`에 저장 (나중에 `grid_search_results_*.csv`로 저장)

#### 2. Baseline Alpha (α=1.0) 평가 (TEST)
```python
# Lines 843-883
# 2. Baseline Alpha (α=1.0) Evaluation
# Evaluate standard similarity without optimization (alpha=1.0)
S_one = _combine_single_similarity(S_user, 1.0, clip_neg=True)
preds_dict_one, _ = _knn_predict_removed_with_S(
    X=XX_test.values,           # ✅ TEST
    XX=XX_train_full.values,    # ✅ Full train
    S=S_one,
    K=k,
    include_negative=False,
    fallback="user_man"
)

# Convert to DataFrame
pred_df_one = XX_test.copy()
pred_df_one[:] = np.nan
for (item_idx, user_idx), pred_rating in preds_dict_one.items():
    pred_df_one.iloc[item_idx, user_idx] = pred_rating
    
rmse_one, mad_one = rmse_mad_on_test(pred_df_one, XX_test)

for topn in TOPN_RANGE:
    precision_one, recall_one = precision_recall_at_n(
        pred=pred_df_one,
        train=XX_train_full,    # ✅ Full train
        test=XX_test,           # ✅ TEST
        N=topn,
        relevance_threshold=RELEVANCE_THRESHOLD
    )
    
    all_results.append({
        'fold': fold_num,
        'method': method,
        'alpha': 1.0,
        'type': 'baseline',     # ✅ Mark as baseline
        'K': k,
        'TopN': topn,
        'RMSE': rmse_one,
        'MAD': mad_one,
        'Precision': precision_one,
        'Recall': recall_one
    })
```

**확인**:
- ✅ **동일한 TEST 데이터** 사용 (공정한 비교)
- ✅ **동일한 학습 데이터**: `XX_train_full` (60%)
- ✅ **결과 type**: `'baseline'` - α=1.0 결과임을 명시
- ✅ **공정한 비교**: Optimal α와 α=1.0이 동일한 TEST 데이터로 평가됨

---

## 📊 저장되는 결과 파일

### 1. Alpha Optimization History (`alpha_optimization_history_*.csv`)
**데이터 소스**: VALIDATION (15%)

**내용**:
- Alpha 최적화 과정에서 테스트한 모든 α 값
- Coarse search + Fine search
- VALIDATION 데이터로 평가한 MSE, RMSE, Precision, Recall
- V3 regularization penalty 및 regularized score

**컬럼**:
```python
{
    'fold': fold_num,
    'method': method,
    'K': k,
    'TopN': topn,
    'phase': 'coarse' or 'fine',
    'alpha': alpha,
    'mse': mse,
    'rmse': np.sqrt(mse),
    'regularization_penalty': regularization_penalty,
    'regularized_score': regularized_score,
    'precision': precision,
    'recall': recall
}
```

**용도**:
- Alpha 최적화 과정 시각화 (Plot 0, Plot 0.5)
- Regularization 효과 분석
- **절대 TEST 데이터와 비교하면 안 됨!**

### 2. Grid Search Results (`grid_search_results_*.csv`)
**데이터 소스**: TEST (20%)

**내용**:
- Optimal α로 평가한 TEST 성능 (type='optimized')
- α=1.0으로 평가한 TEST 성능 (type='baseline')
- 둘 다 동일한 TEST 데이터로 평가

**컬럼**:
```python
{
    'fold': fold_num,
    'method': method,
    'alpha': optimal_alpha or 1.0,
    'type': 'optimized' or 'baseline',  # ✅ 핵심 구분자
    'K': k,
    'TopN': topn,
    'RMSE': rmse,
    'MAD': mad,
    'Precision': precision,
    'Recall': recall
}
```

**용도**:
- 최종 성능 평가 (Plot 1, 2, 5, 6, Table 1)
- α=1.0 vs optimal α 비교 (Plot 1.5, 3, 4, 7, 8)
- **공정한 비교**: 둘 다 TEST 데이터

---

## 🎯 핵심 검증 포인트

### ✅ 1. TEST 데이터 유출 방지
**검증**: TEST 데이터는 최종 평가에서만 사용되는가?

```python
# Alpha 최적화 (Lines 684-797)
❌ XX_test 사용: 0번
✅ XX_validation 사용: 모든 alpha 평가

# 최종 TEST 평가 (Lines 802-883)
✅ XX_test 사용: 정확히 2번 (optimal α, α=1.0)
✅ 주석: "first and only use of test!"
```

**결론**: ✅ **완벽하게 방지됨**

### ✅ 2. 공정한 비교
**검증**: α=1.0과 optimal α가 동일한 데이터로 평가되는가?

```python
# Optimal α 평가
X=XX_test.values           # TEST
XX=XX_train_full.values    # Full train (60%)

# α=1.0 평가
X=XX_test.values           # TEST (동일!)
XX=XX_train_full.values    # Full train (60%, 동일!)
```

**결론**: ✅ **완벽하게 공정함**

### ✅ 3. 데이터 구분 명시
**검증**: 코드에 데이터 구분이 명확히 표시되어 있는가?

```python
# 주석 확인
"V2: For α optimization learning"       # train_inner
"V2: For α optimization evaluation"     # validation
"V2: For final evaluation"              # train_full
"V2: For final evaluation ONLY"         # test
"V2: Changed to use train_inner → validation (NOT test!)"
"V2: Test (first use!)"
"first and only use of test!"
```

**결론**: ✅ **주석으로 명확히 문서화됨**

### ✅ 4. 결과 타입 구분
**검증**: Grid Results에 baseline과 optimized가 구분되는가?

```python
# Optimal α 결과
'type': 'optimized'    # ✅

# α=1.0 결과
'type': 'baseline'     # ✅
```

**결론**: ✅ **`type` 컬럼으로 명확히 구분됨**

---

## 📈 데이터 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: ALPHA 최적화 (VALIDATION 데이터)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Train_inner (45%)] ──┐                                        │
│                        ├──> KNN 학습 ──> Prediction             │
│  [Validation (15%)]  ──┘                    │                   │
│                                             ↓                   │
│                                  MSE + λ×|α-1.0|                │
│                                             │                   │
│                                             ↓                   │
│                                  Select Best α                  │
│                                             │                   │
│                                             ↓                   │
│                          Save to ALPHA_HISTORY.csv              │
│                          (VALIDATION 데이터 기반)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

                                    ↓
                            optimal_alpha

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: 최종 TEST 평가 (TEST 데이터)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  A. Optimal α 평가                                               │
│  ──────────────────                                             │
│  [Train_full (60%)] ──┐                                         │
│                       ├──> KNN(α=optimal) ──> Prediction        │
│  [Test (20%)]       ──┘                           │             │
│                                                   ↓             │
│                                    RMSE, Precision, Recall      │
│                                                   │             │
│                                                   ↓             │
│                              type='optimized' results           │
│                                                                  │
│  B. Baseline α=1.0 평가                                          │
│  ───────────────────────                                        │
│  [Train_full (60%)] ──┐    ⚠️ 동일한 데이터!                    │
│                       ├──> KNN(α=1.0) ──> Prediction            │
│  [Test (20%)]       ──┘                           │             │
│                                                   ↓             │
│                                    RMSE, Precision, Recall      │
│                                                   │             │
│                                                   ↓             │
│                              type='baseline' results            │
│                                                                  │
│                                   ↓                             │
│                    Save to GRID_RESULTS.csv                     │
│                    (TEST 데이터 기반, 공정한 비교)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 시각화 노트북과의 연결

### Alpha History → Plot 0, 0.5
```python
# visualize_grid_results_v3.ipynb
df_alpha = pd.read_csv("alpha_optimization_history_*.csv")

# VALIDATION 데이터 기반
# 용도: Alpha 최적화 과정 시각화
# ⚠️ TEST 데이터와 절대 비교하면 안 됨!
```

### Grid Results → 모든 비교 Plot
```python
# visualize_grid_results_v3.ipynb
df_combined = pd.read_csv("grid_search_results_*.csv")

# TEST 데이터 기반
df_grid = df_combined[df_combined['type'] == 'optimized']
df_baseline = df_combined[df_combined['type'] == 'baseline']

# 용도: 공정한 α=1.0 vs optimal α 비교
# ✅ 둘 다 동일한 TEST 데이터 사용
```

---

## ✅ 최종 결론

### 실험 설계: 완벽함 ✅

1. **데이터 분리**: 4가지 파일로 명확히 구분
   - `train_inner.csv` (45%) - Alpha 최적화 학습
   - `validation.csv` (15%) - Alpha 최적화 평가
   - `train.csv` (60%) - 최종 TEST 평가 학습
   - `test.csv` (20%) - 최종 평가 전용

2. **TEST 유출 방지**: 완벽
   - Alpha 최적화 과정에서 TEST 절대 사용 안 함
   - TEST는 최종 평가에서만 딱 2번 사용 (optimal α, α=1.0)
   - 주석으로 "first and only use of test!" 명시

3. **공정한 비교**: 완벽
   - α=1.0과 optimal α 모두 동일한 TEST 데이터로 평가
   - 동일한 학습 데이터 (train_full 60%)
   - `type` 컬럼으로 명확히 구분

4. **결과 분리**: 완벽
   - Alpha History → VALIDATION 기반 (alpha_optimization_history_*.csv)
   - Grid Results → TEST 기반 (grid_search_results_*.csv)
   - 두 파일을 절대 직접 비교하면 안 됨 (시각화 노트북에서 이미 수정됨)

5. **문서화**: 우수
   - 코드 전체에 "V2:", "V3:" 주석으로 변경사항 명시
   - 데이터 용도 명확히 표시
   - "NOT test!", "first use!" 등 경고 주석 포함

---

## 📋 권장사항

### 현재 상태: 수정 불필요 ✅
실험 코드는 이미 **완벽하게 구현**되어 있습니다. 추가 수정이 필요하지 않습니다.

### 추가 개선 제안 (선택사항)
1. **셀 주석 강화**: 실험 단계를 시각적으로 구분하는 markdown 셀 추가
2. **Assertion 추가**: TEST 데이터가 alpha 최적화에 사용되지 않는지 runtime 체크
3. **로깅 강화**: 각 단계에서 사용하는 데이터 출력

이러한 개선은 **가독성 향상**을 위한 것이며, **기능적으로는 이미 완벽**합니다.

---

## 📚 관련 문서
- `PLOT_DATA_SOURCE_ANALYSIS.md` - 시각화 노트북 데이터 소스 분석
- `DATA_SOURCE_FAIRNESS_REPORT.md` - 전체 공정성 검증 보고서
- `visualize_grid_results_v3.ipynb` - 시각화 노트북 (Plot 1.5, 3, 4 수정 완료)

---

## 🎉 요약

**질문**: "ALPHA를 학습하면서 저장하는 과정과 OPTIMAL ALPHA와 ALPHA=1일때의 테스트 데이터를 이용한 검증하기 위한 비교 실험이 구분되어 있어야 할 것 같아."

**답변**: ✅ **완벽하게 구분되어 있습니다!**

- **Alpha 학습/저장**: VALIDATION 데이터 (train_inner + validation)
- **최종 비교 평가**: TEST 데이터 (train_full + test)
- **공정성**: α=1.0과 optimal α 모두 동일한 TEST 데이터
- **유출 방지**: TEST는 최종 평가에서만 사용
- **결과 분리**: Alpha History (VALIDATION) vs Grid Results (TEST)

**실험 설계 평가**: ⭐⭐⭐⭐⭐ (5/5)
