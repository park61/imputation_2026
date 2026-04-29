# CLAUDE.md — Imputation Project: 실험 데이터 참조 문서

> 이 파일은 Claude가 매 세션 시작 시 참조하여 데이터 출처 혼동을 방지하기 위해 작성되었습니다.
> 마지막 업데이트: 2026-04-23

---

## 1. 프로젝트 개요

- **목적**: 17개 유사도 메서드 × 10-fold CV로 최적 alpha 탐색 및 test 성능 평가
- **현재 분석 범위**: `lambda = 0` 조건만 (1차 실험 완료)
- **그리드**: K ∈ {10,20,...,100}, TopN ∈ {5,10,...,50}, fold ∈ {1,...,10}

## 2. 17개 유사도 메서드 목록

```
acos, ami, ari, chebyshev, cosine, cpcc, euclidean,
ipwr, itr, jaccard, jmsd, kendall_tau_b, manhattan,
msd, pcc, spcc, src
```

- **acos, jmsd, msd**: 버그 수정 재실험 대상 (20260415~16)
- **나머지 14개**: ami, ari, chebyshev, cosine, cpcc, euclidean, ipwr, itr, jaccard, kendall_tau_b, manhattan, pcc, spcc, src

---

## 3. inner_sim 이란?

- **핵심 원칙**: alpha 최적화(Phase 1)는 반드시 `sim_inner_{method}.npy` (train_inner 기반) 를 사용해야 함
- train_inner → validation으로 alpha 최적화 (데이터 누수 없음)
- train_full → test로 최종 성능 평가
- inner_sim을 사용하지 않은 파일(1~2월 생성)은 **validation 누수 문제** 있음 → 사용 금지

---

## 4. 사용해야 하는 파일 (lambda=0 통합 기준)

모든 유효 데이터는 `results/inner_sim/` 경로 안에 있음.

| 메서드 | fold | 사용 파일 | 비고 |
|--------|------|-----------|------|
| acos, jmsd, msd | 1~10 | `results/inner_sim/combined/all_folds_grid_results_20260416_004853.csv` | 20열, 버그수정 버전 |
| 14개 메서드 | 1 | `results/inner_sim/fold_01/grid_search_results_inner_lambda_0.0_v1_20260408_082727.csv` | 17열 (lambda열 포함) |
| 14개 메서드 | 2~6 | `results/inner_sim/combined/all_folds_grid_results_20260415_145918.csv` | 17열 (lambda열 포함) |
| 14개 메서드 | 7 | `results/inner_sim/fold_07/grid_search_results_inner_lambda_0.0_v1_20260408_084959.csv` | 17열 |
| 14개 메서드 | 8 | `results/inner_sim/fold_08/grid_search_results_inner_lambda_0.0_v1_20260408_090701.csv` | 17열 |
| 14개 메서드 | 9 | `results/inner_sim/fold_09/grid_search_results_inner_lambda_0.0_v1_20260408_092409.csv` | 17열 |
| 14개 메서드 | 10 | `results/inner_sim/fold_10/grid_search_results_inner_lambda_0.0_v1_20260408_094122.csv` | 17열 |

**예상 통합 결과**: ~34,000행 × 13열 표준 스키마

---

## 5. 절대 사용 금지 파일

| 파일 | 이유 |
|------|------|
| `results/fold_07~10/grid_search_results_20260128_*.csv` | 1월 생성, **non-inner_sim** (validation 누수) |
| `results/fold_10/grid_search_results_20260129_063658.csv` | 동일 이유 |
| `results/combined/all_folds_grid_results_20260117_003457.csv` | 옛 버전, test 지표만 있고 validation 없음 |
| `results/inner_sim/fold_01/grid_search_results_20260311_132613.csv` | lambda≠0 (penalty에 non-zero 혼재) |
| `results/inner_sim/fold_*/grid_search_results_20260312_*.csv` | lambda≠0 혼재 |
| `results/archive_buggy_20260415/` 하위 전체 | 버그 있는 acos/jmsd/msd 결과 |

---

## 6. 표준 통합 스키마 (13열)

```
fold, method, alpha, type, K, TopN,
validation_rmse, validation_precision, validation_recall,
test_RMSE, test_MAD, test_Precision, test_Recall
```

- **type**: `optimized` 또는 `baseline` (alpha=1.0 고정)
- 원본 파일의 중복 열 (`RMSE`, `MAD`, `Precision`, `Recall`)은 `test_*` 복사본이므로 통합 시 제거
- `lambda` 열이 있는 파일은 검증 후 제거 (모두 0.0임을 확인함)

---

## 7. 컬럼 형식 차이 요약

| 형식 | 열수 | 해당 파일 |
|------|------|-----------|
| 20열 | fold~Recall + RMSE/MAD/Precision/Recall (중복) | 20260416, 20260311_132613, fold_07~10 (20260128) |
| 17열 | fold~Recall + lambda | 20260415, lambda_0.0_v1 파일들 |
| 10열 | fold~Recall (validation 없음) | 20260117 (사용 금지) |

---

## 8. 통합 출력 파일

- **grid 결과**: `results/combined/consolidated_lambda0_final.csv` — 34,000행 × 13열
- **alpha history**: `results/combined/consolidated_alpha_history_lambda0.csv` — 675,300행 × 10열, 67.3 MB
  - **17개 메서드 × 10 fold 전체 포함**
  - acos/jmsd/msd: `inner_sim/combined/all_folds_alpha_history_20260416_004853.csv` (lambda=0 직접 실험)
  - 14개 메서드: `inner_sim/fold_XX/alpha_optimization_history_2026031*.csv` (lambda=0.002 실험이지만 `mse` 컬럼은 raw MSE)
  - 검증: `mse + regularization_penalty == regularized_score` (수식 일치 확인 완료)
  - → `validation_mse` 컬럼으로 lambda=0 최적 alpha 재구성 완전히 유효
- 통합 스크립트: `consolidate_lambda0.py`
- 시각화 노트북: `visualization_lambda0.ipynb`

---

## 9. 주요 실험 스크립트

| 스크립트 | 설명 |
|----------|------|
| `main_experiment_v6_inner_sim.py` | acos/jmsd/msd 전용, fold 1~10, lambda=0 |
| `main_experiment_v6_inner_sim_2_6.py` | 14개 메서드, fold 2~6, lambda=0.002 (구버전) |
| `main_experiment_v6_inner_sim_1_lambda.py` | 14개 메서드, fold 1, lambda=0 재실험용 |
| `main_experiment_v6_inner_sim_lambda.py` | 14개 메서드, lambda 재실험용 |
| `precompute_similarity_inner.py` | sim_inner_{method}.npy 생성 |
| `reeval_openworld_recall.py` | Open-World Recall/Precision 재평가 (sim_inner 사용) |
| `precompute_sim_full.py` | sim_full_{method}.npy 생성 (현재 실험에서는 불필요) |

---

## 10. 데이터 디렉토리 구조

```
results/
├── inner_sim/            ← 유효한 실험 결과 (inner_sim 기반)
│   ├── fold_01/          ← lambda_0.0_v1_20260408 파일 사용
│   ├── fold_02~06/       ← combined/20260415 파일에 통합됨
│   ├── fold_07~10/       ← lambda_0.0_v1_20260408 파일 사용
│   └── combined/
│       ├── all_folds_grid_results_20260416_004853.csv  ← acos/jmsd/msd (fold 1~10)
│       └── all_folds_grid_results_20260415_145918.csv  ← 14개 메서드 (fold 2~6)
├── combined/             ← 구버전 통합 파일 (사용 금지)
├── fold_01~10/           ← 1~2월 비inner_sim 결과 (사용 금지)
└── archive_buggy_20260415/  ← 버그 버전 (사용 금지)

data/movielenz_data/
└── fold_01~10/
    ├── train_inner.csv   ← alpha 최적화용 입력
    ├── validation.csv    ← alpha 최적화 평가 대상
    ├── train.csv         ← test 평가용 입력
    ├── test.csv          ← 최종 성능 평가 대상
    ├── sim_inner_{method}.npy  ← precompute 결과 (train_inner 기반)
    └── sim_full_{method}.npy   ← train.csv 기반 (fold_01/cosine만 존재, 현재 미사용)
```

---

## 11. Open-World Recall/Precision 재평가 실험

### 왜 이 실험이 필요한가?

기존 실험(`consolidated_lambda0_final.csv`)의 `test_Precision`, `test_Recall`은 **Closed-World** 방식으로 계산됨:
- 후보 아이템 = `train에 없고 test에 있는 아이템` → 유저당 평균 ~8개
- TopN = 5~50이므로 **후보 전체가 TopN 안에 들어가는 경우가 대부분** (TopN=50이면 99.4% 유저가 해당)
- 결과: 모든 메서드에서 Recall ≈ 1.0, Precision ≈ 거의 동일 → **메서드 간 차이를 전혀 볼 수 없음**

**해결**: 후보 아이템을 `train에 없는 모든 아이템`(~1,587개)으로 확장하면 TopN=50이어도 후보의 3%만 추천 → 메서드별로 실질적인 랭킹 능력 차이가 드러남.

---

### 실험 설계

| 항목 | Closed-World (기존) | Open-World (이 실험) |
|------|---------------------|----------------------|
| 후보 아이템 | `train.isna() & ~test.isna()` | `train.isna()` |
| 후보 수/유저 | ~8개 | ~1,587개 |
| relevant 정의 | 후보 중 test에서 ≥4.0 | 동일 |
| TopN=50일 때 | 후보 거의 전부 추천 | 후보의 3.2%만 추천 |
| 메서드 간 차이 | 거의 없음 | **실질적으로 드러남** |

---

### Alpha와 Recall의 관계 (중요)

본 실험의 예측 공식은 다음과 같다:

$$\hat{r}(u, i) = \frac{\sum_{v \in \mathcal{N}_K(u,i)} S_{\text{eff}}(u,v) \cdot r(v,i)}{\sum_{v \in \mathcal{N}_K(u,i)} S_{\text{eff}}(u,v)}, \quad S_{\text{eff}}(u,v) = \max(S(u,v),\, 0)^{\alpha}$$

α는 CF 항에 곱해지는 스케일이 아니라, **유사도에 붙는 지수(power exponent)**다.

α가 바뀌면 $S_{\text{eff}}$의 분포와 이웃 집합 $\mathcal{N}_K(u,i)$가 달라지므로, 아이템 간 랭킹이 수학적으로 보존된다는 보장은 없다.

실험에서 baseline(α=1)과 optimized(α=α*)의 Recall이 동일하게 나오는 이유는 두 가지 가능성이 있다:
- **경험적 안정성**: 대부분의 경우 α 변화가 상위 K 이웃 구성과 아이템 간 순위를 실질적으로 바꾸지 않음
- **Closed-World 구조적 한계**: 후보 아이템이 ~8개뿐이라 α와 무관하게 Recall≈1.0에 수렴

→ 이 실험에서 baseline vs optimized Recall 비교는 의미 없음 (스키마 일관성을 위해 양쪽 모두 저장)  
→ **실질 분석 포인트: 메서드 간 비교** (cosine vs euclidean vs pcc 등)

---

### 입력 데이터

- **유사도**: `sim_inner_{method}.npy` 사용 (sim_full 불필요)
  - sim_inner = train_inner 기반 (train_full의 80%)
  - 목적이 메서드 간 비교이므로 80% vs 100% 차이는 결과에 미미한 영향
  - sim_full 계산 비용 (~2.6시간) 절약
- **alpha**: `consolidated_lambda0_final.csv`의 (fold, method, K)별 alpha* 재사용
- **train/test**: `data/movielenz_data/fold_XX/train.csv`, `test.csv`

---

### 알고리즘 (성능 최적화)

```
1. (fold, method, alpha) 단위로 predict_all_unseen_batch_k() 호출
   - K_max(=100)번 반복하면서 (n_items × n_users) 행렬에 이웃 기여 누적
   - K=10, 20, ..., 100 체크포인트에서 예측 행렬 저장
   - Python 유저 루프 없음 — 완전 numpy 행렬 연산

2. 각 K에 대해 precision_recall_batch_topn() 호출
   - argpartition으로 상위 N_max 추출 (1회)
   - TopN=5,10,...,50 전체를 한 번에 계산
   - Python 유저 루프 없음
```

총 alpha call 수: **1,390회** (17메서드 × 10fold × 평균 ~8.2 alpha그룹)  
alpha 그룹 수가 적은 이유: K가 달라도 alpha*가 같은 경우 묶어서 1회 처리

---

### 출력 파일

**`results/combined/openworld_recall_lambda0.csv`**

```
스키마 (8열):
  fold       : 1~10
  method     : 17개 메서드명
  K          : 10, 20, ..., 100
  TopN       : 5, 10, 15, ..., 50
  type       : "baseline" (α=1.0) 또는 "optimized" (α=α*)
  alpha      : 실제 사용된 alpha 값
  recall_ow  : Open-World Recall@TopN (유저 평균)
  precision_ow: Open-World Precision@TopN (유저 평균)

총 행 수: 34,000행 = 17 × 10 × 10 × 10 × 2
저장 방식: (fold, method) 완료 시마다 CSV 전체 재작성 (중간 저장)
중단 후 재실행: python reeval_openworld_recall.py --resume
```

---

### 분석 시 주의사항

1. **baseline vs optimized Recall 차이는 항상 0** — alpha ranking 불변성 때문 (수학적 필연)
2. **meaningful 비교 = 메서드 간 비교** — 같은 (fold, K, TopN, type) 조건에서 메서드별 recall_ow 비교
3. **K 효과**: K가 클수록 더 많은 이웃 사용 → 일반적으로 recall 증가하나 noisy 이웃 포함 가능성도 증가
4. **TopN 효과**: TopN 클수록 recall 증가, precision 감소 (tradeoff)
5. **fold 간 분산**: 10-fold 평균 ± 표준편차로 메서드 신뢰성 평가

---

### 실행 명령

```bash
python reeval_openworld_recall.py          # 전체 실행 (~63분)
python reeval_openworld_recall.py --resume # 중단 후 이어서 실행
python reeval_openworld_recall.py --folds 1 2 --methods cosine pcc  # 부분 실행
```

---

### 예상 소요 시간 (실측 기준)

| 항목 | 시간 |
|------|------|
| 예측 (1,390 calls × 2.7s) | 62분 |
| 평가 (3,400 K평가, 완전 벡터화) | 1.5분 |
| **합계** | **약 63분** |
