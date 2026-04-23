# 실험 재수행 계획 (버그 수정 후)

작성일: 2026-04-15  
작성 배경: `similarities.py`의 `msd`, `acos` 구현 버그 수정 후 해당 메서드에 대한 전체 실험을 재수행하기 위한 계획.

---

## 영향 범위

| 메서드 | 버그 내용 | 영향 |
|--------|-----------|------|
| `msd` | `x/5 - y/5` (rmin=1 무시) → 값 범위 [0.36, 1.0]으로 압축 | `.npy` 재생성 + 실험 재수행 |
| `jmsd` | `msd()` 호출하므로 자동 상속 | `.npy` 재생성 + 실험 재수행 |
| `acos` | user-mean + `nan_to_num` 방식 → 값 범위 [-0.013, +0.059]으로 압축 | `.npy` 재생성 + 실험 재수행 |
| 나머지 14개 | 정상 | **재작업 불필요** |

> **lambda 설정**: `REGULARIZATION_LAMBDA = 0` (페널티 없음). Lambda 최적화는 수행하지 않는다.

---

## Phase 0 — 기존 결과물 아카이빙 + 버기 행 삭제

### 전략: 복사 → 삭제

1. 기존 CSV를 **복사**하여 아카이브 (before/after 비교용 원본 보존)
2. 원본 CSV에서 `msd`/`jmsd`/`acos` 행을 **삭제** → 14개 정상 메서드만 남김
3. 재실험 시 스크립트가 3개 메서드 결과를 새 CSV로 저장
4. 분석 시 같은 폴더의 CSV를 `glob+concat`하면 17개 메서드가 자연스럽게 완성됨

이 방법의 장점:
- **분석 코드 수정 불필요**: glob+concat만으로 정상 14개 + 수정된 3개가 합쳐짐
- **원본 보존**: 아카이브에 복사본이 있으므로 before/after 비교 가능
- **직관적**: 어떤 CSV를 열어도 버기 행이 없음

### 0-1. 버기 `.npy` 파일 백업

각 fold 폴더(`fold_01` ~ `fold_10`)에서 3개 파일의 이름을 변경한다.  
**원본을 삭제하지 말고 이름을 바꿔 보존한다.**

```bash
# 프로젝트 루트에서 실행
for f in $(seq -w 1 10); do
    dir="data/movielenz_data/fold_${f}"
    for m in msd jmsd acos; do
        [[ -f "$dir/sim_inner_${m}.npy" ]] && mv "$dir/sim_inner_${m}.npy" "$dir/sim_inner_${m}_buggy.npy"
        [[ -f "$dir/sim_inner_${m}_meta.json" ]] && mv "$dir/sim_inner_${m}_meta.json" "$dir/sim_inner_${m}_buggy_meta.json"
    done
done
```

### 0-2. 기존 실험 결과 CSV 복사 → 아카이브

기존 `results/inner_sim/`의 **전체 디렉토리 구조를** 아카이브로 복사한다.

> **범위 한정**: `results/fold_*/`(과거 V5 실험)와 `results/*.csv`(최상위 집계)는 inner_sim 실험과 무관하므로 정리 대상에서 제외한다.

```bash
tag=$(date +%Y%m%d)
cp -r results/inner_sim "results/archive_buggy_${tag}"
```

> `results/archive_buggy_YYYYMMDD/`에 원본 그대로 보존됨 (before/after 비교용).

### 0-3. 원본 CSV에서 msd/jmsd/acos 행 삭제

아카이브 복사 후, 원본 CSV에서 3개 메서드 행만 제거한다.

```python
# clean_buggy_rows.py (Phase 0 전용 일회성 스크립트)
# 대상: results/inner_sim/ 만 (results/fold_*/ 와 results/*.csv 는 과거 실험이므로 제외)
import pandas as pd, glob, os

BUGGY_METHODS = {'msd', 'jmsd', 'acos'}
csv_patterns = [
    "results/inner_sim/fold_*/grid_search_results_*.csv",
    "results/inner_sim/fold_*/alpha_optimization_history_*.csv",  # 'method' 컬럼 있음 확인 (main_experiment_v6_inner_sim.py:312)
    "results/inner_sim/combined/all_folds_*.csv",
]

for pattern in csv_patterns:
    for fpath in glob.glob(pattern):
        df = pd.read_csv(fpath)
        if 'method' not in df.columns:
            continue
        before = len(df)
        df_clean = df[~df['method'].isin(BUGGY_METHODS)]
        removed = before - len(df_clean)
        if removed > 0:
            df_clean.to_csv(fpath, index=False)
            print(f"  {fpath}: {removed} rows removed ({before} → {len(df_clean)})")
        else:
            print(f"  {fpath}: no buggy rows found")
```

실행 후 상태:
```
results/inner_sim/fold_02/grid_search_results_20260408_084031.csv
  → msd/jmsd/acos 행 제거됨, 14개 메서드만 남음
results/archive_buggy_YYYYMMDD/fold_02/grid_search_results_20260408_084031.csv
  → 원본 그대로 (17개 메서드, 버기 포함)
```

### 0-4. (참고) Phase 4 재실험 후 자동 병합

재실험 시 스크립트가 3개 메서드 결과를 새 CSV로 저장한다:
```
results/inner_sim/fold_02/grid_search_results_20260416_120000.csv  ← msd/jmsd/acos만
```

분석 시 같은 폴더의 CSV를 `glob+concat`하면:
- 기존 CSV → 14개 정상 메서드
- 새 CSV → 3개 수정 메서드
- 합계 = **17개 메서드 완전체** (추가 병합 스크립트 불필요)

---

## Phase 1 — 코드 수정

### 1-1. `utils/similarities.py`

| 위치 | 현재 (버그) | 수정 후 |
|------|------------|---------|
| `msd()` 본문 | `1 - sum((x[mask]/5 - y[mask]/5)**2) / n` | `1 - sum(((x[mask] - y[mask]) / 4)**2) / n` |
| `acos()` 시그니처 | `acos(x, y)` | `acos(x, y, item_means)` |
| `acos()` 본문 | user-mean + `nan_to_num` | item-mean centering (노트북 정의와 동일) |
| `compute_acos()` | `acos`를 `sim_xy`에 직접 전달 (시그니처 불일치) | `item_means = np.nanmean(R, axis=0)` 계산 후 lambda 클로저로 주입 (아래 참조) |

**올바른 `acos` 구현 (참조: `Similarity_measures.ipynb` Cell 8)**:
```python
def acos(x, y, item_means):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    x_adj = x[mask] - item_means[mask]
    y_adj = y[mask] - item_means[mask]
    numerator = np.sum(x_adj * y_adj)
    denominator = np.sqrt(np.sum(x_adj**2)) * np.sqrt(np.sum(y_adj**2))
    return numerator / denominator if denominator != 0 else np.nan
```

**수정된 `compute_acos()` 구현** — `_pairwise_build`는 `sim_xy(x, y)` 형태로만 호출하므로 lambda 클로저로 `item_means`를 주입한다:
```python
def compute_acos(train_df: pd.DataFrame) -> np.ndarray:
    R = train_df.to_numpy(dtype=float, copy=False)
    item_means = np.nanmean(R, axis=0)
    sim_fn = lambda x, y: acos(x, y, item_means)
    return _pairwise_build(R, sim_xy=sim_fn, fill_diagonal=1.0, out_dtype="float32")
```

> **호출부 확인 결과**: `acos()`를 직접 호출하는 곳은 `compute_acos()` 내의 `_pairwise_build` 경유가 유일 (`similarities.py:573`).  
> 디스패처 딕셔너리(`"acos": compute_acos`, `similarities.py:714`)도 `compute_acos()`를 통하므로 **추가 수정 불요**.

### 1-2. `Similarity_measures.ipynb`

- `msd()` 셀: `/5` → `/4` 수정 (노트북도 동일 버그 있음)
- `acos()`는 노트북이 이미 올바름 → 수정 불필요

### 1-3. `main_experiment_v6_inner_sim.py`

```python
# 수정 전
REGULARIZATION_LAMBDA = 0.002

# 수정 후
REGULARIZATION_LAMBDA = 0.0
```

---

## Phase 2 — 수정 검증 (`similarity_verification.ipynb`)

### 2-1. 토이 예제 재계산

`similarities.py` 수정 후 `importlib.reload(sim_mod)`로 최신 코드 반영 → 토이 예제 재실행.

### 2-2. 핵심 assert

| 항목 | 기대값 | 근거 |
|------|--------|------|
| `msd` 최솟값 ≥ 0 | 수정 전 floor=0.36 → 수정 후 0까지 가능 | 정규화 범위 `[0,1]` |
| `msd` U1↔U4 | `1 - ((3-1)²+(4-1)²+(4-1)²) / (3×16)` ≈ 0.375 | 수계산 |
| `acos` 값 범위 ∈ [-1, +1] | 수정 전 [-0.013, +0.059] → 수정 후 범위 정상화 | 코사인 값 정의역 |
| `similarities.py` acos == 노트북 acos | 수치 일치 | 두 구현이 같은 공식 사용 |

### 2-2-1. jmsd 토이 예제

`jmsd`는 `msd()` × `jaccard()`이므로, `msd()` 버그 수정의 영향이 jmsd까지 전파됨을 직접 검증한다.

```python
import numpy as np
import importlib
import utils.similarities as sim_mod
importlib.reload(sim_mod)

U1 = np.array([3., 4., 4., np.nan])
U2 = np.array([1., 1., 1., 5.])

jmsd_val = sim_mod.jmsd(U1, U2)
# 수계산:
#   msd(U1,U2)  = 1 - ((2/4)² + (3/4)² + (3/4)²) / 3
#               = 1 - 1.375/3 ≈ 0.5417
#   jaccard     = 3/4 = 0.75  (공통 3개 / 합집합 4개)
#   jmsd        = 0.75 × 0.5417 ≈ 0.4063
assert abs(jmsd_val - 0.4063) < 1e-3, f"jmsd 불일치: {jmsd_val:.4f} (예상 ≈ 0.4063)"

# 동일 사용자: jmsd = 1.0
U3 = np.array([3., 4., 4., np.nan])
assert sim_mod.jmsd(U1, U3) == 1.0, "identical users → jmsd should be 1.0"

# 버그 수정 확인: 수정 전 ≈ 0.5303, 수정 후 ≈ 0.4063 (더 큰 차이 반영)
print(f"jmsd(U1, U2) = {jmsd_val:.4f}  (예상 ≈ 0.4063)")
print("[PASS] jmsd 수정 확인 완료")
```

> 수정 전 `msd` 버그(/5 방식)로는 `jmsd(U1, U2) ≈ 0.5303` → 수정 후 `≈ 0.4063`으로 감소.  
> 삭제와 재생성이 필요한 jmsd `.npy`는 Phase 0-1에서 이미 처리된다.

### 2-3. `Similarity_measures.ipynb`와 교차 검증

`similarity_verification.ipynb` (similarities.py 결과) 와  
`Similarity_measures.ipynb` (노트북 구현 결과) 의 `msd`, `acos` 행렬이 일치하면 완료.

---

## Phase 3 — `.npy` 재생성

Phase 0에서 버기 파일의 이름을 `*_buggy.npy`로 변경해뒀으므로,  
`precompute_similarity_inner.py`는 기존 파일이 없다고 판단하고 재계산한다.

```bash
# 프로젝트 루트(imputation_project/)에서 실행
python -W ignore precompute_similarity_inner.py
```

단, `METHODS` 리스트에서 3개만 처리하도록 임시 수정하여 시간을 절약할 수 있다:

```python
# precompute_similarity_inner.py 상단
METHODS = ["msd", "jmsd", "acos"]   # 재생성 대상만
```

**완료 후 검증**:
```python
import numpy as np
for method in ["msd", "jmsd", "acos"]:
    S = np.load(f"data/movielenz_data/fold_01/sim_inner_{method}.npy")
    print(f"{method}: min={S[~np.isnan(S)].min():.4f}  max={S[~np.isnan(S)].max():.4f}")
```

예상 결과:
- `msd`: min < 0.36 (버그 수정 확인)
- `acos`: min < -0.1 (버그 수정 확인)

---

## Phase 4 — 실험 재수행

### 4-1. Nested CV 실험 (3개 메서드만)

`main_experiment_v6_inner_sim.py`에서 `METHODS` 리스트를 임시로 3개만 남기면  
기존 14개 메서드의 결과를 건드리지 않고 재실험할 수 있다:

```python
# main_experiment_v6_inner_sim.py 상단
METHODS = ["msd", "jmsd", "acos"]   # 재수행 대상만

REGULARIZATION_LAMBDA = 0.0          # lambda=0, 페널티 없음
```

```bash
python -W ignore main_experiment_v6_inner_sim.py
```

결과는 `results/inner_sim/fold_XX/` 에 저장된다.  
기존 CSV에는 14개 정상 메서드만 남아있으므로 (Phase 0-3), 새 CSV가 추가되면 자연스럽게 합쳐진다.

### 4-2. Alpha grid search 재확인

실험 완료 후:
- `msd`: α가 여전히 10.0으로 쏠리면 → alpha grid 범위를 `[0, 30]`으로 확장하는 추가 실험 필요
- `acos`: bimodal 현상 (K=10 vs K≥20) 해소 여부 확인

---

## Phase 5 — 결과 비교 분석

### 5-1. Before / After 성능 비교

```
results/archive_buggy_YYYYMMDD/   ← 원본 (버기 3개 메서드 포함)
results/inner_sim/                ← 수정된 결과 (14개 정상 + 3개 수정)
```

`msd`, `jmsd`, `acos`의 RMSE/MAE before-after 비교.

### 5-2. 전체 17개 메서드 최종 순위 재정리

`results/inner_sim/` 폴더를 `glob+concat`하면 14개(기존) + 3개(재수행) = 17개가 자동으로 합쳐진다.  
별도 병합 스크립트 없이 바로 최종 순위 산출 가능.

---

## 다른 컴퓨터로 옮길 때 체크리스트

이 폴더(`imputation_project/`)를 그대로 복사하면 실험을 이어받을 수 있다.

### 필수 포함 항목

| 항목 | 경로 | 비고 |
|------|------|------|
| 코드 (수정 완료 상태) | `utils/similarities.py` | Phase 1 완료 후 |
| 실험 스크립트 | `main_experiment_v6_inner_sim.py` | `LAMBDA=0.0`, `METHODS=3개` 임시 설정 상태 |
| 전처리 스크립트 | `precompute_similarity_inner.py` | `METHODS=3개` 임시 설정 상태 |
| 버기 `.npy` 백업 | `fold_XX/sim_inner_{msd,jmsd,acos}_buggy.npy` | Phase 0-1 완료 후 |
| 정상 `.npy` 14개 | `fold_XX/sim_inner_{나머지}.npy` | 재사용 |
| 기존 결과 CSV (행 삭제 완료) | `results/inner_sim/fold_XX/*.csv` | 14개 메서드만 남은 상태 |
| 기존 결과 아카이브 (원본) | `results/archive_buggy_YYYYMMDD/` | before/after 비교용 |

### Python 환경

```
numpy      (2.4.1 이상 권장)
pandas
scipy
scikit-learn
```

```bash
pip install numpy pandas scipy scikit-learn
```

> **주의**: 실행 시 `python -W ignore` 플래그를 사용 (numpy 2.x + 구버전 sklearn 경고 억제).

### 재개 순서 요약

```
# Phase 0 완료 여부 확인 (버기 파일 이름 변경 확인)
ls data\movielenz_data\fold_01\sim_inner_msd*

# Phase 1 완료 여부 확인 (코드 수정 확인)
grep "/ 4" utils\similarities.py

# Phase 3: .npy 재생성
python -W ignore precompute_similarity_inner.py

# Phase 4: 실험 재수행
python -W ignore main_experiment_v6_inner_sim.py
```

---

## 왜 실험 결과가 꼬이지 않는가

### ① 원본 CSV에서 버기 행을 삭제했다

Phase 0-3에서 기존 CSV의 `msd`/`jmsd`/`acos` 행을 제거했으므로,  
어떤 분석 스크립트가 `glob("*.csv")+concat`을 하더라도 **버기 행이 포함될 수 없다**.  
이것이 가장 근본적인 안전장치다.

### ② 아카이브에 원본이 보존되어 있다

삭제 전 `results/archive_buggy_YYYYMMDD/`에 원본을 복사해뒀으므로,  
before/after 비교가 필요하면 아카이브에서 꺼내 쓸 수 있다.  
아카이브 폴더는 `results/inner_sim/`과 경로가 다르므로 분석 glob에 걸리지 않는다.

### ③ 재실험 결과가 자연스럽게 합쳐진다

Phase 4에서 생성되는 새 CSV에는 3개 수정 메서드만 담긴다:
```
results/inner_sim/fold_02/
  grid_search_results_20260408_084031.csv   ← 14개 정상 메서드 (버기 행 삭제됨)
  grid_search_results_20260416_120000.csv   ← 3개 수정 메서드 (새로 생성)
```
`glob+concat` → **17개 메서드 완전체**, 중복이나 버기 행 없음.

### ④ `.npy` 파일명으로 버기/수정 구분

버기 `.npy`를 `*_buggy.npy`로 rename했으므로 `precompute_similarity_inner.py`가  
`sim_inner_{method}.npy`를 찾을 때 버기 파일을 정상으로 오인하지 않는다.

### ⑤ `METHODS` 범위 제한으로 정상 파일 보호

재실험 시 `METHODS = ["msd", "jmsd", "acos"]`로 한정하여  
정상인 14개 메서드의 `.npy`와 결과 파일은 **코드가 아예 접근하지 않는다**.

### ⑥ lambda=0 통일로 비교 조건 일치

`REGULARIZATION_LAMBDA = 0.0`으로 고정하면 14개 기존 결과(lambda=0 파일)와  
3개 재수행 결과가 **동일한 조건**에서 생성된 것으로 병합 가능하다.

### ⑦ Phase 2 검증 게이트

assert를 통과한 뒤에만 `.npy` 재생성 → Phase 4 실험을 진행하므로  
버그 있는 코드로 파일을 생성할 가능성이 없다.

---

*이 문서의 각 Phase는 순서대로 완료한 뒤 다음 Phase로 진행한다.*
