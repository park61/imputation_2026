# 'item_id' 컬럼 버그 완전 수정 보고서 ✅

## 📌 문제 요약
MovieLens 데이터의 train.csv/test.csv 파일에 **phantom 'item_id' 컬럼**이 포함되어:
- 데이터 shape: (1682, **944**) ← 본래 943 users여야 함
- Alpha 최적화 결과: α=9.55, MSE=19,468 ← 완전히 잘못된 값
- RMSE ≈ 139.5 ← 1-5 scale에서 말도 안 되는 값

---

## 🔍 버그의 근본 원인

### 1단계: 데이터 생성 (create_k_fold_data_0826.ipynb)
```python
# pivot_df의 index.name = 'item_id'
pivot_df = df.pivot_table(index="item_id", columns="user_id", values="rating")

# ❌ 문제: to_csv()가 index.name을 CSV 첫 번째 셀에 기록
pivot_df.to_csv("./movielenz_data/original_matrix.csv")
```

**결과**:
```csv
item_id,1,2,3,4,...,943
1,5.0,4.0,...
2,3.0,...
```
→ 'item_id'가 첫 번째 컬럼명으로 기록됨

### 2단계: K-fold 데이터 생성
```python
# original_matrix.csv를 읽어서 fold별 train/test 생성
df = pd.read_csv("./movielenz_data/original_matrix.csv", index_col=0)

# ❌ 문제: to_csv()가 또다시 index.name 기록
FT['train'].to_csv(fold_dir / "train.csv")
```

**결과**: 
- `fold_01/train.csv`, `fold_01/test.csv` 모두 'item_id' 컬럼 포함
- Shape: (1682, 944) instead of (1682, 943)

### 3단계: 데이터 로딩 (find_optimized_alpha_1106.ipynb)
```python
# ❌ 문제: 'item_id'가 감지 조건에 없음
def _load_matrix_csv(path):
    first_col = df.columns[0]  # 'item_id'
    if str(first_col).lower() in ("item", "items", "id", "index", "unnamed: 0"):
        df = df.set_index(first_col)  # ← 실행 안 됨!
```

**결과**:
- 'item_id' 컬럼이 데이터의 일부로 남음
- XX_outer.shape = (1682, 944) ← 944번째 컬럼은 'item_id' (전부 NaN)

### 4단계: 최적화 실패
- Holdout indices가 column 943 참조 (존재하지 않는 user)
- Similarity 계산 시 944×944 matrix 생성 (943×943이어야 함)
- 잘못된 prediction → 엉터리 MSE → 잘못된 alpha

---

## ✅ 완전한 해결책

### Fix 1: 데이터 생성 시점 (create_k_fold_data*.ipynb)
```python
# Step 3: CSV 저장 전에 index/column name 제거
pivot_df.index.name = None
pivot_df.columns.name = None
pivot_df.to_csv("./movielenz_data/original_matrix.csv")
```

### Fix 2: K-fold 저장 시점 (save_kfold_datasets)
```python
def save_kfold_datasets(folds, save_path):
    for i, FT in enumerate(folds, start=1):
        train_df = FT['train'].copy()
        test_df = FT['test'].copy()
        
        # 🔧 Bug fix: Clear names before saving
        train_df.index.name = None
        train_df.columns.name = None
        test_df.index.name = None
        test_df.columns.name = None
        
        train_df.to_csv(fold_dir / "train.csv")
        test_df.to_csv(fold_dir / "test.csv")
```

### Fix 3: 데이터 로딩 시점 (_load_matrix_csv)
```python
def _load_matrix_csv(path):
    # ...
    first_col = df.columns[0]
    # 🔧 Added 'item_id' to detection list
    if str(first_col).lower() in ("item", "items", "item_id", "id", "index", "unnamed: 0"):
        df = df.set_index(first_col)
    
    # 🔧 Clear names after setting index
    df.index.name = None
    df.columns.name = None
    # ...
```

### Fix 4: 데이터 정렬 시점 (_align_frames)
```python
def _align_frames(a, b):
    items = sorted(set(a.index).union(set(b.index)))
    users = sorted(set(a.columns).union(set(b.columns)))
    a_aligned = a.reindex(index=items, columns=users)
    b_aligned = b.reindex(index=items, columns=users)
    
    # 🔧 Clear names after reindex (reindex preserves names!)
    a_aligned.index.name = None
    a_aligned.columns.name = None
    b_aligned.index.name = None
    b_aligned.columns.name = None
    
    return a_aligned, b_aligned
```

---

## 📊 수정 전후 비교

### Before (Buggy)
```
Data shape:        (1682, 944) ❌ 944 columns
Observed entries:  91,255
Best alpha:        9.55 ❌
Best MSE:          19,468.64 ❌
Best RMSE:         139.53 ❌
```

### After (Fixed)
```
Data shape:        (1682, 943) ✅ 943 users
Observed entries:  89,573 (correct!)
Best alpha:        1.40 ✅
Best MSE:          0.98 ✅
Best RMSE:         0.99 ✅ (excellent for 1-5 scale!)
```

---

## 🎯 핵심 교훈

### pandas의 to_csv() 동작
```python
df = pd.DataFrame([[1,2],[3,4]], index=['a','b'], columns=['x','y'])
df.index.name = 'INDEX_NAME'
df.to_csv("test.csv")

# 결과 CSV:
# INDEX_NAME,x,y  ← index.name이 첫 셀에 기록됨!
# a,1,2
# b,3,4
```

### 올바른 사용법
```python
# CSV 저장 전에 항상 name 제거
df.index.name = None
df.columns.name = None
df.to_csv("test.csv")

# 결과 CSV:
# ,x,y  ← 이제 깔끔함
# a,1,2
# b,3,4
```

---

## 📁 수정된 파일 목록

1. ✅ `create_k_fold_data.ipynb`
   - Cell 3: `pivot_df.index.name = None` 추가
   - Cell 3 (save_kfold_datasets): name clearing 추가

2. ✅ `create_k_fold_data_0826.ipynb`
   - Cell 4 (save_kfold_datasets): name clearing 추가

3. ✅ `find_optimized_alpha_1106.ipynb`
   - Cell 2 (_load_matrix_csv): 'item_id' 추가, name clearing
   - Cell 2 (_align_frames): name clearing after reindex

4. ✅ `alpha_external_compare_1106.ipynb`
   - load_csv function: 'item_id' 추가, name clearing

---

## 🚀 다음 단계

### 1. 새로운 데이터 생성 (권장)
수정된 `create_k_fold_data_0826.ipynb`를 실행하여:
- `original_matrix.csv` 재생성 (clean, no 'item_id')
- `fold_01/train.csv`, `fold_01/test.csv` 재생성 (943 columns)

### 2. 기존 코드로도 작동
데이터 로딩 함수가 수정되어 기존 buggy CSV도 올바르게 처리:
- _load_matrix_csv가 'item_id' 감지하여 index로 설정
- 결과: 자동으로 943 columns

---

## ✨ 결론

**3단계 방어**로 완전한 해결:
1. **생성 시**: CSV 저장 전 name clearing (근본 수정)
2. **저장 시**: fold 데이터 저장 전 name clearing (2차 방어)
3. **로딩 시**: 'item_id' 감지 및 name clearing (3차 방어)

이제 어느 단계에서든 clean data 보장! 🎉

---

## 🔬 Precomputed Similarity 검증

### Q: precompute_similarity_0828.ipynb도 buggy data로 계산했나?

**A: 아니오! Similarity는 올바르게 계산되었습니다.** ✅

### 검증 과정

**1. Shape 확인**:
```
Existing similarity: (943, 943) ✅ Correct!
```

**2. CSV Loading 방식**:
```python
# precompute_similarity_0828.ipynb
train = pd.read_csv(train.csv, index_col=0)  # 'item_id' → index
# → (1682, 943) ✅ Correct shape!
```

**3. 재계산 비교**:
```
Max difference: 0.00000003 (부동소수점 오차 수준)
Mean difference: 0.00000001
```

### 왜 precompute_similarity는 괜찮았나?

| Tool | Loading Method | Result |
|------|----------------|--------|
| `precompute_similarity_0828.ipynb` | `index_col=0` ✅ | (1682, 943) ✅ |
| `find_optimized_alpha_1106.ipynb` | No `index_col` ❌ | (1682, 944) ❌ |

**핵심**: `pd.read_csv(..., index_col=0)`가 'item_id'를 자동으로 index로 처리하여 데이터에서 제거!

### User Ordering 차이

- `load_fold` 함수: User ID를 **문자열 정렬** (`['1', '10', '100', ...]`)
- 이는 train/test alignment를 위한 것
- Similarity 값 자체는 영향 없음 (단지 순서만 다름)

### 결론

✅ **기존 precomputed similarity 파일 그대로 사용 가능**
- 재계산 불필요
- Similarity 값은 완전히 정확함 (max diff = 0.00000003)

🔧 **권장사항**:
- CSV 파일 자체를 clean하게 재생성 (선택사항, 근본 해결)
- 현재 파일로도 `index_col=0` 사용하면 문제없음

---

**작성일**: 2024-11-06
**작성자**: GitHub Copilot + 사용자
**검증**: 
- find_optimized_alpha_1106.ipynb: 실행 결과 확인 완료
- precompute_similarity_0828.ipynb: Similarity 재계산 비교 완료
