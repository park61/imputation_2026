"""
Nested CV 데이터 분할 스크립트

현재 train_inner를 3개의 Inner Sub-fold로 재분할
각 Sub-fold별로 Inner Train / Inner Val / Inner Test 생성
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

def load_matrix_csv(filepath):
    """CSV 파일을 행렬로 로드"""
    df = pd.read_csv(filepath, low_memory=False)
    
    # 첫 열이 인덱스인 경우 처리
    first_col = df.columns[0]
    if str(first_col).lower() in ("item", "items", "item_id", "id", "index", "unnamed: 0"):
        df = df.set_index(first_col)
    
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def create_inner_fold_split(matrix, fold_idx, seed=42):
    """
    행렬을 Inner Train(67%) / Inner Val(17%) / Inner Test(17%)로 분할
    
    Args:
        matrix: (items, users) 형태의 행렬
        fold_idx: Inner Fold 인덱스 (0, 1, 2)
        seed: 랜덤 시드
    
    Returns:
        (inner_train, inner_val, inner_test) 튜플
    """
    
    np.random.seed(seed + fold_idx)  # 각 fold마다 다른 분할
    
    # NaN이 아닌 위치 찾기
    non_nan_positions = list(zip(*np.where(~matrix.isna())))
    
    if not non_nan_positions:
        raise ValueError("행렬에 값이 없습니다")
    
    # 분할할 위치 인덱스 섞기
    n_samples = len(non_nan_positions)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 67% / 17% / 17% 분할
    train_end = int(n_samples * 0.67)
    val_end = int(n_samples * 0.84)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # 분할된 행렬 생성
    inner_train = matrix.copy()
    inner_val = matrix.copy()
    inner_test = matrix.copy()
    
    # Train: 선택된 위치만 유지, 나머지는 NaN
    train_positions = [non_nan_positions[i] for i in train_indices]
    val_positions = [non_nan_positions[i] for i in val_indices]
    test_positions = [non_nan_positions[i] for i in test_indices]
    
    # Train 마스크 생성
    train_mask = pd.isna(matrix)
    for item_idx, user_idx in train_positions:
        train_mask.iloc[item_idx, user_idx] = False
    inner_train[train_mask] = np.nan
    
    # Val 마스크 생성
    val_mask = pd.isna(matrix)
    for item_idx, user_idx in val_positions:
        val_mask.iloc[item_idx, user_idx] = False
    inner_val[val_mask] = np.nan
    
    # Test 마스크 생성
    test_mask = pd.isna(matrix)
    for item_idx, user_idx in test_positions:
        test_mask.iloc[item_idx, user_idx] = False
    inner_test[test_mask] = np.nan
    
    return inner_train, inner_val, inner_test

def create_nested_cv_splits():
    """모든 Fold에 대해 Nested CV 분할 생성"""
    
    print("=" * 100)
    print("🔧 Nested CV 데이터 분할 생성 중")
    print("=" * 100)
    
    # Fold 1-5 처리
    for fold_num in range(1, 6):
        fold_id = f"fold_{fold_num:02d}"
        fold_dir = f"data/movielenz_data/{fold_id}"
        
        if not os.path.exists(fold_dir):
            print(f"\n⚠️  {fold_id} 디렉토리 없음: {fold_dir}")
            continue
        
        # train_inner 로드
        train_inner_path = os.path.join(fold_dir, "train_inner.csv")
        if not os.path.exists(train_inner_path):
            print(f"\n⚠️  {fold_id} train_inner 파일 없음")
            continue
        
        print(f"\n{'='*100}")
        print(f"처리 중: {fold_id}")
        print(f"{'='*100}")
        
        # train_inner 로드
        train_inner = load_matrix_csv(train_inner_path)
        total_ratings = (~train_inner.isna()).sum().sum()
        
        print(f"  train_inner 로드: {train_inner.shape} ({total_ratings:,} ratings)")
        
        # Nested CV 분할 생성
        nested_cv_dir = os.path.join(fold_dir, "nested_cv")
        os.makedirs(nested_cv_dir, exist_ok=True)
        
        split_info = {
            'fold': fold_num,
            'fold_id': fold_id,
            'total_ratings': total_ratings,
            'inner_folds': {}
        }
        
        for inner_fold in range(3):
            print(f"\n  Inner Fold {inner_fold} 분할 중...", end='')
            
            # 분할 생성
            inner_train, inner_val, inner_test = create_inner_fold_split(
                train_inner, 
                fold_idx=inner_fold,
                seed=42
            )
            
            # 파일로 저장
            train_ratings = (~inner_train.isna()).sum().sum()
            val_ratings = (~inner_val.isna()).sum().sum()
            test_ratings = (~inner_test.isna()).sum().sum()
            
            inner_train.to_csv(os.path.join(nested_cv_dir, f"inner_fold_{inner_fold:02d}_train.csv"))
            inner_val.to_csv(os.path.join(nested_cv_dir, f"inner_fold_{inner_fold:02d}_val.csv"))
            inner_test.to_csv(os.path.join(nested_cv_dir, f"inner_fold_{inner_fold:02d}_test.csv"))
            
            split_info['inner_folds'][inner_fold] = {
                'train_ratings': int(train_ratings),
                'val_ratings': int(val_ratings),
                'test_ratings': int(test_ratings),
                'pct_train': f"{train_ratings/total_ratings*100:.1f}%",
                'pct_val': f"{val_ratings/total_ratings*100:.1f}%",
                'pct_test': f"{test_ratings/total_ratings*100:.1f}%"
            }
            
            print(f" ✓")
            print(f"    Train: {train_ratings:,} ({train_ratings/total_ratings*100:.1f}%)")
            print(f"    Val:   {val_ratings:,} ({val_ratings/total_ratings*100:.1f}%)")
            print(f"    Test:  {test_ratings:,} ({test_ratings/total_ratings*100:.1f}%)")
        
        # 메타정보 저장
        metadata_path = os.path.join(nested_cv_dir, "split_info.json")
        with open(metadata_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\n  ✅ {fold_id} Nested CV 분할 완료")
        print(f"     저장 위치: {nested_cv_dir}")
    
    print("\n" + "=" * 100)
    print("✅ 모든 Fold의 Nested CV 분할 완료")
    print("=" * 100)
    
    print("""
다음 단계:
1. nested_cv_grid_search.py를 실행하여 Inner CV 루프에서 Alpha 최적화 수행
2. nested_cv_analysis.py를 실행하여 Inner Val/Test 성능 비교
3. 과적합 여부 판단 및 최종 결론 도출
    """)

if __name__ == "__main__":
    create_nested_cv_splits()
