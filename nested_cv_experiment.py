"""
Nested Cross-Validation for Alpha Optimization

목표: Alpha 선택 과정 자체를 검증하여 Validation Overfitting 여부 명확화

구조:
  Outer CV (1-5 Folds) - Test용
    └─ Outer Train 
        └─ Inner CV (1-3 Sub-folds)
            ├─ Inner Train (Alpha 학습용)
            ├─ Inner Validation (Alpha 최적화용)
            └─ Inner Test (Alpha 검증용)
    └─ Outer Test (최종 성능 평가)

이를 통해:
1. Alpha 선택 과정이 Inner Validation에 과적합되는지 명확히 파악
2. Inner Test ≈ Outer Test 성능 비교로 일반화 능력 평가
3. 공정한 Alpha=1 vs 최적 Alpha 비교 가능
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import time
import json
from pathlib import Path

def analyze_nested_cv_structure():
    """현재 Fold 구조 분석 및 Nested CV 재분할 계획"""
    
    print("=" * 100)
    print("📊 현재 데이터 분할 구조 분석")
    print("=" * 100)
    
    # Fold 1의 데이터 구조 확인
    fold_01_dir = "data/movielenz_data/fold_01"
    
    files_to_check = {
        "train_inner.csv": "Inner Train (Alpha Learning)",
        "validation.csv": "Inner Validation (Alpha Optimization)",
        "train.csv": "Outer Train (Full)",
        "test.csv": "Outer Test (Final Evaluation)"
    }
    
    print("\n현재 구조 (Outer CV Only):")
    print("  Outer Train (train_inner + validation) = 80%")
    print("    ├─ train_inner: Alpha 최적화 학습용")
    print("    └─ validation: Alpha 최적화 평가용")
    print("  Outer Test = 20%")
    
    actual_files = {}
    for fname, description in files_to_check.items():
        fpath = os.path.join(fold_01_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            actual_files[fname] = {
                "shape": df.shape,
                "ratings": (~df.isna()).sum().sum() if df.size > 0 else 0
            }
            print(f"\n  {fname}:")
            print(f"    - Shape: {df.shape}")
            print(f"    - Ratings: {actual_files[fname]['ratings']:,}")
    
    # ========================================
    # Nested CV 계획
    # ========================================
    print("\n" + "=" * 100)
    print("🔧 제안하는 Nested CV 구조")
    print("=" * 100)
    
    print("""
Nested Cross-Validation (Outer CV 5-Fold × Inner CV 3-Fold):

각 Outer Fold마다:
  ├─ Outer Train (80%)
  │   └─ Split into Inner 3-Folds:
  │       For each Inner Fold:
  │       ├─ Inner Train (67%) - Alpha 학습
  │       ├─ Inner Validation (17%) - Alpha 최적화
  │       └─ Inner Test (17%) - Alpha 검증 ← 과적합 여부 판단!
  │
  └─ Outer Test (20%) - 최종 성능 평가

비교:
  1. Inner Validation MSE ≈ Inner Test MSE: Alpha가 잘 일반화됨
  2. Inner Test MSE ≈ Outer Test RMSE: Alpha가 실제로 좋은 성능
  3. 이 두 조건이 모두 만족되어야 신뢰할 수 있음
    """)
    
    # ========================================
    # 데이터 크기 시뮬레이션
    # ========================================
    print("\n" + "=" * 100)
    print("📈 데이터 분할 시뮬레이션")
    print("=" * 100)
    
    # Fold 01 train_inner 로드
    train_inner_path = os.path.join(fold_01_dir, "train_inner.csv")
    if os.path.exists(train_inner_path):
        train_inner_df = pd.read_csv(train_inner_path)
        total_train_inner = (~train_inner_df.isna()).sum().sum()
        
        print(f"\n현재 train_inner 데이터: {total_train_inner:,} ratings")
        
        # Inner CV 분할 시뮬레이션
        print("\nNested CV 분할 후 (3-Fold Inner):")
        inner_train_size = int(total_train_inner * (2/3))
        inner_val_size = int(total_train_inner * (1/6))
        inner_test_size = total_train_inner - inner_train_size - inner_val_size
        
        print(f"  Inner Train: {inner_train_size:,} ratings (67%)")
        print(f"  Inner Val:   {inner_val_size:,} ratings (17%)")
        print(f"  Inner Test:  {inner_test_size:,} ratings (17%)")
    
    # ========================================
    # 계산량 예상
    # ========================================
    print("\n" + "=" * 100)
    print("⏱️  예상 계산량")
    print("=" * 100)
    
    print("""
현재 구조:
  - 5 Folds × 17 Methods × 30 K values × 5 TopN × 2 Alpha phases = ~25,500 실험

Nested CV (3-Fold Inner):
  - 5 Outer Folds × 3 Inner Folds × 17 Methods × 30 K × 5 TopN × 2 phases
  - = 5 × 3 × 17 × 30 × 5 × 2 = 76,500 실험
  - = 현재 대비 3배 증가

시간 예상:
  - 현재: Fold당 ~2시간 (총 10시간)
  - Nested CV: Fold당 ~6시간 (총 30시간)
    """)
    
    # ========================================
    # 구현 전략
    # ========================================
    print("\n" + "=" * 100)
    print("🎯 구현 전략")
    print("=" * 100)
    
    print("""
Step 1: 현재 데이터 분할 유지
  - 기존 Fold 1-5 구조는 Outer Test로 사용
  
Step 2: train_inner 재분할 (Inner CV)
  - 각 Fold의 train_inner를 3개 Sub-fold로 분할
  - 각 Sub-fold별로 Inner Train / Inner Val / Inner Test 생성
  
Step 3: Inner CV 루프 추가
  - 기존 Grid Search 루프를 Inner CV 루프로 감싸기
  - Inner Val MSE로 Alpha 최적화
  - Inner Test로 최적화된 Alpha 검증
  
Step 4: 결과 분석
  - Inner Val vs Inner Test 비교: Overfitting 여부 판단
  - Inner Test vs Outer Test 비교: 일반화 능력 평가
  - 최종 결론 도출

즉시 구현 우선순위:
  1️⃣  Data 재분할 스크립트 생성
  2️⃣  Inner CV 루프 추가
  3️⃣  결과 분석 및 비교
    """)
    
    return actual_files

if __name__ == "__main__":
    print("\n🚀 Nested Cross-Validation for Alpha Optimization\n")
    
    files = analyze_nested_cv_structure()
    
    print("\n" + "=" * 100)
    print("다음 단계")
    print("=" * 100)
    print("""
1. 이 분석 결과 검토
2. nested_cv_data_split.py 실행 → Inner CV 데이터 생성
3. nested_cv_grid_search.py 실행 → Nested CV 실험 수행
4. nested_cv_analysis.py 실행 → 결과 비교 분석
    """)
