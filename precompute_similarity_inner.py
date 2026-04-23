# -*- coding: utf-8 -*-
"""
precompute_similarity_inner.py
----------------------------------------------------------------------
train_inner.csv 기반으로 17개 유사도를 재계산해 sim_inner_{method}.npy 저장.

기존 sim_{method}.npy (train_full 기반)와 구별하기 위해 별도 파일명 사용.

실행:
    python precompute_similarity_inner.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from datetime import datetime

# utils 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from similarities import compute_similarity, available_methods

# 설정
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "movielenz_data")
FOLDS     = list(range(1, 11))
METHODS   = ["msd", "jmsd", "acos"]   # 재생성 대상만 (버그 수정된 3개)

# 데이터 로딩 헬퍼
def load_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    lower = [str(c).lower() for c in df.columns]
    if {"item", "user", "rating"}.issubset(lower):
        def col(name):
            for c in df.columns:
                if str(c).lower() == name: return c
        mat = df.pivot(index=col("item"), columns=col("user"), values=col("rating"))
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat.index   = mat.index.astype(int)
        mat.columns = mat.columns.astype(int)
        return mat.sort_index().sort_index(axis=1)
    first_col = df.columns[0]
    if str(first_col).lower() in ("item","items","item_id","id","index","unnamed: 0"):
        df = df.set_index(first_col)
    df.index.name = None; df.columns.name = None
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index   = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.sort_index().sort_index(axis=1)

# 메인 루프
print("=" * 70)
print("  precompute_similarity_inner.py")
print("  Source: train_inner.csv  |  Output: sim_inner_{method}.npy")
print("  Start: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 70)
print("  Folds  : " + str(FOLDS))
print("  Methods: " + str(METHODS))
print()

total_start = time.time()

for fold in FOLDS:
    fold_id  = f"fold_{fold:02d}"
    fold_dir = os.path.join(DATA_ROOT, fold_id)
    inner_csv = os.path.join(fold_dir, "train_inner.csv")

    if not os.path.exists(inner_csv):
        print(f"[{fold_id}] WARNING: train_inner.csv not found - skipping")
        continue

    print(f"\n[{fold_id}] Loading train_inner.csv...", end=" ", flush=True)
    t0 = time.time()
    df_inner = load_matrix(inner_csv)
    print(f"shape={df_inner.shape}  ({time.time()-t0:.1f}s)")

    for method in METHODS:
        out_npy  = os.path.join(fold_dir, f"sim_inner_{method}.npy")
        out_meta = os.path.join(fold_dir, f"sim_inner_{method}_meta.json")

        if os.path.exists(out_npy):
            print(f"  [{method:15s}] already exists - skipping")
            continue

        t1 = time.time()
        print(f"  [{method:15s}] computing...", end=" ", flush=True)
        try:
            # compute_similarity expects user x item DataFrame (rows=users)
            # df_inner is items x users, so we pass df_inner.T
            S = compute_similarity(df_inner.T, method)
            np.save(out_npy, S.astype(np.float32))
            meta = {
                "method":      method,
                "source":      "train_inner",
                "shape":       list(S.shape),
                "created_at":  datetime.now().isoformat(),
                "elapsed_sec": round(time.time() - t1, 2),
                "fold":        fold_id,
                "data_shape":  list(df_inner.shape),
            }
            with open(out_meta, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"done  shape={S.shape}  ({time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")

print("\n" + "=" * 70)
print(f"  Total elapsed: {(time.time()-total_start)/60:.1f} min")
print("  End: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 70)
