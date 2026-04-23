# -*- coding: utf-8 -*-
"""
precompute_sim_full.py
----------------------------------------------------------------------
train.csv (train_full) 기반으로 17개 유사도를 계산해 sim_full_{method}.npy 저장.

open-world Recall/Precision 재평가 (reeval_openworld_recall.py) 의 사전 작업.

실행:
    python precompute_sim_full.py
    python precompute_sim_full.py --folds 1 2 3     # 특정 fold만
    python precompute_sim_full.py --methods cosine pcc  # 특정 메서드만
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from similarities import compute_similarity

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "movielenz_data")

ALL_METHODS = [
    "acos", "ami", "ari", "chebyshev", "cosine", "cpcc",
    "euclidean", "ipwr", "itr", "jaccard", "jmsd",
    "kendall_tau_b", "manhattan", "msd", "pcc", "spcc", "src",
]

# sim_inner 기반의 실측 계산 시간 (초, fold_01 기준)
ESTIMATED_SECONDS = {
    "ami": 344, "ari": 176, "src": 97, "kendall_tau_b": 94,
    "ipwr": 56, "itr": 30, "spcc": 29, "pcc": 26, "cosine": 17,
    "cpcc": 10, "euclidean": 8, "jaccard": 8, "chebyshev": 7,
    "manhattan": 7, "jmsd": 6, "acos": 4, "msd": 3,
}


# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folds",   nargs="+", type=int, default=list(range(1, 11)))
parser.add_argument("--methods", nargs="+", type=str, default=ALL_METHODS)
parser.add_argument("--force",   action="store_true", help="기존 파일 덮어쓰기")
args = parser.parse_args()

FOLDS   = args.folds
METHODS = [m for m in args.methods if m in ALL_METHODS]


# ── 데이터 로딩 헬퍼 ──────────────────────────────────────────────────────────
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
    if str(first_col).lower() in ("item", "items", "item_id", "id", "index", "unnamed: 0"):
        df = df.set_index(first_col)
    df.index.name = None
    df.columns.name = None
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index   = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.sort_index().sort_index(axis=1)


# ── 예상 시간 출력 ────────────────────────────────────────────────────────────
total_est = sum(ESTIMATED_SECONDS.get(m, 30) for m in METHODS) * len(FOLDS)
print("=" * 70)
print("  precompute_sim_full.py")
print("  Source: train.csv (train_full)  |  Output: sim_full_{method}.npy")
print("  Start: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 70)
print(f"  Folds  : {FOLDS}")
print(f"  Methods: {METHODS}")
print(f"  예상 총 시간: ~{total_est//60}분 ({total_est/3600:.1f}시간)")
print()

total_start = time.time()
done_count  = 0
skip_count  = 0
err_count   = 0

for fold in FOLDS:
    fold_id  = f"fold_{fold:02d}"
    fold_dir = os.path.join(DATA_ROOT, fold_id)
    train_csv = os.path.join(fold_dir, "train.csv")

    if not os.path.exists(train_csv):
        print(f"[{fold_id}] WARNING: train.csv not found - skipping")
        continue

    print(f"\n[{fold_id}] Loading train.csv...", end=" ", flush=True)
    t0 = time.time()
    df_train = load_matrix(train_csv)
    print(f"shape={df_train.shape}  ({time.time()-t0:.1f}s)")

    for method in METHODS:
        out_npy  = os.path.join(fold_dir, f"sim_full_{method}.npy")
        out_meta = os.path.join(fold_dir, f"sim_full_{method}_meta.json")

        if os.path.exists(out_npy) and not args.force:
            print(f"  [{method:15s}] already exists - skipping")
            skip_count += 1
            continue

        t1 = time.time()
        est = ESTIMATED_SECONDS.get(method, 30)
        print(f"  [{method:15s}] computing... (est ~{est}s)", end=" ", flush=True)
        try:
            # compute_similarity expects users x items (rows=users)
            # df_train is items x users, so pass df_train.T
            S = compute_similarity(df_train.T, method)
            np.save(out_npy, S.astype(np.float32))
            elapsed = round(time.time() - t1, 2)
            meta = {
                "method":      method,
                "source":      "train_full",
                "shape":       list(S.shape),
                "created_at":  datetime.now().isoformat(),
                "elapsed_sec": elapsed,
                "fold":        fold_id,
                "data_shape":  list(df_train.shape),
            }
            with open(out_meta, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"done  shape={S.shape}  ({elapsed:.1f}s)")
            done_count += 1
        except Exception as e:
            print(f"ERROR: {e}")
            err_count += 1

elapsed_total = time.time() - total_start
print("\n" + "=" * 70)
print(f"  완료: {done_count}개  건너뜀: {skip_count}개  오류: {err_count}개")
print(f"  총 경과: {elapsed_total/60:.1f}분")
print("  End: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 70)
