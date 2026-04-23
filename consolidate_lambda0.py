# -*- coding: utf-8 -*-
"""
consolidate_lambda0.py
----------------------------------------------------------------------
lambda=0 실험 결과를 두 개의 표준 CSV로 통합.

출력 1: results/combined/consolidated_lambda0_final.csv        (grid 결과)
출력 2: results/combined/consolidated_alpha_history_lambda0.csv (alpha 최적화 이력)

[grid 결과 스키마 13열]
  fold, method, alpha, type, K, TopN,
  validation_rmse, validation_precision, validation_recall,
  test_RMSE, test_MAD, test_Precision, test_Recall

[alpha history 스키마 10열]
  fold, method, K, TopN, phase, alpha,
  validation_mse, validation_rmse, validation_precision, validation_recall
  ※ acos/jmsd/msd만 포함 (14개 메서드는 lambda=0 history 미저장)

데이터 출처 (CLAUDE.md 기준):
  [grid]
  - acos/jmsd/msd fold 1~10  : inner_sim/combined/all_folds_grid_results_20260416_004853.csv (20열)
  - 14개 메서드  fold 1       : inner_sim/fold_01/grid_search_results_inner_lambda_0.0_v1_20260408_082727.csv (17열)
  - 14개 메서드  fold 2~6     : inner_sim/combined/all_folds_grid_results_20260415_145918.csv (17열)
  - 14개 메서드  fold 7       : inner_sim/fold_07/grid_search_results_inner_lambda_0.0_v1_20260408_084959.csv (17열)
  - 14개 메서드  fold 8       : inner_sim/fold_08/grid_search_results_inner_lambda_0.0_v1_20260408_090701.csv (17열)
  - 14개 메서드  fold 9       : inner_sim/fold_09/grid_search_results_inner_lambda_0.0_v1_20260408_092409.csv (17열)
  - 14개 메서드  fold 10      : inner_sim/fold_10/grid_search_results_inner_lambda_0.0_v1_20260408_094122.csv (17열)
데이터 출처 (alpha history - 14개 메서드):
  원본 실험 (20260311~12)은 lambda=0.002로 실행. alpha history 파일의 mse 컬럼은
  raw MSE (penalty 미포함). lambda=0 재해석 시 argmin(mse)로 최적 alpha 선택 가능.
  검증: mse + regularization_penalty == regularized_score ← 수식 일치 확인됨.
  - fold_01: inner_sim/fold_01/alpha_optimization_history_20260311_132613.csv
  - fold_02: inner_sim/fold_02/alpha_optimization_history_20260311_232135.csv
  - fold_03: inner_sim/fold_03/alpha_optimization_history_20260312_051052.csv
  - fold_04: inner_sim/fold_04/alpha_optimization_history_20260312_110249.csv
  - fold_05: inner_sim/fold_05/alpha_optimization_history_20260312_171114.csv
  - fold_06: inner_sim/fold_06/alpha_optimization_history_20260311_202254.csv
  - fold_07: inner_sim/fold_07/alpha_optimization_history_20260312_024741.csv
  - fold_08: inner_sim/fold_08/alpha_optimization_history_20260312_090735.csv
  - fold_09: inner_sim/fold_09/alpha_optimization_history_20260312_153137.csv
  - fold_10: inner_sim/fold_10/alpha_optimization_history_20260312_215403.csv
"""

import os
import sys
import numpy as np
import pandas as pd

BASE   = os.path.dirname(os.path.abspath(__file__))
INNER  = os.path.join(BASE, "results", "inner_sim")
OUTDIR = os.path.join(BASE, "results", "combined")

KEEP_COLS = [
    "fold", "method", "alpha", "type", "K", "TopN",
    "validation_rmse", "validation_precision", "validation_recall",
    "test_RMSE", "test_MAD", "test_Precision", "test_Recall",
]

METHODS_3  = {"acos", "jmsd", "msd"}
METHODS_14 = {
    "ami", "ari", "chebyshev", "cosine", "cpcc", "euclidean",
    "ipwr", "itr", "jaccard", "kendall_tau_b", "manhattan",
    "pcc", "spcc", "src",
}
ALL_17 = METHODS_3 | METHODS_14

# ── 소스 정의 ────────────────────────────────────────────────────────────────
SOURCES = [
    {
        "path"   : os.path.join(INNER, "combined", "all_folds_grid_results_20260416_004853.csv"),
        "methods": METHODS_3,
        "folds"  : set(range(1, 11)),
        "label"  : "acos/jmsd/msd  fold 1~10  (20260416, 버그수정)",
    },
    {
        "path"   : os.path.join(INNER, "fold_01", "grid_search_results_inner_lambda_0.0_v1_20260408_082727.csv"),
        "methods": METHODS_14,
        "folds"  : {1},
        "label"  : "14개 메서드  fold 1  (20260408)",
    },
    {
        "path"   : os.path.join(INNER, "combined", "all_folds_grid_results_20260415_145918.csv"),
        "methods": METHODS_14,
        "folds"  : {2, 3, 4, 5, 6},
        "label"  : "14개 메서드  fold 2~6  (20260415)",
    },
    {
        "path"   : os.path.join(INNER, "fold_07", "grid_search_results_inner_lambda_0.0_v1_20260408_084959.csv"),
        "methods": METHODS_14,
        "folds"  : {7},
        "label"  : "14개 메서드  fold 7  (20260408)",
    },
    {
        "path"   : os.path.join(INNER, "fold_08", "grid_search_results_inner_lambda_0.0_v1_20260408_090701.csv"),
        "methods": METHODS_14,
        "folds"  : {8},
        "label"  : "14개 메서드  fold 8  (20260408)",
    },
    {
        "path"   : os.path.join(INNER, "fold_09", "grid_search_results_inner_lambda_0.0_v1_20260408_092409.csv"),
        "methods": METHODS_14,
        "folds"  : {9},
        "label"  : "14개 메서드  fold 9  (20260408)",
    },
    {
        "path"   : os.path.join(INNER, "fold_10", "grid_search_results_inner_lambda_0.0_v1_20260408_094122.csv"),
        "methods": METHODS_14,
        "folds"  : {10},
        "label"  : "14개 메서드  fold 10  (20260408)",
    },
]

# ── 통합 ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  consolidate_lambda0.py")
print("=" * 70)

frames = []
for src in SOURCES:
    print(f"\n[소스] {src['label']}")
    if not os.path.exists(src["path"]):
        print(f"  ERROR: 파일 없음 → {src['path']}")
        sys.exit(1)

    df = pd.read_csv(src["path"])
    raw_rows = len(df)

    # 메서드 필터
    df = df[df["method"].isin(src["methods"])].copy()
    # fold 필터
    df["fold"] = df["fold"].astype(int)
    df = df[df["fold"].isin(src["folds"])].copy()

    after_filter = len(df)
    print(f"  raw={raw_rows}행  →  필터 후={after_filter}행")

    # lambda 열 검증 후 제거
    if "lambda" in df.columns:
        lambda_vals = df["lambda"].astype(float).unique()
        if not all(v == 0.0 for v in lambda_vals):
            print(f"  ERROR: lambda != 0.0 값 발견: {lambda_vals}")
            sys.exit(1)

    # regularization_penalty 검증 (0이어야 함)
    if "regularization_penalty" in df.columns:
        bad = df[df["regularization_penalty"].astype(float).abs() > 1e-9]
        if len(bad) > 0:
            n_bad = len(bad)
            methods_bad = bad["method"].unique().tolist()
            print(f"  WARNING: regularization_penalty != 0 행 {n_bad}개 (methods={methods_bad}) → baseline type의 0.0 외에는 제거")
            # optimized type 중 penalty != 0인 행은 제거
            df = df[~((df["type"] == "optimized") & (df["regularization_penalty"].astype(float).abs() > 1e-9))].copy()
            print(f"  제거 후: {len(df)}행")

    # 표준 13열만 추출 (없는 열은 NaN)
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[KEEP_COLS].copy()

    # 타입 변환
    df["fold"]  = df["fold"].astype(int)
    df["K"]     = df["K"].astype(int)
    df["TopN"]  = df["TopN"].astype(int)
    df["alpha"] = df["alpha"].astype(float)

    folds_got   = sorted(df["fold"].unique())
    methods_got = sorted(df["method"].unique())
    types_got   = sorted(df["type"].unique())
    print(f"  folds={folds_got}")
    print(f"  methods({len(methods_got)}): {methods_got}")
    print(f"  types: {types_got}")

    frames.append(df)

# ── 결합 ─────────────────────────────────────────────────────────────────────
combined = pd.concat(frames, ignore_index=True)

print("\n" + "=" * 70)
print("  통합 검증")
print("=" * 70)

# 중복 검사
dup_keys = ["fold", "method", "type", "K", "TopN"]
dups = combined.duplicated(subset=dup_keys)
if dups.any():
    print(f"  WARNING: 중복 행 {dups.sum()}개 발견!")
    print(combined[dups][dup_keys + ["alpha"]].head(10))
else:
    print(f"  중복 없음 ✓")

# 커버리지 검사
expected_opt = 17 * 10 * 100   # 17 methods × 10 folds × (10K × 10TopN)
actual_opt   = len(combined[combined["type"] == "optimized"])
print(f"\n  optimized 행 수: {actual_opt}  (기대={expected_opt})")
if actual_opt != expected_opt:
    missing = expected_opt - actual_opt
    print(f"  WARNING: {missing}행 부족")
    # 어떤 (fold, method) 조합이 부족한지 확인
    opt_df = combined[combined["type"] == "optimized"]
    coverage = opt_df.groupby(["fold","method"]).size().reset_index(name="count")
    bad_cov  = coverage[coverage["count"] != 100]
    if len(bad_cov) > 0:
        print(f"\n  100행이 아닌 (fold, method) 조합:")
        print(bad_cov.to_string(index=False))
else:
    print(f"  커버리지 완전 ✓")

# NaN 검사
print(f"\n  NaN 비율:")
for col in ["validation_rmse", "validation_precision", "validation_recall",
            "test_RMSE", "test_MAD", "test_Precision", "test_Recall"]:
    nan_n = combined[col].isna().sum()
    if nan_n > 0:
        print(f"    {col}: {nan_n}행 NaN")
    else:
        print(f"    {col}: NaN 없음 ✓")

# ── grid 결과 저장 ────────────────────────────────────────────────────────────
os.makedirs(OUTDIR, exist_ok=True)
out_path = os.path.join(OUTDIR, "consolidated_lambda0_final.csv")
combined.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1024 / 1024

print(f"\n[SAVED] {out_path}")
print(f"  행 수: {len(combined)}")
print(f"  열 수: {combined.shape[1]}  {list(combined.columns)}")
print(f"  파일 크기: {size_mb:.2f} MB")

# ── alpha history 통합 ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Alpha History 통합 (17개 메서드 전체)")
print("=" * 70)

AHIST_KEEP_COLS = [
    "fold", "method", "K", "TopN", "phase", "alpha",
    "validation_mse", "validation_rmse", "validation_precision", "validation_recall",
]

# 열 이름 매핑 (원본 파일은 mse/rmse/precision/recall 사용)
AHIST_COL_RENAME = {
    "mse"       : "validation_mse",
    "rmse"      : "validation_rmse",
    "precision" : "validation_precision",
    "recall"    : "validation_recall",
}

# ── acos/jmsd/msd (lambda=0으로 실험) ─────────────────────────────────────────
AHIST_SOURCE_3 = os.path.join(INNER, "combined", "all_folds_alpha_history_20260416_004853.csv")

# ── 14개 메서드 (원본 lambda=0.002 실험의 alpha history; mse는 raw MSE) ────────
# 수학적 검증: mse + regularization_penalty == regularized_score ← 확인됨
# → mse 컬럼만 사용하면 lambda=0 해석 완전히 유효
AHIST_SOURCES_14 = [
    (1,  os.path.join(INNER, "fold_01", "alpha_optimization_history_20260311_132613.csv")),
    (2,  os.path.join(INNER, "fold_02", "alpha_optimization_history_20260311_232135.csv")),
    (3,  os.path.join(INNER, "fold_03", "alpha_optimization_history_20260312_051052.csv")),
    (4,  os.path.join(INNER, "fold_04", "alpha_optimization_history_20260312_110249.csv")),
    (5,  os.path.join(INNER, "fold_05", "alpha_optimization_history_20260312_171114.csv")),
    (6,  os.path.join(INNER, "fold_06", "alpha_optimization_history_20260311_202254.csv")),
    (7,  os.path.join(INNER, "fold_07", "alpha_optimization_history_20260312_024741.csv")),
    (8,  os.path.join(INNER, "fold_08", "alpha_optimization_history_20260312_090735.csv")),
    (9,  os.path.join(INNER, "fold_09", "alpha_optimization_history_20260312_153137.csv")),
    (10, os.path.join(INNER, "fold_10", "alpha_optimization_history_20260312_215403.csv")),
]

def _load_ahist(path, label, expected_methods=None, allow_nonzero_penalty=False):
    """alpha history 파일을 로드하고 표준 열로 변환."""
    if not os.path.exists(path):
        print(f"  ERROR: 파일 없음 → {path}")
        return None
    ah = pd.read_csv(path)
    print(f"\n[소스] {label}: {len(ah):,}행")

    # penalty 검증
    if "regularization_penalty" in ah.columns:
        pen_vals = ah["regularization_penalty"].astype(float)
        if not allow_nonzero_penalty and pen_vals.abs().max() > 1e-9:
            print(f"  ERROR: regularization_penalty != 0 발견 (max={pen_vals.abs().max():.2e})")
            return None
        elif allow_nonzero_penalty:
            n_nonzero = (pen_vals.abs() > 1e-9).sum()
            print(f"  regularization_penalty: {n_nonzero:,}행 non-zero (mse 컬럼으로 lambda=0 해석) ✓")
        else:
            print(f"  regularization_penalty: all 0.0 ✓")

    # 메서드 필터
    if expected_methods is not None:
        ah = ah[ah["method"].isin(expected_methods)].copy()
    methods_got = sorted(ah["method"].unique())
    folds_got   = sorted(ah["fold"].astype(int).unique())
    print(f"  methods({len(methods_got)}): {methods_got}")
    print(f"  folds: {folds_got}")

    # 열 이름 표준화
    ah = ah.rename(columns=AHIST_COL_RENAME)
    ah["fold"] = ah["fold"].astype(int)
    ah["K"]    = ah["K"].astype(int)
    ah["TopN"] = ah["TopN"].astype(int)

    for col in AHIST_KEEP_COLS:
        if col not in ah.columns:
            ah[col] = np.nan
    return ah[AHIST_KEEP_COLS].copy()

ah_frames = []

# acos/jmsd/msd
ah3 = _load_ahist(AHIST_SOURCE_3, "all_folds_alpha_history_20260416_004853.csv (acos/jmsd/msd)",
                  expected_methods=METHODS_3, allow_nonzero_penalty=False)
if ah3 is not None:
    ah_frames.append(ah3)

# 14개 메서드 (각 fold별)
for fold_num, path in AHIST_SOURCES_14:
    label = f"fold_{fold_num:02d}/alpha_optimization_history (14개 메서드)"
    ah14 = _load_ahist(path, label, expected_methods=METHODS_14, allow_nonzero_penalty=True)
    if ah14 is not None:
        ah_frames.append(ah14)

if ah_frames:
    ah_all = pd.concat(ah_frames, ignore_index=True)

    # 중복 검사
    dup_ah = ah_all.duplicated(subset=["fold", "method", "K", "TopN", "phase", "alpha"])
    if dup_ah.any():
        print(f"\n  WARNING: alpha history 중복 {dup_ah.sum()}행 → 제거")
        ah_all = ah_all[~dup_ah].copy()
    else:
        print(f"\n  중복 없음 ✓")

    # phase 확인
    phases = sorted(ah_all["phase"].unique())
    methods_all_ah = sorted(ah_all["method"].unique())
    folds_all_ah   = sorted(ah_all["fold"].unique())
    print(f"  phases:   {phases}")
    print(f"  methods({len(methods_all_ah)}): {methods_all_ah}")
    print(f"  folds:    {folds_all_ah}")

    # NaN 확인
    nan_mse = ah_all["validation_mse"].isna().sum()
    print(f"  validation_mse NaN: {nan_mse}행")

    # 저장
    ahist_out = os.path.join(OUTDIR, "consolidated_alpha_history_lambda0.csv")
    ah_all.to_csv(ahist_out, index=False)
    ahist_mb = os.path.getsize(ahist_out) / 1024 / 1024

    print(f"\n[SAVED] {ahist_out}")
    print(f"  행 수: {len(ah_all):,}")
    print(f"  열 수: {ah_all.shape[1]}  {list(ah_all.columns)}")
    print(f"  파일 크기: {ahist_mb:.2f} MB")

print("\n" + "=" * 70)
print("  완료")
print("=" * 70)
