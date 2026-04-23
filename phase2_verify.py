#!/usr/bin/env python3
"""Phase 2 검증 스크립트 — msd / jmsd / acos 수정 확인"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import importlib
import utils.similarities as sim_mod
importlib.reload(sim_mod)
import numpy as np
import pandas as pd

print("=== Phase 2 검증 ===\n")

# ── msd ──────────────────────────────────────────────────────────────────────
U1 = np.array([3., 4., 4., np.nan])
U3 = np.array([1., 1., 1., np.nan])

msd_val = sim_mod.msd(U1, U3)
expected_msd = 1 - ((3-1)**2 + (4-1)**2 + (4-1)**2) / (3 * 16)   # ≈ 0.5417
print(f"msd(U1, U3) = {msd_val:.4f}  (예상 ≈ {expected_msd:.4f})")
assert abs(msd_val - expected_msd) < 1e-6, f"msd 불일치: {msd_val}"

full_mat = np.array([[3., 4., 4., 1.],
                     [1., 1., 1., 5.],
                     [5., 5., 5., 1.]])
vals = [sim_mod.msd(full_mat[i], full_mat[j])
        for i in range(3) for j in range(3) if i != j]
vals = [v for v in vals if not np.isnan(v)]
assert min(vals) >= 0,    f"msd 최솟값 < 0: {min(vals)}"
assert min(vals) < 0.36,  f"msd 최솟값이 여전히 0.36 이상 (버그 미수정): {min(vals)}"
print(f"msd 범위: [{min(vals):.4f}, {max(vals):.4f}]  (수정 전 floor=0.36 → 수정 후 0 가능 확인)")
print("[PASS] msd\n")

# ── jmsd ─────────────────────────────────────────────────────────────────────
U2 = np.array([1., 1., 1., 5.])
jmsd_val     = sim_mod.jmsd(U1, U2)
expected_jmsd = 0.75 * (1 - 1.375 / 3)    # ≈ 0.4063
print(f"jmsd(U1, U2) = {jmsd_val:.4f}  (예상 ≈ {expected_jmsd:.4f})")
assert abs(jmsd_val - expected_jmsd) < 1e-3, f"jmsd 불일치: {jmsd_val}"
assert sim_mod.jmsd(U1, np.array([3., 4., 4., np.nan])) == 1.0, "identical users → jmsd should be 1.0"
print("[PASS] jmsd\n")

# ── acos ─────────────────────────────────────────────────────────────────────
R = np.array([[4., 3., np.nan, 2.],
              [3., np.nan, 2., 4.],
              [np.nan, 2., 5., 3.]], dtype=float)
item_means = np.nanmean(R, axis=0)
v0, v1 = R[0], R[1]

acos_val = sim_mod.acos(v0, v1, item_means)
print(f"acos(row0, row1) = {acos_val:.4f}")
assert -1.0 <= acos_val <= 1.0, f"acos 범위 벗어남: {acos_val}"
print("[PASS] acos 범위 정상")

train_df = pd.DataFrame(R)
S = sim_mod.compute_acos(train_df)
print(f"compute_acos 행렬 shape: {S.shape}")
assert S[0, 0] == 1.0, "대각 원소 != 1.0"
off_diag = S[~np.eye(S.shape[0], dtype=bool)]
valid = off_diag[~np.isnan(off_diag)]
if len(valid) > 0:
    print(f"compute_acos off-diag 범위: [{valid.min():.4f}, {valid.max():.4f}]")
    assert valid.min() >= -1.0 and valid.max() <= 1.0, "acos 범위 벗어남"
print("[PASS] compute_acos\n")

print("=== 모든 Phase 2 검증 PASS ===")
