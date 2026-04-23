import numpy as np
import importlib
import utils.similarities as sim_mod
importlib.reload(sim_mod)

print("=" * 55)
print("  MSD 수정 검증")
print("=" * 55)

# 현재 코드 확인
import inspect
src = inspect.getsource(sim_mod.msd)
print("\n[현재 msd() 소스코드]")
print(src)

# 수계산 케이스: U1=[3,4,4,nan], U2=[1,1,1,5]  공통 3개
# fixed: 1 - ((2/4)^2 + (3/4)^2 + (3/4)^2) / 3 = 1 - 1.375/3 = 0.5417
U1 = np.array([3., 4., 4., np.nan])
U2 = np.array([1., 1., 1., 5.])
msd_val = sim_mod.msd(U1, U2)
expected_fixed = 1 - 1.375 / 3
expected_buggy = 1 - ((2/5)**2 + (3/5)**2 + (3/5)**2) / 3

print("[케이스 1] msd([3,4,4,nan], [1,1,1,5])")
print(f"  현재 결과       : {msd_val:.6f}")
print(f"  fixed 기대값    : {expected_fixed:.6f}  (올바른 /4 공식)")
print(f"  buggy 기대값    : {expected_buggy:.6f}  (잘못된 /5 공식)")
ok = abs(msd_val - expected_fixed) < 1e-6
print(f"  → {'PASS ✓' if ok else 'FAIL ✗'}")

# 극단 케이스: 완전히 반대 (1 vs 5) → fixed=0.0, buggy=0.36
print()
U3 = np.array([1., 1., 1.])
U4 = np.array([5., 5., 5.])
ext = sim_mod.msd(U3, U4)
print("[케이스 2] msd([1,1,1], [5,5,5])  — 완전 반대")
print(f"  현재 결과  : {ext:.6f}  (기대: 0.0)")
print(f"  buggy였다면: {1 - (4/5)**2:.6f}  (= 0.36)")
print(f"  → {'PASS ✓' if abs(ext) < 1e-9 else 'FAIL ✗'}")

# 동일한 사용자 → 1.0
print()
msd_same = sim_mod.msd(U1, np.array([3., 4., 4., np.nan]))
print("[케이스 3] msd([3,4,4,nan], [3,4,4,nan])  — 동일 사용자")
print(f"  현재 결과  : {msd_same:.6f}  (기대: 1.0)")
print(f"  → {'PASS ✓' if abs(msd_same - 1.0) < 1e-9 else 'FAIL ✗'}")

# .npy 파일 범위 확인
print()
from pathlib import Path
npy_path = Path("data/movielenz_data/fold_01/sim_inner_msd.npy")
S = np.load(npy_path)
diag_mask = ~np.eye(S.shape[0], dtype=bool)
vals = S[diag_mask & ~np.isnan(S)]
print(f"[.npy fold_01 sim_inner_msd.npy 범위]")
print(f"  min = {vals.min():.6f}  (buggy였다면 min≥0.36)")
print(f"  max = {vals.max():.6f}")
print(f"  0.36 미만 값 비율: {(vals < 0.36).mean()*100:.2f}%  (0이면 버그)")
print(f"  → {'PASS ✓ — min<0.36 존재, 버그 수정 확인됨' if vals.min() < 0.36 else 'FAIL ✗ — min>=0.36, 버그 미수정 가능성'}")

print()
print("=" * 55)
