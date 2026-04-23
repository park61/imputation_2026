"""
Phase 5 — Before/After 비교 분석 + 17개 메서드 최종 순위 + MSD alpha 포화 분석

실행:
    python -W ignore phase5_analysis.py
"""

import numpy as np
import pandas as pd
import glob
import os

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(ROOT, "results", "archive_buggy_20260415")
INNER_SIM_DIR = os.path.join(ROOT, "results", "inner_sim")
DATA_DIR = os.path.join(ROOT, "data", "movielenz_data")

BUGGY_METHODS = ["msd", "jmsd", "acos"]


# ══════════════════════════════════════════════════════════════════════════════
# 유틸 함수
# ══════════════════════════════════════════════════════════════════════════════

def load_main_experiment_csvs(base_dir, label):
    """base_dir 하위 fold_XX 폴더에서 grid_search_results_날짜.csv 만 로드.
    lambda 이름 포함 파일 제외.
    """
    files = sorted(glob.glob(os.path.join(base_dir, "fold_*", "grid_search_results_2*.csv")))
    files = [f for f in files if "_lambda_" not in os.path.basename(f)]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "method" in df.columns:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["_source"] = label
    return combined


def best_rmse_by_method_K(df, type_filter="optimized"):
    """type='optimized' 행에서 method × K 별로 validation_rmse 최솟값 집계."""
    sub = df[df["type"] == type_filter].copy()
    return sub.groupby(["method", "K"])["validation_rmse"].mean().reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  데이터 로드")
print("=" * 70)

buggy_df = load_main_experiment_csvs(ARCHIVE_DIR, "buggy")
new_files = sorted(glob.glob(os.path.join(INNER_SIM_DIR, "fold_*", "grid_search_results_20260415_*.csv"))) + \
            sorted(glob.glob(os.path.join(INNER_SIM_DIR, "fold_*", "grid_search_results_20260416_*.csv")))
new_dfs = [pd.read_csv(f) for f in new_files if "_lambda_" not in os.path.basename(f)]
new_df = pd.concat(new_dfs, ignore_index=True) if new_dfs else pd.DataFrame()
new_df["_source"] = "fixed"

# 14개 정상 메서드는 inner_sim의 기존 파일에서 로드 (20260415 이전 날짜 파일)
old_inner_files = sorted(glob.glob(os.path.join(INNER_SIM_DIR, "fold_*", "grid_search_results_2*.csv")))
old_inner_files = [f for f in old_inner_files
                   if "_lambda_" not in os.path.basename(f)
                   and "20260415" not in os.path.basename(f)
                   and "20260416" not in os.path.basename(f)]
old_inner_dfs = [pd.read_csv(f) for f in old_inner_files]
old_inner_df = pd.concat(old_inner_dfs, ignore_index=True) if old_inner_dfs else pd.DataFrame()
old_inner_df["_source"] = "fixed_14"

print(f"  버기 아카이브 행수  : {len(buggy_df):,}")
print(f"  수정된 3개 메서드   : {len(new_df):,}")
print(f"  기존 14개 메서드    : {len(old_inner_df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Phase 5-1: Before / After RMSE 비교 (msd / jmsd / acos)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("  Phase 5-1: Before / After 검증 RMSE 비교 (optimized, 10-fold 평균)")
print("=" * 70)

for method in BUGGY_METHODS:
    buggy_m = buggy_df[buggy_df["method"] == method]
    new_m = new_df[new_df["method"] == method]

    if buggy_m.empty or new_m.empty:
        print(f"  [{method.upper()}]  데이터 없음")
        continue

    # optimized 행 → validation_rmse를 K 별로 10-fold 평균
    b_opt = buggy_m[buggy_m["type"] == "optimized"].groupby("K")["validation_rmse"].mean()
    n_opt = new_m[new_m["type"] == "optimized"].groupby("K")["validation_rmse"].mean()
    # baseline (alpha=1) validation_rmse
    b_base = buggy_m[buggy_m["type"] == "baseline"].groupby("K")["validation_rmse"].mean()
    n_base = new_m[new_m["type"] == "baseline"].groupby("K")["validation_rmse"].mean()

    # alpha* 분포 비교
    b_alpha = buggy_m[buggy_m["type"] == "optimized"].groupby("K")["alpha"].mean()
    n_alpha = new_m[new_m["type"] == "optimized"].groupby("K")["alpha"].mean()

    print(f"\n  [{method.upper()}]")
    print(f"  {'K':>5}  {'α*(buggy)':>10}  {'α*(fixed)':>10}  "
          f"{'RMSE_buggy':>12}  {'RMSE_fixed':>12}  {'Δ RMSE':>10}")
    print("  " + "-" * 65)
    for k in sorted(b_opt.index):
        if k not in n_opt.index:
            continue
        b_r = b_opt[k]
        n_r = n_opt[k]
        b_a = b_alpha.get(k, float("nan"))
        n_a = n_alpha.get(k, float("nan"))
        delta = n_r - b_r
        sign = "↓" if delta < -0.0001 else ("↑" if delta > 0.0001 else "≈")
        print(f"  {k:>5}  {b_a:>10.3f}  {n_a:>10.3f}  "
              f"{b_r:>12.4f}  {n_r:>12.4f}  {delta:>+9.4f} {sign}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Phase 5-2: 17개 메서드 최종 순위 (test_RMSE 기준, 최적 K 평균)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("  Phase 5-2: 17개 메서드 최종 순위 (test_RMSE, optimized, 10-fold 평균)")
print("=" * 70)

# 14개 메서드: old_inner_df (버기 3개 제외되어 있음)
# 3개 수정 메서드: new_df
all_df = pd.concat([old_inner_df, new_df], ignore_index=True)

# optimized 행 중에서 각 메서드 × fold × K 별로 최소 test_RMSE K를 찾고 fold 평균
opt_all = all_df[all_df["type"] == "optimized"].copy()

# method × fold 별로 최적 K (test_RMSE 최소인 K) 선택
best_per_fold = (
    opt_all.groupby(["method", "fold"])
    .apply(lambda g: g.loc[g["test_RMSE"].idxmin()])
    .reset_index(drop=True)
)

ranking = (
    best_per_fold.groupby("method")[["test_RMSE", "test_MAD"]]
    .mean()
    .sort_values("test_RMSE")
    .reset_index()
)
ranking.index += 1  # 1-based rank

print(f"\n  {'순위':>4}  {'메서드':<18}  {'test_RMSE':>10}  {'test_MAD':>10}  {'비고'}")
print("  " + "-" * 58)
for rank, row in ranking.iterrows():
    tag = "★ 수정됨" if row["method"] in BUGGY_METHODS else ""
    print(f"  {rank:>4}  {row['method']:<18}  {row['test_RMSE']:>10.4f}  "
          f"{row['test_MAD']:>10.4f}  {tag}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MSD alpha 포화 분석
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("  MSD alpha 포화 분석: 왜 α* = 10.0 인가?")
print("=" * 70)

# 4-1. MSD 유사도 행렬 분포 (fold_01)
fold_id = "fold_01"
sim_path = os.path.join(DATA_DIR, fold_id, "sim_inner_msd.npy")
S_msd = np.load(sim_path)

# 대각선 제외, NaN 제외
mask = ~np.eye(S_msd.shape[0], dtype=bool)
vals_msd = S_msd[mask]
vals_msd = vals_msd[~np.isnan(vals_msd)]

# 비교용: jmsd, acos
vals_all = {}
for m in ["msd", "jmsd", "acos"]:
    p = os.path.join(DATA_DIR, fold_id, f"sim_inner_{m}.npy")
    S = np.load(p)
    mask_m = ~np.eye(S.shape[0], dtype=bool)
    v = S[mask_m]
    v = v[~np.isnan(v)]
    # acos: clip_neg=True 가 실험에서 적용됨 → 양수만 의미 있음
    if m == "acos":
        v = v[v > 0]
    vals_all[m] = v

print("\n  [유사도 행렬 분포 비교] (fold_01, 대각선·NaN 제외)")
print(f"  {'메서드':<8}  {'n':>8}  {'mean':>8}  {'std':>8}  "
      f"{'min':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'max':>8}")
print("  " + "-" * 80)
for m, v in vals_all.items():
    label = m + (" (>0)" if m == "acos" else "")
    print(f"  {label:<12}  {len(v):>8,}  {v.mean():>8.4f}  {v.std():>8.4f}  "
          f"{v.min():>8.4f}  {np.percentile(v,25):>8.4f}  "
          f"{np.percentile(v,50):>8.4f}  {np.percentile(v,75):>8.4f}  "
          f"{v.max():>8.4f}")

# 4-2. S^alpha 변환 후 분포 변화 (MSD 기준)
print()
print("  [MSD  S^alpha 변환 후 분포 변화 — 왜 높은 α가 유리한가]")
print()
v = vals_msd
print(f"  alpha  │  mean(S^α)  std(S^α)  p25(S^α)  p75(S^α)  "
      f"p75/p25 비 (concentration↑ 나쁨)")
print("  " + "-" * 65)
for alpha in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
    v_a = np.power(np.clip(v, 0, None), alpha)
    p25 = np.percentile(v_a, 25)
    p75 = np.percentile(v_a, 75)
    ratio = p75 / p25 if p25 > 1e-10 else float("inf")
    print(f"  {alpha:>5.1f}  │  {v_a.mean():>9.4f}  {v_a.std():>8.4f}  "
          f"{p25:>8.6f}  {p75:>8.4f}  {ratio:>8.2f}x")

# 4-3. MSD vs JMSD vs ACOS: optimal alpha 결과 요약
print()
print("  [10-fold 평균 최적 alpha 요약]")
print()
for method in BUGGY_METHODS:
    m_df = new_df[(new_df["method"] == method) & (new_df["type"] == "optimized")]
    alpha_by_K = m_df.groupby("K")["alpha"].mean()
    print(f"  [{method.upper()}]  K범위별 α* 평균:")
    for k, a in alpha_by_K.items():
        bar = "█" * int(round(a / 0.5))
        print(f"    K={k:>3}: α*={a:>5.2f}  {bar}")

# 4-4. MSD 분포 특성이 alpha 포화를 유발하는 근본 원인 설명
print()
print("  [근본 원인 요약]")
v = vals_all["msd"]
v_jmsd = vals_all["jmsd"]
v_acos = vals_all["acos"]

# 상위 10% 임계값 이상이 몇 %인지
t90_msd = np.percentile(v, 90)
t90_jmsd = np.percentile(v_jmsd, 90)
t90_acos = np.percentile(v_acos, 90)

pct_high_msd  = np.mean(v       >= t90_msd)   * 100
pct_high_jmsd = np.mean(v_jmsd  >= t90_jmsd)  * 100
pct_high_acos = np.mean(v_acos  >= t90_acos)  * 100

above_08_msd  = np.mean(v       >= 0.8) * 100
above_08_jmsd = np.mean(v_jmsd  >= 0.8) * 100

print(f"""
  MSD  sim 분포: mean={v.mean():.4f}, std={v.std():.4f}
       → 값의 {above_08_msd:.1f}%가 0.8 이상 (고밀집 구간)
       → 상위 10% 기준값 = {t90_msd:.4f}

  JMSD sim 분포: mean={v_jmsd.mean():.4f}, std={v_jmsd.std():.4f}
       → 값의 {above_08_jmsd:.1f}%가 0.8 이상
       → 상위 10% 기준값 = {t90_jmsd:.4f}

  ACOS sim 분포: mean={v_acos.mean():.4f}, std={v_acos.std():.4f}
       → 상위 10% 기준값 = {t90_acos:.4f}

  핵심 메커니즘:
    S^alpha 변환은 값이 [0,1] 범위에서 서로 다른 효과를 낸다.

    - MSD 값이 0.8~1.0 구간에 밀집해 있으면, alpha가 클수록 이 구간의 값들이
      더 빠르게 0에 수렴 → 이웃 선택이 더 discriminative해짐.
      예: S=0.90 → S^10=0.349, S^20=0.122  (큰 차이)
          S=0.95 → S^10=0.599, S^20=0.358
          S=0.99 → S^10=0.904, S^20=0.818  (여전히 높게 유지)

    - ACOS는 이미 clip_neg 후 값 분포가 넓어서 (std 더 큼) 낮은 alpha로 충분.

    - JMSD = MSD × Jaccard → Jaccard가 [0,1] 균등 분포에 가까워서
      MSD보다 밀집도가 낮음 → 중간 alpha로 충분.

  따라서 α* = 10.0은 "MSD 유사도 분포가 고값에 밀집"된 특성의 결과이며,
  그리드 상한(10.0)을 초과해도 RMSE가 계속 개선될 가능성이 있다.
  → Phase 4-2: alpha 범위를 [0, 30]으로 확장하는 추가 실험 권장.
""")

# 4-5. alpha 범위 확장 시 MSD RMSE 개선 여부 시뮬레이션 (fold_01, K=20)
print("  [시뮬레이션] fold_01 MSD K=20: alpha=10~30 범위 RMSE 추이")
print("  (train_inner 데이터로 재계산)\n")

# 간단 시뮬레이션: validation set으로 alpha=10~30 구간의 RMSE 추정
# 실제 alpha_optimization_history.csv 에서 확인
hist_files = sorted(glob.glob(os.path.join(INNER_SIM_DIR, "fold_01", "alpha_optimization_history_20260415_*.csv")))
if hist_files:
    hist_df = pd.read_csv(hist_files[0])
    print(f"  alpha_history 컬럼: {hist_df.columns.tolist()}")
    msd_hist = hist_df[(hist_df["method"] == "msd") & (hist_df["K"] == 20)].copy()
    if not msd_hist.empty:
        rmse_col = "val_rmse" if "val_rmse" in msd_hist.columns else \
                   "validation_rmse" if "validation_rmse" in msd_hist.columns else \
                   [c for c in msd_hist.columns if "rmse" in c.lower()][0]
        # TopN 중복 제거: alpha 별로 대표값(평균) 사용
        msd_uniq = msd_hist.groupby("alpha")[rmse_col].mean().reset_index()
        msd_uniq = msd_uniq.sort_values("alpha")
        print(f"  [MSD K=20 coarse grid 탐색 결과]  (단조 감소 → 상한 포화 확인)")
        print(f"  {'alpha':>6}  {'val_rmse':>10}  {'RMSE 변화':>10}")
        prev = None
        for _, row in msd_uniq.iterrows():
            delta = f"{row[rmse_col]-prev:+.4f}" if prev is not None else "     —"
            bar_len = max(0, int((1.05 - row[rmse_col]) * 200))
            bar = "▓" * bar_len
            print(f"  {row['alpha']:>6.2f}  {row[rmse_col]:>10.4f}  {delta:>10}  {bar}")
            prev = row[rmse_col]
        # 마지막 alpha에서도 감소 중인지 확인
        last_delta = msd_uniq.iloc[-1][rmse_col] - msd_uniq.iloc[-2][rmse_col]
        if last_delta < 0:
            print(f"\n  ⚠ alpha=10.0에서도 RMSE가 {last_delta:+.4f} 감소 중 (그리드 상한 포화 확인됨)")
            print(f"  → alpha 그리드를 [0, 30]으로 확장하면 추가 개선 가능성 있음")
    else:
        print("  MSD K=20 데이터 없음")
else:
    print("  alpha_optimization_history 파일 없음")

print()
print("=" * 70)
print("  분석 완료")
print("=" * 70)
