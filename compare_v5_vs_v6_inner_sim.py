# -*- coding: utf-8 -*-
"""
compare_v5_vs_v6_inner_sim.py
──────────────────────────────────────────────────────────────────
v5 (구 실험, full-train 기반 유사도, validation leakage 있음) 와
v6 inner_sim (train_inner 기반 유사도, leakage 없음) 실험 결과 비교.

비교 조건: λ = 0.002 (두 실험 모두 동일)

분석 항목 (test_summary.tex 대응):
  1. 전체 요약 (10-fold × 17 method × all K, TopN 평균)
  2. Fold별 분해
  3. 유사도별 상세 비교 [핵심] – RMSE, MAD, Precision@N, Recall@N
  4. K별 분해
  5. Alpha 최적값 비교
  6. Win-Rate 비교 (optimized vs baseline)

실행:
    python compare_v5_vs_v6_inner_sim.py
"""

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
V5_DIR     = os.path.join(BASE_DIR, "results")
V6_DIR     = os.path.join(BASE_DIR, "results", "inner_sim")

# ── v5 개별 fold 파일 매핑 ─────────────────────────────────────────────────
# fold_01: 구 실험에는 별도 파일 없음 → combined 파일 사용 (컬럼 10개)
V5_FILES = {
    1:  os.path.join(V5_DIR, "combined", "all_folds_grid_results_20260117_003457.csv"),
    2:  os.path.join(V5_DIR, "fold_02", "grid_search_results_20260121_022044.csv"),
    3:  os.path.join(V5_DIR, "fold_03", "grid_search_results_20260121_094002.csv"),
    4:  os.path.join(V5_DIR, "fold_04", "grid_search_results_20260121_184454.csv"),
    5:  os.path.join(V5_DIR, "fold_05", "grid_search_results_20260122_173925.csv"),
    6:  os.path.join(V5_DIR, "fold_06", "grid_search_results_20260128_053240.csv"),
    7:  os.path.join(V5_DIR, "fold_07", "grid_search_results_20260128_090801.csv"),
    8:  os.path.join(V5_DIR, "fold_08", "grid_search_results_20260128_222923.csv"),
    9:  os.path.join(V5_DIR, "fold_09", "grid_search_results_20260129_023436.csv"),
    10: os.path.join(V5_DIR, "fold_10", "grid_search_results_20260129_063658.csv"),
}

# ── v6 inner_sim: fold마다 가장 늦은 타임스탬프 파일 사용 ───────────────────
def _latest_grid_file(fold_dir):
    """fold 디렉토리에서 grid_search_results_*.csv 중 최신 파일 반환"""
    files = [f for f in os.listdir(fold_dir)
             if f.startswith("grid_search_results_") and f.endswith(".csv")
             and "lambda" not in f]
    if not files:
        return None
    return os.path.join(fold_dir, sorted(files)[-1])

V6_FILES = {}
for fn in range(1, 11):
    fold_dir = os.path.join(V6_DIR, f"fold_{fn:02d}")
    if os.path.isdir(fold_dir):
        p = _latest_grid_file(fold_dir)
        if p:
            V6_FILES[fn] = p

# ── 데이터 로드 ────────────────────────────────────────────────────────────
def _normalise_columns(df):
    """컬럼 이름 통일: 구 실험(10열) → 신 실험(20열) 형식에 맞게 매핑"""
    cols_lower = {c.lower(): c for c in df.columns}
    col_map = {}
    # 구 실험: RMSE/MAD/Precision/Recall 컬럼이 test_ 접두어 없이 직접 존재
    # 신 실험: test_RMSE, test_MAD, test_Precision, test_Recall 존재
    # 공통으로 RMSE/MAD/Precision/Recall 이름으로 통일
    if "test_RMSE" in df.columns and "RMSE" not in df.columns:
        col_map["test_RMSE"] = "RMSE"
    if "test_MAD" in df.columns and "MAD" not in df.columns:
        col_map["test_MAD"] = "MAD"
    if "test_Precision" in df.columns and "Precision" not in df.columns:
        col_map["test_Precision"] = "Precision"
    if "test_Recall" in df.columns and "Recall" not in df.columns:
        col_map["test_Recall"] = "Recall"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def load_dataset(file_map, label):
    frames = []
    for fn, path in sorted(file_map.items()):
        if not os.path.exists(path):
            print(f"  [{label}] fold_{fn:02d}: 파일 없음 → {path}")
            continue
        df = pd.read_csv(path)
        df = _normalise_columns(df)
        # fold 별 데이터에서 해당 fold만
        if "fold" in df.columns:
            df = df[df["fold"] == fn].copy()
        else:
            df["fold"] = fn
        frames.append(df)
    if not frames:
        raise RuntimeError(f"[{label}] 데이터 없음")
    out = pd.concat(frames, ignore_index=True)
    print(f"  [{label}] 로드 완료: {len(out):,} rows, fold={sorted(out['fold'].unique())}")
    return out


print("=" * 80)
print("  V5 vs V6 inner_sim 비교 분석  (λ=0.002 동일 조건)")
print("  Start:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 80)

print("\n[데이터 로드]")
v5 = load_dataset(V5_FILES, "v5")
v6 = load_dataset(V6_FILES, "v6-inner")

# 공통 methods만 사용
methods_v5 = set(v5["method"].unique())
methods_v6 = set(v6["method"].unique())
METHODS = sorted(methods_v5 & methods_v6)
v5 = v5[v5["method"].isin(METHODS)]
v6 = v6[v6["method"].isin(METHODS)]

FOLDS = sorted(set(v5["fold"].unique()) & set(v6["fold"].unique()))
print(f"  공통 fold: {FOLDS}")
print(f"  공통 method: {METHODS}")

# ── 헬퍼 ──────────────────────────────────────────────────────────────────
SEP  = "─" * 110
SEP2 = "═" * 110
BOLD = lambda s: f"[{s}]"

def pct(a, b):
    """(a-b)/b*100"""
    if b == 0 or np.isnan(b): return np.nan
    return (a - b) / b * 100


def summarise(df, groupby_cols, metric="RMSE"):
    """type별(baseline/optimized) 평균을 구해 wide pivot 반환"""
    g = df.groupby(groupby_cols + ["type"])[metric].mean().unstack("type")
    g.columns.name = None
    for t in ["baseline", "optimized"]:
        if t not in g.columns:
            g[t] = np.nan
    g["delta_pct"] = (g["optimized"] - g["baseline"]) / g["baseline"] * 100
    return g


def win_rate(df, groupby_cols, metric="RMSE"):
    """optimized가 baseline보다 낮은(개선) 비율"""
    base = df[df["type"] == "baseline"].groupby(groupby_cols)[metric].mean()
    opt  = df[df["type"] == "optimized"].groupby(groupby_cols)[metric].mean()
    # merge on common index
    comb = pd.concat({"base": base, "opt": opt}, axis=1).dropna()
    return float((comb["opt"] < comb["base"]).mean()) * 100


def alpha_summary(df, groupby_cols):
    """optimized alpha의 평균/중앙값"""
    return df[df["type"] == "optimized"].groupby(groupby_cols)["alpha"].mean()


def fmt_float(x, d=5):
    if pd.isna(x): return "  N/A  "
    return f"{x:.{d}f}"

def fmt_pct(x, d=2):
    if pd.isna(x): return "  N/A  "
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{d}f}%"

def fmt_diff(x, d=5):
    if pd.isna(x): return "  N/A  "
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{d}f}"


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 1: 전체 요약
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  섹션 1: 전체 요약  (10-fold × 17 method × all K×TopN 평균)")
print(SEP2)

for label, df in [("v5 (leakage있음)", v5), ("v6 inner_sim (leakage없음)", v6)]:
    base_rmse = df[df["type"] == "baseline"]["RMSE"].mean()
    opt_rmse  = df[df["type"] == "optimized"]["RMSE"].mean()
    base_mad  = df[df["type"] == "baseline"]["MAD"].mean()
    opt_mad   = df[df["type"] == "optimized"]["MAD"].mean()
    wr        = win_rate(df, ["fold", "method", "K", "TopN"], "RMSE")
    avg_alpha = df[df["type"] == "optimized"]["alpha"].mean()
    print(f"\n  {label}")
    print(f"  {'설정':20s}  {'RMSE':>10s}  {'vs Base':>9s}  {'MAD':>10s}  {'WinRate':>9s}  {'Avg α':>8s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*8}")
    print(f"  {'Baseline (α=1)':20s}  {base_rmse:>10.6f}  {'기준':>9s}  {base_mad:>10.6f}  {'─':>9s}  {'1.000':>8s}")
    print(f"  {'Optimized (λ=0.002)':20s}  {opt_rmse:>10.6f}  {fmt_pct(pct(opt_rmse,base_rmse)):>9s}  {opt_mad:>10.6f}  {wr:>8.1f}%  {avg_alpha:>8.3f}")

# v5 vs v6 기준값 비교
v5_base = v5[v5["type"] == "baseline"]["RMSE"].mean()
v6_base = v6[v6["type"] == "baseline"]["RMSE"].mean()
v5_opt  = v5[v5["type"] == "optimized"]["RMSE"].mean()
v6_opt  = v6[v6["type"] == "optimized"]["RMSE"].mean()
print(f"\n  {'':4s}{'구분':20s}  {'v5 RMSE':>10s}  {'v6 RMSE':>10s}  {'v6-v5 차이':>11s}  {'변화율':>9s}")
print(f"  {'':4s}{'─'*20}  {'─'*10}  {'─'*10}  {'─'*11}  {'─'*9}")
print(f"  {'':4s}{'Baseline':20s}  {v5_base:>10.6f}  {v6_base:>10.6f}  {fmt_diff(v6_base-v5_base):>11s}  {fmt_pct(pct(v6_base,v5_base)):>9s}")
print(f"  {'':4s}{'Optimized':20s}  {v5_opt:>10.6f}  {v6_opt:>10.6f}  {fmt_diff(v6_opt-v5_opt):>11s}  {fmt_pct(pct(v6_opt,v5_opt)):>9s}")


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 2: Fold별 분해
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  섹션 2: Fold별 분해")
print(SEP2)

s5f = summarise(v5, ["fold"])
s6f = summarise(v6, ["fold"])

hdr = (f"  {'Fold':>6s}  "
       f"{'v5 Base':>10s}  {'v5 Opt':>10s}  {'v5 Δ%':>9s}  "
       f"{'v6 Base':>10s}  {'v6 Opt':>10s}  {'v6 Δ%':>9s}  "
       f"{'Base변화':>10s}  {'Opt변화':>10s}")
print(hdr)
print("  " + "─" * 108)
for fn in FOLDS:
    vb5 = s5f.loc[fn, "baseline"]  if fn in s5f.index else np.nan
    vo5 = s5f.loc[fn, "optimized"] if fn in s5f.index else np.nan
    vd5 = s5f.loc[fn, "delta_pct"] if fn in s5f.index else np.nan
    vb6 = s6f.loc[fn, "baseline"]  if fn in s6f.index else np.nan
    vo6 = s6f.loc[fn, "optimized"] if fn in s6f.index else np.nan
    vd6 = s6f.loc[fn, "delta_pct"] if fn in s6f.index else np.nan
    base_chg = fmt_diff(vb6 - vb5)
    opt_chg  = fmt_diff(vo6 - vo5)
    print(f"  {fn:>6d}  "
          f"{fmt_float(vb5):>10s}  {fmt_float(vo5):>10s}  {fmt_pct(vd5):>9s}  "
          f"{fmt_float(vb6):>10s}  {fmt_float(vo6):>10s}  {fmt_pct(vd6):>9s}  "
          f"{base_chg:>10s}  {opt_chg:>10s}")


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 3: 유사도별 상세 비교 [핵심]
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  섹션 3: 유사도별 RMSE 상세 비교  [핵심]")
print(f"  집계: 전체 fold({FOLDS}) × 전체 K × 전체 TopN 평균")
print(SEP2)

s5m = summarise(v5, ["method"])
s6m = summarise(v6, ["method"])
# win-rate per method
wr5 = {m: win_rate(v5[v5["method"]==m], ["fold","K","TopN"]) for m in METHODS}
wr6 = {m: win_rate(v6[v6["method"]==m], ["fold","K","TopN"]) for m in METHODS}

hdr = (f"  {'Method':>14s}  "
       f"{'v5 Base':>9s}  {'v6 Base':>9s}  {'B변화':>8s}  "
       f"{'v5 Opt':>9s}  {'v6 Opt':>9s}  {'O변화':>8s}  "
       f"{'v5 Δ%':>8s}  {'v6 Δ%':>8s}  "
       f"{'v5 WR%':>8s}  {'v6 WR%':>8s}")
print(hdr)
print("  " + "─" * 108)

# 정렬: v6 baseline 기준 오름차순
rows = []
for m in METHODS:
    vb5 = s5m.loc[m,"baseline"]  if m in s5m.index else np.nan
    vo5 = s5m.loc[m,"optimized"] if m in s5m.index else np.nan
    vd5 = s5m.loc[m,"delta_pct"] if m in s5m.index else np.nan
    vb6 = s6m.loc[m,"baseline"]  if m in s6m.index else np.nan
    vo6 = s6m.loc[m,"optimized"] if m in s6m.index else np.nan
    vd6 = s6m.loc[m,"delta_pct"] if m in s6m.index else np.nan
    rows.append((m, vb5, vb6, vb6-vb5, vo5, vo6, vo6-vo5, vd5, vd6, wr5[m], wr6[m]))

rows.sort(key=lambda r: r[2] if not np.isnan(r[2]) else 99)

for (m, vb5, vb6, bc, vo5, vo6, oc, vd5, vd6, w5, w6) in rows:
    print(f"  {m:>14s}  "
          f"{fmt_float(vb5):>9s}  {fmt_float(vb6):>9s}  {fmt_diff(bc):>8s}  "
          f"{fmt_float(vo5):>9s}  {fmt_float(vo6):>9s}  {fmt_diff(oc):>8s}  "
          f"{fmt_pct(vd5):>8s}  {fmt_pct(vd6):>8s}  "
          f"{w5:>7.1f}%  {w6:>7.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 3b: 유사도별 MAD 비교
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  섹션 3b: 유사도별 MAD 비교")
print(SEP)

s5m_mad = summarise(v5, ["method"], "MAD")
s6m_mad = summarise(v6, ["method"], "MAD")

hdr = (f"  {'Method':>14s}  "
       f"{'v5 Base':>9s}  {'v6 Base':>9s}  {'B변화':>8s}  "
       f"{'v5 Opt':>9s}  {'v6 Opt':>9s}  {'O변화':>8s}  "
       f"{'v5 Δ%':>8s}  {'v6 Δ%':>8s}")
print(hdr)
print("  " + "─" * 90)

rows_mad = []
for m in METHODS:
    vb5 = s5m_mad.loc[m,"baseline"]  if m in s5m_mad.index else np.nan
    vo5 = s5m_mad.loc[m,"optimized"] if m in s5m_mad.index else np.nan
    vd5 = s5m_mad.loc[m,"delta_pct"] if m in s5m_mad.index else np.nan
    vb6 = s6m_mad.loc[m,"baseline"]  if m in s6m_mad.index else np.nan
    vo6 = s6m_mad.loc[m,"optimized"] if m in s6m_mad.index else np.nan
    vd6 = s6m_mad.loc[m,"delta_pct"] if m in s6m_mad.index else np.nan
    rows_mad.append((m, vb5, vb6, vb6-vb5, vo5, vo6, vo6-vo5, vd5, vd6))
rows_mad.sort(key=lambda r: r[2] if not np.isnan(r[2]) else 99)
for (m, vb5, vb6, bc, vo5, vo6, oc, vd5, vd6) in rows_mad:
    print(f"  {m:>14s}  "
          f"{fmt_float(vb5):>9s}  {fmt_float(vb6):>9s}  {fmt_diff(bc):>8s}  "
          f"{fmt_float(vo5):>9s}  {fmt_float(vo6):>9s}  {fmt_diff(oc):>8s}  "
          f"{fmt_pct(vd5):>8s}  {fmt_pct(vd6):>8s}")


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 3c: 유사도별 Precision@N 비교 (TopN별 평균)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  섹션 3c: 유사도별 Precision (all TopN 평균)")
print(SEP)

s5m_pr = summarise(v5, ["method"], "Precision")
s6m_pr = summarise(v6, ["method"], "Precision")

hdr = (f"  {'Method':>14s}  "
       f"{'v5 Base':>9s}  {'v6 Base':>9s}  {'B변화':>8s}  "
       f"{'v5 Opt':>9s}  {'v6 Opt':>9s}  {'O변화':>8s}  "
       f"{'v5 Δ%':>8s}  {'v6 Δ%':>8s}")
print(hdr)
print("  " + "─" * 90)

rows_pr = []
for m in METHODS:
    vb5 = s5m_pr.loc[m,"baseline"]  if m in s5m_pr.index else np.nan
    vo5 = s5m_pr.loc[m,"optimized"] if m in s5m_pr.index else np.nan
    vd5 = s5m_pr.loc[m,"delta_pct"] if m in s5m_pr.index else np.nan
    vb6 = s6m_pr.loc[m,"baseline"]  if m in s6m_pr.index else np.nan
    vo6 = s6m_pr.loc[m,"optimized"] if m in s6m_pr.index else np.nan
    vd6 = s6m_pr.loc[m,"delta_pct"] if m in s6m_pr.index else np.nan
    rows_pr.append((m, vb5, vb6, vb6-vb5, vo5, vo6, vo6-vo5, vd5, vd6))
rows_pr.sort(key=lambda r: -r[2] if not np.isnan(r[2]) else -99)  # Precision은 높을수록 좋으므로 내림차순
for (m, vb5, vb6, bc, vo5, vo6, oc, vd5, vd6) in rows_pr:
    print(f"  {m:>14s}  "
          f"{fmt_float(vb5):>9s}  {fmt_float(vb6):>9s}  {fmt_diff(bc):>8s}  "
          f"{fmt_float(vo5):>9s}  {fmt_float(vo6):>9s}  {fmt_diff(oc):>8s}  "
          f"{fmt_pct(vd5):>8s}  {fmt_pct(vd6):>8s}")


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 3d: 유사도별 Recall@N 비교
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  섹션 3d: 유사도별 Recall (all TopN 평균)")
print(SEP)

s5m_re = summarise(v5, ["method"], "Recall")
s6m_re = summarise(v6, ["method"], "Recall")

hdr = (f"  {'Method':>14s}  "
       f"{'v5 Base':>9s}  {'v6 Base':>9s}  {'B변화':>8s}  "
       f"{'v5 Opt':>9s}  {'v6 Opt':>9s}  {'O변화':>8s}  "
       f"{'v5 Δ%':>8s}  {'v6 Δ%':>8s}")
print(hdr)
print("  " + "─" * 90)

rows_re = []
for m in METHODS:
    vb5 = s5m_re.loc[m,"baseline"]  if m in s5m_re.index else np.nan
    vo5 = s5m_re.loc[m,"optimized"] if m in s5m_re.index else np.nan
    vd5 = s5m_re.loc[m,"delta_pct"] if m in s5m_re.index else np.nan
    vb6 = s6m_re.loc[m,"baseline"]  if m in s6m_re.index else np.nan
    vo6 = s6m_re.loc[m,"optimized"] if m in s6m_re.index else np.nan
    vd6 = s6m_re.loc[m,"delta_pct"] if m in s6m_re.index else np.nan
    rows_re.append((m, vb5, vb6, vb6-vb5, vo5, vo6, vo6-vo5, vd5, vd6))
rows_re.sort(key=lambda r: -r[2] if not np.isnan(r[2]) else -99)
for (m, vb5, vb6, bc, vo5, vo6, oc, vd5, vd6) in rows_re:
    print(f"  {m:>14s}  "
          f"{fmt_float(vb5):>9s}  {fmt_float(vb6):>9s}  {fmt_diff(bc):>8s}  "
          f"{fmt_float(vo5):>9s}  {fmt_float(vo6):>9s}  {fmt_diff(oc):>8s}  "
          f"{fmt_pct(vd5):>8s}  {fmt_pct(vd6):>8s}")


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 4: K별 분해 (RMSE, optimized 기준)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  섹션 4: K별 분해 (all methods × all folds × all TopN 평균)")
print(SEP2)

s5k = summarise(v5, ["K"])
s6k = summarise(v6, ["K"])

K_vals = sorted(set(s5k.index) | set(s6k.index))
hdr = (f"  {'K':>5s}  "
       f"{'v5 Base':>10s}  {'v5 Opt':>10s}  {'v5 Δ%':>9s}  "
       f"{'v6 Base':>10s}  {'v6 Opt':>10s}  {'v6 Δ%':>9s}  "
       f"{'Base변화':>10s}  {'Opt변화':>10s}")
print(hdr)
print("  " + "─" * 100)
for k in K_vals:
    vb5 = s5k.loc[k,"baseline"]  if k in s5k.index else np.nan
    vo5 = s5k.loc[k,"optimized"] if k in s5k.index else np.nan
    vd5 = s5k.loc[k,"delta_pct"] if k in s5k.index else np.nan
    vb6 = s6k.loc[k,"baseline"]  if k in s6k.index else np.nan
    vo6 = s6k.loc[k,"optimized"] if k in s6k.index else np.nan
    vd6 = s6k.loc[k,"delta_pct"] if k in s6k.index else np.nan
    print(f"  {k:>5d}  "
          f"{fmt_float(vb5):>10s}  {fmt_float(vo5):>10s}  {fmt_pct(vd5):>9s}  "
          f"{fmt_float(vb6):>10s}  {fmt_float(vo6):>10s}  {fmt_pct(vd6):>9s}  "
          f"{fmt_diff(vb6-vb5):>10s}  {fmt_diff(vo6-vo5):>10s}")


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 5: 유사도별 Alpha 최적값 비교
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  섹션 5: 유사도별 최적 Alpha 비교  (K·fold·TopN 전체 평균)")
print(SEP2)

alpha5 = v5[v5["type"] == "optimized"].groupby("method")["alpha"].agg(["mean","median","std"])
alpha6 = v6[v6["type"] == "optimized"].groupby("method")["alpha"].agg(["mean","median","std"])

hdr = (f"  {'Method':>14s}  "
       f"{'v5 avg α':>9s}  {'v5 med α':>9s}  {'v5 std':>8s}  "
       f"{'v6 avg α':>9s}  {'v6 med α':>9s}  {'v6 std':>8s}  "
       f"{'Δ avg α':>9s}")
print(hdr)
print("  " + "─" * 95)

rows_al = []
for m in METHODS:
    a5 = alpha5.loc[m] if m in alpha5.index else pd.Series({"mean":np.nan,"median":np.nan,"std":np.nan})
    a6 = alpha6.loc[m] if m in alpha6.index else pd.Series({"mean":np.nan,"median":np.nan,"std":np.nan})
    rows_al.append((m, a5["mean"], a5["median"], a5["std"], a6["mean"], a6["median"], a6["std"]))

rows_al.sort(key=lambda r: r[1] if not np.isnan(r[1]) else 99)

for (m, am5, me5, st5, am6, me6, st6) in rows_al:
    delta = am6 - am5 if not (np.isnan(am5) or np.isnan(am6)) else np.nan
    print(f"  {m:>14s}  "
          f"{fmt_float(am5):>9s}  {fmt_float(me5):>9s}  {fmt_float(st5):>8s}  "
          f"{fmt_float(am6):>9s}  {fmt_float(me6):>9s}  {fmt_float(st6):>8s}  "
          f"{fmt_diff(delta):>9s}")


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 6: 유사도별 K별 RMSE 세부 (method × K)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  섹션 6: 유사도 × K 별 Optimized RMSE 비교")
print(f"  (v6_opt - v5_opt 차이, + = v6 성능 악화 / - = v6 성능 개선)")
print(SEP2)

# optimized only, average over fold and TopN
v5_mk = v5[v5["type"]=="optimized"].groupby(["method","K"])["RMSE"].mean()
v6_mk = v6[v6["type"]=="optimized"].groupby(["method","K"])["RMSE"].mean()

K_vals_all = sorted(v5[v5["type"]=="optimized"]["K"].unique())

# 헤더
hdr = f"  {'Method':>14s}"
for k in K_vals_all:
    hdr += f"  {'K='+str(k):>8s}"
print(hdr)
print("  " + "─" * (14 + 10 * len(K_vals_all) + 4))

for m in METHODS:
    line = f"  {m:>14s}"
    for k in K_vals_all:
        key = (m, k)
        v5v = v5_mk.get(key, np.nan)
        v6v = v6_mk.get(key, np.nan)
        diff = v6v - v5v if not (np.isnan(v5v) or np.isnan(v6v)) else np.nan
        line += f"  {fmt_diff(diff):>8s}"
    print(line)

print(f"\n  (v5_opt 절대값 참고)")
hdr = f"  {'Method':>14s}"
for k in K_vals_all:
    hdr += f"  {'K='+str(k):>8s}"
print(hdr)
print("  " + "─" * (14 + 10 * len(K_vals_all) + 4))
for m in METHODS:
    line = f"  {m:>14s}"
    for k in K_vals_all:
        v5v = v5_mk.get((m, k), np.nan)
        line += f"  {fmt_float(v5v):>8s}"
    print(line)

print(f"\n  (v6_opt 절대값 참고)")
hdr = f"  {'Method':>14s}"
for k in K_vals_all:
    hdr += f"  {'K='+str(k):>8s}"
print(hdr)
print("  " + "─" * (14 + 10 * len(K_vals_all) + 4))
for m in METHODS:
    line = f"  {m:>14s}"
    for k in K_vals_all:
        v6v = v6_mk.get((m, k), np.nan)
        line += f"  {fmt_float(v6v):>8s}"
    print(line)


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 7: 유사도별 K별 Alpha 세부 (v6 inner_sim)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  섹션 7: 유사도 × K 별 최적 Alpha - v6 inner_sim")
print(SEP2)

v6_mk_al = v6[v6["type"]=="optimized"].groupby(["method","K"])["alpha"].mean()
v5_mk_al = v5[v5["type"]=="optimized"].groupby(["method","K"])["alpha"].mean()

hdr = f"  {'Method':>14s}"
for k in K_vals_all:
    hdr += f"  {'K='+str(k):>6s}"
print(f"  [v6 inner_sim 최적 α]")
print(hdr)
print("  " + "─" * (14 + 8 * len(K_vals_all) + 4))
for m in METHODS:
    line = f"  {m:>14s}"
    for k in K_vals_all:
        v = v6_mk_al.get((m, k), np.nan)
        line += f"  {fmt_float(v):>6s}"
    print(line)

print(f"\n  [v5 구실험 최적 α]")
print(hdr)
print("  " + "─" * (14 + 8 * len(K_vals_all) + 4))
for m in METHODS:
    line = f"  {m:>14s}"
    for k in K_vals_all:
        v = v5_mk_al.get((m, k), np.nan)
        line += f"  {fmt_float(v):>6s}"
    print(line)

print(f"\n  [Δ (v6 - v5) 최적 α 차이]")
print(hdr)
print("  " + "─" * (14 + 8 * len(K_vals_all) + 4))
for m in METHODS:
    line = f"  {m:>14s}"
    for k in K_vals_all:
        v5v = v5_mk_al.get((m, k), np.nan)
        v6v = v6_mk_al.get((m, k), np.nan)
        diff = v6v - v5v if not (np.isnan(v5v) or np.isnan(v6v)) else np.nan
        line += f"  {fmt_diff(diff):>6s}"
    print(line)


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 8: 종합 순위 비교
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  섹션 8: 유사도 성능 순위 비교  (Optimized RMSE 기준, 낮을수록 좋음)")
print(SEP2)

v5_rank = s5m["optimized"].rank().rename("v5_rank")
v6_rank = s6m["optimized"].rank().rename("v6_rank")
rank_df = pd.concat([s5m["optimized"].rename("v5_opt"), s6m["optimized"].rename("v6_opt"),
                     s5m["baseline"].rename("v5_base"), s6m["baseline"].rename("v6_base"),
                     v5_rank, v6_rank], axis=1)
rank_df["rank_change"] = rank_df["v6_rank"] - rank_df["v5_rank"]
rank_df = rank_df.sort_values("v6_opt")

hdr = (f"  {'Method':>14s}  {'v5 Opt':>10s}  {'v6 Opt':>10s}  "
       f"{'v5 Base':>10s}  {'v6 Base':>10s}  "
       f"{'v5 순위':>7s}  {'v6 순위':>7s}  {'순위변화':>8s}")
print(hdr)
print("  " + "─" * 98)
for m, row in rank_df.iterrows():
    chg = int(row["rank_change"])
    chg_str = f"{chg:+d}" if chg != 0 else "  ─ "
    print(f"  {m:>14s}  "
          f"{fmt_float(row['v5_opt']):>10s}  {fmt_float(row['v6_opt']):>10s}  "
          f"{fmt_float(row['v5_base']):>10s}  {fmt_float(row['v6_base']):>10s}  "
          f"{int(row['v5_rank']):>7d}  {int(row['v6_rank']):>7d}  {chg_str:>8s}")

# Spearman 상관
from scipy.stats import spearmanr
common = rank_df.dropna(subset=["v5_rank","v6_rank"])
rho, pval = spearmanr(common["v5_rank"], common["v6_rank"])
print(f"\n  → 순위 Spearman 상관계수: ρ = {rho:.4f}  (p = {pval:.4f})")


print(f"\n{SEP2}")
print("  분석 완료:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print(SEP2)
