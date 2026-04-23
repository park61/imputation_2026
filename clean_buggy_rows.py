# clean_buggy_rows.py  (Phase 0 전용 일회성 스크립트)
# 대상: results/inner_sim/ 만 (results/fold_*/ 와 results/*.csv 는 과거 실험이므로 제외)
import pandas as pd, glob

BUGGY_METHODS = {'msd', 'jmsd', 'acos'}
csv_patterns = [
    "results/inner_sim/fold_*/grid_search_results_*.csv",
    "results/inner_sim/fold_*/alpha_optimization_history_*.csv",
    "results/inner_sim/combined/all_folds_*.csv",
]

total_removed = 0
for pattern in csv_patterns:
    for fpath in sorted(glob.glob(pattern)):
        df = pd.read_csv(fpath)
        if 'method' not in df.columns:
            print(f"  [SKIP] {fpath}: 'method' 컬럼 없음")
            continue
        before = len(df)
        df_clean = df[~df['method'].isin(BUGGY_METHODS)]
        removed = before - len(df_clean)
        if removed > 0:
            df_clean.to_csv(fpath, index=False)
            print(f"  [OK]   {fpath}: {removed}행 삭제 ({before} → {len(df_clean)})")
            total_removed += removed
        else:
            print(f"  [--]   {fpath}: 버기 행 없음")

print(f"\n총 {total_removed}행 삭제 완료.")
