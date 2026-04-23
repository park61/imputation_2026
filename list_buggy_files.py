import glob, csv, os

BUGGY = {'msd', 'jmsd', 'acos'}

# inner_sim 전용: results/fold_*/ (과거 V5 실험)와 results/*.csv (최상위 집계)는 제외
patterns = [
    'results/inner_sim/fold_*/grid_search_results*.csv',
    'results/inner_sim/fold_*/alpha_optimization_history*.csv',
    'results/inner_sim/combined/all_folds_*.csv',
]

found = []
for pat in patterns:
    for fpath in sorted(glob.glob(pat)):
        try:
            with open(fpath, newline='', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or 'method' not in reader.fieldnames:
                    continue
                total = 0
                buggy = 0
                for row in reader:
                    total += 1
                    if row.get('method', '').strip() in BUGGY:
                        buggy += 1
            if buggy > 0:
                found.append((fpath, buggy, total))
        except Exception as e:
            print(f"  ERROR {fpath}: {e}")

print(f"총 {len(found)}개 파일에 msd/jmsd/acos 행 존재:\n")
for fpath, buggy, total in found:
    print(f"  [{buggy:5d}/{total:5d} rows]  {fpath}")
