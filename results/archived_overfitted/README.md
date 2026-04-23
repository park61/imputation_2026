# Archived Overfitted Results

**Date Archived:** 2026-01-16

## Reason for Archival

These results were generated with alpha optimization range of 0.0 to 10.0, which led to **overfitting to the validation set**.

### Key Issues Identified:

1. **Excessive Alpha Range (0~10)**
   - Extreme alpha values (>5) caused overfitting
   - Average validation→test performance gap: 12.71%
   - Worst cases showed 35%+ degradation on test set

2. **Methods Most Affected:**
   - CHEBYSHEV: 100% cases worse than baseline
   - CPCC: 100% cases worse than baseline
   - EUCLIDEAN: 100% cases worse than baseline
   - PCC, SPCC, SRC: 100% cases worse than baseline

3. **Evidence of Overfitting:**
   - High α (>5.0): average gap = 22.11%
   - Low α (≤2.0): average gap = 2.45%
   - 900/1700 configurations (52.94%) performed worse than baseline α=1.0

### Corrective Actions Taken:

**New Configuration:**
- Alpha range reduced to 0.0 ~ 3.0
- Added regularization penalty (λ × |α - 1.0|)
- Prevents extreme alpha values
- Promotes better generalization

### Files in This Archive:

```
all_folds_grid_results_20260115_220436.csv       (3,401 rows - FOLD 01 only)
all_folds_alpha_history_20260115_220436.csv     (69,101 rows - optimization history)
grid_search_results_20260115_220436.csv          (fold_01 individual results)
alpha_optimization_history_20260115_220436.csv   (fold_01 optimization)
best_configuration_*.json                        (overfitted configurations)
method_summary_*.csv                             (summaries based on overfitted data)
```

### Do Not Use These Results For:
- ❌ Final performance comparisons
- ❌ Production recommendations
- ❌ Publication figures
- ❌ Hyperparameter selection

### Can Use These Results For:
- ✅ Understanding overfitting patterns
- ✅ Demonstrating importance of regularization
- ✅ Comparative analysis (old vs new approach)
- ✅ Educational purposes

---

**New results with corrected alpha optimization are in:**
- `../fold_01/` (current results)
- `../combined/` (combined results)
