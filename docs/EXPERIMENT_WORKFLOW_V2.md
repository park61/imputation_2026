# Experimental Workflow Summary - V2 (Fixed Test Leakage)

## 🔄 Complete Workflow

### Phase 0: Archive Previous Version
✅ **Completed (2026-01-13)**
- Backed up notebooks to `notebooks/archive_test_leakage/`
- Backed up results to `results/archive_test_leakage/`
- Created README documenting test leakage problem

---

### Phase 1: Data Preparation

#### 1️⃣ K-Fold Split (Unchanged)
**Notebook**: `create_k_fold_data_1106.ipynb`
- Splits MovieLens into 10 folds
- Output: `fold_XX/train.csv`, `fold_XX/test.csv`
- **Status**: ✅ Already completed, no changes needed

#### 2️⃣ Train/Validation Split (NEW!)
**Notebook**: `create_train_validation_split_0113.ipynb`
- **Purpose**: Split train into train_inner (80%) + validation (20%)
- **Method**: User-wise random holdout (seed=42)
- **Output**: 
  - `fold_XX/train_inner.csv` (for α optimization learning)
  - `fold_XX/validation.csv` (for α optimization evaluation)
- **Status**: ⏳ Ready to run

---

### Phase 2: Main Experiment (MODIFIED)

#### 3️⃣ Grid Search with α Optimization - V2
**Notebook**: `main_experiment_grid_search_0113_v2.ipynb`

**Key Changes**:
- ✅ α optimization uses **validation** (not test!)
- ✅ Final evaluation uses **test** (first and only use!)
- ✅ Uses **train_full** for final test predictions

**Data Flow**:
```
α Optimization:
  train_inner (80%) → KNN learning
  validation (20%)  → α evaluation
  → Select optimal α

Final Evaluation (per K, TopN):
  train_full (100%) → KNN learning
  test              → Evaluation (FIRST USE!)
  → RMSE, MAD, Precision@N, Recall@N
```

**Output**:
- `results/fold_XX/grid_search_results_*.csv`
- `results/fold_XX/alpha_optimization_history_*.csv`
- `results/combined/all_folds_*.csv`

**Status**: ⏳ Ready to run after train/validation split

---

### Phase 3: Baseline Experiment (MODIFIED)

#### 4️⃣ α=1 Baseline - V2
**Notebook**: `compute_alpha_baseline_topn_0113_v2.ipynb`

**Key Changes**:
- ✅ Uses **train.csv** (full) for learning
- ✅ Uses **test.csv** for evaluation only

**Output**:
- `results/alpha_baseline_topn_results_v2.csv`

**Status**: ⏳ Ready to run

---

### Phase 4: Analysis & Visualization

#### 5️⃣ Similarity Measures
**Notebook**: `Similarity_measures_(Confirmed).ipynb`
- Function library (no changes needed)
- **Status**: ✅ Unchanged

#### 6️⃣ Visualization
**Notebook**: `visualize_grid_results_1231.ipynb`
- May need path updates for v2 results
- **Status**: ⏳ Will update after v2 experiments complete

#### 7️⃣ F1 Re-analysis
**Notebook**: `reanalyze_alpha_with_f1_0102.ipynb`
- Re-analyze with F1 instead of RMSE
- **Status**: ⏳ Will run on v2 results

---

## 📊 Data Leakage Fix Summary

### V1 (Old - WRONG)
```
Train → Learning
  ↓
Test → α optimization ❌ LEAKAGE!
  ↓
Test → Final evaluation ❌ Same data reused!
```

### V2 (New - CORRECT)
```
Train → Split → Train_inner (80%) + Validation (20%)
  ↓
Train_inner → Learning for α optimization
  ↓
Validation → α optimization evaluation ✅
  ↓
Train_full → Learning for final evaluation
  ↓
Test → Final evaluation ✅ FIRST USE!
```

---

## 🎯 Next Steps

### To Run Full V2 Experiment:

1. **Create train/validation splits**:
   ```
   Run: notebooks/create_train_validation_split_0113.ipynb
   Output: fold_XX/train_inner.csv, fold_XX/validation.csv
   ```

2. **Run main experiment v2**:
   ```
   Run: notebooks/main_experiment_grid_search_0113_v2.ipynb
   Config: Set FOLDS_TO_RUN and METHODS_TO_TEST
   Expected time: ~hours per fold
   ```

3. **Run baseline v2**:
   ```
   Run: notebooks/compute_alpha_baseline_topn_0113_v2.ipynb
   Output: results/alpha_baseline_topn_results_v2.csv
   ```

4. **Visualize and analyze**:
   ```
   Update paths in: visualize_grid_results_1231.ipynb
   Run F1 analysis: reanalyze_alpha_with_f1_0102.ipynb
   ```

---

## 📝 Key Files

### Notebooks
- ✅ `create_k_fold_data_1106.ipynb` (unchanged)
- 🆕 `create_train_validation_split_0113.ipynb`
- 🆕 `main_experiment_grid_search_0113_v2.ipynb`
- 🆕 `compute_alpha_baseline_topn_0113_v2.ipynb`
- ✅ `Similarity_measures_(Confirmed).ipynb` (unchanged)
- ⏳ `visualize_grid_results_1231.ipynb` (may need updates)
- ⏳ `reanalyze_alpha_with_f1_0102.ipynb` (will use v2 results)

### Data Structure
```
movielenz_data/fold_XX/
  ├─ train.csv          (original - for final evaluation)
  ├─ train_inner.csv    (NEW - 80% for α optimization)
  ├─ validation.csv     (NEW - 20% for α optimization)
  ├─ test.csv           (original - final evaluation ONLY)
  └─ sim_*.npy          (unchanged)
```

---

## 🔬 Expected Outcome Differences

**V1 vs V2 Results**:
- **RMSE**: May be slightly HIGHER in v2 (less overfitting)
- **Precision/Recall**: May show different patterns (no test bias)
- **Optimal α values**: May differ (validated properly)
- **Generalization**: V2 results are MORE TRUSTWORTHY

---

## 📅 Version History

- **V1** (2025-12-29): Original experiments with test leakage
  - Archived in `archive_test_leakage/`
  
- **V2** (2026-01-13): Fixed test leakage
  - Proper train/validation/test split
  - No data reuse
  - Scientifically valid experimental design
