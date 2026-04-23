# -*- coding: utf-8 -*-
"""
reeval_openworld_recall.py
----------------------------------------------------------------------
Open-World 프로토콜로 Recall@N / Precision@N 재평가.

목적: 메서드 간 Recall/Precision 차이를 비교
  (alpha baseline vs optimized 비교는 ranking 불변성으로 항상 0% 차이 → RMSE/MAD로만 판단)

현재 실험의 Closed-World 문제:
    cand_mask = train_col.isna() & ~test_col.isna()  # 후보: test 아이템만 (~8개/유저)

변경 사항 (Open-World):
    cand_mask  = train_col.isna()                    # 후보: 전체 미평가 아이템 (~1,587개/유저)
    relevant   = cand_mask & (test_col >= 4.0)       # relevant: test 아이템 중 ≥4.0만

유사도 행렬: sim_inner_{method}.npy 사용 (sim_full 불필요)
  근거: 목적이 메서드 간 비교이므로 train_inner 기반 유사도로 충분
        (sim_full과 차이 미미: train_full의 80% vs 100%)

RMSE/MAD는 기존 결과(consolidated_lambda0_final.csv)를 그대로 사용.
Alpha는 consolidated_lambda0_final.csv에서 (fold, method, K)별 최적값 재사용.

출력:
    results/combined/openworld_recall_lambda0.csv
    스키마: fold, method, K, TopN, type, alpha, recall_ow, precision_ow

실행:
    python reeval_openworld_recall.py
    python reeval_openworld_recall.py --folds 1 2   # 특정 fold만
    python reeval_openworld_recall.py --methods cosine pcc
    python reeval_openworld_recall.py --resume      # 이어서 실행
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATA_ROOT    = os.path.join(os.path.dirname(__file__), "data", "movielenz_data")
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results", "combined")
ALPHA_CSV    = os.path.join(RESULTS_ROOT, "consolidated_lambda0_final.csv")
OUT_CSV      = os.path.join(RESULTS_ROOT, "openworld_recall_lambda0.csv")

ALL_METHODS = [
    "acos", "ami", "ari", "chebyshev", "cosine", "cpcc",
    "euclidean", "ipwr", "itr", "jaccard", "jmsd",
    "kendall_tau_b", "manhattan", "msd", "pcc", "spcc", "src",
]
K_RANGE    = list(range(10, 101, 10))
TOPN_RANGE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
RELEVANCE_THRESHOLD = 4.0


# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folds",   nargs="+", type=int, default=list(range(1, 11)))
parser.add_argument("--methods", nargs="+", type=str, default=ALL_METHODS)
parser.add_argument("--resume",  action="store_true", help="기존 결과 이어서 실행")
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


# ── 유사도 적용 (alpha 지수 변환) ─────────────────────────────────────────────
def apply_alpha(S: np.ndarray, alpha: float) -> np.ndarray:
    """S_eff = clip(S, 0)^alpha, diagonal=0"""
    S_eff = np.clip(np.asarray(S, dtype=float), 0.0, None)
    S_eff = np.power(S_eff, float(alpha))
    np.fill_diagonal(S_eff, 0.0)
    return S_eff


# ── Open-World KNN 예측 — 완전 행렬 연산 (사용자 루프 없음) ────────────────
def predict_all_unseen_batch_k(train: pd.DataFrame,
                                S_eff: np.ndarray,
                                k_values: list) -> dict:
    """
    전체를 행렬 연산으로 처리 — Python 사용자 루프 없음.

    아이디어: K_max번 반복하면서 각 스텝에서 "각 유저의 k번째 이웃"의 기여를
    (n_items × n_users) 행렬에 누적. K 체크포인트에서 예측 행렬 저장.

    예전 방식 (사용자 루프 943회): O(n_users × n_items × K)
    현재 방식 (이웃 순위 루프 K_max회): O(K_max × n_items × n_users)
    → 실질적 동일 복잡도지만 numpy 벡터 연산이 Python 루프보다 ~20-50x 빠름.

    train shape : items × users
    S_eff shape : users × users (대각=0, 양수만)
    반환       : {K: pred_DataFrame}
    """
    X          = train.values.astype(np.float64)   # (n_items, n_users)
    n_items, n_users = X.shape
    K_max      = min(max(k_values), n_users - 1)
    k_set      = set(k_values)

    user_means  = np.nanmean(X, axis=0)            # (n_users,)
    global_mean = float(np.nanmean(X))
    fallback    = np.where(np.isfinite(user_means), user_means, global_mean)  # (n_users,)

    unseen_mask = np.isnan(X)                      # (n_items, n_users) — 예측 대상

    # 유사도 내림차순 정렬 인덱스 — (n_users × n_users)
    sorted_nbrs = np.argsort(-S_eff, axis=1)       # (n_users, n_users)

    cum_num  = np.zeros((n_items, n_users), dtype=np.float64)
    cum_den  = np.zeros((n_items, n_users), dtype=np.float64)

    pred_mats = {}

    for k in range(K_max):
        # k번째 이웃 인덱스 및 유사도 (모든 유저 동시 처리)
        nbr_k     = sorted_nbrs[:, k]              # (n_users,): user j의 k번째 이웃
        sim_k     = S_eff[np.arange(n_users), nbr_k]  # (n_users,): 해당 유사도

        # 유사도 > 0인 유저만 기여
        valid     = sim_k > 0.0                    # (n_users,)

        # k번째 이웃이 각 아이템에 준 평점 → (n_items, n_users)
        nbr_ratings = X[:, nbr_k]                  # (n_items, n_users)
        rated       = ~np.isnan(nbr_ratings)       # (n_items, n_users) bool

        # 가중치 = sim_k × valid (브로드캐스트: (n_items, n_users) × (n_users,))
        w           = sim_k * valid                # (n_users,)

        cum_num += np.where(rated, nbr_ratings, 0.0) * w   # (n_items, n_users)
        cum_den += rated.view(np.uint8) * w                # (n_items, n_users)

        # K 체크포인트: 예측 저장
        if (k + 1) in k_set:
            valid_den = cum_den > 0.0
            pred = np.where(valid_den,
                            cum_num / np.where(valid_den, cum_den, 1.0),
                            fallback)              # (n_items, n_users)
            # 알려진 평점은 원래 값 유지
            pred = np.where(unseen_mask, pred, X)
            pred_mats[k + 1] = pd.DataFrame(pred, index=train.index, columns=train.columns)

    return pred_mats


# ── Open-World Recall/Precision 계산 (벡터화 — 모든 TopN 한 번에) ────────────
def precision_recall_batch_topn(pred_full: pd.DataFrame,
                                 train: pd.DataFrame,
                                 test: pd.DataFrame,
                                 topn_list: list,
                                 relevance_threshold: float = 4.0) -> dict:
    """
    Open-World 평가 (완전 벡터화 — Python 유저 루프 없음).
    모든 TopN 값을 한 번의 argsort로 처리.

    반환: {TopN: (precision, recall)} 딕셔너리
    """
    X_pred  = pred_full.values                     # (n_items, n_users)
    X_train = train.values
    X_test  = test.values

    cand_mask     = np.isnan(X_train)              # (n_items, n_users)
    # NaN 비교는 False → test에 없는 아이템은 자동 제외
    relevant_mask = cand_mask & (X_test >= relevance_threshold)  # (n_items, n_users)
    n_rel         = relevant_mask.sum(axis=0)      # (n_users,) — 유저별 relevant 수
    has_cand      = cand_mask.any(axis=0)          # (n_users,) — 후보 있는 유저
    has_rel       = (n_rel > 0) & has_cand         # (n_users,) — recall 계산 가능

    # 후보가 아닌 아이템은 -inf로 마스킹 → argsort에서 뒤로 밀림
    N_max   = max(topn_list)
    scores  = np.where(cand_mask, X_pred, -np.inf) # (n_items, n_users)

    # argsort: 내림차순 정렬 → top_idx[:N, :]가 진짜 상위 N개 보장
    # (argpartition은 상위 N_max를 무순서로 뽑아 [:N]이 임의 N개가 됨 → 버그)
    top_idx = np.argsort(-scores, axis=0)[:N_max, :]  # (N_max, n_users), 내림차순 정렬

    results = {}
    for N in topn_list:
        # 상위 N개 인덱스에서 relevant 여부 확인
        top_n = top_idx[:N, :]                     # (N, n_users)
        hits  = relevant_mask[top_n, np.arange(pred_full.shape[1])]  # (N, n_users)
        n_hits = hits.sum(axis=0)                  # (n_users,)

        prec  = n_hits / N                         # (n_users,)
        rec   = np.where(has_rel,
                         n_hits / np.where(n_rel > 0, n_rel, 1),
                         np.nan)                   # (n_users,)

        # 방법론 기준: Precision/Recall 모두 U* (has_rel) 에서만 평균
        # (relevant item이 없는 유저는 양쪽 지표 모두 제외)
        prec_mean = float(np.mean(prec[has_rel])) if has_rel.any() else np.nan
        rec_mean  = float(np.nanmean(rec[has_rel])) if has_rel.any()  else np.nan
        results[N] = (prec_mean, rec_mean)

    return results


# ── Alpha Lookup 테이블 구성 ──────────────────────────────────────────────────
print("Loading alpha lookup from consolidated CSV...")
df_alpha = pd.read_csv(ALPHA_CSV)
df_alpha = df_alpha[df_alpha["type"] == "optimized"]

# (fold, method, K) → alpha* 딕셔너리 (TopN-independent이므로 첫 번째 값 사용)
alpha_lookup = (
    df_alpha.groupby(["fold", "method", "K"])["alpha"]
    .first()
    .to_dict()
)
print(f"  alpha 조합 수: {len(alpha_lookup):,}  (fold × method × K)")


# ── 기존 결과 로드 (resume 옵션) ──────────────────────────────────────────────
done_keys = set()
if args.resume and os.path.exists(OUT_CSV):
    df_existing = pd.read_csv(OUT_CSV)
    for _, row in df_existing.iterrows():
        done_keys.add((int(row["fold"]), row["method"], int(row["K"]), int(row["TopN"]), row["type"]))
    print(f"  Resume: 기존 {len(df_existing):,}행 로드, {len(done_keys):,} 조합 완료")


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  reeval_openworld_recall.py")
print("  Candidate pool: ALL unseen items (~1,587/user) — Open-World")
print("  유사도: sim_inner_{method}.npy  (sim_full 불필요 — 메서드 간 비교 목적)")
print("  최적화: K 누적 패스 1회 / TopN 전체 argsort 1회 / 완전 벡터화")
print("  Start: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 80)
print(f"  Folds  : {FOLDS}")
print(f"  Methods: {METHODS}")
print(f"  K      : {K_RANGE}")
print(f"  TopN   : {TOPN_RANGE}")
print(f"  예측 실행 횟수: {len(FOLDS)}×{len(METHODS)}×2(alpha) = "
      f"{len(FOLDS)*len(METHODS)*2:,}회  "
      f"(K는 1회 패스에서 처리, TopN은 평가 단계만)\n")

all_results = []
if args.resume and os.path.exists(OUT_CSV):
    all_results = [pd.read_csv(OUT_CSV)]

global_start = time.time()
total_runs   = len(FOLDS) * len(METHODS) * 2   # fold × method × 2 alpha (baseline/optimized)
done_runs    = 0

for fold_num in FOLDS:
    fold_id  = f"fold_{fold_num:02d}"
    fold_dir = os.path.join(DATA_ROOT, fold_id)

    t_fold = time.time()
    print(f"\n[{fold_id}] Loading data...", end=" ", flush=True)
    train = load_matrix(os.path.join(fold_dir, "train.csv"))
    test  = load_matrix(os.path.join(fold_dir, "test.csv"))
    print(f"train={train.shape}, test={test.shape}  ({time.time()-t_fold:.1f}s)")

    for method in METHODS:
        sim_path = os.path.join(fold_dir, f"sim_inner_{method}.npy")
        if not os.path.exists(sim_path):
            print(f"  [{method}] WARNING: sim_inner_{method}.npy 없음 — skipping")
            print(f"    → 먼저 python precompute_similarity_inner.py 를 실행하세요.")
            continue

        S = np.load(sim_path).astype(float)          # (users, users)

        fold_method_results = []
        t_method = time.time()

        # ── baseline(alpha=1) + optimized(alpha=alpha*) 각각 처리 ────────────
        # 주의: Recall/Precision은 alpha에 무관(ranking 불변)하지만
        #       실험 스키마 일관성을 위해 양쪽 모두 명시적으로 계산.
        # alpha*는 K마다 다를 수 있으므로 K별로 grouping.
        for run_type in ["baseline", "optimized"]:
            if run_type == "baseline":
                # baseline: alpha=1.0 → K 전체 공통
                alpha_per_k = {K: 1.0 for K in K_RANGE}
            else:
                # optimized: K마다 다른 alpha* 가능
                alpha_per_k = {}
                for K in K_RANGE:
                    a = alpha_lookup.get((fold_num, method, K), np.nan)
                    if not np.isnan(a):
                        alpha_per_k[K] = a

            # alpha 값이 동일한 K들을 묶어서 처리
            from itertools import groupby
            alpha_to_ks = {}
            for K, a in alpha_per_k.items():
                alpha_to_ks.setdefault(round(a, 6), []).append(K)

            for alpha, k_group in alpha_to_ks.items():
                if args.resume and all(
                    (fold_num, method, K, topn, run_type) in done_keys
                    for K in k_group for topn in TOPN_RANGE
                ):
                    done_runs += 1
                    continue

                # ── 예측: 모든 k_group K를 누적 패스 1회 ────────────────────
                S_eff     = apply_alpha(S, alpha)
                pred_by_k = predict_all_unseen_batch_k(train, S_eff, k_group)

                elapsed = time.time() - global_start
                rate    = done_runs / elapsed if (elapsed > 0 and done_runs > 0) else 0.01
                eta     = (total_runs - done_runs) / rate / 60 if rate > 0 else 0
                print(f"  [{method} {run_type:9s} alpha={alpha:.3f} K={k_group}]  "
                      f"({done_runs}/{total_runs}, ETA {eta:.0f}분)", flush=True)

                # ── 평가: 모든 K × TopN (벡터화) ─────────────────────────────
                for K in k_group:
                    if args.resume and (fold_num, method, K, TOPN_RANGE[0], run_type) in done_keys:
                        continue
                    pred_full = pred_by_k[K]
                    # 모든 TopN을 한 번에 평가 (argsort 1회)
                    topn_results = precision_recall_batch_topn(
                        pred_full, train, test, TOPN_RANGE, RELEVANCE_THRESHOLD
                    )
                    for topn in TOPN_RANGE:
                        prec, rec = topn_results[topn]
                        fold_method_results.append({
                            "fold":         fold_num,
                            "method":       method,
                            "K":            K,
                            "TopN":         topn,
                            "type":         run_type,
                            "alpha":        alpha,
                            "recall_ow":    rec,
                            "precision_ow": prec,
                        })
                done_runs += 1

        # fold × method 단위 중간 저장
        if fold_method_results:
            all_results.append(pd.DataFrame(fold_method_results))
            df_out = pd.concat(all_results, ignore_index=True)
            df_out.to_csv(OUT_CSV, index=False)

        print(f"  [{method}] fold={fold_num} 완료  ({time.time()-t_method:.1f}s)")

# ── 최종 저장 ─────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_ROOT, exist_ok=True)
df_final = pd.concat(all_results, ignore_index=True)
df_final.to_csv(OUT_CSV, index=False)

elapsed_total = time.time() - global_start
print("\n" + "=" * 80)
print(f"  저장 완료: {OUT_CSV}")
print(f"  총 행 수: {len(df_final):,}")
print(f"  총 경과: {elapsed_total/60:.1f}분")
print("  End: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 80)
