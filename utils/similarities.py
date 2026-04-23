# similarities.py
# ------------------------------------------------------------
# - 공통 pairwise 빌더는 '아주 얇게'만 감싸 반복 중복만 줄임
# - compute_similarity(train_df, method)만 실험 코드에서 호출
# ------------------------------------------------------------
from __future__ import annotations   # Python 3.9: X | Y 타입 힌트 지원

from typing import Callable, Dict, List
import numpy as np
import pandas as pd

# =========================
# 1) PCC 원자 함수
# =========================
def pcc(x, y):
    """
    Computes Pearson correlation coefficient (PCC) between two users x and y,
    centering by each user's mean over all rated items (not just common ones).
    """
    # User means (over their own non-NaN ratings)
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)

    # Identify co-rated items
    mask = ~np.isnan(x) & ~np.isnan(y)

    if np.sum(mask) < 2:
        return np.nan  # Not enough common ratings

    # Centered ratings on co-rated items
    x_centered = x[mask] - mean_x
    y_centered = y[mask] - mean_y

    # Compute PCC
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2)) * np.sqrt(np.sum(y_centered**2))

    return numerator / denominator if denominator != 0 else np.nan

# =========================
# 2)  Cosine 원자 함수
# =========================
def cos(x, y):
    """
    Compute cosine similarity between x and y:
    - Numerator: dot product over co-rated items
    - Denominator: full norm of x and y over their non-NaN entries
    """
    # Co-rated mask
    mask_common = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask_common) == 0:
        return np.nan

    # Numerator: dot product over co-rated items
    numerator = np.nansum(x[mask_common] * y[mask_common])

    # Denominator: full norms (not just on co-rated items)
    norm_x = np.sqrt(np.nansum(x[~np.isnan(x)]**2))
    norm_y = np.sqrt(np.nansum(y[~np.isnan(y)]**2))

    denominator = norm_x * norm_y

    return numerator / denominator if denominator != 0 else np.nan


# =========================
# ACOS (Adjusted Cosine)
# =========================
def acos(x: np.ndarray, y: np.ndarray, item_means: np.ndarray) -> float:
    """
    Adjusted Cosine similarity:
    - Center each user vector by item-mean (column mean of rating matrix).
    - Computed only on co-rated items.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan

    x_adj = x[mask] - item_means[mask]
    y_adj = y[mask] - item_means[mask]
    numerator = np.sum(x_adj * y_adj)
    denominator = np.sqrt(np.sum(x_adj ** 2)) * np.sqrt(np.sum(y_adj ** 2))
    return numerator / denominator if denominator != 0 else np.nan


def cpcc(x: np.ndarray, y: np.ndarray, max_rating: int) -> float:
    """
    CPCC similarity (Shardanand & Maes, 1995):
    - 사용자 평균 대신 rating scale의 중앙값으로 중심화
    - co-rated item 기준
    """
    # 중앙값 (예: rating=1~5이면 median=3)
    median = (max_rating + 1) / 2

    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan

    # 중심화
    x_centered = x[mask] - median
    y_centered = y[mask] - median

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2)) * np.sqrt(np.sum(y_centered**2))

    return numerator / denominator if denominator != 0 else np.nan

def jaccard(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two users' rating vectors.
    - 공통 평가 아이템(co-rated): 둘 다 NaN이 아닌 경우
    - 전체 평가 고려 아이템(union): 둘 중 하나라도 NaN이 아닌 경우
    """
    mask_inter = ~np.isnan(x) & ~np.isnan(y)
    mask_union = ~np.isnan(x) | ~np.isnan(y)

    num = np.sum(mask_inter)
    den = np.sum(mask_union)

    if den == 0:
        return np.nan # Not enough common ratings
    return num / den

def spcc(x: np.ndarray, y: np.ndarray, scale: float = 2.0) -> float:
    """
    Computes Pearson correlation coefficient (PCC) between two users x and y,
    centering by each user's mean over all rated items (not just common ones).
    """
    # User means (over their own non-NaN ratings)
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)

    # Identify co-rated items
    mask = ~np.isnan(x) & ~np.isnan(y)

    if np.sum(mask) < 2:
        return np.nan  # Not enough common ratings

    # Centered ratings on co-rated items
    x_centered = x[mask] - mean_x
    y_centered = y[mask] - mean_y

    # Compute PCC
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2)) * np.sqrt(np.sum(y_centered**2))

    return numerator / denominator * 1/(1+np.exp(-np.sum(mask)/2)) if denominator != 0 else np.nan

def msd(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Mean Squared Difference (MSD) similarity between two users x and y.
    Ratings are normalized to [0,1] scale using (r - rmin) / (rmax - rmin) = r/4
    for a 1-5 rating scale (range = 4).
    """
    mask = ~np.isnan(x) & ~np.isnan(y)

    if np.sum(mask) < 2:
        return np.nan  # Not enough common ratings

    return 1 - np.sum(((x[mask] - y[mask]) / 4) ** 2) / np.sum(mask)

def jmsd(x: np.ndarray, y: np.ndarray) -> float:
    """
    Jaccard Mean Squared Difference (JMSD) similarity.
    Combines Jaccard similarity (binary overlap) and MSD similarity (rating distance).
    """
    jac = jaccard(x, y)
    m = msd(x, y)
    if np.isnan(jac) or np.isnan(m):
        return np.nan
    return jac * m

# -----------------------------
# 스케일(최소/최대) 자동 추정
# -----------------------------
def _infer_rating_bounds(train_df: pd.DataFrame) -> tuple[float, float]:
    vals = train_df.to_numpy(dtype=float, copy=False)
    obs = vals[~np.isnan(vals)]
    if obs.size == 0:
        # 비상 상황: 값이 전혀 없으면 0~1 가정
        return 0.0, 1.0
    vmin = float(np.nanmin(obs))
    vmax = float(np.nanmax(obs))
    # 정수 스케일 힌트가 있으면 경계 반올림
    if np.allclose(obs, np.round(obs), atol=1e-6):
        vmin = round(vmin)
        vmax = round(vmax)
    return vmin, vmax


# -----------------------------
# PIP 구성요소 (벡터화 버전)
# -----------------------------
def _agreement_vec(r1: np.ndarray, r2: np.ndarray, med: float) -> np.ndarray:
    # True if (r1 - med) * (r2 - med) >= 0
    return (r1 - med) * (r2 - med) >= 0



def _proximity_vec(r1: np.ndarray, r2: np.ndarray, vmin: float, vmax: float,
                   agree: np.ndarray) -> np.ndarray:
    # d = |r1-r2| (agree) or 2|r1-r2| (disagree)
    d = np.abs(r1 - r2)
    d = np.where(agree, d, 2.0 * d)
    return (2.0 * np.abs(vmax - vmin) + 1.0 - d) ** 2

def _impact_vec(r1: np.ndarray, r2: np.ndarray, med: float,
                agree: np.ndarray) -> np.ndarray:
    # (|r-med|+1)(|r'-med|+1) if agree; reciprocal if disagree
    a = np.abs(r1 - med) + 1.0
    b = np.abs(r2 - med) + 1.0
    base = a * b
    return np.where(agree, base, 1.0 / base)

def _popularity_vec(r1: np.ndarray, r2: np.ndarray, item_means: np.ndarray,
                    vmax: float, vmin: float) -> np.ndarray:
    # 1 + ((r1/scale + r2/scale)/2 - mu/scale)^2  if same side wrt mu; else 1
    # (원 코드: mu = item 평균, scale=5 고정 → 여기선 동적 스케일 사용)
    # scale = vmax - vmin if vmax > vmin else 1.0
    scale = 5
    mu = item_means
    same_side = (r1 - mu) * (r2 - mu) > 0
    term = ((r1 / scale + r2 / scale) / 2.0) - (mu / scale)
    val = 1.0 + term**2
    return np.where(same_side, val, 1.0)


# -----------------------------
# 원자 PIP: 두 사용자 벡터 x,y → 스칼라
# -----------------------------
def pip(x: np.ndarray, y: np.ndarray,
             item_means: np.ndarray, vmin: float, vmax: float) -> float:
    """
    Ahn(2008) PIP: sum_i Proximity(i) * Impact(i) * Popularity(i) over co-rated items i.
    - x,y: 사용자 평점 벡터 (NaN 허용)
    - item_means: 아이템별 평균(열 평균)
    - vmin,vmax: 평점 스케일 하한/상한
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan

    r1 = x[mask]
    r2 = y[mask]
    mu_i = item_means[mask]
    med = (vmin + vmax) / 2.0

    agree = _agreement_vec(r1, r2, med)
    prox  = _proximity_vec(r1, r2, vmin, vmax, agree)
    imp   = _impact_vec(r1, r2, med, agree)
    pop   = _popularity_vec(r1, r2, mu_i, vmax, vmin)

    pip_vals = prox * imp * pop
    return float(np.sum(pip_vals))

# -----------------------------
# AMI (Adjusted Mutual Information)
# -----------------------------

from sklearn.metrics import adjusted_mutual_info_score

def ami(u_ratings: np.ndarray, v_ratings: np.ndarray) -> float:
    """
    Computes Adjusted Mutual Information (AMI) between two users' ratings.

    Args:
        u_ratings: numpy array of ratings by user u (np.nan for missing).
        v_ratings: numpy array of ratings by user v (np.nan for missing).

    Returns:
        AMI score (float), np.nan if insufficient data.
    """
    # Identify co-rated items
    mask = ~np.isnan(u_ratings) & ~np.isnan(v_ratings)
    if np.sum(mask) < 2:
        return np.nan  # Not enough overlap

    u_clean = u_ratings[mask]
    v_clean = v_ratings[mask]

    # 만약 평점이 정수 스케일(예: 1~5)이면 int 변환
    if np.all(u_clean == np.round(u_clean)) and np.all(v_clean == np.round(v_clean)):
        u_clean = u_clean.astype(int)
        v_clean = v_clean.astype(int)
    else:
        # 연속형 평점일 경우 binning 고려 필요 (예: round)
        u_clean = np.round(u_clean).astype(int)
        v_clean = np.round(v_clean).astype(int)

    return adjusted_mutual_info_score(u_clean, v_clean)

# -----------------------------
# ARI (Adjusted Rand Index)
# -----------------------------
from sklearn.metrics import adjusted_rand_score

def ari(u_ratings: np.ndarray, v_ratings: np.ndarray) -> float:
    """
    Computes Adjusted Rand Index (ARI) between two users' ratings over co-rated items.
    """
    mask = ~np.isnan(u_ratings) & ~np.isnan(v_ratings)
    if np.sum(mask) < 2:
        return np.nan

    u_clean = u_ratings[mask].astype(int)
    v_clean = v_ratings[mask].astype(int)

    return adjusted_rand_score(u_clean, v_clean)

# -----------------------------
# SRC (Spearman Rank Correlation) 원자 함수
# -----------------------------
from scipy.stats import spearmanr
def src(u_ratings: np.ndarray, v_ratings: np.ndarray) -> float:
    """
    Compute Spearman rank correlation between two users' ratings
    over co-rated items (ignoring NaN).
    """
    mask = ~np.isnan(u_ratings) & ~np.isnan(v_ratings)
    if np.sum(mask) < 2:
        return np.nan  # Not enough overlap

    u_clean = u_ratings[mask]
    v_clean = v_ratings[mask]
    #print(u_clean)
    #print(v_clean)

    corr, _ = spearmanr(u_clean, v_clean)
    return float(corr) if corr is not None else np.nan

from scipy.stats import kendalltau

def kendall_tau_b(u_ratings, v_ratings):
    """
    Computes Kendall's Tau-b rank correlation between two users' ratings over co-rated items.

    Args:
        u_ratings: numpy array of user u's ratings (np.nan for missing).
        v_ratings: numpy array of user v's ratings (np.nan for missing).

    Returns:
        Kendall's Tau-b coefficient (float), np.nan if insufficient data.
    """
    # Mask for co-rated items
    mask = ~np.isnan(u_ratings) & ~np.isnan(v_ratings)

    if np.sum(mask) < 2:
        return np.nan  # Not enough co-rated items

    u_clean = u_ratings[mask]
    v_clean = v_ratings[mask]

    tau, _ = kendalltau(u_clean, v_clean, variant='b')
    return tau

def euclidean(u_ratings: np.ndarray, v_ratings: np.ndarray) -> float:
    """
    Computes Euclidean similarity between two users:
    similarity = 1 / (1 + Euclidean distance over co-rated items)
    """
    # Mask for co-rated items
    mask = ~np.isnan(u_ratings) & ~np.isnan(v_ratings)

    if np.sum(mask) < 1:
        return np.nan  # No co-rated items

    distance = np.sqrt(np.sum((u_ratings[mask] - v_ratings[mask]) ** 2))
    similarity = 1.0 / (1.0 + distance)

    return similarity

def manhattan(u_ratings: np.ndarray, v_ratings: np.ndarray) -> float:
    """
    Computes Manhattan similarity between two users:
    similarity = 1 / (1 + Manhattan distance over co-rated items)
    """
    mask = ~np.isnan(u_ratings) & ~np.isnan(v_ratings)

    if np.sum(mask) < 1:
        return np.nan  # No co-rated items

    distance = np.sum(np.abs(u_ratings[mask] - v_ratings[mask]))
    similarity = 1.0 / (1.0 + distance)

    return similarity

def chebyshev(u_ratings: np.ndarray, v_ratings: np.ndarray) -> float:
    """
    Computes Chebyshev similarity between two users:
    similarity = 1 / (1 + Chebyshev distance over co-rated items)
    """
    mask = ~np.isnan(u_ratings) & ~np.isnan(v_ratings)

    if np.sum(mask) < 1:
        return np.nan  # No co-rated items

    distance = np.max(np.abs(u_ratings[mask] - v_ratings[mask]))
    similarity = 1.0 / (1.0 + distance)

    return similarity


# -----------------------------
# URP (User Rating Profile similarity factor)
# -----------------------------
def urp(x: np.ndarray, y: np.ndarray) -> float:
    # Extract valid ratings
    x_rated = x[~np.isnan(x)]
    y_rated = y[~np.isnan(y)]

    if len(x_rated) == 0 or len(y_rated) == 0:
        return np.nan

    mu_x = np.mean(x_rated)
    mu_y = np.mean(y_rated)
    s_x = np.std(x_rated, ddof=0)  # population std
    s_y = np.std(y_rated, ddof=0)

    diff = (mu_x - mu_y) * (s_x - s_y)
    return 1 - 1 / (1 + np.exp(-diff))


# -----------------------------
# Triangle similarity
# -----------------------------
def triangle_similarity(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan

    x_c = x[mask]
    y_c = y[mask]

    norm_diff = np.linalg.norm(x_c - y_c)
    norm_sum = np.linalg.norm(x_c) + np.linalg.norm(y_c)

    return 1 - (norm_diff / norm_sum) if norm_sum != 0 else np.nan


# -----------------------------
# ITR similarity (Integration of Triangle & URP)
# -----------------------------
def itr(x: np.ndarray, y: np.ndarray) -> float:
    t_sim = triangle_similarity(x, y)
    u_sim = urp(x, y)
    if np.isnan(t_sim) or np.isnan(u_sim):
        return np.nan
    return t_sim * u_sim

# -----------------------------
# rpb: rating profile based (cos of |Δmean|·|Δstd|)
# -----------------------------
def rpb(x: np.ndarray, y: np.ndarray) -> float:
    x_rated = x[~np.isnan(x)]
    y_rated = y[~np.isnan(y)]

    if x_rated.size == 0 or y_rated.size == 0:
        return np.nan

    mu_x = np.mean(x_rated)
    mu_y = np.mean(y_rated)
    s_x  = np.std(x_rated, ddof=0)  # population std
    s_y  = np.std(y_rated, ddof=0)

    product = abs(mu_x - mu_y) * abs(s_x - s_y)
    return float(np.cos(product))


# -----------------------------
# ipcc: item-mean–informed PCC 변형
# -----------------------------
def ipcc(x: np.ndarray, y: np.ndarray, item_means: np.ndarray) -> float:
    mask_x = ~np.isnan(x)
    mask_y = ~np.isnan(y)
    mask_common = mask_x & mask_y

    if np.sum(mask_common) < 2:
        return np.nan

    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)

    # 공통 아이템에서의 편차 차(사용자평균 보정 vs 아이템평균 보정의 차)
    diff_x = (x[mask_common] - mean_x) - (x[mask_common] - item_means[mask_common])
    diff_y = (y[mask_common] - mean_y) - (y[mask_common] - item_means[mask_common])
    numerator = np.sum(diff_x * diff_y)

    # 각 사용자 전체(자기 평가 아이템)에 대한 분모
    full_diff_x = (x[mask_x] - mean_x) - (x[mask_x] - item_means[mask_x])
    full_diff_y = (y[mask_y] - mean_y) - (y[mask_y] - item_means[mask_y])
    denom_x = np.sqrt(np.sum(full_diff_x**2))
    denom_y = np.sqrt(np.sum(full_diff_y**2))

    if denom_x == 0 or denom_y == 0:
        return np.nan

    return float(numerator / (denom_x * denom_y))


# -----------------------------
# ipwr = rpb × ipcc  (원자 유사도)
# -----------------------------
def ipwr(x: np.ndarray, y: np.ndarray, item_means: np.ndarray) -> float:
    a = rpb(x, y)
    b = ipcc(x, y, item_means)
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a * b)


# =========================
# 공통 pairwise 빌더 
# =========================
def _pairwise_build(R: np.ndarray, sim_xy: Callable[[np.ndarray, np.ndarray], float],
                    fill_diagonal: float = 1.0, out_dtype: str = "float32") -> np.ndarray:
    """
    행=사용자, 열=아이템 행렬 R(U×I)에 대해, 사용자-사용자 유사도 행렬(U×U)을 만듦.
    - sim_xy: (x, y) -> float  형태의 '원자 유사도 함수'
      (PCC/코사인 모두 같은 방식으로 호출)
    - 직관성을 위해 단순 이중 for 루프 사용(약간 비효율 OK).
    """
    U = R.shape[0]
    S = np.full((U, U), np.nan, dtype=out_dtype)
    
    for u in range(U):
        S[u, u] = fill_diagonal
        x = R[u]
        for v in range(u + 1, U):
            y = R[v]
            s = sim_xy(x, y)
            S[u, v] = s
            S[v, u] = s
    return S



# =========================
# 4) 공개 함수: compute_pcc / compute_cosine
# =========================
def compute_pcc(train_df: pd.DataFrame) -> np.ndarray:
    """
    DataFrame(U×I, NaN=미관측) -> ndarray(U×U) PCC 유사도.
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, pcc, fill_diagonal=1.0, out_dtype="float32")

def compute_cosine(train_df: pd.DataFrame) -> np.ndarray:
    """
    DataFrame(U×I, NaN=미관측) -> ndarray(U×U) 코사인 유사도.
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, cos, fill_diagonal=1.0, out_dtype="float32")

def compute_acos(train_df: pd.DataFrame) -> np.ndarray:
    """
    user×item DataFrame(NaN=missing) -> user×user ACOS similarity (ndarray, float32)
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    item_means = np.nanmean(R, axis=0)
    sim_fn = lambda x, y: acos(x, y, item_means)
    return _pairwise_build(R, sim_xy=sim_fn, fill_diagonal=1.0, out_dtype="float32")

def compute_cpcc(train_df: pd.DataFrame, max_rating: int = 5) -> np.ndarray:
    """
    user×item DataFrame(NaN=missing) -> user×user CPCC similarity (ndarray, float32)
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    sim_fn = lambda a, b: cpcc(a, b, max_rating)
    return _pairwise_build(R, sim_xy=sim_fn, fill_diagonal=1.0, out_dtype="float32")

def compute_jaccard(train_df: pd.DataFrame) -> np.ndarray:
    """
    user×item DataFrame(NaN=missing) → user×user Jaccard similarity
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=jaccard, fill_diagonal=1.0, out_dtype="float32")

def compute_spcc(train_df: pd.DataFrame) -> np.ndarray:
    """
    user×item DataFrame(NaN=missing) → user×user SPCC similarity (float32)
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=spcc, fill_diagonal=1.0, out_dtype="float32")

def compute_msd(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute MSD similarity matrix for all users in train_df.
    - train_df: user × item rating matrix (NaN = missing)
    - return: user × user similarity matrix
    """
    R = train_df.to_numpy(dtype=float)
    return _pairwise_build(R, msd, fill_diagonal=1.0, out_dtype="float32")

def compute_jmsd(train_df: pd.DataFrame) -> np.ndarray:
    """
    user×item DataFrame(NaN=missing) → user×user JMSD similarity
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=jmsd, fill_diagonal=1.0, out_dtype="float32")

def compute_pip(train_df: pd.DataFrame,
                vmin: float | None = None,
                vmax: float | None = None) -> np.ndarray:
    """
    user×item DataFrame(NaN=missing) → user×user PIP similarity (float32)
    - vmin, vmax 미지정 시 데이터에서 자동 추정
    - Popularity는 아이템 평균(열 평균)을 사용
    - Ahn(2008) 원식은 합(sum) 형태이므로, 정규화가 필요하면 후처리에서 나눠 쓰세요.
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    item_means = np.nanmean(R, axis=0)

    if vmin is None or vmax is None:
        _vmin, _vmax = _infer_rating_bounds(train_df)
    else:
        _vmin, _vmax = float(vmin), float(vmax)
    
    print(item_means)
    print(_vmin, _vmax)

    # 클로저로 item_means/스케일 캡쳐
    sim_fn = lambda a, b: pip(a, b, item_means=item_means, vmin=_vmin, vmax=_vmax)
    return _pairwise_build(R, sim_xy=sim_fn, fill_diagonal=1.0, out_dtype="float32")

def compute_ami(train_df: pd.DataFrame) -> np.ndarray:
    """
    user×item DataFrame(NaN=missing) → user×user JMSD similarity
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=ami, fill_diagonal=1.0, out_dtype="float32")

def compute_ari(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute Adjusted Rand Index (ARI) similarity matrix for all users.
    Args:
        train_df: user × item rating DataFrame (NaN = missing)
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=ari, fill_diagonal=1.0, out_dtype="float32")

def compute_src(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute Spearman rank correlation (SRC) similarity matrix for all users.
    - train_df: user × item rating DataFrame (NaN = missing)
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=src, fill_diagonal=1.0, out_dtype="float32")

def compute_kendall_tau_b(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute Kendall's Tau-b similarity matrix for all users.
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy= kendall_tau_b, fill_diagonal=1.0, out_dtype="float32")

def compute_euclidean(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute Euclidean similarity matrix for all users.
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=euclidean, fill_diagonal=1.0, out_dtype="float32")

def compute_manhattan(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute Manhattan similarity matrix for all users.
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=manhattan,
                           fill_diagonal=1.0, out_dtype="float32")

def compute_chebyshev(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute Chebyshev similarity matrix for all users.
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=chebyshev,
                           fill_diagonal=1.0, out_dtype="float32")

def compute_itr(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute ITR similarity matrix for all users.
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    return _pairwise_build(R, sim_xy=itr, fill_diagonal=1.0, out_dtype="float32")

def compute_ipwr(train_df: pd.DataFrame) -> np.ndarray:
    """
    user×item DataFrame(NaN=missing) → user×user IPWR similarity (float32)
    IPWR = rpb × ipcc, where ipcc uses item_means (column-wise means).
    """
    R = train_df.to_numpy(dtype=float, copy=False)
    item_means = np.nanmean(R, axis=0)
    sim_fn = lambda a, b: ipwr(a, b, item_means=item_means)
    return _pairwise_build(R, sim_xy=sim_fn, fill_diagonal=1.0, out_dtype="float32")

# =========================
# 5) 이름 → 함수 매핑 & 진입점
# =========================
SIMILARITIES: Dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "pcc": compute_pcc,
    "cosine": compute_cosine,
    "acos": compute_acos,
    "cpcc": compute_cpcc, 
    "jaccard": compute_jaccard,
    "spcc": compute_spcc,
    "msd": compute_msd,   
    "jmsd": compute_jmsd,
    "ami": compute_ami,
    "ari": compute_ari,
    "src": compute_src,
    "kendall_tau_b": compute_kendall_tau_b,
    "euclidean": compute_euclidean,
    "manhattan": compute_manhattan,
    "chebyshev": compute_chebyshev,
    "itr": compute_itr,
    "ipwr": compute_ipwr,
    #"pip": compute_pip,
    # 나중에 유사도 늘리면 여기 한 줄만 추가: "foo": compute_foo,
}

def compute_similarity(train_df: pd.DataFrame, method: str) -> np.ndarray:
    """
    실험 코드에서 사용하는 단일 진입점.
    - method: {"pcc","cosine", ...}
    """
    key = method.strip().lower()
    if key not in SIMILARITIES:
        raise ValueError(f"Unknown similarity '{method}'. Available: {sorted(SIMILARITIES)}")
    return SIMILARITIES[key](train_df)

def available_methods() -> List[str]:
    """디버그용: 등록된 유사도 이름 리스트."""
    return sorted(SIMILARITIES)
