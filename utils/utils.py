"""Computational utilities for connectivity analysis"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple

def compute_static_metrics(mat: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute avg_weight, mean_strength, density, var_weight for connectivity matrix."""
    tri = mat[np.triu_indices_from(mat, 1)]
    avg_weight = float(tri.mean())
    var_weight = float(tri.var())
    strength = mat.sum(axis=0)
    mean_strength = float(strength.mean())
    density = float(np.count_nonzero(tri) / tri.size)
    return avg_weight, mean_strength, density, var_weight

def create_metrics_dataframe(sc_matrices: Dict[str, np.ndarray], 
                           vs_ipps: List[str], mcs_ipps: List[str]) -> pd.DataFrame:
    """Create standardized metrics DataFrame from SC matrices"""
    records = []
    for ipp, mat in sc_matrices.items():
        avg_w, mean_s, dens, var_w = compute_static_metrics(mat)
        group = "VS" if ipp in vs_ipps else "MCS"
        records.append({
            "IPP": ipp,
            "Group": group,
            "avg_weight": avg_w,
            "mean_strength": mean_s,
            "density": dens,
            "var_weight": var_w,
        })
    return pd.DataFrame.from_records(records)

def compute_distance_matrix(sc_dict: Dict[str, np.ndarray], 
                          metric: str = "correlation") -> Tuple[List[str], np.ndarray]:
    """Return (IPP list, square distance matrix) for all SCs using any pdist metric."""
    ipps = list(sc_dict.keys())
    vecs = [sc_dict[ipp][np.triu_indices_from(sc_dict[ipp], 1)] for ipp in ipps]
    dist_mat = squareform(pdist(vecs, metric=metric))
    return ipps, dist_mat

def group_comparison(df: pd.DataFrame, group_col: str = "Group") -> pd.DataFrame:
    """Compare VS vs MCS groups for each metric using appropriate statistical tests"""
    results = {}
    metrics = ["avg_weight", "mean_strength", "density", "var_weight"]
    
    for metric in metrics:
        vs_vals = df.loc[df[group_col] == "VS", metric]
        mcs_vals = df.loc[df[group_col] == "MCS", metric]

        # Choose test based on normality (project standard approach)
        norm_vs = stats.shapiro(vs_vals).pvalue > 0.05 if len(vs_vals) >= 3 else False
        norm_mcs = stats.shapiro(mcs_vals).pvalue > 0.05 if len(mcs_vals) >= 3 else False

        if norm_vs and norm_mcs:
            stat, p = stats.ttest_ind(vs_vals, mcs_vals, equal_var=False)
            test_name = "Welch t‑test"
        else:
            stat, p = stats.mannwhitneyu(vs_vals, mcs_vals, alternative="two-sided")
            test_name = "Mann‑Whitney U"

        results[metric] = {"Test": test_name, "Statistic": stat, "p_value": p}

    return pd.DataFrame(results).T

def compute_identifiability_matrix(bootstrapped_patients: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute identifiability matrix from bootstrap samples using project method"""
    patients = list(bootstrapped_patients.keys())
    n = len(patients)
    
    # Split bootstraps in half for cross-validation approach  
    n_boot = bootstrapped_patients[patients[0]].shape[0]
    half = n_boot // 2

    # Vectorize & average the two halves for each patient
    vec1, vec2 = [], []
    for p in patients:
        mats = bootstrapped_patients[p]
        # Flatten each SC into a vector (upper triangle approach)
        v1 = mats[:half].reshape(half, -1).mean(axis=0)
        v2 = mats[half:].reshape(half, -1).mean(axis=0)
        vec1.append(v1)
        vec2.append(v2)
    
    vec1 = np.vstack(vec1)
    vec2 = np.vstack(vec2)

    # Build identifiability matrix
    ident = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr = np.corrcoef(vec1[i], vec2[j])[0, 1]
            ident[i, j] = corr

    return pd.DataFrame(ident, index=patients, columns=patients)

def compute_identifiability_stats(vec1: np.ndarray, vec2: np.ndarray, 
                                n_perm: int = 5000) -> Tuple[float, float]:
    """Compute identifiability statistics with permutation test"""
    N = vec1.shape[0]
    
    # Build the full N×N correlation matrix
    corr_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            corr_mat[i, j] = np.corrcoef(vec1[i], vec2[j])[0, 1]

    # Compute observed identifiability difference
    I_self_obs = np.mean(np.diag(corr_mat))
    sum_all = corr_mat.sum()
    sum_diag = np.diag(corr_mat).sum()
    I_others_obs = (sum_all - sum_diag) / (N * (N - 1))
    Idiff_obs = I_self_obs - I_others_obs

    # Permutation test
    null_idiffs = np.zeros(n_perm)
    for k in range(n_perm):
        perm = np.random.permutation(N)
        permuted = corr_mat[:, perm]
        
        diag_p = np.diag(permuted)
        I_self_p = diag_p.mean()
        total_sum = permuted.sum()
        I_others_p = (total_sum - diag_p.sum()) / (N * (N - 1))
        
        null_idiffs[k] = I_self_p - I_others_p

    # p-value with continuity correction
    p_val = (np.sum(null_idiffs >= Idiff_obs) + 1) / (n_perm + 1)
    
    return Idiff_obs, p_val