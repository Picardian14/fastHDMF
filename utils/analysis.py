"""
Analysis module for FC grid correlation analysis.

This module provides functions to compute correlation matrices from a grid of functional connectivity (FC) matrices.

Each FC matrix is typically a 2D numpy array. The function in this module 
vectorizes each FC matrix and computes the Pearson correlation coefficient 
between all pairs of FC matrices in the grid.
"""

import numpy as np


def get_cross_correlation(dataset_fcs: np.ndarray, target_fcs: np.ndarray) -> np.ndarray:
    """Computes a correlation matrix between two sets of FC matrices.
    
    Parameters:
        dataset_fcs (array-like): A grid (e.g., numpy array or list) of functional connectivity (FC) matrices.
                                  Each element should be a 2D numpy array.
        target_fcs (array-like): A grid (e.g., numpy array or list) of target functional connectivity (FC) matrices.
                                  Each element should be a 2D numpy array.
    
    Returns:
        np.ndarray: A square matrix of Pearson correlation coefficients between vectorized FC matrices.
                    The element (i, j) is the correlation between the i-th and j-th FC matrix in the flattened grid.
    """
    
    # flatten only the upper‐triangle of each FC matrix
    triu_idx = np.triu_indices_from(dataset_fcs[0], k=1)
    flat_ds = np.vstack([mat[triu_idx] if mat is not None else np.zeros(triu_idx[0].shape) for mat in dataset_fcs])
    flat_tg = np.vstack([mat[triu_idx] if mat is not None else np.zeros(triu_idx[0].shape) for mat in target_fcs])

    # compute the cross‐correlation matrix in one shot
    corr_matrix = np.corrcoef(flat_ds, flat_tg)[:len(dataset_fcs), len(dataset_fcs):]
    
    return corr_matrix

def get_identificability(corr_matrix: np.ndarray) -> float:
    """Computes the identifiability score from a correlation matrix.
    
    The identifiability score is defined as the difference between the mean of the diagonal elements
    (self-correlations) and the mean of the off-diagonal elements (cross-correlations).
    
    Parameters:
        corr_matrix (np.ndarray): A square matrix of Pearson correlation coefficients.
    
    Returns:
        float: The identifiability score.
    """
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("Input correlation matrix must be square.")
    
    n = corr_matrix.shape[0]
    diag_mean = np.mean(np.diag(corr_matrix))
    off_diag_mean = (np.sum(corr_matrix) - np.sum(np.diag(corr_matrix))) / (n * (n - 1))
    
    identifiability_score = diag_mean - off_diag_mean
    return identifiability_score

def get_structural_differentiabiltiy(sc_matrices: np.ndarray) -> float:
    """Computes the structural differentiability score from a set of structural connectivity (SC) matrices.
    
    The structural differentiability score is defined as the mean of the pairwise Pearson correlation coefficients
    between all pairs of SC matrices.
    
    Parameters:
        sc_matrices (np.ndarray): A 3D numpy array where each element along the first axis is a 2D SC matrix.
    
    Returns:
        float: The structural differentiability score.
    """
    n = sc_matrices.shape[0]
    if n < 2:
        raise ValueError("At least two SC matrices are required to compute structural differentiability.")
    
    # flatten only the upper‐triangle of each SC matrix
    triu_idx = np.triu_indices_from(sc_matrices[0], k=1)
    flat_sc = np.vstack([mat[triu_idx] for mat in sc_matrices])
    
    # compute the full correlation matrix
    corr_matrix = np.corrcoef(flat_sc)
    
    # compute the mean of the off-diagonal elements
    off_diag_mean = (np.sum(corr_matrix) - np.sum(np.diag(corr_matrix))) / (n * (n - 1))
    
    return off_diag_mean    

def identifiability_gradient(fc_grid: np.ndarray) -> np.ndarray:
    """Computes the gradient of identifiability scores across a grid of FC matrices.
    
    Parameters:
        fc_grid (np.ndarray): A 2D numpy array where each element is a functional connectivity (FC) matrix.
    
    Returns:
        np.ndarray: A 2D numpy array of the same shape as fc_grid, containing the identifiability scores.
    """
    n_rows, _ = fc_grid.shape
    identifiability_scores = np.zeros((n_rows,))
    
    for i in range(n_rows-1):
    
        # Create dataset and target excluding the current element
        dataset = fc_grid[i, :]
        target = fc_grid[i+1, :]
        # Compute correlation matrix
        corr_matrix = get_cross_correlation(dataset, target)
        # Compute identifiability score
        identifiability_scores[i] = get_identificability(corr_matrix)

    return identifiability_scores
