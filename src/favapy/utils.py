"""Utility functions for data preprocessing and correlation analysis."""

import logging
import time
import numpy as np
import pandas as pd
import anndata
from scipy import stats


def _preprocess_expression(x):
    """
    Apply log2 transformation and min-max normalization to expression data.
    
    Combines both preprocessing steps:
    1. Log2 transformation (skipped if negative values detected)
    2. Min-max normalization to [0, 1] range
    
    Parameters
    ----------
    x : np.ndarray
        Expression matrix with shape (n_genes, n_features).
    
    Returns
    -------
    x : np.ndarray
        Preprocessed expression matrix with values in [0, 1] range.
    """
    # Step 1: Apply log2 transformation if data is non-negative
    if np.any(x < 0):
        logging.warning(
            "Negative values detected, skipping log2 normalization."
        )
    else:
        x = np.log2(1 + x)
    
    # Step 2: Apply robust min-max normalization
    constant = 1e-8  # Small constant to avoid division by zero
    row_min = np.min(x, axis=1, keepdims=True)
    row_max = np.max(x, axis=1, keepdims=True)
    x = (x - row_min) / (row_max - row_min + constant)
    x = np.nan_to_num(x)  # Handle NaN values
    
    return x


def _load_data(input_file, data_type):
    """
    Loads and preprocesses data from a file.

    Parameters
    ----------
    input_file : str
        Path to the input file.
    data_type : str
        Type of the data file ('tsv' or 'csv').

    Returns
    -------
    expr : np.ndarray
        Processed data array.
    row_names : list
        List of row names corresponding to the data.
    """
    row_names = []
    array = []
    with open(input_file, "r", encoding="utf-8") as infile:
        next(infile)
        for line in infile:
            if data_type == "tsv":
                line = line.split("\t")
            else:
                line = line.split(",")
            row_names.append(line[0])
            array.append(line[1:])

    expr = np.asarray(array, dtype=np.float32)

    # Return raw data without preprocessing - cook() will handle it
    return expr, row_names


def _extract_data(data, layer=None):
    """
    Extract matrix and gene names from AnnData or pandas DataFrame.
    
    Parameters
    ----------
    data : anndata.AnnData or pd.DataFrame
        Input data. AnnData with genes in var, cells in obs.
        DataFrame with genes as index (rows), cells as columns.
    layer : str, optional
        For AnnData input, which layer to use. If None, uses X (default layer).
    
    Returns
    -------
    x : np.ndarray
        Expression matrix with shape (n_genes, n_cells).
    row_names : list
        Gene/protein names.
    
    Raises
    ------
    ValueError
        If input type is not supported or data structure is invalid.
    """
    if isinstance(data, anndata.AnnData):
        # AnnData input
        if layer is not None:
            if layer not in data.layers:
                raise ValueError(
                    f"Layer '{layer}' not found in AnnData object. "
                    f"Available layers: {list(data.layers.keys())}"
                )
            x_matrix = data.layers[layer]
        else:
            x_matrix = data.X
        
        # Convert sparse to dense if needed
        if hasattr(x_matrix, 'toarray'):
            x = x_matrix.toarray()
        else:
            x = np.asarray(x_matrix)
        
        # Transpose to (genes, cells) - AnnData stores (cells, genes)
        x = x.T
        row_names = data.var.index.tolist()
        
    elif isinstance(data, pd.DataFrame):
        # pandas DataFrame input (index = genes, columns = cells)
        x = data.values
        row_names = data.index.tolist()
        
    else:
        raise ValueError(
            f"Unsupported input type: {type(data).__name__}. "
            f"Expected anndata.AnnData or pd.DataFrame."
        )
    
    # Ensure float32 dtype and validate 2D shape
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x.shape}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"Empty data matrix with shape {x.shape}.")
    
    return x, row_names


def _create_protein_pairs(x_test_encoded, row_names, correlation_type="pearson",
                          interaction_count=100000, CC_cutoff=None):
    """
    Create pairs of proteins based on their encoded latent spaces.

    FAVA uses ALL THREE VAE encoder outputs (z_mean, z_log_sigma, z) concatenated
    to create a richer representation for correlation analysis. This captures:
    - z_mean: Mean latent embedding
    - z_log_sigma: Uncertainty/variance in latent space
    - z: Stochastic sample from the distribution

    Parameters
    ----------
    x_test_encoded : np.ndarray
        Encoded latent spaces with shape (3, n_samples, latent_dim)
        where dim 0 contains [z_mean, z_log_sigma, z]
    row_names : list
        List of row names corresponding to the data.
    correlation_type : str
        Type of correlation to use (Pearson or Spearman).
    interaction_count : int, optional
        Maximum number of interactions to include, by default 100000.
    CC_cutoff : float, optional
        Correlation Coefficient cutoff, by default None.

    Returns
    -------
    pairs_df : pd.DataFrame
        DataFrame containing filtered protein pairs and correlation scores,
        sorted by score descending.
    """
    start_time = time.time()
    
    # Reshape from (3, n_samples, latent_dim) to (n_samples, 3*latent_dim)
    n_outputs, n_samples, latent_dim = x_test_encoded.shape
    x_concat = x_test_encoded.transpose(1, 0, 2).reshape(n_samples, -1)
    
    # Compute correlation matrix using NumPy/SciPy (faster than pandas)
    # x_concat is (n_genes, 3*latent_dim), correlate rows (genes)
    if correlation_type == "pearson":
        corr_matrix = np.corrcoef(x_concat)
    else:  # spearman
        # Transpose so genes are columns, correlate along axis 0 (across features)
        corr_matrix, _ = stats.spearmanr(x_concat.T, axis=0)
    
    # Extract upper triangle only (avoids AB-BA duplicates and self-loops)
    n_genes = len(row_names)
    upper_idx = np.triu_indices(n_genes, k=1)
    scores = corr_matrix[upper_idx]
    
    # Sort once by score descending
    sort_idx = np.argsort(scores)[::-1]
    sorted_i = upper_idx[0][sort_idx]
    sorted_j = upper_idx[1][sort_idx]
    sorted_scores = scores[sort_idx]
    
    # Filter by cutoff or top N before creating DataFrame
    if CC_cutoff is not None and isinstance(CC_cutoff, (int, float)):
        # Filter by correlation cutoff (scores already sorted descending)
        logging.info(f" A cut-off of {CC_cutoff} is applied.")
        mask = sorted_scores >= CC_cutoff
        filtered_i = sorted_i[mask]
        filtered_j = sorted_j[mask]
        filtered_scores = sorted_scores[mask]
    elif interaction_count is not None:
        # Get top N by score (already sorted)
        top_n = min(interaction_count, len(sorted_scores))
        filtered_i = sorted_i[:top_n]
        filtered_j = sorted_j[:top_n]
        filtered_scores = sorted_scores[:top_n]
        logging.warning(f" The number of interactions in the output file is {len(filtered_scores)}.")
    else:
        # No filtering, return all pairs (already sorted)
        filtered_i = sorted_i
        filtered_j = sorted_j
        filtered_scores = sorted_scores
    
    # Create DataFrame only for filtered pairs
    pairs_df = pd.DataFrame({
        "Protein_1": [row_names[i] for i in filtered_i],
        "Protein_2": [row_names[j] for j in filtered_j],
        "Score": filtered_scores
    })
    
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    return pairs_df
