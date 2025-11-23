"""Utility functions for data preprocessing and correlation analysis."""

import logging
import time
import numpy as np
import pandas as pd
import anndata


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


def _create_protein_pairs(x_test_encoded, row_names, correlation_type="pearson"):
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

    Returns
    -------
    correlation_df : pd.DataFrame
        DataFrame containing protein pairs and correlation scores.
    """
    start_time = time.time()
    
    # Reshape from (3, n_samples, latent_dim) to (n_samples, 3*latent_dim)
    # This concatenates z_mean, z_log_sigma, and z horizontally
    n_outputs, n_samples, latent_dim = x_test_encoded.shape
    x_concatenated = x_test_encoded.transpose(1, 0, 2).reshape(n_samples, -1)

    # Correlation of the latent space: Pearson or Spearman
    corr = pd.DataFrame(x_concatenated.T).corr(method=correlation_type)
    corr.columns = corr.index = row_names

    correlation_df = corr.stack().reset_index()

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time taken: {total_time} seconds")
    correlation_df.columns = ["Protein_1", "Protein_2", "Score"]
    return correlation_df


def _pairs_after_cutoff(correlation, interaction_count=100000, CC_cutoff=None):
    """
    Filter protein pairs based on correlation scores and cutoffs.
    
    Removes duplicate pairs in both directions (AB-BA) before applying filters.
    For duplicate pairs, keeps the one with the higher score.

    Parameters
    ----------
    correlation : pd.DataFrame
        DataFrame containing protein pairs and correlation scores.
    interaction_count : int, optional
        Maximum number of interactions to include, by default 100000.
    CC_cutoff : float, optional
        Correlation Coefficient cutoff, by default None.

    Returns
    -------
    correlation_df_new : pd.DataFrame
        Filtered DataFrame with selected protein pairs (no duplicate directions).
    """
    # Remove duplicate pairs in both directions (AB-BA)
    # Sort by score descending, then keep first occurrence of each canonical pair
    correlation_sorted = correlation.sort_values("Score", ascending=False)
    
    # Create canonical pairs (sorted tuple so AB == BA)
    canonical = correlation_sorted.apply(
        lambda row: tuple(sorted([row["Protein_1"], row["Protein_2"]])), axis=1
    )
    
    # Drop duplicates keeping first (highest score)
    correlation_dedup = correlation_sorted[~canonical.duplicated(keep="first")]
    
    # Apply cutoff or count filter
    if CC_cutoff is not None and isinstance(CC_cutoff, (int, float)):
        logging.info(f" A cut-off of {CC_cutoff} is applied.")
        correlation_df_new = correlation_dedup.loc[(correlation_dedup["Score"] >= CC_cutoff)]
    else:
        correlation_df_new = correlation_dedup.iloc[:interaction_count, :]
        logging.warning(
            f" The number of interactions in the output file is {interaction_count} (duplicate directions removed)."
        )
    
    # Reset index to sequential 0, 1, 2, ...
    return correlation_df_new.reset_index(drop=True)

