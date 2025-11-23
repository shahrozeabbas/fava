import logging
import argparse
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
import pandas as pd

from .models import VAE
from .utils import (
    _preprocess_expression,
    _extract_data,
    _load_data,
    _create_protein_pairs,
    _pairs_after_cutoff,
)


# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Infer Functional Associations using Variational Autoencoders on -Omics data."
    )
    parser.add_argument("input_file", type=str, help="The absolute path of the data.")
    parser.add_argument(
        "output_file", type=str, help="The absolute path where the output will be saved"
    )
    parser.add_argument(
        "-t",
        "--data_type",
        type=str,
        default="tsv",
        choices=["tsv", "csv"],
        help="Type of input data.",
    )
    parser.add_argument(
        "-n",
        dest="interaction_count",
        type=int,
        default=100000,
        help="The number of interactions in the output file.",
    )
    parser.add_argument(
        "-c",
        dest="CC_cutoff",
        type=float,
        default=None,
        help="The cut-off on the Correlation scores.",
    )
    parser.add_argument(
        "-d",
        dest="hidden_layer",
        default=None,
        type=int,
        help="Intermediate/hidden layer dimensions",
    )
    parser.add_argument(
        "-l", dest="latent_dim", default=None, type=int, help="Latent space dimensions"
    )
    parser.add_argument(
        "-e", dest="epochs", type=int, default=50, help="How many epochs?"
    )
    parser.add_argument(
        "-b", dest="batch_size", type=int, default=32, help="batch_size"
    )
    parser.add_argument(
        "-cor",
        "--correlation_type",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Type of correlation to use (Pearson or Spearman).",
    )

    args = parser.parse_args()
    return args


def cook(
    data,
    log2_normalization=True,
    hidden_layer=None,
    latent_dim=None,
    epochs=50,
    batch_size=32,
    interaction_count=100000,
    correlation_type="pearson",
    CC_cutoff=None,
    layer=None,
):
    """
    Preprocess data, train a Variational Autoencoder (VAE), and create filtered protein pairs.

    Parameters
    ----------
    data : anndata.AnnData or pd.DataFrame
        Input data. Can be:
        - AnnData object with genes in var and cells in obs
        - pandas DataFrame with genes as index (rows), cells as columns
    log2_normalization : bool, optional
        Whether to apply log2 normalization and min-max scaling, by default True.
    hidden_layer : int, optional
        Number of units in the hidden layer, by default None (auto-determined).
    latent_dim : int, optional
        Dimension of the latent space, by default None (auto-determined).
    epochs : int, optional
        Number of training epochs, by default 50.
    batch_size : int, optional
        Batch size for training, by default 32.
    interaction_count : int, optional
        Maximum number of interactions to include, by default 100000.
    correlation_type : str, optional
        Type of correlation to use ('pearson' or 'spearman'), by default 'pearson'.
    CC_cutoff : float, optional
        Correlation Coefficient cutoff, by default None.
    layer : str, optional
        For AnnData input, which layer to use. If None, uses X (default layer).

    Returns
    -------
    final_pairs : pd.DataFrame
        Filtered protein pairs based on correlation and cutoffs.
    
    Raises
    ------
    ValueError
        If input type is not supported or data dimensions are invalid.
    """
    # Step 1: Extract matrix and gene names from input
    x, row_names = _extract_data(data, layer=layer)
    
    # Step 2: Apply preprocessing if enabled
    if log2_normalization:
        x = _preprocess_expression(x)
    
    # Step 3: Determine architecture dimensions
    original_dim = x.shape[1]
    
    if hidden_layer is None:
        if original_dim >= 2000:
            hidden_layer = 1000
        elif original_dim >= 500:
            hidden_layer = 500
        else:
            hidden_layer = max(50, original_dim // 2)
    
    if latent_dim is None:
        if hidden_layer >= 1000:
            latent_dim = 100
        elif hidden_layer >= 500:
            latent_dim = 50
        else:
            latent_dim = max(5, hidden_layer // 10)
    
    # Step 4: Train VAE and compute correlations
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.001)
    x_train = x_test = np.array(x)
    
    vae = VAE(
        opt, x_train, x_test, batch_size, 
        original_dim, hidden_layer, latent_dim, epochs
    )
    
    x_test_encoded = np.array(vae.encoder.predict(x_test, batch_size=batch_size))
    correlation = _create_protein_pairs(x_test_encoded, row_names, correlation_type)
    
    # Step 5: Filter and return results
    final_pairs = correlation[correlation.iloc[:, 0] != correlation.iloc[:, 1]]
    final_pairs = final_pairs.sort_values(by=["Score"], ascending=False)
    
    return _pairs_after_cutoff(
                CC_cutoff=CC_cutoff,
                correlation=final_pairs, 
                interaction_count=interaction_count
            )

def main():
    """
    Main function for preprocessing data, training VAE, and saving results.
    
    This function loads data from file, calls cook() to process it,
    and saves the results.
    """
    args = argument_parser()
    
    # Load raw data from file
    x, row_names = _load_data(args.input_file, args.data_type)
    
    # Convert to DataFrame for consistency with cook()
    df = pd.DataFrame(x, index=row_names)
    
    # Process using cook() - handles all preprocessing, VAE training, and correlation
    final_pairs = cook(
        data=df,
        log2_normalization=True,
        hidden_layer=args.hidden_layer,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        interaction_count=args.interaction_count,
        correlation_type=args.correlation_type,
        CC_cutoff=args.CC_cutoff,
    )
    
    # Round and save results
    final_pairs.Score = final_pairs.Score.astype(float).round(5)
    logging.warning(
        " If it is not the desired cut-off, please check again the value assigned to the related parameter (-n or interaction_count | -c or CC_cutoff)."
    )
    logging.info(" Saving the file with the interactions in the chosen directory ...")
    np.savetxt(args.output_file, final_pairs, fmt="%s")
    logging.info(
        f" Congratulations! A file is waiting for you here: {args.output_file}"
    )


if __name__ == "__main__":
    main()
