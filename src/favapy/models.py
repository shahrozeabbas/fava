"""Variational Autoencoder (VAE) model for FAVA."""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class VAE(tf.keras.Model):
    """
    Variational Autoencoder model class.

    Parameters
    ----------
    opt : tf.keras.optimizers.Optimizer
        Optimizer for the model.
    x_train : np.ndarray
        Training data.
    x_test : np.ndarray
        Test data.
    batch_size : int
        Batch size for training.
    original_dim : int
        Dimension of the input data.
    hidden_layer : int
        Number of units in the hidden layer.
    latent_dim : int
        Dimension of the latent space.
    epochs : int
        Number of training epochs.
    """

    def __init__(
        self,
        opt,
        x_train,
        x_test,
        batch_size,
        original_dim,
        hidden_layer,
        latent_dim,
        epochs,
    ):
        super(VAE, self).__init__()
        inputs = tf.keras.Input(shape=(original_dim,))
        h = layers.Dense(hidden_layer, activation="relu")(inputs)

        z_mean = layers.Dense(latent_dim)(h)
        z_log_sigma = layers.Dense(latent_dim)(h)

        # Sampling
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(
                shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=0.1
            )
            return z_mean + K.exp(z_log_sigma) * epsilon

        z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

        # Create encoder
        encoder = tf.keras.Model(inputs, [z_mean, z_log_sigma, z], name="encoder")
        self.encoder = encoder
        # Create decoder
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
        x = layers.Dense(hidden_layer, activation="relu")(latent_inputs)

        outputs = layers.Dense(original_dim, activation="sigmoid")(x)
        decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")
        self.decoder = decoder

        # instantiate VAE model with custom loss
        outputs = decoder(encoder(inputs)[2])
        
        # Convert original_dim to a constant tensor for use in Lambda layer
        original_dim_tensor = K.constant(original_dim, dtype='float32')
        
        # Define custom loss function inside Lambda layer
        def vae_loss_fn(args):
            inputs_t, outputs_t, z_mean_t, z_log_sigma_t, orig_dim = args
            # Reconstruction loss
            recon_loss = K.mean(K.square(inputs_t - outputs_t), axis=-1) * orig_dim
            # KL divergence loss
            kl = K.sum(1 + z_log_sigma_t - K.square(z_mean_t) - K.exp(z_log_sigma_t), axis=-1) * -0.5
            # Combined: 90% reconstruction + 10% KL
            return K.mean(0.9 * recon_loss + 0.1 * kl)
        
        # Wrap loss computation in Lambda layer
        vae_loss = layers.Lambda(
            vae_loss_fn,
            output_shape=(1,),
            name='vae_loss'
        )([inputs, outputs, z_mean, z_log_sigma, original_dim_tensor])
        
        # Create model and add custom loss
        vae = tf.keras.Model(inputs, outputs, name="vae_mlp")
        vae.add_loss(vae_loss)
        
        # Compile without specifying loss (custom loss already added)
        vae.compile(optimizer=opt, metrics=["mse"])
        
        vae.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, x_test),
        )

