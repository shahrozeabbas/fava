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

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = tf.keras.Model(inputs, outputs, name="vae_mlp")

        # loss
        reconstruction_loss = K.mean(K.square(inputs - outputs), axis=-1)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(0.9 * (reconstruction_loss) + 0.1 * (kl_loss))
        vae.add_loss(vae_loss)

        vae.compile(optimizer=opt, loss="mean_squared_error", metrics=["accuracy"])
        
        vae.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, x_test),
        )

