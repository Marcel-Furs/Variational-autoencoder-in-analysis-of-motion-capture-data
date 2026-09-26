import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Hiperparametry architektury
LATENT_DIM = 4 
WIDTH_FIRST = 128
#LATENT_DIM = 16
#WIDTH_FIRST = 256
BETA_KL = 0.01
DROPOUT = 0.2


class Sampling(layers.Layer):
    """Reparameterization trick: z = μ + ε·σ, ε ~ N(0, I)."""
    def call(self, inputs):
        z_mean, z_logvar = inputs
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

def build_encoder(input_dim):
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(WIDTH_FIRST, activation="relu")(encoder_inputs)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(WIDTH_FIRST // 2, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(input_dim):
    latent_input = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(WIDTH_FIRST // 2, activation="relu")(latent_input)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(WIDTH_FIRST, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    decoder_output = layers.Dense(input_dim, activation="linear")(x)
    return keras.Model(latent_input, decoder_output, name="decoder")

# KLASA

class VAE(keras.Model):
    """β-VAE: loss = MSE + β·KL."""
    def __init__(self, encoder, decoder, beta=BETA_KL, **kw):
        super().__init__(**kw)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_tracker = keras.metrics.Mean(name="mse")
        self.kl_tracker = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_tracker, self.kl_tracker]

    def call(self, x):
        _, _, z = self.encoder(x)
        return self.decoder(z)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mu, logvar, z = self.encoder(x, training=True)
            logvar = tf.clip_by_value(logvar, -10.0, 10.0)
            recon = self.decoder(z, training=True)
            mse = tf.reduce_mean(tf.square(x - recon), axis=1)
            kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            loss = tf.reduce_mean(mse + self.beta * kl)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(tf.reduce_mean(mse))
        self.kl_tracker.update_state(tf.reduce_mean(kl))
        return {
            "loss": self.loss_tracker.result(),
            "mse": self.mse_tracker.result(),
            "kl": self.kl_tracker.result(),
        }

    def test_step(self, x):
        mu, logvar, z = self.encoder(x, training=False)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        recon = self.decoder(z, training=False)
        mse = tf.reduce_mean(tf.square(x - recon), axis=1)
        kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        loss = tf.reduce_mean(mse + self.beta * kl)
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(tf.reduce_mean(mse))
        self.kl_tracker.update_state(tf.reduce_mean(kl))
        return {
            "loss": self.loss_tracker.result(),
            "mse": self.mse_tracker.result(),
            "kl": self.kl_tracker.result(),
        }