import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Hiperparametry architektury
LATENT_DIM   = 4
HIDDEN_DIM   = 128     
DROPOUT      = 0.4
BETA_KL      = 0.01


class Sampling(layers.Layer):
    """Reparameterization trick: z = μ + ε·σ, ε ~ N(0, I)."""
    def call(self, inputs):
        z_mean, z_logvar = inputs
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

def build_encoder_lstm(seq_len, n_features):
    """
    Encoder: LSTM przeczyta sekwencję, weźmie ostatni hidden state, 
    z niego dwa Dense → (mu, logvar) → z (Sampling).
    """
    encoder_inputs = keras.Input(shape=(seq_len, n_features))
    x = layers.LSTM(HIDDEN_DIM, return_sequences=False, dropout=DROPOUT)(encoder_inputs)
    z_mean    = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_lstm")

def build_decoder_lstm(seq_len, n_features):
    """
    Decoder: latent → RepeatVector seq_len razy -> LSTM -> TimeDistributed Dense → rekonstrukcja.
    """
    latent_input = keras.Input(shape=(LATENT_DIM,))
    x = layers.RepeatVector(seq_len)(latent_input)
    x = layers.LSTM(HIDDEN_DIM, return_sequences=True, dropout=DROPOUT)(x)
    decoder_output = layers.TimeDistributed(
        layers.Dense(n_features, activation="linear")
    )(x)
    return keras.Model(latent_input, decoder_output, name="decoder_lstm")


# KLASA

class LSTM_VAE(keras.Model):
    """β-VAE dla sequence input. Loss = MSE + β·KL."""
    def __init__(self, encoder, decoder, beta=BETA_KL, **kw):
        super().__init__(**kw)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_tracker  = keras.metrics.Mean(name="mse")
        self.kl_tracker   = keras.metrics.Mean(name="kl")

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
            mse = tf.reduce_mean(tf.square(x - recon), axis=[1, 2])
            kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            loss = tf.reduce_mean(mse + self.beta * kl)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(tf.reduce_mean(mse))
        self.kl_tracker.update_state(tf.reduce_mean(kl))
        return {"loss": self.loss_tracker.result(),
                "mse":  self.mse_tracker.result(),
                "kl":   self.kl_tracker.result()}

    def test_step(self, x):
        mu, logvar, z = self.encoder(x, training=False)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        recon = self.decoder(z, training=False)
        mse = tf.reduce_mean(tf.square(x - recon), axis=[1, 2])  # 3D
        kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        loss = tf.reduce_mean(mse + self.beta * kl)
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(tf.reduce_mean(mse))
        self.kl_tracker.update_state(tf.reduce_mean(kl))
        return {"loss": self.loss_tracker.result(),
                "mse":  self.mse_tracker.result(),
                "kl":   self.kl_tracker.result()}


def evaluate_lstm_vae(X_train_raw, X_val_raw, X_test_normal_raw, X_test_anomaly_raw, seed,
                     batch_size=32, lr=1e-3, epochs_max=1000, patience=30):
    """
    Input shapes: (n, seq_len, n_features) — np. (n, 100, 78).
    Standaryzacja per-feature na flattened.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    n_train, T, F = X_train_raw.shape

    # standaryzacja: fit na flat train, transform pozostałych, reshape z powrotem
    scaler = StandardScaler()
    X_train  = scaler.fit_transform(X_train_raw.reshape(n_train, -1)).reshape(n_train, T, F)
    X_val    = scaler.transform(X_val_raw.reshape(len(X_val_raw), -1)).reshape(-1, T, F)
    X_tn     = scaler.transform(X_test_normal_raw.reshape(len(X_test_normal_raw), -1)).reshape(-1, T, F)
    X_ta     = scaler.transform(X_test_anomaly_raw.reshape(len(X_test_anomaly_raw), -1)).reshape(-1, T, F)

    # build model
    encoder = build_encoder_lstm(seq_len=T, n_features=F)
    decoder = build_decoder_lstm(seq_len=T, n_features=F)
    vae = LSTM_VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

    # train z EarlyStopping
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True)
    vae.fit(X_train, X_train, validation_data=(X_val, X_val),
            batch_size=batch_size, epochs=epochs_max,
            callbacks=[es], verbose=0)

    # scoring: MSE per sample (axis=[1, 2] bo 3D)
    def compute_mse_3d(X):
        mu, logvar, z = encoder.predict(X, verbose=0)
        recon = decoder.predict(z, verbose=0)
        return np.mean((X - recon)**2, axis=(1, 2))

    score_normal  = compute_mse_3d(X_tn)
    score_anomaly = compute_mse_3d(X_ta)

    y_true  = np.concatenate([np.zeros(len(score_normal)), np.ones(len(score_anomaly))])
    y_score = np.concatenate([score_normal, score_anomaly])
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap":  float(average_precision_score(y_true, y_score)),
        "n_train": len(X_train), "n_val": len(X_val),
        "n_test_normal": len(X_tn), "n_test_anomaly": len(X_ta),
    }