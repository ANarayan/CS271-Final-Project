import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.layers import LSTM
import tensorflow_addons as tfa


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(d_model),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class MLP(layers.Layer):
    def __init__(self, max_len, d_model, dropout_rate, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)

        self.mlp = keras.Sequential(
            [
                layers.Dense(units=max_len),
                tfa.layers.GELU(),
                layers.Dense(units=d_model),
                layers.Dropout(rate=dropout_rate),
            ]
        )

    def call(self, inputs):
        return self.mlp(inputs)


def LSTMEncoder(d_model, **kwargs):
    return LSTM(units=d_model, **kwargs)


def TransformerEncoder(d_model, num_heads, ff_dim, rate, **kwargs):
    transformer = TransformerBlock(d_model, num_heads, ff_dim, rate)
    return transformer


def MLPEncoder(max_len, d_model, dropout_rate, *args, **kwargs):
    return MLP(max_len, d_model, dropout_rate, *args, **kwargs)


ENCODER_REGISTRY = {"lstm": LSTMEncoder, "transformer": TransformerEncoder}
