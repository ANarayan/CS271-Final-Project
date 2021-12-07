import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
)
from einops import rearrange
from tensorflow import keras



class LinearEmbed(keras.layers.Layer):
    def __init__(
        self,
        d_input,
        d_model,
        kernel_size,
    ):
        """
        input: (B, C, S)
        output: (B, S // 1, d_model)
        """
        super().__init__()
        self.d_input = d_input
        self.emb = Dense(d_model)

    def call(self, x):
        b, c, s = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        # assert c == self.d_input, f"Patchsize expected {self.d_input}
        # channels got {c} channels"
        print(tf.shape(x)[0])
        print(tf.shape(x)[1])
        print(tf.shape(x)[2])
        x = tf.reshape(x, [b, c * s // self.d_input, 1])
        return self.emb(x)
    
    
class Conv1DEmbed(keras.layers.Layer):
    def __init__(self, d_input, d_model, kernel_size):
        """
        input: (B, C, S)
        """
      
        self.emb = Conv1D(filters=d_model, kernel_size=kernel_size)

    def call(self, x):
        return rearrange(self.emb(x), "b c s -> b s c")



EMBEDDING_REGISRY = {"conv1d": Conv1DEmbed, "linear": LinearEmbed}
