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
        patch_size=1,
    ):
        """
        input: (B, C, S)
        output: (B, S // 1, d_model)
        """
        super().__init__()
        self.d_input = d_input
        self.reshape = tf.keras.layers.Reshape((d_input * 1000 // patch_size, patch_size))
        self.emb = Dense(d_model)

    def call(self, x):
        b, c, s = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        # assert c == self.d_input, f"Patchsize expected {self.d_input}
        # channels got {c} channels"
        x = self.reshape(x)
        return self.emb(x)
    
    
class Conv1DEmbed(keras.layers.Layer):
    def __init__(self, d_input, d_model, kernel_size, patch_size):
        """
        input: (B, C, S)
        """
      
        super().__init__()
        self.emb = Conv1D(filters=d_model, kernel_size=kernel_size, strides=2)

    def call(self, x):
        x = tf.transpose(x, perm=[0,2,1])
        x = self.emb(x)
        print(x.shape)
        return x



EMBEDDING_REGISRY = {"conv1d": Conv1DEmbed, "linear": LinearEmbed}
