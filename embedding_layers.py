import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
)
from einops import rearrange
from tensorflow import keras


def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)


class AddPositionalEncoding(layers.Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal


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
