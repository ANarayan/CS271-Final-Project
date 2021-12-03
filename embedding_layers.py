from keras.layers import (
    Dense,
    Conv1D,
)


def Conv1DEmbed(d_input, d_model, kernel_size):
    # for PTB-XL, d_input=12
    return Conv1D(filters=d_model, kernel_size=kernel_size)


def LinearEmbed(d_input, d_model):
    return Dense(d_model)


EMBEDDING_REGISRY = {"conv1d": Conv1DEmbed, "linear": LinearEmbed}
