import tensorflow as tf
import pickle
from data import load_data
from tensorflow import keras
from embedding_layers import Conv1DEmbed, LinearEmbed
from encoder_layers import LSTMEncoder, TransformerEncoder, MLPEncoder
from tensorflow.keras import layers
from data import load_data


EMBEDDING_REGISTRY = {"conv1d": Conv1DEmbed, "linear": LinearEmbed}
ENCODER_REGISTRY = {"lstm": LSTMEncoder, "transformer": TransformerEncoder, "mlp": MLPEncoder}

# EXPERIMENT PARAMETERS
d_input = 12 # number of channels in input
max_len = 1000
d_model = 256
data_dir = "/juice/scr/avanika/unagi/physionet.org/files/ptb-xl/1.0.1/"
embedding_layer = "linear"
encoder_layer = "lstm"
kernel_size=4
num_classes=5
num_heads = 6
ff_dim = 128
rate=0.1


EMBED_PARAMS = {
    'd_input' : d_input,
    'd_model' : d_model,
    'kernel_size': kernel_size
}

ENCODER_PARAMS = {
    'd_input' : d_input,
    'd_model' : d_model,
    'kernel_size': kernel_size,
    'num_heads': num_heads, 
    'ff_dim' : ff_dim , 
    'rate' : rate
}

"""train_data, val_data, test_data = load_data(data_dir)
breakpoint()
pickle.dump(train_data, open("train_data.pkl", "wb"))
pickle.dump(test_data, open("test_data.pkl", "wb"))
pickle.dump(val_data, open("val_data.pkl", "wb"))"""

train_data = pickle.load(open("train_data.pkl", "rb"))
test_data = pickle.load(open("test_data.pkl", "rb"))
val_data = pickle.load(open("val_data.pkl", "rb"))

x_train, y_train = train_data[0], train_data[1]
x_val, y_val =  val_data[0], val_data[1]
x_test, y_test = test_data[0], test_data[1]


inputs = layers.Input(shape=(d_input,max_len, ))
embed_layer = EMBEDDING_REGISTRY[embedding_layer](**EMBED_PARAMS)
encoder_layer = ENCODER_REGISTRY[encoder_layer](**ENCODER_PARAMS)
avg_pool_layer = layers.GlobalAveragePooling1D()
dropout_layer = layers.Dropout(0.1)
prediction_layer = layers.Dense(num_classes, activation="softmax")

x = embed_layer(inputs)
x = encoder_layer(x)
x = avg_pool_layer(x)
x = dropout_layer(x)
outputs = prediction_layer(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val)
)
model.save(f"{embed_layer}_{encoder_layer}")
