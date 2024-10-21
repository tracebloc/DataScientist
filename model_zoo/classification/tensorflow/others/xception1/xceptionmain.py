import tensorflow as tf
from tensorflow.keras import layers, utils
from xception_methods import *

framework = 'tensorflow'
main_method = ''
input_shape = 'inp'
output_classes = 3
model_type = ""
category = "image_classification"


# define final model

inp=(224, 224, 3)

input = layers.Input(shape=inp)

x = entry_flow(input)

# middle flow is repeated 8 times
for _ in range(8):
    x = middle_flow(x)

output = exit_flow(x, output_classes)
model_xception = tf.keras.Model(input, output)