import tensorflow as tf
from tensorflow.keras import layers, models

framework = "tensorflow"
main_method = "MyModel"
input_shape = 224
batch_size = 16
output_classes = 3
category = "image_classification"

# define lenet model


def MyModel(input_shape=(224, 224, 3), classes=output_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(6, 5, activation="tanh", input_shape=input_shape))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation("sigmoid"))
    model.add(layers.Conv2D(16, 5, activation="tanh"))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation("sigmoid"))
    model.add(layers.Conv2D(120, 5, activation="tanh"))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation="tanh"))
    model.add(layers.Dense(classes, activation="softmax"))

    return model
