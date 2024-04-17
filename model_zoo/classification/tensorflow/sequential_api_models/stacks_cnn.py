import tensorflow as tf
from tensorflow.keras import layers, models

framework = "tensorflow"
main_method = "MyModel"
input_shape = "input_shape"
output_classes = "classes"
model_type = ""

# define lenet model


def MyModel(input_shape=(224, 224, 3), classes=3):
    model = models.Sequential()
    # layer conv 1
    model.add(layers.Conv2D(32, 3, activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    # layer conv 2
    model.add(layers.Conv2D(64, 3, activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    # layer conv 3
    model.add(layers.Conv2D(128, 3, activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    # layer conv 4 --> reduce channels to 1 using 1x1 kernel
    model.add(layers.Conv2D(1, 1, activation="relu", input_shape=input_shape))
    # Flatten the feature maps to serve dense
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(classes, activation="softmax"))

    return model
