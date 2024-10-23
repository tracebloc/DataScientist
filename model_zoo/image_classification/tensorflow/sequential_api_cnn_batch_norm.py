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
    # layer conv 1
    model.add(layers.Conv2D(32, 3, activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    layers.BatchNormalization()
    # layer conv 2
    model.add(layers.Conv2D(64, 3, activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    layers.BatchNormalization()
    # layer conv 3
    model.add(layers.Conv2D(128, 3, activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    layers.BatchNormalization()
    # layer conv 4
    model.add(layers.Conv2D(264, 3, activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    layers.BatchNormalization()
    # layer conv 5
    model.add(layers.Conv2D(1, 1, activation="relu", input_shape=input_shape))
    layers.BatchNormalization()
    # Flatten the feature maps to serve dense
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation="softmax"))

    return model
