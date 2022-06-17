import tensorflow as tf
from tensorflow.keras import layers, models

# define lenet model

def MyModel(input_shape=(224,224,3),classes=3):

    model = models.Sequential()
    # layer conv 1
    model.add(layers.Conv2D(264, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    # Flatten the feature maps to serve dense
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation='softmax'))

    return model