import tensorflow as tf
from tensorflow.keras import layers, utils

framework = "tensorflow"
main_method = "MyModel"
input_shape = 224
batch_size = 16
output_classes = 3
category = "image_classification"


def MyModel(input_shape=(224, 224, 3), classes=output_classes):
    model = tf.keras.models.Sequential(
        [
            layers.Conv2D(
                filters=96,
                kernel_size=(11, 11),
                strides=4,
                activation="relu",
                input_shape=input_shape,
            ),
            layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same"),
            layers.Conv2D(
                filters=256, kernel_size=(5, 5), padding="same", activation="relu"
            ),
            layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            layers.Conv2D(
                filters=384, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=384, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=256, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            layers.Flatten(),
            # Fully connected layers
            layers.Dense(units=4096, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(units=4096, activation="relu"),
            layers.Dropout(0.5),
            # Units in last layer are 1000 per imagenet dataset
            layers.Dense(units=classes, activation="softmax"),
        ]
    )

    return model
