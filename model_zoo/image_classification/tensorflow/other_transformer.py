import subprocess

subprocess.call(["pip", "install", "vit-keras"])
subprocess.call(["pip", "install", "tensorflow_addons"])

from vit_keras import vit
import tensorflow as tf

ramework = "tensorflow"
main_method = "MyModel"
input_shape = 224
batch_size = 16
output_classes = 3
category = "image_classification"


def MyModel(input_shape=(224, 224, 3), output_classes=output_classes):
    vit_model = vit.vit_b16(
        image_size=input_shape[0],
        activation="softmax",
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=output_classes,
    )
    model = tf.keras.Sequential(
        [
            vit_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(11, activation=tf.keras.activations.gelu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(output_classes, "softmax"),
        ],
        name="vision_transformer",
    )

    return model
