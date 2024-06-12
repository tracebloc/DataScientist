import tensorflow as tf
from tensorflow.keras import layers, utils

# Dense Block
framework = "tensorflow"
main_method = "MyModel"
model_type = ""
input_shape = "input_shape"
output_classes = 2
category = "image_classification"


def dense_block(input_tensor, k, block_reps):
    """
    tensor: input tensor from the previous layers
    k: growth rate
    block_reps: Number of times the block is repeated
    Return the concatenated tensors
    """
    for _ in range(block_reps):
        x = layers.BatchNormalization()(input_tensor)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=4 * k, kernel_size=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=k, kernel_size=3, padding="same")(x)

        output_tensor = layers.Concatenate()([input_tensor, x])

        return output_tensor


# Transition Layers


def transition_layers(input_tensor, theta=0.5):
    """
    input_tensor: tensor from the previous dense block
    theta: compression factor, to be multiplied to the output feature maps...
    of the previous dense block.
    return the output tensor
    """

    filters = input_tensor.shape[-1] * theta

    x = layers.BatchNormalization()(input_tensor)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters, kernel_size=1)(x)
    output_tensor = layers.AveragePooling2D(pool_size=2, strides=2)(x)

    return output_tensor


# DenseNet-169 - final model to return


def MyModel(input_shape=(224, 224, 3), classes=output_classes):
    k = 32  # growth rate

    input = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(input)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=2 * k, kernel_size=7, strides=2)(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = dense_block(x, 32, 6)
    x = transition_layers(x)

    x = dense_block(x, 32, 12)
    x = transition_layers(x)

    x = dense_block(x, 32, 32)
    x = transition_layers(x)

    x = dense_block(x, 32, 32)

    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(classes, activation="softmax")(x)

    model_densenet = tf.keras.Model(input, output)

    return model_densenet
