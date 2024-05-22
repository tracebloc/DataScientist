from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import math

framework = "tensorflow"
main_method = "MyModel"
input_shape = "input_shape"
output_classes = 2
model_type = ""
category = "image_classification"



def mbconv_block(
    input, filters_in, filters_out, kernel_size=3, strides=1, exp_ratio=6, se_ratio=0.25
):
    """
    Arguments:
    -------
    input: input tensor
    filters_in: input filters
    filters_out: output filters
    kernel_size: the size/dimension of convolution filters
    strides: integer, the stride of convolution. If strides=2, padding in depthwise conv is 'valid'.
    exp_ratio: expansion ration, an integer for scaling the input filters/filters_in
    se_ratio: a float between 0 and 1 for squeezing the input filters
    -------
    """

    # Expansion layer: exp_ratio is integer >=1

    filters = filters_in * exp_ratio
    if exp_ratio != 1:
        x = layers.Conv2D(filters, kernel_size=1, padding="same")(input)
        x = layers.BatchNormalization()(x)
        x = keras.activations.swish(x)

    else:
        x = input

    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=strides, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.swish(x)

    # Squeeze and excitation
    if se_ratio > 0 and se_ratio <= 1:
        filters_se = max(
            1, int(filters_in * se_ratio)
        )  # max with 1 to make sure filters are not less than 1
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Reshape((1, 1, filters))(se)
        se = layers.Conv2D(
            filters_se, kernel_size=1, padding="same", activation="swish"
        )(se)
        se = layers.Conv2D(
            filters, kernel_size=1, padding="same", activation="sigmoid"
        )(se)
        x = layers.multiply([x, se])

    x = layers.Conv2D(filters_out, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Add identity shortcut if strides=2 and in & filters are same
    if strides == 1 and filters_in == filters_out:
        x = layers.add([x, input])

    return x


def scale_number_of_blocks(block_repeats, depth_coefficient=1):
    """Scale and round the number of block repeatitions(scale depth in other words)"""

    scaled_blocks = int(math.ceil(block_repeats * depth_coefficient))

    return scaled_blocks


def scale_width(filters, width_coefficient=1, depth_divisor=8):
    filters *= width_coefficient
    new_filters = (filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)

    # make sure that scaled filters down does not go down by more than 10%
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor

    return int(new_filters)


# final Model


def MyModel(input_shape=(224, 224, 3), classes=output_classes):
    # Setting some hyperparameters for EfficientNet-B0

    input_shape = input_shape

    # If using the scaling utility functions, use the following hyperparamaters for EfficientNet-B0
    # depth_coefficient = 1.0
    # width_coefficient = 1.0
    # depth_divisor = 8

    # The stem of network
    input = layers.Input(input_shape)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = keras.activations.swish(x)

    # MBConv blocks
    # Block 1: input filters=32, output filters=16, kernel size=3, block repeats=1
    x = mbconv_block(
        x, filters_in=32, filters_out=16, kernel_size=3, strides=1, exp_ratio=1
    )

    # Block 2: input filters=16, output filters=24, kernel size=3, strides=2, block repeats=2
    # the first block of every stage has stride of 1
    x = mbconv_block(
        x, filters_in=16, filters_out=24, kernel_size=3, strides=1, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=16, filters_out=24, kernel_size=3, strides=2, exp_ratio=6
    )

    # Block 3: input filters=24, output filters=40, kernel size=5, strides=2, block repeats=2
    x = mbconv_block(
        x, filters_in=24, filters_out=40, kernel_size=5, strides=1, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=24, filters_out=40, kernel_size=5, strides=2, exp_ratio=6
    )

    # Block 4: input filters=40, output filters=80, kernel size=3, strides=2, block repeats=3
    x = mbconv_block(
        x, filters_in=40, filters_out=80, kernel_size=3, strides=1, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6
    )

    # Block 5: input filters=80, output filters=112, kernel size=5, strides=1, block repeats=3
    x = mbconv_block(
        x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6
    )

    # Block 6: input filters=112, output filters=192, kernel size=5, strides=2, block repeats=4
    x = mbconv_block(
        x, filters_in=112, filters_out=192, kernel_size=5, strides=1, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6
    )
    x = mbconv_block(
        x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6
    )

    # Block 7: input filters=192, output filters=320, kernel size=3, strides = 1, block repeats=1
    x = mbconv_block(
        x, filters_in=192, filters_out=320, kernel_size=3, strides=1, exp_ratio=6
    )

    # Classification head
    x = layers.Conv2D(filters=1280, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(units=classes, activation="softmax")(x)

    model_efficientnet = keras.Model(inputs=input, outputs=output)

    return model_efficientnet
