import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

framework = 'tensorflow'
main_method = 'MyModel'
input_shape = 'input_shape'
output_classes = 'classes'

# Identity Block
def identity_block(input_tensor, filters, strides=1):
	"""
	input tensor
	filters: a tuple of 3 filters to be used in convolutional layers
	strides = 1 for all conv layers
	"""
	f1, f2, f3 = filters

	x = layers.Conv2D(filters=f1, kernel_size=(1,1), strides=strides)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.Conv2D(filters=f2, kernel_size=(3,3), strides=strides, padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.Conv2D(filters=f3, kernel_size=(1,1), strides=strides)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Add()([x, input_tensor])
	output_tensor = layers.ReLU()(x)

	return output_tensor

# Projection Block
def projection_block(input_tensor, filters, strides=2):
	"""
	A projection block is a block that has 1x1 conv layer in the shortcut connection
	1x1 convolution at shortcut connection is used for increasing the input dimension.
	input_tensor: input tensor
	filters: a tuple of 3 filters to be used in 3 conv layers at the main path, the 1x1conv shortcut takes third filter
	strides of 2 at the first conv layers in conv block conv3_1, conv4_1, and conv5_1 for downsampling purpose
	"""

	f1, f2, f3 = filters
	x = layers.Conv2D(filters=f1, kernel_size=(1,1), strides=strides)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.Conv2D(filters=f2, kernel_size=(3,3), strides=1, padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.Conv2D(filters=f3, kernel_size=(1,1), strides=1)(x)
	x = layers.BatchNormalization()(x)

	# 1x1 conv projection shortcut
	shortcut = layers.Conv2D(filters=f3, kernel_size=(1,1), strides=strides)(input_tensor)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.Add()([x, shortcut])
	output_tensor = layers.ReLU()(x)

	return output_tensor

# Final model to return 

def MyModel(input_shape=(224,224,3),classes=3):

	input = layers.Input(shape=input_shape)

	x = layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same')(input)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

	x = projection_block(x, (64, 64, 256))
	x = identity_block(x, (64, 64, 256))
	x = identity_block(x, (64, 64, 256))

	x = projection_block(x, (128, 128, 512))
	x = identity_block(x, (128, 128, 512))
	x = identity_block(x, (128, 128, 512))
	x = identity_block(x, (128, 128, 512))

	x = projection_block(x, (256, 256, 1024))
	x = identity_block(x, (256, 256, 1024))
	x = identity_block(x, (256, 256, 1024))
	x = identity_block(x, (256, 256, 1024))
	x = identity_block(x, (256, 256, 1024))
	x = identity_block(x, (256, 256, 1024))

	x = projection_block(x, (512, 512, 2048))
	x = identity_block(x, (512, 512, 2048))
	x = identity_block(x, (512, 512, 2048))

	x = layers.GlobalAvgPool2D()(x)
	x = layers.Dense(classes, activation='softmax')(x)

	model_resnet50 = tf.keras.Model(input, x, name='ResNet-50')

	return model_resnet50