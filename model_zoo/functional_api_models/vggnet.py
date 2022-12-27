import tensorflow as tf
from tensorflow.keras import layers, utils


main_method = 'MyModel'
input_shape = 'input_shape'
output_classes = 'classes'

def MyModel(input_shape=(224,224,3),classes=3):
	## VGG-16

	input = tf.keras.Input(shape=input_shape)

	# Convolutional Block 1 
	"""
	2 Convolutional layers: 64 filters each, same padding, relu activation
	1 Maxpooling layer: pool size of 2, stride of 2
	"""
	x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_conv1')(input)
	x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_conv2')(x)
	block1_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool1')(x)

	# Convolutional Block 2
	"""
	2 Convolutional layers: 128 filters each, same padding, relu activation
	1 Maxpooling layer: pool size of 2, stride of 2
	"""
	x = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_conv1')(block1_end)
	x = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_conv2')(x)
	block2_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool2')(x)


	# Convolutional Block 3
	"""
	3 Convolutional layers: 256 filters each, same padding, relu activation
	1 Maxpooling layer: pool size of 2, stride of 2
	"""
	x = layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv1')(block2_end)
	x = layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv2')(x)
	x = layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv3')(x)
	block3_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool3')(x)


	# Convolutional Block 4
	"""
	3 Convolutional layers: 512 filters each, same padding, relu activation
	1 Maxpooling layer: pool size of 2, stride of 2
	"""
	x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv1')(block3_end)
	x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv2')(x)
	x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv3')(x)
	block4_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool4')(x)


	# Convolutional Block 5
	"""
	3 Convolutional layers: 512 filters each, same padding, relu activation
	1 Maxpooling layer: pool size of 2, stride of 2
	"""
	x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv1')(block4_end)
	x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv2')(x)
	x = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv3')(x)
	block5_end = layers.MaxPooling2D(pool_size=2, strides=2, name='pool5')(x)

	# Flattening layer

	x = layers.Flatten(name='flatten')(block5_end)

	# Fully connected layers
	"""
	2 fully connected layers of 4096 each
	A dropout layer with 0.5 probability after above each fully connected layer
	1 classification layer of 1000 units per imagenet with softmax activation
	"""

	x = layers.Dense(units=4096, activation='relu', name='fc1')(x)
	x = layers.Dropout(0.5, name='drop1')(x)
	x = layers.Dense(units=4096, activation='relu',name='fc2')(x)
	x = layers.Dropout(0.5, name='drop2')(x)
	x = layers.Dense(units=classes, activation='softmax',name='classification')(x)

	# Create a model

	vgg_16_model = tf.keras.Model(inputs=input, outputs=x)

	return vgg_16_model
