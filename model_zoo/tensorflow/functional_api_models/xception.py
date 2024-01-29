import tensorflow as tf
from tensorflow.keras import layers, utils

framework = 'tensorflow'
main_method = 'MyModel'
input_shape = 'input_shape'
output_classes = 'classes'
model_type = ''

# entry flow block

def entry_flow(input):
	"""
	Entry flow is the first block of Xception architecture
	input: input image of shape (299, 299, 3)
	"""

	x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding='same')(input)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	shortcut = layers.Conv2D(filters=128, kernel_size=(1,1), strides=2, padding='same')(x)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.SeparableConv2D(filters=128, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=128, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

	x = layers.Add()([x, shortcut])

	shortcut = layers.Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='same')(x)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=256, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=256, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

	x = layers.Add()([x, shortcut])

	shortcut = layers.Conv2D(filters=728, kernel_size=(1,1), strides=2, padding='same')(x)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

	output_tensor = layers.Add()([x, shortcut])

	return output_tensor

# middle flow block

def middle_flow(input_tensor):
	"""
	Middle flow of the Xception Architecture
	input_tensor: output tensor of the the entry flow
	Middle flow will be repeated 8 times.
	"""
	shortcut = input_tensor

	x = layers.ReLU()(input_tensor)
	x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)

	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)

	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)

	x = layers.Add()([x, shortcut])

	return x

def exit_flow(input_tensor, classes):
	"""
	Exit flow of Xception architecture
	input_tensor: output tensor of the middle flow
	Fully connected layers before classification layer are optional
	"""

	shortcut = layers.Conv2D(filters=1024, kernel_size=(1,1), strides=2, padding='same')(input_tensor)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.ReLU()(input_tensor)
	x = layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=1024, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
	x = layers.Add()([x, shortcut])

	x = layers.SeparableConv2D(filters=1536, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.SeparableConv2D(filters=2048, kernel_size=(3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.GlobalAveragePooling2D()(x)
	classifier = layers.Dense(classes, activation='softmax')(x)

	return classifier
# define final model

def MyModel(input_shape=(224,224,3),classes=3):
	
	input = layers.Input(shape=input_shape)

	x = entry_flow(input)

	#middle flow is repeated 8 times
	for _ in range(8):
		x = middle_flow(x)

	output = exit_flow(x,classes)
	model_xception = tf.keras.Model(input, output)
	return model_xception