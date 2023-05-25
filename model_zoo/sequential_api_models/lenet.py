import tensorflow as tf
from tensorflow.keras import layers, models

framework = 'tensorflow'
main_method = 'MyModel'
input_shape = 'input_shape'
output_classes = 'classes'

# define lenet model

def MyModel(input_shape=(224,224,3),classes=3):

	model = models.Sequential()
	model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=input_shape))
	model.add(layers.AveragePooling2D(2))
	model.add(layers.Activation('sigmoid'))
	model.add(layers.Conv2D(16, 5, activation='tanh'))
	model.add(layers.AveragePooling2D(2))
	model.add(layers.Activation('sigmoid'))
	model.add(layers.Conv2D(120, 5, activation='tanh'))
	model.add(layers.Flatten())
	model.add(layers.Dense(84, activation='tanh'))
	model.add(layers.Dense(classes, activation='softmax'))
	
	return model