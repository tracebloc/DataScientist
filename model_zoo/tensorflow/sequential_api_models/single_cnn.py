import tensorflow as tf
from tensorflow.keras import layers, models

framework = 'tensorflow'
main_method = 'MyModel'
input_shape = 'input_shape'
output_classes = 'classes'
model_type = ''

# define lenet model

def MyModel(input_shape=(224,224,3),classes=2):

    model = models.Sequential()
    # layer conv 1
    model.add(layers.Conv2D(264, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    # Flatten the feature maps to serve dense
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation='softmax'))

    return model