from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D
from tensorflow.keras import Sequential


def MyModel(input_shape=(48,48,3),classes=3):

    multi_disease_model = Sequential()
    multi_disease_model.add(Conv2D(3,3,input_shape=input_shape))
    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(512))
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(classes, activation = 'sigmoid'))
    return multi_disease_model