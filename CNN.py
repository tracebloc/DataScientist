from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras import Sequential


def MyModel(input_shape=(224,224,15),classes=2):

    base_mobilenet_model = MobileNet(input_shape =  input_shape,
                                 include_top = False, weights = None)
    multi_disease_model = Sequential()
    multi_disease_model.add(base_mobilenet_model)
    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(512))
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(classes, activation = 'sigmoid'))
    return multi_disease_model
