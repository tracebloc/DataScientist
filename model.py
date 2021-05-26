import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',name='conv_layer')
        self.pool = tf.keras.layers.MaxPool2D(strides=2,padding='same',name ='pooling_layer')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(256,activation='relu',name='fc1')
        self.d2 = tf.keras.layers.Dense(128,activation='relu',name='fc2')
        self.d3 = tf.keras.layers.Dense(1,activation='sigmoid')
        
    def call(self,x):
        
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x
 
