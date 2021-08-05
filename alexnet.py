import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Dropout, Flatten

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        #Conv Layers
        self.conv1 = Conv2D(96,(11,11),strides=4,padding='valid',
                            activation='relu',name='conv_96')
        self.conv2 = Conv2D(256,(5,5), strides=1,padding='same',
                            activation='relu',name= 'conv_256')
        self.conv3 = Conv2D(384,(3,3), strides=1, padding='same', 
                            activation='relu',name= 'conv_384')
        self.conv4 = Conv2D(384,(3,3), strides=1, padding='same', 
                            activation='relu',name= 'conv_384')
        self.conv5 = Conv2D(256, (3,3), strides=1, padding='same', 
                           activation='relu', name= 'conv_256_3')
        
        #Pooling layers
        self.pool2 = MaxPool2D(strides=2,padding='valid')
        self.pool3 = MaxPool2D(pool_size=(3,3),strides=2, 
                               padding='valid')
        
        #Batchnormalization layers
        self.batch1 = BatchNormalization()
        self.batch2 = BatchNormalization()
        self.batch3 = BatchNormalization()
        self.batch4 = BatchNormalization()
        self.batch5 = BatchNormalization()
        self.batch_dense = BatchNormalization()
        self.batch_dense2 = BatchNormalization()
        
        #Dense layers
        self.fc1 = Dense(4096,activation='relu',input_shape=(32,32,3,))
        self.fc2 = Dense(4096,activation='relu')
        
        self.flatten = Flatten()
        self.drop = Dropout(0.4)
        self.ou = Dense(4,activation='softmax')
        
    def call(self,x):
        
        x = self.conv1(x)
        x = self.pool2(x)
        x = self.batch1(x)
        
        x = self.conv2(x)
        x = self.pool3(x)
        x = self.batch2(x)
        
        x = self.conv3(x)
        x = self.batch3(x)
        
        x = self.conv4(x)
        x = self.batch4(x)
        
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.batch5(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.batch_dense(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.batch_dense2(x)
        x = self.ou(x)
        
        return x
