import tensorflow as tf

class IdentityBlock(tf.keras.Model):
    def __init__(self,f,filters,stage,block):
        """
    Implementation of the identity block
    
    Arguments:
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
        super(IdentityBlock, self).__init__(name='')
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        #Retrieve filters
        F1,F2,F3 = filters
        
        self.conv1 = tf.keras.layers.Conv2D(filters=F1, kernel_size = (1,1),
                                            strides=(1,1),padding='valid',
                                           name=conv_name_base+'2a')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=3,name=bn_name_base+'2a')
        
        
        self.conv2 = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f,f),strides=(1,1),
                                            padding='same', name=conv_name_base+'2b')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=3,name=bn_name_base+'2b')
        
        
        self.conv3 = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1),
                                           padding='valid',name=conv_name_base+'2c')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2c')
        
        
        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()
        
    def call(self,input_tensor):
        #First component of main path
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)
        
        #second component of main path
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        
        
        #third component of main path
        x = self.conv3(x)
        x = self.bn3(x)
        
        #shortcut path Final step
        x = self.add([x, input_tensor])
        x = self.act(x)
        
        return x

class ConvolutionBlock(tf.keras.Model):
    def __init__(self,f,filters,stage,block,s =2):
        """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
        super(ConvolutionBlock, self).__init__(name='')
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        #Retrieve filters
        F1,F2,F3 = filters
        
        self.conv1 = tf.keras.layers.Conv2D(filters=F1, kernel_size = (1,1),
                                            strides=(s,s),
                                            name=conv_name_base+'2a')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=3,name=bn_name_base+'2a')
        
        
        self.conv2 = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f,f),strides=(1,1),
                                            padding='same', name=conv_name_base+'2b')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=3,name=bn_name_base+'2b')
        
        
        self.conv3 = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1),
                                           padding='valid',name=conv_name_base+'2c')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2c')
        
        self.conv_short = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1,1),
                                                 strides=(s,s),padding='valid',
                                                 name=conv_name_base+'1')
        self.bns = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'1')
        
        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()
        
    def call(self,input_tensor):
        #First component of main path
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)
        
        #second component of main path
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        #third component of main path
        x = self.conv3(x)
        x = self.bn3(x)
        
        #shortcut path Final step
        input_tensor = self.conv_short(input_tensor)
        input_tensor = self.bns(input_tensor)
        x = self.add([x, input_tensor])
        x = self.act(x)
        
        return x

class MyModel(tf.keras.Model):
    def __init__(self,classes=1):
        super(MyModel, self).__init__()
        
        self.conv = tf.keras.layers.Conv2D(64,7, strides = 2,name ='conv1')
        self.bn = tf.keras.layers.BatchNormalization(axis = 3, name= 'bn_conv1')
        self.act = tf.keras.layers.Activation('relu')
        self.maxpool = tf.keras.layers.MaxPool2D((3,3),strides=(2,2))
        
        self.conblock1 = ConvolutionBlock(f=3, filters=[64, 64, 256],
                                          stage=2, block='a', s=1)
        self.idenblock1 = IdentityBlock(3, [64, 64, 256], stage=2, block='b')
        self.idenblock2 = IdentityBlock(3, [64, 64, 256], stage=2, block='c')
        
        self.conblock2 = ConvolutionBlock(f = 3, filters = [128, 128, 512],
                                          stage = 3, block='a', s = 2)
        self.idenblock3 = IdentityBlock(3, [128, 128, 512], stage=3, block='b')
        self.idenblock4 = IdentityBlock(3, [128, 128, 512], stage=3, block='c')
        self.idenblock5 = IdentityBlock(3, [128, 128, 512], stage=3, block='d')
        
        self.conblock3 = ConvolutionBlock(f = 3, filters = [256, 256, 1024],
                                          stage = 4, block='a', s = 2)
        self.idenblock6 = IdentityBlock(3, [256, 256, 1024], stage=4, block='b')
        self.idenblock7 = IdentityBlock(3, [256, 256, 1024], stage=4, block='c')
        self.idenblock8 = IdentityBlock(3, [256, 256, 1024], stage=4, block='d')
        self.idenblock9 = IdentityBlock(3, [256, 256, 1024], stage=4, block='e')
        self.idenblock10 = IdentityBlock(3, [256, 256, 1024], stage=4, block='f')
        
        self.conblock4 = ConvolutionBlock(f = 3, filters = [512, 512, 2048],
                                          stage = 5, block='a', s = 2)
        self.idenblock11 = IdentityBlock(3, [512, 512, 2048], stage=5, block='b')
        self.idenblock12 = IdentityBlock(3, [512, 512, 2048], stage=5, block='c')
        
        
        
        self.avg_pool = tf.keras.layers.AveragePooling2D((2,2), name="avg_pool")
        self.flat = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(classes, activation='softmax',
                                                name='fc' + str(classes))
        
    def call(self, x):
        
        # Stage 1
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)
        
        # Stage 2
        x = self.conblock1(x)
        x = self.idenblock1(x)
        x = self.idenblock2(x)
        
        # Stage 3
        x = self.conblock2(x)
        x = self.idenblock3(x)
        x = self.idenblock4(x)
        x = self.idenblock5(x)
        
        # Stage 4
        x = self.conblock3(x)
        x = self.idenblock6(x)
        x = self.idenblock7(x)
        x = self.idenblock8(x)
        x = self.idenblock9(x)
        x = self.idenblock10(x)
        
        # Stage 5
        x = self.conblock4(x)
        x = self.idenblock11(x)
        x = self.idenblock12(x)
        
        # AVGPOOL
        x = self.avg_pool(x)
        
        # output layer
        x = self.flat(x)
        x = self.classifier(x)
        return x

