import tensorflow as tf 
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.optimizers import Adam 
from keras.losses import CategoricalCrossentropy
from keras.regularizers import L2
from keras.initializers import RandomNormal

class VGG16(tf.keras.Model):
    '''
    My implementation of VGG16
    '''
    def __init__(self,input_shape=(224,224),reg_param=0.0005,dropout=0.5,num_classes=100):
        '''
        Initializes the model and layers
        '''
        super(VGG16,self).__init__()
        ## First Conv block
        self.conv_1 = Conv2D(3,64,1,padding='same',activation='relu',kernel_regularizer=L2(reg_param),name=f"ConvBlock1Conv1")
        self.max_pool = MaxPool2D((2,2))
        ## Second Conv Block
        self.conv_2 = Conv2D(3,128,1,padding='same',activation='relu',kernel_regularizer=L2(reg_param),name=f"ConvBlock2Conv1")
        ## Third Conv Block
        self.conv_3 = Conv2D(3,256,1,padding='same',activation='relu',kernel_regularizer=L2(reg_param),name=f"ConvBlock3Conv1")
        self.conv_4 = Conv2D(3,256,1,padding='same',activation='relu',kernel_regularizer=L2(reg_param),name=f"ConvBlock3Conv2")
        ## Fourth Conv Block
        self.conv_5 = Conv2D(3,512,1,padding='same',activation='relu',kernel_regularizer=L2(reg_param),name=f"ConvBlock4Conv1")
        self.conv_6 = Conv2D(3,512,1,padding='same',activation='relu',kernel_regularizer=L2(reg_param),name=f"ConvBlock4Conv2")
        ## Fifth Conv Block
        self.conv_7 = Conv2D(3,512,1,padding='same',activation='relu',kernel_regularizer=L2(reg_param),name=f"ConvBlock5Conv1")
        self.conv_8 = Conv2D(3,512,1,padding='same',activation='relu',kernel_regularizer=L2(reg_param),name=f"ConvBlock5Conv2")
        ## Fully Connected Layers
        self.flatten = Flatten()
        self.fc_1 = Dense(4096,activation='relu',kernel_regularizer=L2(reg_param),kernel_initializer=RandomNormal(0.0,0.001),name='FC1')
        self.fc_2 = Dense(4096,activation='relu',kernel_regularizer=L2(reg_param),kernel_initializer=RandomNormal(0.0,0.001),name='FC2')
        self.output_layer = Dense(num_classes,activation='sigmoid',kernel_regularizer=L2(reg_param),kernel_initializer=RandomNormal(0.0,0.001),name='OutputLayer')

    def __call__(self,x,training=True):
        
        ## First Conv Block
        x = self.max_pool(self.conv_1(x))
        ## Second Conv Block
        x = self.max_pool(self.conv_2(x))
        ## Third Conv Block
        x = self.conv_3(x)
        x = self.max_pool(self.conv_4(x))
        ## Fourth Conv Block
        x = self.conv_5(x)
        x = self.max_pool(self.conv_6(x))
        ## Fifth Conv Block
        x = self.conv_7(x)
        x = self.max_pool(self.conv_8(x))
        ## Classifier
        flatten = self.flatten(x)
        x = self.fc_1(flatten)
        output = self.output(x)
        return output