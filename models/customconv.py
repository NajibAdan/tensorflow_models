from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization
from keras.layers import Layer
class CustomConv(Layer):
    def __init__(self,filters,kernel_size,strides,activation):
        super(CustomConv,self).__init__()
        self.conv = Conv2D(filters,kernel_size,strides=1,activation='relu',padding='same')
        self.batch_norm = BatchNormalization()
    def call(self,x,training=True):
        result = self.batch_norm(self.conv(x),training=training)
        return result