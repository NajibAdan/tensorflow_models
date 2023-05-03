import tensorflow as tf 
from tensorflow import keras 
from customconv import CustomConv
from keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, BatchNormalization, DepthwiseConv2D, AvgPool2D, Reshape
class SqueezeExcite(Layer):
    def __init__(self,r =1):
        super(SqueezeExcite, self).__init__(name='SqueezeExcite')
        self.r = r
    def call(self,x):
        c = x.shape[-1]
        # Squeezes out 
        x_out = GlobalAveragePooling2D()(x)

        x_out = Reshape((x_out.shape[-1],))(x_out)
        # Excites
        x_out = Dense(int(c/self.r), activation='relu')(x_out)
        x_out = Dense(c,activation='sigmoid')(x_out)

        x_out = Reshape((1,1,x_out.shape[-1]))(x_out)

        return x_out * x
