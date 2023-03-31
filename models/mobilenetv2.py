import tensorflow as tf 
from tensorflow import keras 
from keras.layers import Layer,Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, BatchNormalization, DepthwiseConv2D, AvgPool2D
class CustomConv(Layer):
    def __init__(self,filters,kernel_size,strides,activation='relu6'):
        super(CustomConv,self).__init__()
        self.conv = Conv2D(filters,kernel_size,strides=1,padding='same')
        self.batch_norm = BatchNormalization()
        self.activation = activation
    def call(self,x,training=True):
        result = self.batch_norm(self.conv(x),training=training)
        if self.activation == 'relu6':
            result = tf.nn.relu6(result)
        else:
            result = tf.keras.activations.linear(result)
        return result
class CustomDepthWise(Layer):
    def __init__(self,stride=1,padding='same',alpha=1):
        super(CustomDepthWise,self).__init__(name='CustomDepthWise')
        self.depth = DepthwiseConv2D(kernel_size=(3,3),strides=stride,padding=padding,depth_multiplier=alpha)
        self.batch_norm = BatchNormalization()
    def call(self,x,training):
        x_out = self.depth(x)
        x_out = self.batch_norm(x_out,training)
        x_out = tf.nn.relu6(x_out)
        return x_out 
class InvertedResidualBlock(Layer):
    def __init__(self,filters,strides=1,alpha=1):
        super(InvertedResidualBlock,self).__init__(name='InvertedResidualBlock')
        self.conv_1 = CustomConv(filters,(1,1),strides=1)
        self.dw_1 = CustomDepthWise(strides,alpha=alpha)
        self.conv_2 = CustomConv(filters,(1,1),strides=1)
        self.skip = (strides==1)
        if self.skip:
            self.conv_3 = CustomConv(filters,(1,1),strides=1)
    def call(self,x,training=True):
        x_out = self.conv_1(x)
        x_out = self.dw_1(x_out)
        x_out = self.conv_2(x_out)
        if self.skip:
            shortcut = self.conv_3(x)
            x_out = tf.keras.layers.add([shortcut + x])
        return x_out
class MobileNetV2(keras.Model):
    def __init__(self,num_classes=10,alpha=1):
        super(MobileNetV2,self).__init__(name='MobileNetV2')
        self.conv_1 = CustomConv(32,(3,3),2)
        self.net = []
        config = [(1,  16, 1, 1),
                  (6,  24, 2, 2),
                  (6,  32, 3, 2),
                  (6,  64, 4, 2),
                  (6,  96, 3, 1),
                  (6, 160, 3, 2),
                  (6, 320, 1, 1)]
        for expansion, channels, num_blocks, stride in config:
            for block_num in range(num_blocks):
                if block_num>1:
                    self.net.append(InvertedResidualBlock(channels*expansion,1))
                else:
                    self.net.append(InvertedResidualBlock(channels*expansion,stride))
        self.conv_2 = CustomConv(1280,(1,1),1)
        self.avg_pool = GlobalAveragePooling2D()
        self.classifer = Dense(num_classes,activation='softmax')
    def call(self,x,training=True):
        x_out = self.conv_1(x)
        for layer in self.net:
            x_out = layer(x_out)
        x_out = self.avg_pool(self.conv_2(x_out))
        x_out = self.classifer(x_out)
        return x_out
