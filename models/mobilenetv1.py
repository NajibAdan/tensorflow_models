import tensorflow as tf 
from tensorflow import keras 
from customconv import CustomConv
from keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, BatchNormalization, DepthwiseConv2D, AvgPool2D
class CustomDepthWise(Layer):
    def __init__(self,stride=1,padding='valid',alpha=1):
        super(CustomDepthWise,self).__init__(name='CustomDepthWise')
        self.depth = DepthwiseConv2D(kernel_size=(3,3),strides=stride,padding=padding,depth_multiplier=alpha)
        self.batch_norm = BatchNormalization()
    def call(self,x,training):
        x_out = self.depth(x)
        x_out = self.batch_norm(x_out,training)
        x_out = tf.nn.relu(x_out)
        return x_out 
class CustomDepthWiseBlock(Layer):
    def __init__(self,filters,strides=1,alpha=1):
        super(CustomDepthWiseBlock,self).__init__(name='CustomDepthWiseBlock')
        self.dw_1 = DepthwiseConv2D(strides,alpha=alpha)
        self.conv_1 = CustomConv(int(filters*alpha),(1,1),1)
    def call(self,x,training=True):
        x_out = self.dw_1(x)
        x_out = self.conv_1(x_out)
        return x_out
class MobileNetV1(keras.Model):
    def __init__(self,num_classes=10,alpha=1):
        super(MobileNetV1,self).__init__(name='MobileNetV1')
        self.conv_1 = CustomConv(32*alpha,(3,3),2)
        self.dw_1 = CustomDepthWiseBlock(64,alpha=alpha)
        self.dw_2 = CustomDepthWiseBlock(128,2,alpha)
        self.dw_3 = CustomDepthWiseBlock(128,alpha=alpha)
        self.dw_4 = CustomDepthWiseBlock(256,2,alpha)
        self.dw_5 = CustomDepthWiseBlock(256,alpha=alpha)
        self.dw_6 = CustomDepthWiseBlock(512,2,alpha)
        self.dw_chunk = [CustomDepthWiseBlock(512,alpha=alpha) for _ in range(5)]
        self.dw_7 = CustomDepthWiseBlock(1024,2,alpha)
        self.dw_8 = CustomDepthWiseBlock(1024,2,alpha)
        self.avg_pool = GlobalAveragePooling2D()
        self.classifer = Dense(num_classes,activation='softmax')
    def call(self,x,training=True):
        x_out = self.conv_1(x)
        x_out = self.dw_3(self.dw_2(self.dw_1(x_out)))
        x_out = self.dw_6(self.dw_5(self.dw_4(x_out)))
        for dw in self.dw_chunk:
            x_out = dw(x_out)
        x_out = self.dw_8(self.dw_7(x_out))
        x_out = self.classifer(self.avg_pool(x_out))
        return x_out