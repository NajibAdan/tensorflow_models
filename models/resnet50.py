'''
My implementation of ResNet50
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import tensorflow as tf 
from tensorflow import keras 
from keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, BatchNormalization

class CustomConv(Layer):
    '''
    Custom Conv Layer that batch normalizes before passing it to the activation function
    '''
    def __init__(self,filters,kernel_size,strides=1):
        super(CustomConv,self).__init__()
        self.conv = Conv2D(filters,kernel_size,strides=strides,padding='same')
        self.batch_norm = BatchNormalization()
    def call(self,x,training=True):
        x_out = self.conv(x)
        x_out = self.batch_norm(x_out,training)
        x_out = tf.nn.relu(x_out)
        return x_out 

class ResNetBlock(Layer):
    '''
    Residual Layer with Identity Shortcut
    '''
    def __init__(self,input_size,output_size,strides=1):
        super(ResNetBlock,self).__init__(name="ResNetBlock")
        self.conv_1 = CustomConv(input_size,(3,3),strides)
        self.conv_2 = CustomConv(input_size,(3,3),1)
        self.conv_3 = CustomConv(output_size,(1,1),1)
        # Shortcut 
        self.flag = (strides!=1)
        if self.flag:
            self.conv_4 = CustomConv(input_size,(1,1),strides)
    def call(self,x,training=None):
        x_out = self.conv_1(x)
        x_out = self.conv_2(x_out)
        x_out = self.conv_3(x_out)
        ## Identity Shortcut
        if self.flag:
            mapped_x = self.conv_4(x)
        else:
            mapped_x = x
        x_out = mapped_x + x_out
        x_out = tf.nn.relu(x_out)
        return x_out
    

class ResNet50(keras.Model):
    '''
    My implementation of the ResNet50
    '''
    def __init__(self,num_classes):
        super(ResNet50,self).__init__(name="ResNet50")
        self.conv_1 = CustomConv(64,(7,7),strides=2)
        self.max_pool = MaxPool2D((3,3),2)
        self.net = []
        blocks_per_chunk = [3,4,6,3]
        channels = [(64,256),(128,512),(256,1024),(512,2048)]
        for idx,chunk in enumerate(blocks_per_chunk):
            for block in range(chunk):
                input_size, output_size = channels[idx]
                if idx == 0:
                    self.net.append(ResNetBlock(input_size,output_size))
                else:
                    self.net.append(ResNetBlock(input_size,output_size,strides=2))

        ## Classifier
        self.avg_pool = GlobalAveragePooling2D()
        self.fc_1 = Dense(512,activation='relu')
        self.dropout = Dropout(0.5)
        self.fc_2 = Dense(num_classes,activation='softmax')
    def call(self, x):
        x_out = self.max_pool(self.conv_1(x))
        for layer in self.net:
            x_out = layer(x_out)
        x_out = self.dropout(self.fc_1(self.avg_pool(x_out)))
        x_out = self.fc_2(x_out)
        return x_out
