import tensorflow as tf 
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, AveragePooling2D
from inception_net import InceptionBlock
from customconv import CustomConv
class AuxClassifier(keras.layers.Layer):
    def __init__(self,num_classes):
        super(AuxClassifier,self).__init__()
        self.aux_avg_pool = AveragePooling2D((5,5),3,padding='same')
        self.aux_conv = Conv2D(128,(1,1),strides=1,activation='relu',padding='same')
        self.aux_fc = Dense(1024,activation='relu')
        self.aux_drop = Dropout(0.7)
        self.aux_out = Dense(num_classes,activation='softmax')
    def call(self,x):
        x_1 = self.aux_conv(self.aux_avg_pool(x))
        x_1 = self.aux_fc(x_1)
        result = self.aux_out(self.aux_drop(x_1))
        return result
class GoogLeNet(keras.Model):
    def __init__(self,num_classes):
        super(GoogLeNet,self).__init__()
        self.conv_1 = CustomConv(64,(7,7),strides=2,activation='relu')
        self.max_pool = MaxPool2D((3,3),strides=2,padding='same')
        ## Mini Inception Block
        self.conv_2 = CustomConv(64,(1,1),strides=1,activation='relu')
        self.conv_3 = CustomConv(192,(3,3),strides=1,activation='relu')
        self.inception_1 = InceptionBlock(0)
        self.inception_2 = InceptionBlock(1)
        self.inception_3 = InceptionBlock(2)
        self.inception_4 = InceptionBlock(3)
        self.inception_5 = InceptionBlock(4)
        self.inception_6 = InceptionBlock(5)
        self.inception_7 = InceptionBlock(6)
        self.inception_8 = InceptionBlock(7)
        self.inception_9 = InceptionBlock(8)
        
        ### Classifier
        self.avg_pool = AveragePooling2D((7,7),1,padding='same')
        self.dropout = Dropout(0.4)
        self.fc_1 = Dense(num_classes,activation='softmax')
        self.flatten = Flatten()
        # self.fc_2 = Dense(num_classes,activation='softmax')
        ### Auxilary Classifier
        self.aux_1 = AuxClassifier(num_classes)
        self.aux_2 = AuxClassifier(num_classes)
        self.aux_3 = AuxClassifier(num_classes)
    def __call__(self,x,training):
        x_1 = self.max_pool(self.conv_1(x))
        x_1 = tf.nn.local_response_normalization(x_1)
        ## Mini Inception
        x_2 = self.conv_3(self.conv_2(x_1))
        x_2 = tf.nn.local_response_normalization(x_2)
        ## InceptionBlock
        x_3 = self.inception_1(self.max_pool(x_2))
        x_4 = self.inception_2(x_3)
        x_4 = self.max_pool(x_4)
        x_5 = self.inception_7(self.inception_6(self.inception_5(self.inception_4(self.inception_4(self.inception_3(x_4))))))
        x_5 = self.max_pool(x_5)
        x_6 = self.inception_9(self.inception_8(x_5))
        ## Aux Classifier
        # aux_1 = self.aux_1(x_3)
        
        ## Classifier
        x_7 = self.dropout(self.avg_pool(x_6))
        x_7 = self.flatten(x_7)
        classifier = self.fc_1(x_7)

        # print(aux_1,classifier)
        return classifier