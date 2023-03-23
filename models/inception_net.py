import tensorflow as tf 
from tensorflow import keras
from keras.layers import MaxPool2D
from customconv import CustomConv
NUM_OF_FILTER = [[64,96,128,16,32,32],
               [128,128,192,32,96,64],
               [192,96,208,16,48,64],
               [160,112,224,24,64,64],
               [128,128,256,24,64,64],
               [112,144,288,32,64,64],
               [256,160,320,32,128,128],
               [256,160,320,32,128,128],
               [384,192,384,48,128,128]]
class InceptionBlock(keras.layers.Layer):
    def __init__(self,position):
        super(InceptionBlock,self).__init__()
        ### Branch 1
        self.conv_1 = CustomConv(NUM_OF_FILTER[position][0],(1,1),strides=1,activation='relu')
        ### Branch 2
        self.conv_2 = CustomConv(NUM_OF_FILTER[position][1],(1,1),strides=1,activation='relu')
        self.conv_3 = CustomConv(NUM_OF_FILTER[position][2],(3,3),strides=1,activation='relu')
        ### Branch 3
        self.conv_4 = CustomConv(NUM_OF_FILTER[position][3],(1,1),strides=1,activation='relu')
        self.conv_5 = CustomConv(NUM_OF_FILTER[position][4],(5,5),strides=1,activation='relu')
        ### Branch 4
        self.max_pool = MaxPool2D((3,3),strides=1,padding='same')
        self.conv_6 = CustomConv(NUM_OF_FILTER[position][5],(1,1),strides=1,activation='relu')
    def call(self,input):
        branch_1 = self.conv_1(input)
        branch_2 = self.conv_3(self.conv_2(input))
        branch_3 = self.conv_5(self.conv_4(input))
        branch_4 = self.conv_6(self.max_pool(input))
        result = tf.keras.layers.concatenate([branch_1,branch_2,branch_3,branch_4],axis=-1)
        return result