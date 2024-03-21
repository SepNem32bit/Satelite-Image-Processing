import keras
from keras.models import Model
from keras.layers import Input, Conv2D,MaxPooling2D,UpSampling2D,Convolution2DTranspose
from keras.layers import concatenate,BatchNormalization,Dropout,Lambda

from keras.backend import backend as K


class Unet(keras.Model):
    def __init__(self,image_height,image_width, image_channels,n_classes):
        #By calling super().__init__(), you ensure that the initialization logic of the superclass is executed before any additional initialization logic in the subclass. This helps to maintain the integrity of the 
        #superclass's functionality while allowing you to add custom initialization logic specific to the subclass.
        super(Unet, self).__init__()
        self.image_height=image_height
        self.image_width=image_width
        self.image_channels=image_channels
        self.n_classes=n_classes
    def down_conv_layer(self,n_filter,input):
        conv=Conv2D(n_filter,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(input)
        #Dropout involves randomly disabling nodes during the training phase that prevents overfitting
        conv=Dropout(0.2)(conv)
        conv=Conv2D(n_filter,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv)
        pool=MaxPooling2D((2,2))(conv)
        return pool,conv

    def up_conv_layer(self,n_filter,input, concatenate_conv_input):
        conv=Convolution2DTranspose(n_filter,(2,2),strides=(2,2),padding='same')(input)
        #Concatnate: It takes as input a list of tensors, all of the same shape except for the concatenation axis, 
        #and returns a single tensor that is the concatenation of all inputs.(default axis=-1)
        conv= concatenate([conv,concatenate_conv_input])
        conv=Conv2D(n_filter,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv)
        conv=Dropout(0.2)(conv)
        conv=Conv2D(n_filter,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv)
        return conv
    def Unet_model(self,n_filter):
        input= Input((self.image_height,self.image_width, self.image_channels))
        down_layer_1,concatenate_conv_1=self.down_conv_layer(n_filter,input)
        down_layer_2,concatenate_conv_2=self.down_conv_layer(n_filter*2,down_layer_1)
        down_layer_3,concatenate_conv_3=self.down_conv_layer(n_filter*4,down_layer_2)
        down_layer_4,concatenate_conv_4=self.down_conv_layer(n_filter*8,down_layer_3)
        down_layer_5, _ =self.down_conv_layer(n_filter*16,down_layer_4)
        up_layer_6=self.up_conv_layer(n_filter*8,down_layer_5,concatenate_conv_4)
        up_layer_7=self.up_conv_layer(n_filter*4,up_layer_6,concatenate_conv_3)
        up_layer_8=self.up_conv_layer(n_filter*2,up_layer_7,concatenate_conv_2)
        up_layer_9=self.up_conv_layer(n_filter,up_layer_8,concatenate_conv_1)
        outputs=Conv2D(self.n_classes,(1,1),activation='softmax',kernel_initializer='he_normal',padding='same')(up_layer_9)
        model= Model(inputs=[input],outputs=[outputs])
        return model
