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
    






# def Unet_model(n_classes,image_height,image_width, image_channels):
#     input= Input((image_height,image_width, image_channels))

#     #Layer1
#     conv1=Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(input)
#     #Dropout involves randomly disabling nodes during the training phase that prevents overfitting
#     conv1=Dropout(0.2)(conv1)
#     conv1=Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv1)
#     pool1=MaxPooling2D((2,2))(conv1)

#     #Layer2
#     conv2=Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool1)
#     conv2=Dropout(0.2)(conv2)
#     conv2=Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv2)
#     pool2=MaxPooling2D((2,2))(conv2)

#     #Layer3
#     conv3=Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool2)
#     conv3=Dropout(0.2)(conv3)
#     conv3=Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv3)
#     pool3=MaxPooling2D((2,2))(conv3)

#     #Layer4
#     conv4=Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool3)
#     conv4=Dropout(0.2)(conv4)
#     conv4=Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv4)
#     pool4=MaxPooling2D((2,2))(conv4)

#     #Layer5
#     conv5=Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool4)
#     conv5=Dropout(0.2)(conv5)
#     conv5=Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv5)
#     pool5=MaxPooling2D((2,2))(conv5)

#     #Layer 6
#     #Mathematically, instead of multiplying two 3x3 matrices, we can multiply each value in the input layer by the 3x3 kernel to yield a 3x3 matrix. 
#     #Then, we just combine all of them together according to the initial positions in the input layer, and sum the overlapped values together
#     up6=Convolution2DTranspose(128,(2,2),strides=(2,2),padding='same')(conv5)
#     #Concatnate: It takes as input a list of tensors, all of the same shape except for the concatenation axis, 
#     #and returns a single tensor that is the concatenation of all inputs.(default axis=-1)
#     up6= concatenate([up6,conv4])
#     up6=Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(up6)
#     up6=Dropout(0.2)(up6)
#     up6=Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(up6)

#     #Layer 7
#     up7=Convolution2DTranspose(64,(2,2),strides=(2,2),padding='same')(up6)
#     up7= concatenate([up7,conv3])
#     up7=Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(up7)
#     up7=Dropout(0.2)(up7)
#     up7=Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(up7)

#     #Layer 8
#     up8=Convolution2DTranspose(32,(2,2),strides=(2,2),padding='same')(up7)
#     up8= concatenate([up8,conv2])
#     up8=Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(up8)
#     up8=Dropout(0.2)(up8)
#     up8=Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(up8)

#     #Layer 9
#     up9=Convolution2DTranspose(16,(2,2),strides=(2,2),padding='same')(up8)
#     up9= concatenate([up9,conv1])
#     up9=Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(up9)
#     up9=Dropout(0.2)(up9)
#     up9=Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(up9)
    
#     #Final Layer
#     outputs=Conv2D(n_classes,(1,1),activation='softmax',kernel_initializer='he_normal',padding='same')(up9)

#     model= Model(inputs=[input],outputs=[outputs])
#     return model