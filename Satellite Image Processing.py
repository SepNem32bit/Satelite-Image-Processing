import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import random
from Class_Encoding import map_class_encoding


##Part1: Preprocessing
data_root_folder='D:\Data Science\Python Assignment\Computer Vision\Data'
dataset_name='Semantic segmentation dataset'

image_patch_size=256

i_name=[]
m_name=[]

image_dataset=[]
mask_dataset=[]
#Folders only in the dataset
for path,subdir,files in os.walk(os.path.join(data_root_folder,dataset_name)):
    #Last element of all directories (name of folders) 
    dir_name=path.split(os.path.sep)[-1]
    # print(dir_name)
    if dir_name=='images':
        images=os.listdir(path)
        # print(images)
        # print(type(path))
        for i,image_name in enumerate(images):
            if image_name.endswith('.jpg'):
                # print(path+'\\'+image_name)
                image=cv2.imread(path+'\\'+image_name)
                #We keep the BGR format for IMAGE
                if image is not None:
                    # image_dataset.append(image)
                    #Defining new shape based on the patch size
                    #the floor division // rounds the result down to the nearest whole number
                    size_x=(image.shape[1]//image_patch_size)*image_patch_size
                    size_y=(image.shape[0]//image_patch_size)*image_patch_size
                    # print('Original Size:{}---Patch Size:{},{}'.format(image.shape,size_y,size_x))
                    #The cv2 read the image as numpy array. Now we want to convert it back to image to crop it
                    image_orginal=Image.fromarray(image)
                    image_orginal_crop=image_orginal.crop((0,0,size_x,size_y))
                    #Converting back to numpy for patching
                    image=np.array(image_orginal_crop)
                    #extracting patches from an image or a set of images
                    #step= It determines how much the extraction window moves horizontally and vertically between each patch.
                    patched_images=patchify(image,(image_patch_size,image_patch_size,3),step=image_patch_size)
                    # print(patched_images.shape)
                    for i in range(patched_images.shape[0]):
                        for j in range(patched_images.shape[1]):
                            #Not considering the first and second dimension
                            individual_patched_image=patched_images[i,j,:,:]
                            # print(individual_patched_image.shape)

                            #Scaling the data between zero and one
                            minMaxScale=MinMaxScaler()
                            #reshape(-1):This is particularly useful when you want to flatten an array without explicitly 
                            #specifying the size of the resulting array along a particular dimension.
                            #i_x.reshape(-1,image.shape[-1]):To reshape it three channel (65536,3) 
                            #then we reshape it back to the original (1, 256, 256, 3) by .reshape(-1,image.shape[-1]))
                            scalled_individual_patched_image=minMaxScale.fit_transform(individual_patched_image.reshape(-1,individual_patched_image.shape[-1])).reshape(individual_patched_image.shape)
                            #Passing the first dimension which is one
                            scalled_individual_patched_image=scalled_individual_patched_image[0]
                            image_dataset.append(scalled_individual_patched_image)

    elif dir_name=='masks':
        masks=os.listdir(path)
        for i,mask_name in enumerate(masks):
            if mask_name.endswith('.png'):
                mask=cv2.imread(path+'\\'+mask_name)
                #The cv output is BGR color format so we convert it to RGB only for MASK
                mask=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                if mask is not None:
                    ##Same approach as images but no need for scaling 
                    size_x=(mask.shape[1]//image_patch_size)*image_patch_size
                    size_y=(mask.shape[0]//image_patch_size)*image_patch_size
                    # print('Original Size:{}---Patch Size:{},{}'.format(image.shape,size_y,size_x))
                    mask_orginal=Image.fromarray(mask)
                    mask_orginal_crop=mask_orginal.crop((0,0,size_x,size_y))
                    #Converting back to numpy for patching
                    mask=np.array(mask_orginal_crop)
                    patched_masks=patchify(mask,(image_patch_size,image_patch_size,3),step=image_patch_size)
                    # print(patched_images.shape)
                    for i in range(patched_masks.shape[0]):
                        for j in range(patched_masks.shape[1]):
                            individual_patched_mask=patched_masks[i,j,:,:]
                            # print(individual_patched_image.shape)
                            individual_patched_mask=individual_patched_mask[0]
                            mask_dataset.append(individual_patched_mask)

#Converting the list to array
image_dataset=np.array(image_dataset)
mask_dataset=np.array(mask_dataset)


# print(image_dataset[0].shape)
# print(mask_dataset[0].shape)

#Showing our image/mask patches
# random_image_id=random.randint(0,len(image_dataset)-1)
# plt.figure(figsize=(14,8))
# plt.subplot(121)
# plt.imshow(image_dataset[random_image_id])
# plt.subplot(122)
# plt.imshow(mask_dataset[random_image_id])
# plt.show()



##Encoding the classes

encoder=map_class_encoding()
#Converting rgb lables to digit ids
labels=[]
for i in range(mask_dataset.shape[0]):
    label=encoder.rgb_to_label(mask_dataset[i])
    labels.append(label)
labels=np.array(labels)


print(labels[0].shape)
print(mask_dataset[0].shape)
print(labels.shape)
#will result in a 4D array with dimensions (height, width, channels, 1). 
#The new axis added at position 3 has size 1, effectively creating a new singleton dimension: [1,1,1]-> [[1],[1],[1]]
labels=np.expand_dims(labels,axis=3)
print(labels.shape)
print(labels[1][0:2,0:2,0])

# print(np.unique(labels))
##Showing the image and the processed labels
# random_image_id=random.randint(0,len(image_dataset)-1)
# plt.figure(figsize=(14,8))
# plt.subplot(121)
# plt.imshow(image_dataset[random_image_id])
# plt.subplot(122)
##[:,:,0] selects all rows and columns along the first two dimensions of the array and only the elements corresponding to index 0 along the third dimension. In the context of image data, 
##this often means selecting only the data from the first channel of the image. Just ignoring the 4th dimension
# plt.imshow(labels[random_image_id][:,:,0])
# plt.show()



##Creating Training and Test Datasets
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical

master_training_dataset=image_dataset
total_classes=len(np.unique(labels))
#One hot coding of label class
categorical_labels=to_categorical(labels,num_classes=total_classes)
print(categorical_labels.shape)
x_train,x_test,y_train,y_test=train_test_split(master_training_dataset,categorical_labels,test_size=0.2,random_state=100)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

image_height=x_train.shape[1]
image_width=x_train.shape[2]
image_channels=x_train.shape[3]
n_classes=6


##Part2: Designing Deep Learning Model (Unet)
import keras
from keras.models import Model
from keras.layers import Input, Conv2D,MaxPooling2D,UpSampling2D,Convolution2DTranspose
from keras.layers import concatenate,BatchNormalization,Dropout,Lambda

from keras.backend import backend as K

#Jaccardindex (IoU): intersection (overlap) over union
def IoU(y_true,y_pred):
    y_true_flatten=K.flatten(y_true)
    y_pred_flatten=K.flatten(y_pred)
    #calculates the number of overlapping pixels between the ground truth and predicted segmentation masks,
    #sum()Computes the (weighted) sum of the given values in the tensor
    intersection=K.sum(y_true_flatten*y_pred_flatten)
    #Adding 1.0 to both the numerator and denominator is a common technique to prevent division by zero,
    # which could occur when both the ground truth and predicted segmentation areas are empty.
    iou=(intersection+1.0)/(K.sum(y_true_flatten)+K.sum(y_pred_flatten)-intersection+1.0)
    return iou

from Unet import Unet 
n_filter=16
model=Unet(image_height,image_width, image_channels,n_classes).Unet_model(n_filter)


metrics=['accuracy',IoU]



    
##Loss Function
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
#Focal Loss (FL) is an improved version of Cross-Entropy Loss (CE) 
# that tries to handle the class imbalance problem by assigning more weights to hard or easily misclassified examples
#"Focal Loss" represents the loss calculated using the Focal Loss function, which is commonly used in classification tasks to address class imbalance and focus more on hard, misclassified examples.
#"Dice Loss" represents the loss calculated using the Dice Loss function, which measures the dissimilarity between the predicted and ground truth segmentation masks at the pixel level in semantic segmentation tasks.


#1/n_classes
weights=[0.166,0.166,0.166,0.166,0.166,0.166]
diceLoss=sm.losses.DiceLoss(class_weights=weights)
focalLoss=sm.losses.CategoricalFocalLoss()
totalLoss=diceLoss+focalLoss

# totalLoss=keras.losses.BinaryFocalCrossentropy(
#     apply_class_balancing=False,
#     alpha=0.25,
#     gamma=2.0,
#     from_logits=False,
#     label_smoothing=0.0,
#     axis=-1,
#     reduction="sum_over_batch_size",
#     name="binary_focal_crossentropy")

#Visualizing the Unet network
from keras.utils.vis_utils import plot_model
plot_model(model,to_file='Unet Satelitte.png',show_shapes=True, show_layer_names=True)

##Model Compilation

import tensorflow as tf
tf.keras.backend.clear_session()
model.compile(optimizer='adam',loss=totalLoss,metrics=metrics)


#Creating a callback class
from Ipython.display import clear_output
%matplotlib inline
class PlotLoss(keras.callbacks.callback):
    def on_train_begin(self,logs={}):
        self.i=0
        self.x=[]
        self.losses=[]
        self.val_losses=[]
        self.fig=plt.figure()
        self.logs=[]
    def on_epoch_end(self,epoch,logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i+=1

        clear_output(wait=True)
        plt.plot(self.x,self.losses,label='Loss')
        plt.plot(self.x,self.val_losses,label='Val Loss')
        plt.legend()
        plt.show();

plotLoss=PlotLoss()

model_progress_history=model.fit(x_train,y_train,batch_size=15, 
                                 verbose=1,
                                 epochs=10,
                                 validation_data=(x_test,y_test),
                                 callback=[plotLoss],
                                 shuffle=True)

##Visualization of loss

loss=model_progress_history.history['loss']
val_loss=model_progress_history.history['val_loss']

epochs=range(0,len(loss)+1)
plt.plot(epochs,loss,'r',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legends()
plt.show()

iou=model_progress_history.history['IoU']
val_iou=model_progress_history.history['val_IoU']
epochs=range(0,len(loss)+1)
plt.plot(epochs,loss,'r',label='IoU')
plt.plot(epochs,val_loss,'b',label='Validation IoU')
plt.title('Training IoU vs Validation IoU')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legends()
plt.show()


##Model Prediction

y_pred=model.predict(x_test)
#returns indices of the max element of the array in a particular axis. 
#np.argmax(y_pred, axis=3) applies argmax along the last axis (axis=3), which represents the class probabilities. 
#This operation returns the index of the class with the highest probability for each pixel in the image. The resulting array y_pred_argmax will have the same shape as y_pred, 
#except the last axis (n_clases) is removed, leaving an array of shape (num_samples, height, width) 
#containing the predicted class labels for each pixel.
y_pred_argmax=np.argmax(y_pred,axis=3)
y_test_argmax=np.argmax(y_test,axis=3)


##Comparing images
image_number=random.randit(0,len(x_test))
test_image=x_test[image_number]
ground_truth_image=y_test_argmax[image_number]
#The function np.expand_dims(test_image, 0) is used to add an extra dimension to the input test_image array at the specified position (position 0 in this case). 
#This is often necessary when you want to pass a single sample to a model that expects input data in a certain shape
test_image_input=np.expand_dims(test_image,0)

prediction=model.predict(test_image_input)
pred_image=np.argmax(prediction,axis=3)[0,:,:]

plt.figure(figsize=(14,8))
plt.sublot(231)
plt.title('Original Image')
plt.imshow(test_image)
plt.sublot(232)
plt.title('Ground Truth Image')
plt.imshow(ground_truth_image)
plt.sublot(233)
plt.title('Predicted Image')
plt.imshow(pred_image)



##Saving Model
model.save('D:\Data Science\Python Assignment\Computer Vision\Model\Unet Satelitte Segmentation.h5')

