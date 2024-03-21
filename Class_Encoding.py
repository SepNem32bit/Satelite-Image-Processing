import numpy as np


#Encoding the classes


class map_class_encoding():
    def __init__(self):
        self.Building='#3C1098'
        self.Land ='#8429F6'
        self.Road='#6EC1E4'
        self.Vegetation='#FEDD3A'
        self.Water='#E2A929'
        self.Unlabeled='#9B9B9B'    
    def encoding(self,class_code):
        class_code=class_code.lstrip('#')
        #Converting it to integer
        rgb_class_code=np.array(tuple(int(class_code[i:i+2],16) for i in [0,2,4]))
        return rgb_class_code
    def rgb_to_label(self,mask_dataset_element):
        label_segment=np.zeros(mask_dataset_element.shape,dtype=np.uint8)
        #Test whether all array elements along a given axis evaluate to True.
        #the result is aggregated along the last axis or channel or dimension, resulting in a boolean array where each element corresponds 
        #to whether all elements along the last axis of the original array are equal to class code in rgb
        label_segment[np.all(mask_dataset_element==self.encoding(self.Water),axis=-1)]=0
        label_segment[np.all(mask_dataset_element==self.encoding(self.Land),axis=-1)]=1
        label_segment[np.all(mask_dataset_element==self.encoding(self.Road),axis=-1)]=2
        label_segment[np.all(mask_dataset_element==self.encoding(self.Building),axis=-1)]=3
        label_segment[np.all(mask_dataset_element==self.encoding(self.Vegetation),axis=-1)]=4
        label_segment[np.all(mask_dataset_element==self.encoding(self.Unlabeled),axis=-1)]=5
        #This line of code will ensure that only the first channel of the label_segment array is retained, effectively converting the RGB image into a grayscale image
        label_segment=label_segment[:,:,0]
        return label_segment
    
