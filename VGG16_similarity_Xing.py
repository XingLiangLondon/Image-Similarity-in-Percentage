# -*- coding: utf-8 -*-
"""
x.liang@greenwich.ac.uk
25th March, 2020
Image Similarity using VGG16
"""
import os
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial import distance
'''
def get_feature_vector(img):
 img1 = cv2.resize(img, (224, 224))
 feature_vector = feature_model.predict(img1.reshape(1, 224, 224, 3))
 return feature_vector
'''

# fc2(Dense)output shape: (None, 4096) 
def get_feature_vector_fromPIL(img):
 feature_vector = feature_model.predict(img)
 assert(feature_vector.shape == (1,4096))
 return feature_vector

def calculate_similarity_cosine(vector1, vector2):
 #return 1- distance.cosine(vector1, vector2)
 return cosine_similarity(vector1, vector2) 

# This distance can be in range of [0,∞]. And this distance is converted to a [0,1]
def calculate_similarity_euclidean(vector1, vector2):
 #return distance.euclidean(vector1, vector2)     #distance.euclidean is slower
 return 1/(1+np.linalg.norm(vector1 - vector2))   #np.linalg.norm is faster
    

# Use VGG16 model as an image feature extractor 
image_input = Input(shape=(224, 224, 3))
model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
layer_name = 'fc2'
feature_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)


# Load images in the images folder into array
cwd_path = os.getcwd()
data_path =cwd_path + '/imgs'
data_dir_list = os.listdir(data_path)

img_data_list=[]
for dataset in data_dir_list:

		img_path = data_path + '/'+ dataset
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		img_data_list.append(x)

#vector_VGG16 =get_feature_vector_fromPIL(img_data_list[6])

# Caculate cosine similarity: [-1,1], that is, [completedly different,same]
image_similarity_cosine = calculate_similarity_cosine(get_feature_vector_fromPIL(img_data_list[31]), get_feature_vector_fromPIL(img_data_list[11]))
# Cacluate euclidean similarity: range from [0, 1], that is, [completedly different, same]
image_similarity_euclidean = calculate_similarity_euclidean(get_feature_vector_fromPIL(img_data_list[31]), get_feature_vector_fromPIL(img_data_list[11]))

print('VGG16 image similarity_euclidean:',image_similarity_euclidean)
print("VGG16 image similarity_cosine: {:.2f}%".format(image_similarity_cosine[0][0]*100))