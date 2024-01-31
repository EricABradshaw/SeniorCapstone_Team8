import glob
import os
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import pathlib
import random
import pandas as pd
import numpy as np
from skimage import img_as_float

import time
from datetime import datetime
from os.path import join
from tensorflow.keras import layers, optimizers
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#import tensorflow_datasets as tfds

def get_model_paths(directory):
    """
    Each subdirectory of /models/ is concatenated to itself twice, creating a
    path that TensorFlow can use to restore the saved model.

    directory: Full path to /models/.

    Returns: a list of all model paths.
    """
    model_folders = []
    items = os.listdir(directory)

    for item in items:
        item_path = os.path.join(directory, item)
        #item_path = os.path.join(item_path, item)
        model_folders.append(item_path)
        #if os.path.isdir(item_path):
        #    model_folders.append(item)

    return model_folders

def preprocess_image(img_path):
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    return img_as_float(img)

# def preprocess_image(image_data):
#     img = Image.fromarray(image_data)
    
#     # If the image has an alpha (transparency) channel, remove it
#     if img.mode != 'RGB':
#         img = img.convert('RGB')

#     # Resize the image
#     img = img.resize((224, 224))

#     return img_as_float(img)

if __name__ == '__main__':
    index = 0
        
    #coverImage = base64_to_image(coverImageString)
    #secretImage = base64_to_image(secretImageString)
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    modelPaths = get_model_paths(modelsDir)
    inputModelPath = modelPaths[index]
    print(f'Attempting to load model from {inputModelPath}')
    #load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = keras.models.load_model(inputModelPath)
    
    coverImage = "C:\\Users\\jacob\\Dropbox\\2024 Spring\\SeniorCapstone_Team8\\Application\\python\\Test Images\\Parachute.png"
    secretImage = "C:\\Users\\jacob\\Dropbox\\2024 Spring\\SeniorCapstone_Team8\\Application\\python\\Test Images\\Pens.png"
    
    coverImagePreproc = preprocess_image(coverImage)
    secretImagePreproc = preprocess_image(secretImage)
        
    coverImagePreproc = np.expand_dims(coverImagePreproc, axis=0)
    secretImagePreproc = np.expand_dims(secretImagePreproc, axis=0)
    
    
    
    # secret, cover
    stegoImage = model([secretImagePreproc, coverImagePreproc])
    stegoImage = tf.sigmoid(stegoImage)
    
    print('Done')
