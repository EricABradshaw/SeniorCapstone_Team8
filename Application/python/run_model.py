# from model_loader import *

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float
from PIL import Image, ImageOps
from SteGuz import deploy_hide_image_op, deploy_reveal_image_op, sess, saver, preprocess_image, verbose
import os
# suppress deprecation warnings and other outputs
import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings('ignore')



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
        item_path = os.path.join(item_path, item)
        model_folders.append(item_path)
        # if os.path.isdir(item_path):
        #    model_folders.append(item)

    return model_folders


def run_model(index=0, coverImage=None, secretImage=None):
    """
    Runs a SteGuz model with a provided cover and secret image to create a 
    Stego Image.

    index: Which model to use.
    cover: Path to a .png image.
    secret: Path to a .png image.

    Returns: a Stego Image created by the specified SteGuz model.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    parentDir = os.path.dirname(cwd)
    modelsDir = os.path.join(parentDir, 'models')
    modelPaths = get_model_paths(modelsDir)

    testing = True
    coverImagePreproc = None
    secretImagePreproc = None
    outputDir = os.path.join(cwd, 'Test Output')

    # if coverImage & secretImage are None, load a test image and create a Stego Image from them.
    testDir = os.path.join(cwd, 'Test Images')
    if coverImage is not None and secretImage is not None:
        testing = False
    else:
        coverImage = os.path.join(testDir, 'Parachute.png')
        # print(f'coverImage is {coverImage}')
        secretImage = os.path.join(testDir, 'Pens.png')
        # print(f'secretImage is {secretImage}')

    # model to use = index of the ComboBox
    inputModelPath = modelPaths[index]
    if verbose:
        for i in range(len(modelPaths)):
            print(modelPaths[i])

    try:
        saver.restore(sess, inputModelPath)
        tf.train.load_checkpoint(inputModelPath)
        if verbose:
            print(f'Successfully loaded model from {inputModelPath}.')
    except:
        print('This model cannot be restored or does not exist.')
        # run a default model?

    # Image preprocessing
    coverImagePreproc = preprocess_image(coverImage)
    secretImagePreproc = preprocess_image(secretImage)

    # Generate Stego Image
    stegoImage = sess.run(deploy_hide_image_op,
                          feed_dict={"input_prep:0": [secretImagePreproc], "input_hide:0": [coverImagePreproc]})
    if testing:
        stegoImage = np.clip(stegoImage, 0, 1)
        plt.imsave(os.path.join(outputDir, "test_stego_generation.png"),
                   np.reshape(stegoImage.squeeze(), (224, 224, 3)))
    else:
        print('stego image generated successfully.')
        # return JSON or image data that can be interpreted by React

    # Extract Hidden Image
    extractedImage = sess.run(deploy_reveal_image_op,
                              feed_dict={"deploy_covered:0": stegoImage})
    if testing:
        extractedImage = np.clip(extractedImage, 0, 1)
        plt.imsave(os.path.join(outputDir, "test_stego_extraction.png"),
                   np.reshape(extractedImage.squeeze(), (224, 224, 3)))
    else:
        print('secret image extracted successfully.')
        # return JSON or image data that can be interpreted by React

    # Calculate metrics
    # TODO


def run_model_batch(index=0, imgs=None):
    print('hello world')


run_model(0)
