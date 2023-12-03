#!/usr/bin/env python
# coding: utf-8

# #SteGuz : Image Steganography using CNN
# ### A Tensorflow Implementation
# #### Published June 2022
#

verbose = False

# ## Imports

# from IPython import get_ipython
# get_ipython().run_line_magic('pylab', 'inline')
# get_ipython().run_line_magic('load_ext', 'tensorboard')
import glob
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pathlib
import random
import pandas as pd
import io

# disabling newest version of tensorflow, since this code was written before TF2
# import tensorflow as tf

if not verbose:
    import logging
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
import tensorflow.compat.v1 as tf

if not verbose:
    logging.getLogger('tensorflow').disabled = True
    import warnings
    warnings.filterwarnings('ignore')

import tensorflow_datasets as tfds
import time
from datetime import datetime
from os.path import join

# image metrics:
import numpy as np
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# for use in mounting google drive image dataset
# from google.colab import drive
# drive.mount('/content/drive')

tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)


# ## Configuration
# All Configuration related information is represented in CAPS

ROOT_PATH = "/content/drive/MyDrive/"
GDRIVE_PATH = ROOT_PATH + "Yashi/"
TRAIN_PATH = ROOT_PATH + "Training-Data/"
LOGS_PATH = GDRIVE_PATH + "logs/"
CHECKPOINTS_PATH = GDRIVE_PATH + "checkpoints_NoNL/"
SAVED_STEGO_MODEL_DIRECTORY_PATH = GDRIVE_PATH + "Saved-Stego-Models/"
MODEL_PATH = SAVED_STEGO_MODEL_DIRECTORY_PATH
METRIC_TESTING_IMAGES_PATH = GDRIVE_PATH + "Results/Final Network/"
SHOULD_CONTINUE_TRAINING_NETWORK = True
model_paths_list = ['/content/drive/MyDrive/Yashi/Saved-Stego-Models/']
# Batch size refers to the number of examples from the training dataset that are used in the estimate of the error
# gradient. Smaller batch sizes are preferred over larger ones because small batch sizes are noisy, which offer a regularizing
# effect and lower generalization error. They also make it easier to fit one batch worth of training data in memory.
# BATCH_SIZE = 8
BATCH_SIZE = 32

EPOCHS = 1
# The learning rate is a hyperparameter that controls how much the model changes in response to the estimated error
# each time the model weights are updated. A small learning rate may result in a long training process, while a large
# learning rate may result in learning a sub-optimal set of weights too quickly or an unstable training process.
# The learning rate may be the most important hyperparameter when configuring the neural network.
#
# Source: https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
# LEARNING_RATE = .0001
LEARNING_RATE = .0002

# Momentum is used to increase the speed of the optimization process.
BETA = .75
#BETA = .25

# Save the model as Trained_Model_Month_Day_Year_HH_MM_SS
# currentDateTime = datetime.now().strftime("%B_%m_%Y_%H_%M_%S")
# Changed by @Khalifa
currentDateTime = datetime.now().strftime("%m-%d-%Y")
EXP_NAME = f"Final_Trained_Model_{currentDateTime}"


# ## Method definitions
# The images are first converted to float values between 0 and 1.
#
#
#
#
def get_img_batch(files_list, batch_size=BATCH_SIZE, size=(224, 224)):
    # Populates a batch of images with random training images.
    #
    # param   files_list                            List of files to obtain the random images from
    # param   batch_size                            Batch size (defaults to specified BATCH_SIZE)
    # param   size                                  Used to set the size of the cropped images (default 224 x 224 px).
    #
    # return  batch_cover,batch_secret              A tuple with the batch of cover and secret images ready for processing

    batch_cover = []
    batch_secret = []

    for i in range(batch_size):
        img_secret_path = random.choice(files_list)
        img_cover_path = random.choice(files_list)

        img_secret = Image.open(img_secret_path).convert("RGB")
        img_cover = Image.open(img_cover_path).convert("RGB")

        # ImageOps returns a sized and cropped version of the image, cropped to the requested aspect ratio (img_secret) and size.
        # The np.array method converts the image to sized and cropped image into an array of float32 values
        img_secret = np.array(ImageOps.fit(img_secret, size), dtype=np.float32)
        img_cover = np.array(ImageOps.fit(img_cover, size), dtype=np.float32)

        # TODO: Why is this here?
        img_secret /= 255.
        img_cover /= 255.

        batch_cover.append(img_cover)
        batch_secret.append(img_secret)

    batch_cover, batch_secret = np.array(batch_cover), np.array(batch_secret)

    return batch_cover, batch_secret


def get_prep_network_op(secret_tensor):
    # The preparatory network prepares a single image to be hidden: secret_tensor

    # tf(.compat.v1).variable_scope is used to declare a variable with variable scope
    # This means that code elsewhere can call tf.get_variable to retrieve the variable,
    # or call tf.Variable to create the variable.
    #
    # Source: https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
    with tf.variable_scope('prep_net'):

        with tf.variable_scope("3x3_conv_branch"):
            # param     secret_tensor    Tensor input into the 2D convolution layer.
            # param     filters          Integer that specifies the dimensionality of the output space (i.e., # of filters in the convolution) (default 50)
            # param     kernel_size      Integer or tuple of two integers that specify the height/width of the 2D convolution window. If single int, height == width
            # param     padding          If 'valid', no padding. If 'same', padding evenly applied to the left/right or up/down such that output has the same height/width
            #                            dimension as the input.
            # param     name             String that represents the name of the layer
            # param     activation       Activation function used for the 2D convolution layer
            conv_3x3 = tf.layers.conv2d(
                inputs=secret_tensor, filters=50, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3,      filters=50, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3,      filters=50, kernel_size=3, padding='same', name="3", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3,      filters=50, kernel_size=3, padding='same', name="4", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3,      filters=50, kernel_size=3, padding='same', name="5", activation=tf.nn.relu)
            if verbose:
                print("Shape of prep 3x3 branch tensor: " +
                    " ".join(map(str, conv_3x3.get_shape().as_list())))

        with tf.variable_scope("4x4_conv_branch"):
            conv_4x4 = tf.layers.conv2d(
                inputs=secret_tensor, filters=50, kernel_size=4, padding='same', name="1", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4,      filters=50, kernel_size=4, padding='same', name="2", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4,      filters=50, kernel_size=4, padding='same', name="3", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4,      filters=50, kernel_size=4, padding='same', name="4", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4,      filters=50, kernel_size=4, padding='same', name="5", activation=tf.nn.relu)
            if verbose:
                print("Shape of prep 4x4 branch tensor: " +
                    " ".join(map(str, conv_4x4.get_shape().as_list())))

        with tf.variable_scope("5x5_conv_branch"):
            conv_5x5 = tf.layers.conv2d(
                inputs=secret_tensor, filters=50, kernel_size=5, padding='same', name="1", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5,      filters=50, kernel_size=5, padding='same', name="2", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5,      filters=50, kernel_size=5, padding='same', name="3", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5,      filters=50, kernel_size=5, padding='same', name="4", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5,      filters=50, kernel_size=5, padding='same', name="5", activation=tf.nn.relu)
            if verbose:
                print("Shape of prep 5x5 branch tensor: " +
                    " ".join(map(str, conv_5x5.get_shape().as_list())))

        # tf.concat() concatenates tensors along one dimension.
        # Parameter axis determines along which dimension to concatenate.
        concat_1 = tf.concat(
            [conv_3x3, conv_4x4, conv_5x5], axis=3, name='concat_1')
        if verbose:
            print("Shape of prep initial concat tensor: " +
                " ".join(map(str, concat_1.get_shape().as_list())))

        conv_5x5 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=5,
                                    padding='same', name="final_5x5", activation=tf.nn.relu)
        conv_4x4 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=4,
                                    padding='same', name="final_4x4", activation=tf.nn.relu)
        conv_3x3 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=3,
                                    padding='same', name="final_3x3", activation=tf.nn.relu)

        concat_final = tf.concat(
            [conv_5x5, conv_4x4, conv_3x3], axis=3, name='concat_final')
        if verbose:
            print("Shape of prep final concat tensor: " +
                " ".join(map(str, concat_final.get_shape().as_list())))

        return concat_final


def get_hiding_network_op(cover_tensor, prep_output):
    if verbose:
        print("Shape of hiding cover tensor: " +
            " ".join(map(str, cover_tensor.get_shape().as_list())))
        print("Shape of hiding secret tensor: " +
            " ".join(map(str, prep_output.get_shape().as_list())))

    with tf.variable_scope('hide_net'):
        concat_input = tf.concat(
            [cover_tensor, prep_output], axis=3, name='images_features_concat')
        if verbose:
            print("Shape of hiding concat_input tensor: " +
                " ".join(map(str, concat_input.get_shape().as_list())))

        with tf.variable_scope("3x3_conv_branch"):
            conv_3x3 = tf.layers.conv2d(
                inputs=concat_input, filters=50, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3,     filters=50, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3,     filters=50, kernel_size=3, padding='same', name="3", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3,     filters=50, kernel_size=3, padding='same', name="4", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3,     filters=50, kernel_size=3, padding='same', name="5", activation=tf.nn.relu)
            if verbose:
                print("Shape of hiding 3x3 concat tensor: " +
                    " ".join(map(str, conv_3x3.get_shape().as_list())))

        with tf.variable_scope("4x4_conv_branch"):
            conv_4x4 = tf.layers.conv2d(
                inputs=concat_input, filters=50, kernel_size=4, padding='same', name="1", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4,     filters=50, kernel_size=4, padding='same', name="2", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4,     filters=50, kernel_size=4, padding='same', name="3", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4,     filters=50, kernel_size=4, padding='same', name="4", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4,     filters=50, kernel_size=4, padding='same', name="5", activation=tf.nn.relu)
            if verbose:
                print("Shape of hiding 4x4 concat tensor: " +
                    " ".join(map(str, conv_4x4.get_shape().as_list())))

        with tf.variable_scope("5x5_conv_branch"):
            conv_5x5 = tf.layers.conv2d(
                inputs=concat_input, filters=50, kernel_size=5, padding='same', name="1", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5,     filters=50, kernel_size=5, padding='same', name="2", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5,     filters=50, kernel_size=5, padding='same', name="3", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5,     filters=50, kernel_size=5, padding='same', name="4", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5,     filters=50, kernel_size=5, padding='same', name="5", activation=tf.nn.relu)
            if verbose:
                print("Shape of hiding 5x5 concat tensor: " +
                    " ".join(map(str, conv_5x5.get_shape().as_list())))

        concat_1 = tf.concat(
            [conv_3x3, conv_4x4, conv_5x5], axis=3, name='concat_1')
        if verbose:
            print("Shape of hiding initial concat tensor: " +
                " ".join(map(str, concat_1.get_shape().as_list())))

        conv_5x5 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=5,
                                    padding='same', name="final_5x5", activation=tf.nn.relu)
        conv_4x4 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=4,
                                    padding='same', name="final_4x4", activation=tf.nn.relu)
        conv_3x3 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=3,
                                    padding='same', name="final_3x3", activation=tf.nn.relu)

        concat_final = tf.concat(
            [conv_5x5, conv_4x4, conv_3x3], axis=3, name='concat_final')
        output = tf.layers.conv2d(
            inputs=concat_final, filters=3, kernel_size=1, padding='same', name='output')
        if verbose:
            print("Shape of hiding final concat tensor: " +
                " ".join(map(str, output.get_shape().as_list())))

        return output


def get_reveal_network_op(container_tensor):

    with tf.variable_scope('reveal_net'):

        with tf.variable_scope("3x3_conv_branch"):
            conv_3x3 = tf.layers.conv2d(inputs=container_tensor, filters=50,
                                        kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3, filters=50, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3, filters=50, kernel_size=3, padding='same', name="3", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3, filters=50, kernel_size=3, padding='same', name="4", activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(
                inputs=conv_3x3, filters=50, kernel_size=3, padding='same', name="5", activation=tf.nn.relu)

        with tf.variable_scope("4x4_conv_branch"):
            conv_4x4 = tf.layers.conv2d(inputs=container_tensor, filters=50,
                                        kernel_size=4, padding='same', name="1", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4, filters=50, kernel_size=4, padding='same', name="2", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4, filters=50, kernel_size=4, padding='same', name="3", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4, filters=50, kernel_size=4, padding='same', name="4", activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(
                inputs=conv_4x4, filters=50, kernel_size=4, padding='same', name="5", activation=tf.nn.relu)

        with tf.variable_scope("5x5_conv_branch"):
            conv_5x5 = tf.layers.conv2d(inputs=container_tensor, filters=50,
                                        kernel_size=5, padding='same', name="1", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5, filters=50, kernel_size=5, padding='same', name="2", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5, filters=50, kernel_size=5, padding='same', name="3", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5, filters=50, kernel_size=5, padding='same', name="4", activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(
                inputs=conv_5x5, filters=50, kernel_size=5, padding='same', name="5", activation=tf.nn.relu)

        concat_1 = tf.concat(
            [conv_3x3, conv_4x4, conv_5x5], axis=3, name='concat_1')

        conv_5x5 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=5,
                                    padding='same', name="final_5x5", activation=tf.nn.relu)
        conv_4x4 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=4,
                                    padding='same', name="final_4x4", activation=tf.nn.relu)
        conv_3x3 = tf.layers.conv2d(inputs=concat_1, filters=50, kernel_size=3,
                                    padding='same', name="final_3x3", activation=tf.nn.relu)

        concat_final = tf.concat(
            [conv_5x5, conv_4x4, conv_3x3], axis=3, name='concat_final')
        if verbose:
            print("Shape of reveal initial concat tensor: " +
                " ".join(map(str, concat_final.get_shape().as_list())))

    output = tf.layers.conv2d(
        inputs=concat_final, filters=3, kernel_size=1, padding='same', name='output')
    if verbose:
        print("Shape of hiding output tensor: " +
            " ".join(map(str, output.get_shape().as_list())))

    return output


def get_noise_layer_op(tensor, std=0.0):
    # A noise layer is introduced into the network such that the networks do not simply encode the
    # information inside the least significant bits (LSB) of the cover image. This noise is added
    # to the output of the hiding network during the training process.
    with tf.variable_scope("noise_layer"):
        # tf.random_normal returns random values from a normal distribution
        return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32)


def get_loss_op(secret_true, secret_pred, cover_true, cover_pred, beta=.5):
    # The network is trained by reducing the error (loss from the hidden to the extracted image)
    # coverImg = cover_true
    # embeddedImg = cover_pred
    # stegoImg = secret_true
    # extractedImg = secret_pred
    with tf.variable_scope("losses"):
        # tf.constant method creates a constant tensor from a tensor-like object.
        beta = tf.constant(beta, name="beta")

        # TODO: the author speculated that the MSE used in the error term can be
        # substituted for another measure, which can make for a good addition.
        cover_psnr = tf.image.psnr(cover_true, cover_pred, 255)
        secret_psnr = tf.image.psnr(secret_true, secret_pred, 255)

        cover_ssim = tf.image.ssim(cover_true, cover_pred, 255)
        secret_ssim = tf.image.ssim(secret_true, secret_pred, 255)

        # DEBUG:
        # print("Shape of MSE tensor: " + tf.shape(tf.losses.mean_squared_error(cover_true, cover_pred)))
        # print("Shape of PSNR tensor: " + tf.shape(cover_psnr))
        # print("Shape of SSIM tensor: " + tf.shape(cover_ssim))
        '''reshapedSecretMSE = tf.losses.mean_squared_error(secret_true, secret_pred)#tf.reshape(tf.losses.mean_squared_error(secret_true, secret_pred), 0)
        reshapedSecretPSNR = tf.reshape(secret_psnr, [])
        reshapedSecretSSIM = tf.reshape(secret_ssim, [])
        reshapedCoverMSE = tf.losses.mean_squared_error(cover_true, cover_pred)#tf.reshape(tf.losses.mean_squared_error(cover_true, cover_pred), 0)
        reshapedCoverPSNR = tf.reshape(cover_psnr, [])
        reshapedCoverSSIM = tf.reshape(cover_ssim, [])
        print("Shape of reshaped MSE tensor: " + tf.strings.as_string(tf.shape(reshapedCoverMSE)))
        print("Shape of reshaped PSNR tensor: " + tf.strings.as_string(tf.shape(reshapedCoverPSNR)))
        print("Shape of reshaped SSIM tensor: " + tf.strings.as_string(tf.shape(reshapedCoverSSIM)))
        print(tf.strings.as_string(reshapedCoverMSE))
        print(tf.strings.as_string(reshapedCoverPSNR))
        print(tf.strings.as_string(reshapedCoverSSIM))'''
        # END DEBUG

        # ||S - S'|| (applies to all three networks, the preparatory, hiding, and reveal networks)
        # tf.losses.mean_squared_error method adds a Sum-of-Squares loss to the training procedure.
        # secret_mse = tf.losses.mean_squared_error(secret_true, secret_pred)# + (1 / secret_psnr) + (1 / secret_ssim)
        # secret_mse = reshapedSecretPSNR + reshapedSecretSSIM
        # secret_mse = secret_psnr + secret_ssim
        # secret_mse = tf.reshape(secret_psnr, [16]) + tf.reshape(secret_ssim, [16])
        # secret_mse = tf.stack([ tf.reshape(secret_psnr, []), tf.reshape(secret_ssim, []) ], 0)
        # secret_mse = tf.stack([ secret_psnr, secret_ssim ], 0)

        # ||C - C'|| (applies to only the preparatory and hiding networks)
        # cover_mse = tf.losses.mean_squared_error(cover_true, cover_pred)# + (1 / cover_psnr) + (1 / cover_ssim)
        # cover_mse = reshapedCoverMSE + reshapedCoverPSNR + reshapedCoverSSIM
        # cover_mse = reshapedCoverPSNR + reshapedCoverSSIM
        # cover_mse = cover_psnr + cover_ssim
        # cover_mse = tf.reshape(cover_psnr, [16]) + tf.reshape(cover_ssim, [16])
        # cover_mse = tf.stack([ tf.reshape(cover_psnr, []), tf.reshape(cover_ssim, []) ], 0)
        # cover_mse = tf.stack([ cover_psnr, cover_ssim ], 0)

        # sumLossSecret = tf.stack([tf.losses.mean_squared_error(secret_true, secret_pred), secret_psnr, secret_ssim], axis=0)

        # final_loss = cover_mse + beta * secret_mse

        ############################## OLD VERSION OF LOSS FUNCTION ##############################
        # secret_mse = tf.reduce_mean(tf.image.ssim(secret_true, secret_pred, 255)) + tf.reduce_mean(tf.image.psnr(secret_true, secret_pred, 255))
        # cover_mse = tf.reduce_mean(tf.image.ssim(cover_true, cover_pred, 255)) + tf.reduce_mean(tf.image.psnr(cover_true, cover_pred, 255))
        # final_loss = cover_mse + secret_mse

        ############################## NEW VERSION OF LOSS FUNCTION ##############################
        secret_mse = tf.reduce_mean(
            tf.image.ssim(secret_true, secret_pred, 255))
        cover_mse = tf.reduce_mean(tf.image.ssim(cover_true, cover_pred, 255))
        final_loss = cover_mse + secret_mse

        # DEBUG
        if verbose:
            print("Shape of reshaped MSE tensor: " +
                tf.strings.as_string(cover_mse))
            print("Shape of reshaped PSNR tensor: " +
                tf.strings.as_string(secret_mse))
            print("Shape of reshaped SSIM tensor: " +
                tf.strings.as_string(final_loss))
        # END DEBUG

        # return final_loss, secret_mse, cover_mse
        return final_loss, secret_mse, cover_mse


def get_tensor_to_img_op(tensor):
    with tf.variable_scope("", reuse=True):
        t = tensor

        # tf.clip_by_value clips a tensor to a given min and max value (0 and 1, respectively)
        return tf.clip_by_value(t, 0, 1)


def prepare_training_graph(secret_tensor, cover_tensor, global_step_tensor):

    prep_output_op = get_prep_network_op(secret_tensor)
    hiding_output_op = get_hiding_network_op(
        cover_tensor=cover_tensor, prep_output=prep_output_op)

    # Directly use the hiding_output_op without adding noise
    noise_add_op = hiding_output_op

    reveal_output_op = get_reveal_network_op(noise_add_op)

    # get_loss_op gets the error terms between the secret images, cover images, and overall error, modified by beta.
    loss_op, secret_loss_op, cover_loss_op = get_loss_op(
        secret_tensor, reveal_output_op, cover_tensor, hiding_output_op, beta=BETA)

    # tf.train.AdamOptimizer: Optimizer that implements the Adam algorithm.
    # minimize() computes the gradients and applies them to the variables
    maximize_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
        (-1 * loss_op), global_step=global_step_tensor)

    # tf.summary.scalar: writes a scalar summary
    if verbose:
        tf.summary.scalar('loss', loss_op, family='train')
        tf.summary.scalar('reveal_net_loss', secret_loss_op, family='train')
        tf.summary.scalar('cover_net_loss', cover_loss_op, family='train')

    # tf.summary.image: writes an image summary
    if verbose:
        tf.summary.image('secret', get_tensor_to_img_op(
            secret_tensor), max_outputs=1, family='train')
        tf.summary.image('cover', get_tensor_to_img_op(
            cover_tensor), max_outputs=1, family='train')
        tf.summary.image('hidden', get_tensor_to_img_op(
            hiding_output_op), max_outputs=1, family='train')
        tf.summary.image('hidden_noisy', get_tensor_to_img_op(
            noise_add_op), max_outputs=1, family='train')
        tf.summary.image('revealed', get_tensor_to_img_op(
            reveal_output_op), max_outputs=1, family='train')

    # tf.summary.image: merges all summaries collected in the default graph.
    merged_summary_op = tf.summary.merge_all()

    return maximize_op, merged_summary_op


def prepare_test_graph(secret_tensor, cover_tensor):
    with tf.variable_scope("", reuse=True):

        prep_output_op = get_prep_network_op(secret_tensor)
        hiding_output_op = get_hiding_network_op(
            cover_tensor=cover_tensor, prep_output=prep_output_op)
        reveal_output_op = get_reveal_network_op(hiding_output_op)

        loss_op, secret_loss_op, cover_loss_op = get_loss_op(
            secret_tensor, reveal_output_op, cover_tensor, hiding_output_op)

        if verbose:
            tf.summary.scalar('loss',            loss_op,        family='test')
            tf.summary.scalar('reveal_net_loss', secret_loss_op, family='test')
            tf.summary.scalar('cover_net_loss',  cover_loss_op,  family='test')

            tf.summary.image('secret',   get_tensor_to_img_op(
                secret_tensor),    max_outputs=1, family='test')
            tf.summary.image('cover',    get_tensor_to_img_op(
                cover_tensor),     max_outputs=1, family='test')
            tf.summary.image('hidden',   get_tensor_to_img_op(
                hiding_output_op), max_outputs=1, family='test')
            tf.summary.image('revealed', get_tensor_to_img_op(
                reveal_output_op), max_outputs=1, family='test')

        merged_summary_op = tf.summary.merge_all()

        return merged_summary_op


def prepare_deployment_graph(secret_tensor, cover_tensor, covered_tensor):
    with tf.variable_scope("", reuse=True):

        prep_output_op = get_prep_network_op(secret_tensor)
        hiding_output_op = get_hiding_network_op(
            cover_tensor=cover_tensor, prep_output=prep_output_op)

        reveal_output_op = get_reveal_network_op(covered_tensor)

        return hiding_output_op, reveal_output_op


def runFullNetwork():
    # Define the starting epoch and step
    starting_epoch = 0
    starting_step = 0
    global_step = 0

    # Try to restore the model from the latest checkpoint
    checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH)
    if checkpoint:
        saver.restore(sess, checkpoint)
        if verbose:
            print("Model restored from:", checkpoint)
        # Assuming the checkpoint name ends with '-step'
        last_step = int(checkpoint.split('-')[-1])
        starting_epoch = last_step // total_steps
        starting_step = last_step % total_steps
        global_step = last_step

    else:
        print(CHECKPOINTS_PATH + " not found, training from scratch.")

    # Training loop
    for ep in range(starting_epoch, EPOCHS):
        for step in range(starting_step, total_steps):
            print("Executing: EPOCH " + str(ep) + " STEP " + str(step) + "\n")

            covers, secrets = get_img_batch(
                files_list=files_list, batch_size=BATCH_SIZE)
            sess.run([train_op], feed_dict={
                     "input_prep:0": secrets, "input_hide:0": covers})

            if step % 10 == 0:
                summary, global_step = sess.run([summary_op, global_step_tensor],
                                                feed_dict={"input_prep:0": secrets, "input_hide:0": covers})
                writer.add_summary(summary, global_step)

            if step % 100 == 0:
                covers, secrets = get_img_batch(
                    files_list=files_list, batch_size=1)
                summary, global_step = sess.run([test_op, global_step_tensor],
                                                feed_dict={"input_prep:0": secrets, "input_hide:0": covers})
                writer.add_summary(summary, global_step)

        save_path = saver.save(sess, join(
            CHECKPOINTS_PATH + ".chkp"), global_step=global_step)
        # save_path = saver.save(sess, join(MODEL_PATH, str(ep)), global_step = global_step)

        # Reset the starting step for the next epoch
        starting_step = 0

    save_path = saver.save(sess, join(MODEL_PATH, EXP_NAME),
                           global_step=global_step, write_meta_graph=True)
    print("Training completed!", save_path)


def runAbbreviatedNetwork():
  # Run the networks for an abbreviated duration
  # try:
  #  save_path = saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
  #  print("Model successfully restored.")
  # except:
    # print(MODEL_PATH + " not found, training from scratch.")
    for step in range(total_steps):
        covers, secrets = get_img_batch(
            files_list=files_list, batch_size=BATCH_SIZE)
        sess.run([train_op], feed_dict={
                 "input_prep:0": secrets, "input_hide:0": covers})

        if step % 10 == 0:

            summary, global_step = sess.run([summary_op, global_step_tensor], feed_dict={
                                            "input_prep:0": secrets, "input_hide:0": covers})
            writer.add_summary(summary, global_step)

        if step % 100 == 0:

            covers, secrets = get_img_batch(
                files_list=files_list, batch_size=1)
            summary, global_step = sess.run([test_op, global_step_tensor], feed_dict={
                                            "input_prep:0": secrets, "input_hide:0": covers})
            writer.add_summary(summary, global_step)

    # save_path = saver.save(sess, join(CHECKPOINTS_PATH, EXP_NAME + ".chkp"), global_step = global_step)
    save_path = saver.save(sess, MODEL_PATH, global_step=global_step)


# ## Network Definitions
# The three networks are identical in terms of structure.
#
# 1. The Prepare network takes in the **Secret Image** and outputs a (BATCH_SIZE,INPUT_HEIGHT,INPUT_WEIGHT,150) tensor.
#
# 2. The Cover network takes in the output from 1. , and a *Cover Image*. It concatenates these two tensors , giving a (BATCH_SIZE,INPUT_HEIGHT,INPUT_WEIGHT,153) tensor. Then it performs Convolutions , and outputs a (BATCH_SIZE,INPUT_HEIGHT,INPUT_WEIGHT,3) image.
#
# 3. The Reveal Network Takes in the output image from Cover Network , and outputs the Revealed Image (which is supposed to look like the **Secret Image**
#

# ORIGINAL
# files_list = glob.glob(join("drive/My Drive/Colab Notebooks/Training-Data/imagenette/","*"))
# testing_files_list = glob.glob(join("drive/My Drive/Colab Notebooks/Training-Data/**/*", "*.jpg"), recursive=True)
# files_list = files_list + testing_files_list

# Modified by @AK in Jan. 2023
imgNett_files_list = glob.glob(join(TRAIN_PATH + "imagenette/", "*"))
Linn_files_list = glob.glob(
    join(TRAIN_PATH + "Linnaeus/**/*", "*.jpg"), recursive=True)
files_list = imgNett_files_list + Linn_files_list
"""
glob.glob(join(TRAIN_PATH + "dog/","*"))
flower_files_list = glob.glob(join(TRAIN_PATH + "flower/","*"))
berry_files_list = glob.glob(join(TRAIN_PATH + "berry/","*"))
bird_files_list = glob.glob(join(TRAIN_PATH + "bird/","*"))
files_list = imgNett_files_list + dog_files_list + flower_files_list + bird_files_list + berry_files_list
"""

# Train with only len(files_list) // BATCH_SIZE number of files
total_steps = len(files_list) // BATCH_SIZE
if verbose:
    print("The training of " + str(len(files_list)) +
        "should be done in  " + str(total_steps) + " steps.")

# Begins the TensorFlow session for use in interactive contexts, such as a shell.
# tf.InteractiveSession.close(sess)
sess = tf.InteractiveSession(graph=tf.Graph())

# tf.placeholder: Inserts a placeholder for a tensor that will be always fed.
secret_tensor = tf.placeholder(
    shape=[None, 224, 224, 3], dtype=tf.float32, name="input_prep")

cover_tensor = tf.placeholder(
    shape=[None, 224, 224, 3], dtype=tf.float32, name="input_hide")

global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

train_op, summary_op = prepare_training_graph(
    secret_tensor, cover_tensor, global_step_tensor)

writer = tf.summary.FileWriter(join(LOGS_PATH, EXP_NAME), sess.graph)

test_op = prepare_test_graph(secret_tensor, cover_tensor)

covered_tensor = tf.placeholder(
    shape=[None, 224, 224, 3], dtype=tf.float32, name="deploy_covered")
deploy_hide_image_op, deploy_reveal_image_op = prepare_deployment_graph(
    secret_tensor, cover_tensor, covered_tensor)

# Save up to 5 of the last checkpoints
saver = tf.train.Saver(max_to_keep=10)
sess.run(tf.global_variables_initializer())


# # Network Training
# runFullNetwork()
# runAbbreviatedNetwork()


# loading a fully trained model to use for Experiements
def load_fully_trained_model():
    inputModelPath = input("Enter the model path to load: ")
    if (inputModelPath == ""):
        # inputModelPath = "TrainedModel-10-02-2022_01_37_02-4455-11881"
        # AK : CHANGED to most recent!
        inputModelPath = "Final_Trained_Model_10-09-2023-1996"
        # inputModelPath = "AK_Trained_Model_01-14-2023-9655"

    MODEL_PATH = SAVED_STEGO_MODEL_DIRECTORY_PATH + inputModelPath

    try:
        saver.restore(sess, MODEL_PATH)
        tf.train.load_checkpoint(MODEL_PATH)
        print("Model " + MODEL_PATH + " successfully restored.")
    except:
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
        inputModelPath = "TrainedModel-" + dt_string
        MODEL_PATH = SAVED_STEGO_MODEL_DIRECTORY_PATH + inputModelPath
        print(
            "This model cannot be restored or does not exist. Defaulting to " + MODEL_PATH)


# # Experimental Results
# ### Calculate Performance metrics on sample (Cover/Secret) image pairs

# Author : Amal Khalifa
# Created Dec. 2022

# Experimental Results
# Calculate Performance metrics, such as PSNR, SSIM, MSE, etc.
# 10 pairs of (cover, secret) --> 10 pairs of (hidden, revealed)
# The (hidden, revealed) images will be created and saved in RES_IMG_PATH

def calculate_performance_metrics_on_sample_image_pairs():
    TEST_IMG_PATH = '/content/drive/MyDrive/Yashi/Test Images/'
    RES_IMG_PATH = '/content/drive/MyDrive/Yashi/Results/'

    coverImgList = ["Baboon", "Berries", "Chainsaws", "Church", "Dog",
                    "Fish", "French Horn", "Garbage Truck", "Gas Pump", "Golf Balls"]
    secretImgList = ["Graffiti", "Karaoke", "Lena", "Lotus", "Parachute",
                     "Parrot", "Pens", "Peppers", "Stained Glass", "Thistle"]

    for i in range(len(coverImgList)):
        # load images
        cover = Image.open(
            TEST_IMG_PATH + coverImgList[i] + ".png").convert("RGB")
        secret = Image.open(
            TEST_IMG_PATH + secretImgList[i] + ".png").convert("RGB")

        # convert into float
        cover = np.array(ImageOps.fit(cover, (224, 224)), dtype=np.float32)
        cover /= 255.
        secret = np.array(ImageOps.fit(secret, (224, 224)), dtype=np.float32)
        secret /= 255.

        secrets = [secret]
        covers = [cover]

        # Use the model to hide..
        hidden = sess.run(deploy_hide_image_op, feed_dict={
            'input_prep:0': secrets, 'input_hide:0': covers})
        # hidden = np.clip(hidden, 0, 1)
        # plt.imsave(RES_IMG_PATH + "Stego_"+str(i)+".png", np.reshape(hidden.squeeze(),(0.8784, 0.8784, 0.0118)))

        # Use the model to extract..
        revealed = sess.run(deploy_reveal_image_op, feed_dict={
                            'deploy_covered:0': hidden})
        # revealed = np.clip(revealed, 0, 1)
        # plt.imsave(RES_IMG_PATH + "retrieved_"+str(i)+".png", np.reshape(revealed.squeeze(),(0.8784, 0.8784, 0.0118)))

        # show images
        # Cover...
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.imshow(cover)
        plt.show()
        # Secret...
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.imshow(secret)
        plt.show()
        # stego...
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.imshow(hidden.squeeze())
        plt.show()
        # revealed...
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.imshow(revealed.squeeze())
        plt.show()

        # Computing Performance Metrics
        # PSNR
        coverAndHiddenPSNR = psnr(cover, hidden.squeeze())
        print("PSNR between cover and hidden image (imperceptibility): " +
              str(coverAndHiddenPSNR))

        # SSIM
        secretAndExtractedSSIM = ssim(
            secret, revealed.squeeze(), multichannel=True)
        print("SSIM between secret and extracted image (recoverability): " +
              str(secretAndExtractedSSIM))

# Author : Guzman
# Last Updated: April 2022

# Testing the trained model on ONE image


def test_trained_model_on_one_image():
    covers, secrets = get_img_batch(files_list=files_list, batch_size=1)
    ROOT_PATH = '/content/drive/MyDrive/Yashi/'

    # sample images
    coverimgName = "Golf Balls"
    hiddenimgName = "Thistle"

    # start debug
    if (True):
        covers = Image.open(ROOT_PATH + "testing_images/" +
                            "riginal Cover.png").convert("RGB")
        secrets = Image.open(ROOT_PATH + "testing_images/" +
                             "Original Hidden.png").convert("RGB")

        secrets = np.array(ImageOps.fit(secrets, (224, 224)), dtype=np.float32)
        covers = np.array(ImageOps.fit(covers, (224, 224)), dtype=np.float32)

        secrets /= 255.
        covers /= 255.

        secrets = [secrets]
        covers = [covers]

        covers, secrets = np.array(covers), np.array(secrets)
    # end debug

    cover = covers.squeeze()
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(cover)
    0
    plt.show()

    secret = secrets.squeeze()
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(secret)
    plt.show()

    hidden = sess.run(deploy_hide_image_op, feed_dict={
        'input_prep:0': secrets, 'input_hide:0': covers})
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(hidden.squeeze())
    plt.show()
    # plt.imsave("/content/drive/MyDrive/Colab Notebooks/testing_images/Control Images/Results/" + coverimgName + ".jpg", np.reshape(hidden.squeeze(), (224, 224, 3)))

    revealed = sess.run(deploy_reveal_image_op, feed_dict={
                        'deploy_covered:0': hidden})
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(revealed.squeeze())
    # plt.imsave("/content/drive/MyDrive/Colab Notebooks/testing_images/Control Images/Results/" + hiddenimgName + ".jpg", revealed.squeeze())
    plt.show()


# Author : Guzman
# Last Updated: April 2022

# Calculate the STego performance metrics, such as PSNR, SSIM, MSE, etc.
# 10 pairs of cover secret --> 10 oairs of hidden revealed
# The images has been created already and saved in METRIC_TESTING_IMAGES_PATH

def calculate_stego_performance_metrics():
    avgCoverAndHiddenSSIM = 0
    avgSecretAndExtractedSSIM = 0
    avgCoverAndHiddenMSE = 0
    avgSecretAndExtractedMSE = 0
    avgCoverAndHiddenPSNR = 0
    avgSecretAndExtractedPSNR = 0
    avgCoverAndHiddenNCC = 0
    avgSecretAndExtractedNCC = 0

    numberOfImages = 20
    for imageNumber in range(1, numberOfImages):

        coverImg = img_as_float(Image.open(
            METRIC_TESTING_IMAGES_PATH + str(imageNumber) + "_cover.png"))
        hiddenImg = img_as_float(Image.open(
            METRIC_TESTING_IMAGES_PATH + str(imageNumber) + "_hidden.png"))
        stegoImg = img_as_float(Image.open(
            METRIC_TESTING_IMAGES_PATH + str(imageNumber) + "_stego.png"))
        extractedImg = img_as_float(Image.open(
            METRIC_TESTING_IMAGES_PATH + str(imageNumber) + "_uncovered.png"))

        print("Test results #" + imageNumber)

        # SSIM
        coverAndHiddenSSIM = ssim(coverImg, stegoImg, multichannel=True)
        secretAndExtractedSSIM = ssim(
            hiddenImg, extractedImg, multichannel=True)
        avgCoverAndHiddenSSIM = avgCoverAndHiddenSSIM + coverAndHiddenSSIM
        avgSecretAndExtractedSSIM = avgSecretAndExtractedSSIM + secretAndExtractedSSIM
        print("SSIM between cover and hidden image (imperceptibility): " +
              str(coverAndHiddenSSIM))
        print("SSIM between secret and extracted image (recoverability): " +
              str(secretAndExtractedSSIM))

        # PSNR
        coverAndHiddenPSNR = psnr(coverImg, stegoImg)
        secretAndExtractedPSNR = psnr(hiddenImg, extractedImg)
        avgCoverAndHiddenPSNR = avgCoverAndHiddenPSNR + coverAndHiddenPSNR
        avgSecretAndExtractedPSNR = avgSecretAndExtractedPSNR + secretAndExtractedPSNR
        print("PSNR between cover and hidden image (imperceptibility): " +
              str(coverAndHiddenPSNR))
        print("PSNR between secret and extracted image (recoverability): " +
              str(secretAndExtractedPSNR))

        # MSE
        coverAndHiddenMSE = mean_squared_error(coverImg, stegoImg)
        secretAndExtractedMSE = mean_squared_error(hiddenImg, extractedImg)
        avgCoverAndHiddenMSE = avgCoverAndHiddenMSE + coverAndHiddenMSE
        avgSecretAndExtractedMSE = avgSecretAndExtractedMSE + secretAndExtractedMSE
        print("MSE between cover and hidden image (imperceptibility): " +
              str(coverAndHiddenMSE))
        print("MSE between secret and extracted image (recoverability): " +
              str(secretAndExtractedMSE))

        # NCC
        '''a = coverImg
    b = stegoImg
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(np.ndarray.flatten(a), np.ndarray.flatten(b), 'full')
    coverAndHiddenNCC = c
    a = hiddenImg
    b = extractedImg
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'full')
    secretAndExtractedNCC = c
    avgCoverAndHiddenNCC = avgCoverAndHiddenNCC + coverAndHiddenNCC
    avgSecretAndExtractedNCC = avgSecretAndExtractedNCC + secretAndExtractedNCC
    print("NCC between cover and hidden image (imperceptibility): " + str(coverAndHiddenNCC))
    print("NCC between secret and extracted image (recoverability): " + str(secretAndExtractedNCC))'''

    print("\n\n\n")
    print("Average SSIM between cover and hidden images: " +
          str(avgCoverAndHiddenSSIM / numberOfImages))
    print("Average SSIM between secret and extracted images: " +
          str(avgSecretAndExtractedSSIM / numberOfImages))
    print("Average PSNR between cover and hidden images: " +
          str(avgCoverAndHiddenPSNR / numberOfImages))
    print("Average PSNR between secret and extracted images: " +
          str(avgSecretAndExtractedPSNR / numberOfImages))
    print("Average MSE between cover and hidden images: " +
          str(avgCoverAndHiddenMSE / numberOfImages))
    print("Average MSE between secret and extracted images: " +
          str(avgSecretAndExtractedMSE / numberOfImages))
    print("Average NCC between cover and hidden images: " +
          str(avgCoverAndHiddenNCC / numberOfImages))
    print("Average NCC between secret and extracted images: " +
          str(avgSecretAndExtractedNCC / numberOfImages))

# Testing the Model
# Calculate the network metrics, such as PSNR, SSIM, MSE, etc.
# Created by Anthony


def test_model():
    coverImgs = ["Baboon", "Berries", "Chainsaws", "Church", "Dog",
                 "Fish", "French Horn", "Garbage Truck", "Gas Pump", "Golf Balls"]
    hiddenImgs = ["Graffiti", "Karaoke", "Lena", "Lotus", "Parachute",
                  "Parrot", "Pens", "Peppers", "Stained Glass", "Thistle"]
    fileType = ".png"

    for x in range(10):

        coverImgFileName = "/content/drive/MyDrive/Yashi/Test Images/" + \
            coverImgs[x] + fileType
        stegoImgFileName = "/content/drive/MyDrive/Yashi/Test Images//" + \
            coverImgs[x] + fileType
        hiddenImgFileName = "/content/drive/MyDrive/Yashi/Test Images//" + \
            hiddenImgs[x] + fileType
        extractedImgFileName = "/content/drive/MyDrive/Yashi/Test Images//" + \
            hiddenImgs[x] + fileType

        # coverImgFileName = "/content/drive/MyDrive/Colab Notebooks/testing_images/Original Version Output Cover"  + fileType
        # stegoImgFileName = "/content/drive/MyDrive/Colab Notebooks/testing_images/Original Version Output Stego"  + fileType
        # hiddenImgFileName = "/content/drive/MyDrive/Colab Notebooks/testing_images/Original Version Output Hidden"  + fileType
        # extractedImgFileName = "/content/drive/MyDrive/Colab Notebooks/testing_images/Original Version Output Extracted"  + fileType

        coverImg = img_as_float(Image.open(coverImgFileName))
        stegoImg = img_as_float(Image.open(stegoImgFileName))
        hiddenImg = img_as_float(Image.open(hiddenImgFileName))
        extractedImg = img_as_float(Image.open(extractedImgFileName))

        # print("Cover: " + coverImgs[x] +  "     Hidden: " + hiddenImgs[x])

        OriginalNetworkStegoSSIM = ssim(coverImg, stegoImg, multichannel=True)
        print("SSIM between cover and stego image: " +
              str(float(format(OriginalNetworkStegoSSIM, '.5f'))))

        # PSNR
        OriginalNetworkStegoPSNR = psnr(coverImg, stegoImg)
        # OriginalNetworkRecoveredPSNR = peak_signal_noise_ratio(hiddenImg, extractedImg)
        print("PSNR between cover and stego image: " +
              str(float(format(OriginalNetworkStegoPSNR, '.5f'))))
        # print("PSNR between hidden and extracted image: " + str(OriginalNetworkRecoveredPSNR))

        # MSE
        # coverAndHiddenMSE = mean_squared_error(coverImg, stegoImg)
        OriginalNetworkRecoveredSSIM = ssim(
            hiddenImg, extractedImg, multichannel=True)
        print("SSIM between hidden and extracted image: " +
              str(float(format(OriginalNetworkRecoveredSSIM, '.5f'))))
        secretAndExtractedMSE = mean_squared_error(hiddenImg, extractedImg)
        # print("MSE between cover and stego image: " + str(coverAndHiddenMSE))
        print("MSE between hidden and extracted image: " +
              str(float(format(secretAndExtractedMSE, '.5f'))))
        print("")


def preprocess_image(image_data):
    # Open the image
    # img = Image.open(image_data)

    img = Image.fromarray(image_data)

    # If input is a byte stream instead of a file path, make sure we're at the beginning
    # if isinstance(image_data, io.BytesIO):
    #     image_data.seek(0)
    
    # If the image has an alpha (transparency) channel, remove it
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image
    img = img.resize((224, 224))

    return img_as_float(img)


def run_model_with_test_images():
    results_list = []

    coverImgs = ["Baboon", "Berries", "Chainsaws", "Church", "Dog",
                 "Fish", "French Horn", "Garbage Truck", "Gas Pump", "Golf Balls"]
    hiddenImgs = ["Graffiti", "Karaoke", "Lena", "Lotus", "Parachute",
                  "Parrot", "Pens", "Peppers", "Stained Glass", "Thistle"]
    fileType = ".png"

    for x in range(10):
        coverImgFileName = "/content/drive/MyDrive/Yashi/Test Images/" + \
            coverImgs[x] + fileType
        hiddenImgFileName = "/content/drive/MyDrive/Yashi/Test Images/" + \
            hiddenImgs[x] + fileType

        coverImg = preprocess_image(coverImgFileName)
        hiddenImg = preprocess_image(hiddenImgFileName)

        # Generate stego image and extract the hidden image
        # Generate stego image and extract the hidden image
        # Generate stego image
        stegoImg = sess.run(deploy_hide_image_op,
                            feed_dict={"input_prep:0": [hiddenImg], "input_hide:0": [coverImg]})
        # Extract the hidden image from the stego image
        extractedImg = sess.run(deploy_reveal_image_op,
                                feed_dict={"deploy_covered:0": stegoImg})

        # Calculate metrics
        OriginalNetworkStegoSSIM = ssim(coverImg, stegoImg[0], channel_axis=-1)
        OriginalNetworkStegoPSNR = psnr(coverImg, stegoImg[0])
        OriginalNetworkRecoveredSSIM = ssim(
            hiddenImg, extractedImg[0], channel_axis=-1)
        secretAndExtractedMSE = mean_squared_error(hiddenImg, extractedImg[0])

        # Append results to results_list
        results_list.append(["SteGuz", "Stego", coverImgs[x],
                            OriginalNetworkStegoSSIM, OriginalNetworkStegoPSNR])
        results_list.append(["SteGuz", "Extracted", hiddenImgs[x],
                            OriginalNetworkRecoveredSSIM, secretAndExtractedMSE])

    # Convert results_list to DataFrame and save to CSV
    df = pd.DataFrame(results_list, columns=[
        'Model', 'ImageType', 'ImageName', 'SSIM', 'PSNR'])
    df.to_csv(
        "/content/drive/MyDrive/Yashi/Saved-Stego-Models/Results.csv", index=False)
