# #SteGuz : Image Steganography using CNN
# ### A Tensorflow Implementation
# #### Published June 2022
# 

verbose = False

import glob
import os
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import pathlib
import random
import pandas as pd

if not verbose:
    import logging
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
import tensorflow as tf

if not verbose:
    logging.getLogger('tensorflow').disabled = True
    import warnings
    warnings.filterwarnings('ignore')

import tensorflow_datasets as tfds
import time
from datetime import datetime
from os.path import join
from tensorflow.keras import layers, optimizers

#image metrics:
import numpy as np
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


#for use in mounting google drive image dataset
#from google.colab import drive
#drive.mount('/content/drive')

#tf.enable_eager_execution()
#tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)


# ## Configuration
# All Configuration related information is represented in CAPS
ROOT_PATH = "/content/drive/MyDrive/"
GDRIVE_PATH = ROOT_PATH + "Yashi/"
TRAIN_PATH = ROOT_PATH + "Training-Data/"
LOGS_PATH = GDRIVE_PATH + "logs/"
CHECKPOINTS_PATH = GDRIVE_PATH + "Saved-Stego-Models/checkpoints_NoNL/"
SAVED_STEGO_MODEL_DIRECTORY_PATH = GDRIVE_PATH + "Saved-Stego-Models/"
MODEL_PATH = SAVED_STEGO_MODEL_DIRECTORY_PATH
METRIC_TESTING_IMAGES_PATH = GDRIVE_PATH + "Results/Final Network/"
SHOULD_CONTINUE_TRAINING_NETWORK = True
model_paths_list = ['/content/drive/MyDrive/Yashi/Saved-Stego-Models/']
# Batch size refers to the number of examples from the training dataset that are used in the estimate of the error
# gradient. Smaller batch sizes are preferred over larger ones because small batch sizes are noisy, which offer a regularizing
# effect and lower generalization error. They also make it easier to fit one batch worth of training data in memory.
#BATCH_SIZE = 50
BATCH_SIZE = 32

EPOCHS = 1
# The learning rate is a hyperparameter that controls how much the model changes in response to the estimated error
# each time the model weights are updated. A small learning rate may result in a long training process, while a large
# learning rate may result in learning a sub-optimal set of weights too quickly or an unstable training process.
# The learning rate may be the most important hyperparameter when configuring the neural network.
#
# Source: https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
#LEARNING_RATE = .0001
LEARNING_RATE = .0002
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
# Momentum is used to increase the speed of the optimization process.
BETA = .75

# Save the model as Trained_Model_Month_Day_Year_HH_MM_SS
#currentDateTime = datetime.now().strftime("%B_%m_%Y_%H_%M_%S")
#Changed by @Khalifa
currentDateTime = datetime.now().strftime("%m-%d-%Y")
EXP_NAME = f"Final_Trained_Model_P75_TF2_{currentDateTime}"


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
    # Defining convolutional layers for different branches
    conv_branches = []
    for kernel_size in [(3, 3), (4, 4), (5, 5)]:
        conv = secret_tensor
        for _ in range(5):
            conv = layers.Conv2D(50, kernel_size, padding='same', activation='relu')(conv)
        conv_branches.append(conv)

    # Concatenating branches
    concat = layers.Concatenate(axis=3)(conv_branches)

    # Additional convolution layers after concatenation
    conv_final = [layers.Conv2D(50, size, padding='same', activation='relu')(concat) for size in [(5, 5), (4, 4), (3, 3)]]

    # Final concatenation
    concat_final = layers.Concatenate(axis=3)(conv_final)

    return concat_final


def get_hiding_network_op(cover_tensor, prep_output):
    #print("Shape of hiding cover tensor:", cover_tensor.shape)
    #print("Shape of hiding secret tensor:", prep_output.shape)

    # Concatenating cover and prepared secret tensors
    concat_input = layers.Concatenate(axis=3)([cover_tensor, prep_output])
    #print("Shape of hiding concat_input tensor:", concat_input.shape)

    # Defining convolutional layers for different branches
    conv_branches = []
    for kernel_size in [(3, 3), (4, 4), (5, 5)]:
        conv = concat_input
        for _ in range(5):
            conv = layers.Conv2D(50, kernel_size, padding='same', activation='relu')(conv)
        conv_branches.append(conv)
        #print(f"Shape of hiding {kernel_size[0]}x{kernel_size[1]} concat tensor:", conv.shape)

    # Concatenating branches
    concat_1 = layers.Concatenate(axis=3)(conv_branches)
    #print("Shape of hiding initial concat tensor:", concat_1.shape)

    # Additional convolution layers after concatenation
    conv_final = [layers.Conv2D(50, size, padding='same', activation='relu')(concat_1) for size in [(5, 5), (4, 4), (3, 3)]]

    # Final concatenation
    concat_final = layers.Concatenate(axis=3)(conv_final)
    output = layers.Conv2D(3, (1, 1), padding='same', activation='relu', name='hiding_output')(concat_final)
    #print("Shape of hiding final concat tensor:", output.shape)

    return output


def get_reveal_network_op(container_tensor):
    # Defining convolutional layers for different branches
    conv_branches = []
    for kernel_size in [(3, 3), (4, 4), (5, 5)]:
        conv = container_tensor
        for _ in range(5):
            conv = layers.Conv2D(50, kernel_size, padding='same', activation='relu')(conv)
        conv_branches.append(conv)

    # Concatenating branches
    concat_1 = layers.Concatenate(axis=3)(conv_branches)

    # Additional convolution layers after concatenation
    conv_final = [layers.Conv2D(50, size, padding='same', activation='relu')(concat_1) for size in [(5, 5), (4, 4), (3, 3)]]

    # Final concatenation
    concat_final = layers.Concatenate(axis=3)(conv_final)
    output = layers.Conv2D(3, (1, 1), padding='same', name='reveal_output')(concat_final)

    #print("Shape of reveal initial concat tensor:", concat_final.shape)
    #print("Shape of reveal output tensor:", output.shape)

    return output


def get_loss_op(secret_true, secret_pred, cover_true, cover_pred, BETA):
    # Weight for PSNR in the combined loss calculation
    psnr_weight = 1.0 / 40.0

    # Calculate SSIM, PSNR, and MSE for secret and cover images
    secret_ssim = tf.reduce_mean(tf.image.ssim(secret_true, secret_pred, max_val=255))
    secret_psnr = tf.reduce_mean(tf.image.psnr(secret_true, secret_pred, max_val=255))
    # Use TensorFlow operations to compute MSE
    secret_mse = tf.reduce_mean(tf.square(secret_true - secret_pred))

    cover_ssim = tf.reduce_mean(tf.image.ssim(cover_true, cover_pred, max_val=255))
    cover_psnr = tf.reduce_mean(tf.image.psnr(cover_true, cover_pred, max_val=255))
    # Use TensorFlow operations to compute MSE
    cover_mse = tf.reduce_mean(tf.square(cover_true - cover_pred))

    # Combine SSIM and PSNR for secret and cover images, and apply beta weighting
    secret_loss = BETA * ((psnr_weight * secret_psnr) + secret_ssim) + (1 - BETA) * secret_mse
    cover_loss = BETA * ((psnr_weight * cover_psnr) + cover_ssim) + (1 - BETA) * cover_mse

    # Calculate final loss
    final_loss = cover_loss + secret_loss

    return -1 * final_loss, secret_mse, cover_mse


def get_tensor_to_img_op(tensor):
    # Clip the tensor values to be in the range [0, 1]
    return tf.clip_by_value(tensor, 0, 1)


def prepare_training_graph(secret_tensor, cover_tensor, global_step_tensor):
    prep_output_op = get_prep_network_op(secret_tensor)
    hiding_output_op = get_hiding_network_op(cover_tensor, prep_output_op)
    reveal_output_op = get_reveal_network_op(hiding_output_op)

    loss_op, secret_loss_op, cover_loss_op = get_loss_op(secret_tensor, reveal_output_op, cover_tensor, hiding_output_op, BETA)

    # Define the optimizer and the training operation
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    maximize_op = optimizer.minimize((-1 * loss_op), global_step=global_step_tensor)

    # Create summary operations for TensorBoard
    with tf.name_scope('train'):
        tf.summary.scalar('loss', -loss_op)
        tf.summary.scalar('reveal_net_loss', secret_loss_op)
        tf.summary.scalar('cover_net_loss', cover_loss_op)
        tf.summary.image('secret', get_tensor_to_img_op(secret_tensor), max_outputs=1)
        tf.summary.image('cover', get_tensor_to_img_op(cover_tensor), max_outputs=1)
        tf.summary.image('hidden', get_tensor_to_img_op(hiding_output_op), max_outputs=1)
        tf.summary.image('revealed', get_tensor_to_img_op(reveal_output_op), max_outputs=1)

    merged_summary_op = tf.summary.merge_all()

    return maximize_op, merged_summary_op


def prepare_test_graph(secret_tensor, cover_tensor):
    prep_output_op = get_prep_network_op(secret_tensor)
    hiding_output_op = get_hiding_network_op(cover_tensor, prep_output_op)
    reveal_output_op = get_reveal_network_op(hiding_output_op)

    loss_op, secret_loss_op, cover_loss_op = get_loss_op(secret_tensor, reveal_output_op, cover_tensor, hiding_output_op)

    # Create summary operations for TensorBoard
    with tf.name_scope('test'):
        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('reveal_net_loss', secret_loss_op)
        tf.summary.scalar('cover_net_loss', cover_loss_op)
        tf.summary.image('secret', get_tensor_to_img_op(secret_tensor), max_outputs=1)
        tf.summary.image('cover', get_tensor_to_img_op(cover_tensor), max_outputs=1)
        tf.summary.image('hidden', get_tensor_to_img_op(hiding_output_op), max_outputs=1)
        tf.summary.image('revealed', get_tensor_to_img_op(reveal_output_op), max_outputs=1)

    merged_summary_op = tf.summary.merge_all()

    return merged_summary_op


def prepare_deployment_graph(secret_tensor, cover_tensor, covered_tensor):
    prep_output_op = get_prep_network_op(secret_tensor)
    hiding_output_op = get_hiding_network_op(cover_tensor, prep_output_op)
    reveal_output_op = get_reveal_network_op(covered_tensor)

    return hiding_output_op, reveal_output_op

# Assuming other functions (get_img_batch, get_prep_network_op, get_hiding_network_op, get_reveal_network_op, get_loss_op) are defined elsewhere\
class CustomModel(tf.keras.Model):
    def call(self, inputs, training=False):
        secret_tensor, cover_tensor = inputs
        prep_output = get_prep_network_op(secret_tensor)
        hiding_output = get_hiding_network_op(cover_tensor, prep_output)
        reveal_output = get_reveal_network_op(hiding_output)
        return reveal_output


def runFullNetwork(train_dataset):
    # Variables to track the last executed epoch and step
    current_epoch = tf.Variable(0, dtype=tf.int64)
    current_step = tf.Variable(0, dtype=tf.int64)

    model = CustomModel()
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=lambda y_true, y_pred: get_loss_op(y_true, y_pred, y_true, y_pred, BETA),
              metrics=['accuracy'])

    # Checkpointing
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer, current_epoch=current_epoch, current_step=current_step)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINTS_PATH, max_to_keep=5)

    # Restore the latest checkpoint
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        if verbose:
            print(f"Restored from {manager.latest_checkpoint}, epoch {current_epoch.numpy()}, step {current_step.numpy()}")
    else:
        if verbose:
            print("Training from scratch")

    # Training loop
    for epoch in range(current_epoch.numpy(), EPOCHS):
        for step, (secret_batch, cover_batch) in enumerate(train_dataset, start=current_step.numpy()):
            print("Executing: EPOCH " + str(epoch) + " STEP " + str(step) + "\n")
            with tf.GradientTape() as tape:
                predictions = model([secret_batch, cover_batch])
                loss, _, _ = get_loss_op(secret_batch, predictions, cover_batch, predictions, BETA)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            current_step.assign(step)  # Update the current step
            manager.save()
            # Logging
            #with summary_writer.as_default():
                #tf.summary.scalar('loss', loss, step=epoch * (len(files_list) // BATCH_SIZE) + step)

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.numpy()}")

        current_epoch.assign(epoch + 1)  # Update the current epoch
        current_step.assign(0)  # Reset step at the start of each new epoch

    if verbose:
        print("Training completed!")

# ## Network Definitions
# The three networks are identical in terms of structure.
# 
# 1. The Prepare network takes in the **Secret Image** and outputs a (BATCH_SIZE,INPUT_HEIGHT,INPUT_WEIGHT,150) tensor.
# 
# 2. The Cover network takes in the output from 1. , and a *Cover Image*. It concatenates these two tensors , giving a (BATCH_SIZE,INPUT_HEIGHT,INPUT_WEIGHT,153) tensor. Then it performs Convolutions , and outputs a (BATCH_SIZE,INPUT_HEIGHT,INPUT_WEIGHT,3) image.
# 
# 3. The Reveal Network Takes in the output image from Cover Network , and outputs the Revealed Image (which is supposed to look like the **Secret Image**
# 

# Data loading
def train_model():
    imgNett_files_list = glob.glob(join(TRAIN_PATH + "imagenette/", "*"))
    Linn_files_list = glob.glob(join(TRAIN_PATH + "Linnaeus/**/*", "*.jpg"), recursive=True)
    files_list = imgNett_files_list + Linn_files_list

    data_generator(files_list)
    
    train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 224, 224, 3], [None, 224, 224, 3]))
    train_dataset = train_dataset.take(len(files_list) // BATCH_SIZE)
    
    runFullNetwork(train_dataset)


def data_generator(files_list):
   while True:
       yield get_img_batch(files_list, BATCH_SIZE)



# # Network Training



#runAbbreviatedNetwork()

#Author : Guzman
#Last Updated: April 2022

#Calculate the STego performance metrics, such as PSNR, SSIM, MSE, etc.
#10 pairs of cover secret --> 10 oairs of hidden revealed
#The images has been created already and saved in METRIC_TESTING_IMAGES_PATH

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
        coverImg =    img_as_float(Image.open(METRIC_TESTING_IMAGES_PATH + str(imageNumber) + "_cover.png"))
        hiddenImg =   img_as_float(Image.open(METRIC_TESTING_IMAGES_PATH + str(imageNumber) + "_hidden.png"))
        stegoImg =   img_as_float(Image.open(METRIC_TESTING_IMAGES_PATH + str(imageNumber) + "_stego.png"))
        extractedImg = img_as_float(Image.open(METRIC_TESTING_IMAGES_PATH + str(imageNumber) + "_uncovered.png"))

        print("Test results #" + imageNumber)

        #SSIM
        coverAndHiddenSSIM = ssim(coverImg, stegoImg, multichannel=True)
        secretAndExtractedSSIM = ssim(hiddenImg, extractedImg, multichannel=True)
        avgCoverAndHiddenSSIM = avgCoverAndHiddenSSIM + coverAndHiddenSSIM
        avgSecretAndExtractedSSIM = avgSecretAndExtractedSSIM + secretAndExtractedSSIM
        print("SSIM between cover and hidden image (imperceptibility): " + str(coverAndHiddenSSIM))
        print("SSIM between secret and extracted image (recoverability): " + str(secretAndExtractedSSIM))

        #PSNR
        coverAndHiddenPSNR = psnr(coverImg, stegoImg)
        secretAndExtractedPSNR = psnr(hiddenImg, extractedImg)
        avgCoverAndHiddenPSNR = avgCoverAndHiddenPSNR + coverAndHiddenPSNR
        avgSecretAndExtractedPSNR = avgSecretAndExtractedPSNR + secretAndExtractedPSNR
        print("PSNR between cover and hidden image (imperceptibility): " + str(coverAndHiddenPSNR))
        print("PSNR between secret and extracted image (recoverability): " + str(secretAndExtractedPSNR))

        #MSE
        coverAndHiddenMSE = mean_squared_error(coverImg, stegoImg)
        secretAndExtractedMSE = mean_squared_error(hiddenImg, extractedImg)
        avgCoverAndHiddenMSE = avgCoverAndHiddenMSE + coverAndHiddenMSE
        avgSecretAndExtractedMSE = avgSecretAndExtractedMSE + secretAndExtractedMSE
        print("MSE between cover and hidden image (imperceptibility): " + str(coverAndHiddenMSE))
        print("MSE between secret and extracted image (recoverability): " + str(secretAndExtractedMSE))

    #NCC
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
    print("Average SSIM between cover and hidden images: " + str(avgCoverAndHiddenSSIM / numberOfImages))
    print("Average SSIM between secret and extracted images: " + str(avgSecretAndExtractedSSIM / numberOfImages))
    print("Average PSNR between cover and hidden images: " + str(avgCoverAndHiddenPSNR / numberOfImages))
    print("Average PSNR between secret and extracted images: " + str(avgSecretAndExtractedPSNR / numberOfImages))
    print("Average MSE between cover and hidden images: " + str(avgCoverAndHiddenMSE / numberOfImages))
    print("Average MSE between secret and extracted images: " + str(avgSecretAndExtractedMSE / numberOfImages))
    print("Average NCC between cover and hidden images: " + str(avgCoverAndHiddenNCC / numberOfImages))
    print("Average NCC between secret and extracted images: " + str(avgSecretAndExtractedNCC / numberOfImages))


#Testing the Model
#Calculate the network metrics, such as PSNR, SSIM, MSE, etc.
#Created by Anthony
def test_model():
    coverImgs = ["Baboon", "Berries", "Chainsaws", "Church", "Dog", "Fish", "French Horn", "Garbage Truck", "Gas Pump", "Golf Balls"]
    hiddenImgs = ["Graffiti", "Karaoke", "Lena", "Lotus", "Parachute", "Parrot", "Pens", "Peppers", "Stained Glass", "Thistle"]
    fileType = ".png"

    for x in range(10):

        coverImgFileName = "/content/drive/MyDrive/Yashi/Test Images/" + coverImgs[x] + fileType
        stegoImgFileName = "/content/drive/MyDrive/Yashi/Test Images//" + coverImgs[x] + fileType
        hiddenImgFileName = "/content/drive/MyDrive/Yashi/Test Images//" + hiddenImgs[x] + fileType
        extractedImgFileName = "/content/drive/MyDrive/Yashi/Test Images//" + hiddenImgs[x] + fileType

        #coverImgFileName = "/content/drive/MyDrive/Colab Notebooks/testing_images/Original Version Output Cover"  + fileType
        #stegoImgFileName = "/content/drive/MyDrive/Colab Notebooks/testing_images/Original Version Output Stego"  + fileType
        #hiddenImgFileName = "/content/drive/MyDrive/Colab Notebooks/testing_images/Original Version Output Hidden"  + fileType
        #extractedImgFileName = "/content/drive/MyDrive/Colab Notebooks/testing_images/Original Version Output Extracted"  + fileType

        coverImg =     img_as_float(Image.open(coverImgFileName))
        stegoImg =     img_as_float(Image.open(stegoImgFileName))
        hiddenImg =    img_as_float(Image.open(hiddenImgFileName))
        extractedImg = img_as_float(Image.open(extractedImgFileName))

        #print("Cover: " + coverImgs[x] +  "     Hidden: " + hiddenImgs[x])

        OriginalNetworkStegoSSIM = ssim(coverImg, stegoImg, multichannel=True)
        print("SSIM between cover and stego image: " + str(float(format(OriginalNetworkStegoSSIM, '.5f'))))

        #PSNR
        OriginalNetworkStegoPSNR = psnr(coverImg, stegoImg)
        #OriginalNetworkRecoveredPSNR = psnr(hiddenImg, extractedImg)
        print("PSNR between cover and stego image: " + str(float(format(OriginalNetworkStegoPSNR, '.5f'))))
        #print("PSNR between hidden and extracted image: " + str(OriginalNetworkRecoveredPSNR))

        #MSE
        #coverAndHiddenMSE = mean_squared_error(coverImg, stegoImg)
        OriginalNetworkRecoveredSSIM = ssim(hiddenImg, extractedImg, multichannel=True)
        print("SSIM between hidden and extracted image: " + str(float(format(OriginalNetworkRecoveredSSIM, '.5f'))))
        secretAndExtractedMSE = mean_squared_error(hiddenImg, extractedImg)
        #print("MSE between cover and stego image: " + str(coverAndHiddenMSE))
        print("MSE between hidden and extracted image: " + str(float(format(secretAndExtractedMSE, '.5f'))))
        print("")


def preprocess_image(img_path):
    # Open the image
    img = Image.open(img_path)

    # If the image has an alpha (transparency) channel, remove it
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image
    img = img.resize((224, 224))

    return img_as_float(img)

# def calculate_performance_metrics_on_sample_image_pairs():
#     results_list = []

#     coverImgs = ["Baboon", "Berries", "Chainsaws", "Church", "Dog", "Fish", "French Horn", "Garbage Truck", "Gas Pump", "Golf Balls"]
#     hiddenImgs = ["Graffiti", "Karaoke", "Lena", "Lotus", "Parachute", "Parrot", "Pens", "Peppers", "Stained Glass", "Thistle"]
#     fileType = ".png"

#     for x in range(10):
#         coverImgFileName = "/content/drive/MyDrive/Yashi/Test Images/" + coverImgs[x] + fileType
#         hiddenImgFileName = "/content/drive/MyDrive/Yashi/Test Images/" + hiddenImgs[x] + fileType

#         coverImg = preprocess_image(coverImgFileName)
#         hiddenImg = preprocess_image(hiddenImgFileName)

#         # Generate stego and extracted images
#         stegoImg = sess.run(deploy_hide_image_op, feed_dict={"input_prep:0": [hiddenImg], "input_hide:0": [coverImg]})
#         extractedImg = sess.run(deploy_reveal_image_op, feed_dict={"deploy_covered:0": stegoImg})

#         # Display images
#         images = [coverImg, hiddenImg, stegoImg[0], extractedImg[0]]
#         titles = ["Cover", "Secret", "Stego", "Revealed"]

#         for img, title in zip(images, titles):
#             plt.axis('off')
#             plt.tight_layout(pad=0)
#             plt.imshow(img.squeeze())
#             plt.title(title)
#             plt.show()

#         # Compute metrics
#         coverAndStegoPSNR = psnr(coverImg, stegoImg[0])
#         print("PSNR between cover and stego image (imperceptibility): " + str(coverAndStegoPSNR))

#         secretAndExtractedSSIM = ssim(hiddenImg, extractedImg[0], multichannel=True)
#         print("SSIM between secret and extracted image (recoverability): " + str(secretAndExtractedSSIM))

#         # Append results to results_list
#         results_list.append(["SteGuz", "Stego", coverImgs[x], coverAndStegoPSNR, "N/A", "No"])
#         results_list.append(["SteGuz", "Extracted", hiddenImgs[x], "N/A", secretAndExtractedSSIM, "No"])

#     # Convert results_list to DataFrame and save to CSV
#     df = pd.DataFrame(results_list, columns=['Model', 'ImageType', 'ImageName', 'PSNR', 'SSIM', 'Noise Layer'])
#     df.to_csv("/content/drive/MyDrive/Yashi/Saved-Stego-Models/Results_P75.csv", index=False)

