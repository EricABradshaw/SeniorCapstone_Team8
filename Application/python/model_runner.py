import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from SteGuz import deploy_hide_image_op, deploy_reveal_image_op, sess, saver, preprocess_image, verbose
import os
import argparse
import time
import uuid

# Suppress deprecation warnings and other outputs
if not verbose:
    import logging
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
if not verbose:
    logging.getLogger('tensorflow').disabled = True
    import warnings
    warnings.filterwarnings('ignore')


def generate_filename():
    """
    Returns a string based on the current time and a random Universal Unique IDentifier.
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid4()
    return f'{timestamp}_{unique_id}'


def output_png(outputDir, filename, image):
    """
    Saves Stego Image output to the specified path (if not testing, to /Application/temp)
    """
    image = image.squeeze()
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    #plt.imsave(os.path.join(outputDir, filename), np.reshape(image.squeeze(), (224, 224, 3)))
    plt.imsave(os.path.join(outputDir, filename), image)


def open_image(imagePath):
    """
    Open the provided .png and convert it to a NumPy array.

    imagePath: The file path of the image.

    Returns: the opened image transformed into a NumPy array.
    """
    with Image.open(imagePath) as img:
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
            
        return img_array


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


def extract_hidden_image(index=0, stegoImage=None):
    """
    Runs a SteGuz model on the provided Stego Image and
    extracts the hidden image.
    
    index: Which model to use.
    stegoImage: Path to the stegoImage JSON (.png)
    
    Returns: the name of the file containing the extracted hidden image.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Models are always found in /Application/models/
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    
    # Get a list of the full paths to each model.
    modelPaths = get_model_paths(modelsDir)
   
    # File outputs will always be in /Application/temp/
    outputDir = os.path.join(os.path.dirname(cwd), 'temp')

    # Model to use = index selected in the ComboBox
    inputModelPath = modelPaths[index]

    # Load the model
    try:
        saver.restore(sess, inputModelPath)
        tf.train.load_checkpoint(inputModelPath)
        #if verbose:
        #    print(f'Successfully loaded model from {inputModelPath}.')
    except:
        print('This model cannot be restored or does not exist.')

    # Open the Stego Image
    stegoImage = open_image(stegoImage)

    # Extract the Hidden Image using the loaded model
    extractedImage = sess.run(deploy_reveal_image_op,
                              feed_dict={"deploy_covered:0": stegoImage})
    
    # Limit the values to between 0 and 1
    extractedImage = np.clip(extractedImage, 0, 1)
        
    outputFilename = generate_filename() + '.png'
    output_png(outputDir, outputFilename, extractedImage)
    
    # Print filename to the console so front-end knows which image to grab.
    print(outputFilename)
    

def create_stego_image(index=0, coverImage=None, secretImage=None):
    """
    Runs a SteGuz model with a provided cover and secret image to create a 
    Stego Image.

    index: Which model to use.
    coverImage: Path to a .png image.
    secretImage: Path to a .png image.

    Returns: the name of the file containing the created Stego Image.
    """
    
    # open everything in imageList(png? json?)
    # mylist = []
    # mylist.append(123234)
    # mylist.append('fart')
    # once everything's opened, append it to a list
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Models are always found in /Application/models/
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    
    # Get a list of the full paths to each model.
    modelPaths = get_model_paths(modelsDir)
   
    # File outputs will always be in /Application/temp/
    outputDir = os.path.join(os.path.dirname(cwd), 'temp')

    # Model to use = index selected in the ComboBox
    inputModelPath = modelPaths[index]

    # Load the model
    try:
        saver.restore(sess, inputModelPath)
        tf.train.load_checkpoint(inputModelPath)
        #if verbose:
        #    print(f'Successfully loaded model from {inputModelPath}.')
    except:
        print('This model cannot be restored or does not exist.')

    # Preprocess the images
    coverImagePreproc = preprocess_image(coverImage)
    secretImagePreproc = preprocess_image(secretImage)

    # Generate the Stego Image using the loaded model
    stegoImage = sess.run(deploy_hide_image_op,
                          feed_dict={"input_prep:0": [secretImagePreproc], "input_hide:0": [coverImagePreproc]})
    
    # Limit the values to between 0 and 1
    #stegoImage = np.clip(stegoImage, 0, 1)
    
    
    outputFilename = generate_filename() + '.png'
    output_png(outputDir, outputFilename, stegoImage)
    
    # Print filename to the console so front-end knows which image to grab.
    print(outputFilename)
    

if __name__ == '__main__':
    # Parse all the input args
    parser = argparse.ArgumentParser(description='Set input parameters for SteGuz model.')
    
    # Which model to use - required for all commands
    parser.add_argument('--index', type=int, default=-1, help='Which model to use.')
    
    # Additional required parameters for creating a Stego Image
    parser.add_argument('--coverImage', type=str, default=None, help='Cover Image path.')
    parser.add_argument('--secretImage', type=str, default=None, help='Secret Image path.')
    
    # Additional required parameters for extracting a Secret Image
    parser.add_argument('--stegoImage', type=str, default=None, help='Stego Image path.')



    args = parser.parse_args()

    print('howdyu')
    
    if args.index == -1:
        print('Error: no model selected.')
        exit()

    # Look at what CLAs were provided to determine which function to run.
    if args.coverImage is not None and args.secretImage is not None:
        # Run the model with the parsed args
        create_stego_image(args.index, args.coverImage, args.secretImage)
    elif args.stegoImage is not None:
        extract_hidden_image(args.index, args.stegoImage)


    
    
    
    
