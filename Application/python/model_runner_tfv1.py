import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from SteGuz import deploy_hide_image_op, deploy_reveal_image_op, sess, saver, preprocess_image, verbose
import os
import argparse
import time
import uuid
import io
import base64
import json

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

def image_to_base64(imageArray):
    """
    Converts a NumPy array into an image encoded in base64.
    
    imageArray: a NumPy array containing image data.
    
    Returns: an image encoded in base64.
    """
    imagePIL = Image.fromarray((imageArray * 255).astype('uint8'))
    
    # put image in a buffer
    buffer = io.BytesIO()
    imagePIL.save(buffer, format='PNG')
    buffer.seek(0)
    
    imageBase64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return imageBase64


def base64_to_image(base64String):
    """
    Decodes a base64 string into as a NumPy array.
    
    base64String: a Stego Image encoded as a base64 string.
    
    Returns: the decoded string as a NumPy array.
    """
    imgData = base64.b64decode(base64String)
    image = Image.open(io.BytesIO(imgData))
    return np.array(image)


def load_json_images(jsonFile):
    images = []
    
    with open(jsonFile, 'r') as file:
        data = json.load(file)
        
    for b64String in data["images"]:
        image = base64_to_image(b64String)
        images.append(image)
    
    return images


def generate_filename():
    """
    Returns a string based on the current time and a random Universal Unique IDentifier.
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid4()
    return f'{timestamp}_{unique_id}'


def output_png(outputDir, filename, image, isBatch=False):
    """
    Saves Stego Image output to the specified path (if not testing, to /Application/temp)
    """
    image = image.squeeze()
    image = np.clip(image, 0, 1)
    # TODO TODO TODO TODO 
    image = (image * 255).astype(np.uint8) # do this in single stego image creation only?
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
    

def batch_stego_images(index=0, coverImages=None, secretImage=None):
    cwd = os.path.dirname(os.path.abspath(__file__))
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    modelPaths = get_model_paths(modelsDir)
    outputDir = os.path.join(os.path.dirname(cwd), 'temp')
    inputModelPath = modelPaths[index]

    coverImages = []
    data = None
    
    tempDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp\\')
    with open (tempDir + '1700351751_e1e15102-524b-415d-ac85-e4fec3511f43.json', 'r') as file:
        data = json.load(file)
        
    for item in data['images']:
        image = base64_to_image(item)
        image = image / 255.0
        image = 1.0 - image
        coverImages.append(image)
        
    for i in range(len(coverImages)):
        coverImages[i] = coverImages[i].astype(np.float64)
        coverImages[i] = coverImages[i][:, :, [0,1,2]]

    # Load the model
    try:
        saver.restore(sess, inputModelPath)
        tf.train.load_checkpoint(inputModelPath)
    except:
        print('This model cannot be restored or does not exist.')

    # Preprocess the images
    # coverImagesPreproc = []
    # for i in range (len(coverImages)):
    #     coverImagesPreproc.append(preprocess_image(coverImages[i]))
    secretImagePreproc = preprocess_image(secretImage)

    # Generate the Stego Image using the loaded model
    stegoImages = []
    for i in range(len(coverImages)):
        stegoImage = sess.run(deploy_hide_image_op,
                        feed_dict={"input_prep:0": [secretImagePreproc], "input_hide:0": [coverImages[i]]})
        stegoImages.append(stegoImage)
    
    for item in stegoImages:
        outputFilename = generate_filename() + '.png'
        output_png(outputDir, outputFilename, item, isBatch=True)
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
    
    # Get the parent folder (/Application/)
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
    
    outputFilename = generate_filename() + '.png'
    output_png(outputDir, outputFilename, stegoImage)
    
    # Print filename to the console so front-end knows which image to grab.
    print(outputFilename)


# do not use pls
def create_big_json():
    images_base64 = []
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    inputDir = os.path.join(cwd, 'Test Images')
    outputDir = os.path.join(os.path.dirname(cwd), 'temp\\')
    
    for file in os.listdir(inputDir):
        if file.endswith('.png'):
            filepath = os.path.join(inputDir, file)
            img = Image.open(filepath)
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img = np.array(img)
            images_base64.append(image_to_base64(img))
            
    filename = generate_filename()
    with open(outputDir + filename + '.json', 'w') as json_file:
        json.dump({'images': images_base64}, json_file)
        

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

    # Additional required parameters for batch creating Stego Images
    parser.add_argument('--batchCovers', type=str, default=None, help='JSON file containing base64-encoded .PNG images.')

    args = parser.parse_args()

    test_create = True
    if test_create:
        args.index = 0
        args.coverImage = "C:\\Users\\jacob\\Dropbox\\2023 Fall\\SeniorCapstone_Team8\\Application\\python\\Test Images\\Parachute.png"
        args.secretImage = "C:\\Users\\jacob\\Dropbox\\2023 Fall\\SeniorCapstone_Team8\\Application\\python\\Test Images\\Pens.png"

    # Look at what CLAs were provided to determine which function to run.
    if args.index == -1:
        print('Error: no model selected.')
        exit()
    if args.coverImage is not None and args.secretImage is not None:
        # Run the model with the parsed args
        create_stego_image(args.index, args.coverImage, args.secretImage)
    # elif args.stegoImage is not None:
    #     extract_hidden_image(args.index, args.stegoImage)


    
    
    
    
