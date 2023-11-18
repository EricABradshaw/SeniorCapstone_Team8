import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from SteGuz import deploy_hide_image_op, deploy_reveal_image_op, sess, saver, preprocess_image, verbose
import os
import base64
import io
import json 
# suppress deprecation warnings and other outputs
if not verbose:
    import logging
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
if not verbose:
    logging.getLogger('tensorflow').disabled = True
    import warnings
    warnings.filterwarnings('ignore')


def image_to_base_64(imageArray):
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

# use temp .json files or .pngs instead of raw strings
def run_model(index=0, stegoBase64=None):
    """
    Runs a SteGuz model with a provided cover and secret image to create a 
    Stego Image.

    index: Which model to use.
    stego: a Stego Image created by the model specified in index.

    Returns: the secret image hidden in the Stego image.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    parentDir = os.path.dirname(cwd)
    modelsDir = os.path.join(parentDir, 'models')
    modelPaths = get_model_paths(modelsDir)

    testing = True
    outputDir = os.path.join(cwd, 'Test Output')

    # model to use = index of the ComboBox
    inputModelPath = modelPaths[index]
    if verbose:
        print(f'Found {len(modelPaths)} models:')
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

    stegoImage = base64_to_image(stegoBase64)
    
    # Extract Hidden Image
    extractedImage = sess.run(deploy_reveal_image_op,
                              feed_dict={"deploy_covered:0": stegoImage})
    if testing:
        extractedImage = np.clip(extractedImage, 0, 1)
        plt.imsave(os.path.join(outputDir, "test_stego_extraction.png"),
                   np.reshape(extractedImage.squeeze(), (224, 224, 3)))
    else:
        print('secret image extracted successfully.')
        # plt.imsave(os.path.join(outputDir, "test_stego_extraction.png"), np.reshape(extractedImage.squeeze(), (224, 224, 3)))
        # extractedImage = np.clip(extractedImage, 0, 1)
        # extractedImage = np.reshape(extractedImage.squeeze(), (224, 224, 3))
        extractedImage_base64 = image_to_base_64(extractedImage)
        print(json.dumps({"image": extractedImage_base64}))

    # Calculate metrics
    # TODO

run_model(0)
