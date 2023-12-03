import os
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt
import base64
import io 
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_psnr(stegoImage, coverImage):
    return psnr(coverImage, stegoImage.squeeze())


def get_ssim(extractedImage, secretImage):
    return ssim(secretImage, extractedImage.squeeze(), multichannel=True)


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


def generate_filename():
    """
    Returns a string based on the current time and a random Universal Unique IDentifier.
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid4()
    return f'{timestamp}_{unique_id}'


def open_image(imagePath):
    """
    Open the provided .png and convert it to a NumPy array.

    imagePath: The file path of the image.

    Returns: the opened image transformed into a NumPy array.
    """
    img_array = imagePath / 255.0
    img_array = np.expand_dims(img_array, axis=0)
        
    return img_array


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
    
    
def base64_to_image(base64String):
    """
    Decodes a base64 string into as a NumPy array.
    
    base64String: a Stego Image encoded as a base64 string.
    
    Returns: the decoded string as a NumPy array.
    """
    imgData = base64.b64decode(base64String)
    image = Image.open(io.BytesIO(imgData))
    return np.array(image)

def image_to_base64(imageArray):
    """
    Converts a NumPy array into an image encoded in base64.
    
    imageArray: a NumPy array containing image data.
    
    Returns: an image encoded in base64.
    """
    # imageArray = (imageArray * 255).astype(np.uint8)
    # print(0)
    # imagePIL = Image.fromarray(imageArray)
    # print(1)
    # # put image in a buffer
    # buffer = io.BytesIO()
    # print(2)
    # imagePIL.save(buffer, format='PNG')
    # print(3)
    # buffer.seek(0)
    # print(4)
    
    # imageBase64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    imageBase64 = base64.b64encode(imageArray)
    
    return imageBase64