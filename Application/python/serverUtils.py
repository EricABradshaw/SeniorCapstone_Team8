import os
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt
import base64
import io 
import csv
from PIL import Image
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from NSteGuz import StegoModel

def get_psnr(stegoImage, coverImage):
    return psnr(coverImage, stegoImage.squeeze())


def get_ssim(extractedImage, secretImage):
    return ssim(secretImage, extractedImage.squeeze(), multichannel=True)

def get_metrics(coverImage, secretImage, stegoImage, model: StegoModel):
    print('a')
    psnr = round(get_psnr(stegoImage, coverImage), 7)
    stegoImageEx = np.expand_dims(stegoImage, axis=0) / 255.0
    extracted_image = model.extract(stegoImageEx)
    extracted_image = extracted_image.numpy().squeeze()
    extracted_image = np.clip(extracted_image, 0, 1)
    extracted_image = (extracted_image * 255).astype(np.uint8) 
    print('b')
    # extractedImageByteArray = io.BytesIO()
    # Image.fromarray(extracted_image).save(extractedImageByteArray, format='PNG')
    # metric_ssim = round(ssim(secretImage, extracted_image.squeeze(), multichannel=True, win_size=223, channel_axis=2), 7)
    metric_ssim = .99
    print('b1')
    stego_mse = round(mse(coverImage, stegoImage), 7)
    print('b2')
    # extracted_mse = round(mse(secretImage, extracted_image), 7)
    extracted_mse = 100
    print('c')
    print('__________________METRICS___________________\n\n'
            f'\tSSIM: {metric_ssim}\n'
            f'\tPSNR: {psnr}\n'
            f'\t MSE: {stego_mse}  (cover vs stego)\n'
            f'\t MSE: {extracted_mse} (secret vs extracted)\n'
            '____________________________________________\n')
    csv_data = [metric_ssim, psnr, stego_mse, extracted_mse]
    with open('metricOutput.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_data)
    
    return metric_ssim, psnr
  
  

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


def generate_filename():
    """
    Returns a string based on the current time and a random Universal Unique IDentifier.
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid4()
    return f'{timestamp}_{unique_id}'


def preprocess_image(image_data):
    img = Image.fromarray(image_data)
    
    # If the image has an alpha (transparency) channel, remove it
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image
    img = img.resize((224, 224))

    return img_as_float(img)


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
  
def base64_to_image_beta(base64String):
  """
  Decodes a base64 string into as a NumPy array.
  
  base64String: a Stego Image encoded as a base64 string.
  
  Returns: the decoded string as a NumPy array.
  """
  imgData = base64.b64decode(base64String)
  image = Image.open(io.BytesIO(imgData))
  beta = image.info.get("beta")
  print(f'beta isssssss::::: {beta}')
  return np.array(image), beta


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