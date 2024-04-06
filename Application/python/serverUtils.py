import os
import glob
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt
import base64
import io 
import csv
from PIL import Image
from skimage import img_as_float, transform
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from NSteGuz import StegoModel
from typing import Tuple, Optional, Union

def get_appropriate_model_path_and_closest_beta(beta: str) -> Optional[Tuple[str, float]]:
    beta = float(beta)

    # Load the appropriate model based on the provided beta value
    targetBetas = [0.25, 0.50, 0.625, 0.75, 0.85]
    closestBeta = min(targetBetas, key=lambda x: abs(x - beta))
    print(f'CLOSEST BETA IS {closestBeta}')

    # Navigate to /Application/models/ and prepare the model with the appropriate beta value
    cwd = os.path.dirname(os.path.abspath(__file__))
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    betaFolderName = f"b{closestBeta:.2f}"
    print(f'Attempting to load model with {betaFolderName}')
    
    modelFolder = glob.glob(os.path.join(modelsDir, betaFolderName + "*"))
    if not modelFolder:
        return None
    inputModelPath = modelFolder[0]
    
    return inputModelPath, closestBeta


def get_psnr(stegoImage: np.ndarray, coverImage: np.ndarray) -> float:
    return psnr(coverImage, stegoImage.squeeze())


def get_ssim(extractedImage: np.ndarray, secretImage: np.ndarray) -> float:
    return ssim(secretImage, extractedImage.squeeze(), multichannel=True)

def get_metrics(coverImage, secretImage, stegoImage, model: StegoModel):
    if secretImage.shape != coverImage.shape:
        secretImage = secretImage[:,:,:3]
        print(coverImage.shape)
        print(secretImage.shape)
        
    psnr = round(get_psnr(stegoImage, coverImage), 7)
    stegoImageEx = np.expand_dims(stegoImage, axis=0) / 255.0
    extracted_image = model.extract(stegoImageEx)
    extracted_image = extracted_image.numpy().squeeze()
    extracted_image = np.clip(extracted_image, 0, 1)
    extracted_image = (extracted_image * 255).astype(np.uint8) 
  
    metric_ssim = round(ssim(secretImage, extracted_image.squeeze(), multichannel=True, win_size=223, channel_axis=2), 7)
    stego_mse = round(mse(coverImage, stegoImage), 7)
    extracted_mse = round(mse(secretImage, extracted_image), 7)
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

def preprocess_image(image_data: np.ndarray) -> np.ndarray:
    """
    Converts a numpy array containing image data in the format RGBA to RGB.
    """
    img = Image.fromarray(image_data)
    
    # If the image has an alpha (transparency) channel, remove it
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image
    img = img.resize((224, 224))

    return img_as_float(img)
  
    
def base64_to_image(base64String: str, getBeta: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Union[str, None]]]:
    """
    Decodes a base64 string into as a NumPy array.
    
    base64String: a Stego Image encoded as a base64 string.
    
    Returns: the decoded string as a NumPy array.
    """
    imgData = base64.b64decode(base64String)
    image = Image.open(io.BytesIO(imgData))
    if getBeta:
        beta = image.info.get("beta")
        print(f'Found beta value of: {beta}')
        return np.array(image), beta
    else:
        return np.array(image)


def image_to_base64(imageArray: np.ndarray) -> bytes:
    """
    Converts a NumPy array into an image encoded in base64.
    
    imageArray: a NumPy array containing image data.
    
    Returns: an image encoded in base64.
    """
    imageBase64 = base64.b64encode(imageArray)
    
    return imageBase64