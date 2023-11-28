import os
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt


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