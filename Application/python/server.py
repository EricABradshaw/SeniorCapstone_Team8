from flask import Flask, request, jsonify, Response
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from SteGuz import deploy_hide_image_op, deploy_reveal_image_op, sess, saver, preprocess_image, verbose
import os
import argparse
import io
import base64
import json
from serverUtils import *
if not verbose:
    import logging
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
if not verbose:
    logging.getLogger('tensorflow').disabled = True
    import warnings
    warnings.filterwarnings('ignore')

Debug = True

app = Flask(__name__)

@app.route('/create_stego_image', methods=['POST'])
def create_stego_image():
    if 'coverImage' not in request.files or 'secretImage' not in request.files:
        return jsonify({"error": "Both cover and secret images must be provided"}), 400
    
    coverImageFile = request.files['coverImage']
    secretImageFile = request.files['secretImage']
    index = request.form.get('index', type=int, default=0)
    
    coverImage = io.BytesIO(coverImageFile.read())
    secretImage = io.BytesIO(secretImageFile.read())
    
    # Get the parent folder (/Application/)
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Models are always found in /Application/models/
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    
    # Get a list of the full paths to each model.
    modelPaths = get_model_paths(modelsDir)
   
    # Model to use = index selected in the ComboBox
    inputModelPath = modelPaths[index]

    # Load the model here since the index being sent in may vary -
    # each index corresponds to a different model.
    try:
        saver.restore(sess, inputModelPath)
        tf.train.load_checkpoint(inputModelPath)
        
        # Preprocess the images
        coverImagePreproc = preprocess_image(coverImage)
        secretImagePreproc = preprocess_image(secretImage)

        # Generate the Stego Image using the loaded model
        stegoImage = sess.run(deploy_hide_image_op,
                            feed_dict={"input_prep:0": [secretImagePreproc], "input_hide:0": [coverImagePreproc]})
        
        # Clean up the image so it's a proper PNG
        stegoImage = stegoImage.squeeze()
        stegoImage = np.clip(stegoImage, 0, 1)
        stegoImage = (stegoImage * 255).astype(np.uint8)
        
        # Now convert it into a byte stream
        stegoImageByteArray = io.BytesIO()
        Image.fromarray(stegoImage).save(stegoImageByteArray, format='PNG')
        stegoImageByteArray = stegoImageByteArray.getvalue()
        
        # Return the stego image
        return Response(response=stegoImage, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": "Model could not be loaded . Details: " + str(e)}), 500
    
if __name__ == '__main__':
    # tensorflow prepping is done when SteGuz.py is imported
    app.run(Debug)