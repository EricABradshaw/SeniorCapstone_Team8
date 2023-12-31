from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
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
tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
if not verbose:
    logging.getLogger('tensorflow').disabled = True
    import warnings
    warnings.filterwarnings('ignore')

Debug = True

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:9000"]}})


# @app.route('/create_stego_image', methods=['POST'])
# def create_stego_image():
#   try:
#     print("Flask request received")
#     cover_image_string = request.json.get('coverString', '')
#     secret_image_string = request.json.get('secretString', '')
    
#     cover_image = base64_to_image(cover_image_string)
#     secret_image = base64_to_image(secret_image_string)
    
#     print("Images received!")
#     return 'Images received!', 200
#   except Exception as e:
#     print(f"Error: {str(e)}")
#     return 'Error!', 500 


@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():
    if 'secretImage' not in request.files:
        return jsonify({"error": "Secret image must be provided"}), 400
    
    if not request.files.getlist('coverImages[]'):
        return jsonify({"error": "No cover images provided"}), 400
    
    secretImageFile = request.files['secretImage']
    secretImageString = io.BytesIO(secretImageFile.read())
    index = request.form.get('index', type=int, default=0)
    
    secretImage = base64_to_image(secretImageString)
    
    # Navigate to /Application/models/ and prepare models/folderIndex/ for loading
    cwd = os.path.dirname(os.path.abspath(__file__))
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    modelPaths = get_model_paths(modelsDir)
    inputModelPath = modelPaths[index]

    # Load the model here since the index being sent in may vary -
    # each index corresponds to a different model.
    metrics_list = []  
    try:
        saver.restore(sess, inputModelPath)
        tf.train.load_checkpoint(inputModelPath)
        
        # Preprocess the secret image
        secretImagePreproc = preprocess_image(secretImage)

        # for each provided cover image...
        for coverImageFile in request.files.getlist('coverImage[]'):
            metrics = []
            
            coverImageString = io.BytesIO(coverImageFile.read())
            coverImage = base64_to_image(coverImageString)
            coverImagePreproc = preprocess_image(coverImage)
            
            # Generate the Stego Image using the loaded model
            stegoImage = sess.run(deploy_hide_image_op,
                                feed_dict={"input_prep:0": [secretImagePreproc], "input_hide:0": [coverImagePreproc]})
        
            # Clean up the image
            stegoImage = stegoImage.squeeze()
            stegoImage = np.clip(stegoImage, 0, 1)
            # stegoImage = (stegoImage * 255).astype(np.uint8)
            
            # we've hidden the image, now get the PSNR
            # any additional preprocessing we need to do?
            psnr = get_psnr(stegoImage, coverImagePreproc)
            metrics.append(psnr)
            
            # extract the image -> get SSIM
            
            # Extract the Hidden Image using the loaded model
            extractedImage = sess.run(deploy_reveal_image_op,
                                    feed_dict={"deploy_covered:0": stegoImage})
            
            # Clean up the image
            extractedImage = np.clip(extractedImage, 0, 1)
                    
            # we've extracted the image, now get the SSIM
            # any additional preprocessing we need to do?
            ssim = get_ssim(extractedImage, secretImage)
            metrics.append(ssim)
            
            metrics_list.append(metrics)
            
        # Return the metrics list
        return jsonify(metrics_list)
    except Exception as e:
        return jsonify({"error": "Model could not be loaded . Details: " + str(e)}), 500


@app.route('/extract_hidden_image', methods=['POST'])
def extract_hidden_image():
    # if 'stegoImage' not in request.files:
    #     return jsonify({"error": "Stego image must be provided"}), 400

    # stegoImageFile = request.files['stegoImage']
    index = request.form.get('index', type=int, default=0)
    data = json.loads(request.data.decode('utf-8'))
    
    stegoImage = base64_to_image(data)
    stegoImage = np.expand_dims(stegoImage, axis=0)
    stegoImage = stegoImage[:, :, :, :3]
    stegoImage = stegoImage / 255.0
    # stegoImage = preprocess_image(stegoImage)
    # Navigate to /Application/models/ and prepare models/folderIndex/ for loading
    cwd = os.path.dirname(os.path.abspath(__file__))
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    modelPaths = get_model_paths(modelsDir)
    inputModelPath = modelPaths[index]
    # Load the model here since the index being sent in may vary -
    # each index corresponds to a different model.
    try:
        with sess.graph.as_default():
          saver.restore(sess, inputModelPath)
        tf.train.load_checkpoint(inputModelPath)

        # Extract the Hidden Image using the loaded model
        extractedImage = sess.run(deploy_reveal_image_op,
                                feed_dict={"deploy_covered:0": stegoImage})

        # Limit the values to between 0 and 1
        # Clean up the image so it's a proper PNG
        extractedImage = extractedImage.squeeze()
        extractedImage = np.clip(extractedImage, 0, 1)
        extractedImage = (extractedImage * 255).astype(np.uint8)

        # Now convert it into a byte stream
        extractedImageByteArray = io.BytesIO()
        Image.fromarray(extractedImage).save(extractedImageByteArray, format='PNG')
        extractedImageByteArray = extractedImageByteArray.getvalue()
        extractedImageBase64 = base64.b64encode(extractedImageByteArray).decode('utf-8')
        # Return the stego image
        # return Response(response=stegoImage, mimetype='image/png')
        return jsonify({"message":"Success", "hiddenImage":extractedImageBase64}), 200
    except Exception as e:
        print(str(e))
        return jsonify({"error": "Model could not be loaded . Details: " + str(e)}), 500


@app.route('/create_stego_image_b64', methods=['POST'])
def create_stego_image():
    tf.enable_eager_execution()
    tf.compat.v1.enable_eager_execution()
    tf.config.run_functions_eagerly(True)
    coverImageString = request.json.get('coverString', '')
    secretImageString = request.json.get('secretString', '')
        
    if Debug:
        print("Flask request received.")
    
    if not coverImageString:
        return 'Error! No cover image provided', 500 
        
    if not secretImageString:
        return 'Error! No secret image provided', 500 
    
    index = 0
    #index = request.form.get('index', type=int, default=0)
    
    #coverImageString = io.BytesIO(coverImageFile.read())
    #secretImageString = io.BytesIO(secretImageFile.read())
    #coverImage.show()
    #secretImage.show()
    
    coverImage = base64_to_image(coverImageString)
    secretImage = base64_to_image(secretImageString)
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    modelPaths = get_model_paths(modelsDir)
    inputModelPath = modelPaths[index]
    # Load the model here since the index being sent in may vary -
    # each index corresponds to a different model.
    try:
        with sess.graph.as_default():
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
        stegoImageBase64 = base64.b64encode(stegoImageByteArray).decode('utf-8')
        # Return the stego image
        # return Response(response=stegoImage, mimetype='image/png')
        return jsonify({"message":"Success", "stegoImage":stegoImageBase64}), 200
    except Exception as e:
        return jsonify({"error": "Model could not be loaded . Details: " + str(e)}), 500
      
if __name__ == '__main__':
    # tensorflow prepping is done when SteGuz.py is imported
    PORT = 5000
    print(f"Flask server running on port {PORT}")
    app.run(debug=Debug,port=PORT)
