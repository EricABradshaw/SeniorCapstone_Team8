from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from NSteGuz import StegoModel
import io
import base64
import json
from serverUtils import *

Debug = True

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:9000"]}})

@app.route('/extract_hidden_image', methods=['POST'])
def extract_hidden_image():  
    data = json.loads(request.data.decode('utf-8')) 
    
    stegoImage, beta = base64_to_image(data, getBeta=True)
    stegoImage = np.expand_dims(stegoImage, axis=0)
    stegoImage = stegoImage[:, :, :, :3]
    stegoImage = stegoImage / 255.0
    if beta is None:
        beta = .75
    # Assign/validate beta value from stegoImage header
    try:
        if not (0 <= float(beta) <= 1):
            raise ValueError('Beta value must be between 0 and 1.')
    except ValueError as e:
        return jsonify({"error": "Invalid beta value. Details: " + str(e)}), 400
    print(f'BETA IS {beta}')

    modelFolder, _ = get_appropriate_model_path_and_closest_beta(beta)
    
    # Load the model
    try:
        print(f'Attempting to load model from {modelFolder}')
        model = StegoModel()
        
        if modelFolder is None:
             raise FileNotFoundError('Unable to find Stego model. Folder does not exist.')
        
        model.load_weights(modelFolder)
        
        # Extract the secret image
        extractedImage = model.extract((stegoImage))
               
        # Limit the values to between 0 and 1; clean up the image so it's a proper PNG in RGB format
        extractedImage = extractedImage.numpy().squeeze()
        extractedImage = np.clip(extractedImage, 0, 1)
        extractedImage = (extractedImage * 255).astype(np.uint8)
        
        # Now convert it into a byte stream for return to the front end
        extractedImageByteArray = io.BytesIO()
        Image.fromarray(extractedImage).save(extractedImageByteArray, format='PNG')
        extractedImageByteArray = extractedImageByteArray.getvalue()
        extractedImageBase64 = base64.b64encode(extractedImageByteArray).decode('utf-8')
        
        # Return the extracted image
        return jsonify({"message":"Success", "hiddenImage":extractedImageBase64}), 200
    except Exception as e:
        print(str(e))
        return jsonify({"error": "Model could not be loaded . Details: " + str(e)}), 500


@app.route('/create_stego_image_b64', methods=['POST'])
def create_stego_image():
    coverImageString = request.json.get('coverString', '')
    secretImageString = request.json.get('secretString', '')

    # Check for beta value sent from front end; default to 0.5 if not provided
    beta = request.json.get('beta', 0.50)/100
    print(f'BETA IS {beta}')
    print(2)
    if Debug:
        print('Flask request received.')
    
    # Get cover/secret image
    if not coverImageString:
        return 'Error! No cover image provided', 500 
        
    if not secretImageString:
        return 'Error! No secret image provided', 500 
    print(3)
    coverImage = base64_to_image(coverImageString)
    secretImage = base64_to_image(secretImageString)
    print(4)
    # Assign/validate beta value
    try:
        if not (0 <= float(beta) <= 1):
            raise ValueError('Beta value must be between 0 and 1.')
    except ValueError as e:
        return jsonify({"error": "Invalid beta value. Details: " + str(e)}), 400

    modelFolder, closestBeta = get_appropriate_model_path_and_closest_beta(beta)
    print(f'modelFolder is {modelFolder}')
    print(f'closestBeta is {closestBeta}')
    # Load the model
    try:
        print(f'Attempting to load model from {modelFolder}')
        
        model = None
        
        if beta >= 0.65:
            model = model_75
        elif beta >= 0.45:
            model = model_50
        else:
            model = model_375
        
        #model = StegoModel()
        #model.load_weights(modelFolder)
        
        # Preprocess the images 
        coverImagePreproc = preprocess_image(coverImage).astype(np.float32)
        secretImagePreproc = preprocess_image(secretImage).astype(np.float32)
        print(8)
        # Generate the Stego Image using the loaded model
        stegoImage = model.hide((
            np.expand_dims(secretImagePreproc, axis=0),
            np.expand_dims(coverImagePreproc, axis=0)
        ))     
        # Clean up the image so it's a proper PNG in RGB format
        stegoImage = stegoImage.numpy().squeeze()
        stegoImage = np.clip(stegoImage, 0, 1)
        stegoImage = (stegoImage * 255).astype(np.uint8)
        # Now convert it into a byte stream
        stegoImageByteArray = io.BytesIO()
        # Add metadata
        metadata = PngInfo()
        metadata.add_text("beta", str(closestBeta))
        Image.fromarray(stegoImage).save(stegoImageByteArray, format='PNG', pnginfo=metadata)
        stegoImageByteArray = stegoImageByteArray.getvalue()
        # Now convert to base64
        stegoImageBase64 = base64.b64encode(stegoImageByteArray).decode('utf-8')
        # Get metrics
        ssim, psnr = get_metrics(coverImage, secretImage, stegoImage, model)
        # Return the stego image
        return jsonify({
                        "message":"Success",
                        "stegoImage":stegoImageBase64,
                        "ssim": ssim,
                        "psnr": psnr
                        }), 200
    except Exception as e:
        return jsonify({"error": "Model could not be loaded . Details: " + str(e)}), 500
    
model_375 = None
model_50 = None
model_75 = None
    
if __name__ == '__main__':
    PORT = 5000
    
    beta = 0.75
    modelFolder, closestBeta = get_appropriate_model_path_and_closest_beta(beta)
    model_75 = StegoModel()
    model_75.load_weights(modelFolder)
    
    beta = 0.375
    modelFolder, closestBeta = get_appropriate_model_path_and_closest_beta(beta)
    model_375 = StegoModel()
    model_375.load_weights(modelFolder)
    
    beta = 0.50
    modelFolder, closestBeta = get_appropriate_model_path_and_closest_beta(beta)
    model_50 = StegoModel()
    model_50.load_weights(modelFolder)
    
    print(f"Flask server running on port {PORT}")
    app.run(debug=Debug,port=PORT)
