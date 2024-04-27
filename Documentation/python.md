# Deploying CNN-Driven Image Steganography Techniques in a Web Application

## Python Folder
This markup describes the back-end Flask server located in `./Application` of the capstone project.
This folder is designated to handle methods and requests to our flask server, defined in two python programs:
1. server.py
2. serverUtils.py

### **server.py**
**extract_hidden_image**
`POST`: Post Request for `/extract_hidden_image`
The extract_hidde_image method handles the extraction process (extraction page) of the capstone project:
```python
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
        
        model = None
        
        if beta >= 0.65:
            model = model_75
        elif beta >= 0.45:
            model = model_50
        else:
            model = model_375
        
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

```
- If no beta is found in the stegoImage header, the model with 0.75 beta is selected.
- If beta is found in the stegoImage header, the beta value is the assigned and validated.
- Based on beta, the correct trained model is selected.
- The extraction process from the chossen model is used to extract the secretImage from the stegoImage.
- The extractedImage (which is will be similar to the orinigal secretImage) is cleaned then convereted to a byte stream, then base64 representation in order to be returned to the front-end.
- Returns the base64 string extractedImage
  
**create_stego_image**
`POST`: Post Request for `/extract_stego_image_b64`
The create_stego_image is used during the hiding process (hiding page) of the capstone project:
```python
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
        
        # Preprocess the images 
        coverImagePreproc = preprocess_image(coverImage).astype(np.float32)
        secretImagePreproc = preprocess_image(secretImage).astype(np.float32)
        print(8)
        # Generate the Stego Image using the loaded model
        stegoImage = model.hide((
            np.expand_dims(secretImagePreproc, axis=0),
            np.expand_dims(coverImagePreproc, axis=0)
        ))
        print(9)
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
        print(10)
        # Get metrics
        ssim, psnr = get_metrics(coverImage, secretImage, stegoImage, model)
        print(11)
        # Return the stego image
        return jsonify({
                        "message":"Success",
                        "stegoImage":stegoImageBase64,
                        "ssim": ssim,
                        "psnr": psnr
                        }), 200
    except Exception as e:
        return jsonify({"error": "Model could not be loaded . Details: " + str(e)}), 500
```
- Will use the beta value from the beta slider on the hiding page on the front-end (default to 0.5 beta)
- Will use the coverImage and secretImage provided by User chossen by the image gallery or by their own upload.
- Load the appropriate model based on the closest beta value chossen:
  1. 0 - 0.44 => Model trained with 0.75 beta
  2. 0.45 - 0.64 => Model trained with 0.5 beta
  3. 0.65 - 1 => Model trained with 0.375 beta
- The coverImage and secretImage are processed into float32 normalized images, then hides the secretImage within the coverImage using the appropriate model's hide() method.
- The stegoImage is cleaned then coverted into a byte stream.
- In the metadata header of the stegoImage, the beta is attached for usage in the extraction process within the extract_hidden_image() method
- stegoImage is converted to base64 string representation before getting SSIM and PSNR perfomrance metrics.
- Returns the stegoImage to the front-end along with the PSNR and SSIM.

**Main block of server.py**
The Flask server port is startedon port 5000. The 3 trained models are initialized and then loaded:
```python
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
```
- The Flask server will run in debug mode.


#### **serverUtils.py**
There are two metrics used to measure the performance of our hiding and extraction techniques:

`PSNR`: Peak signal-noise ratio
- The PSNR measures noise within the stegoimage compared to the cover image

`SSIM`: Structural Similarity Index Measure
- The SSIM measures the similarity (perceived quality) between the extracted image and secret image

**get_psnr and get_ssim**
```python
def get_psnr(stegoImage, coverImage):
    return psnr(coverImage, stegoImage.squeeze())


def get_ssim(extractedImage, secretImage):
    return ssim(secretImage, extractedImage.squeeze(), multichannel=True)
```
- Returns PSNR and SSIM performance metrics respectively.

**get_appropriate_model_path_and_closest_beta**

The get_appropraite_model_path_and_closest_beta method will choose the model with the closest beta value, then prepare the appropriate model in `Application/models/`. This is also detailed in the models.md markup document:
```python
def get_appropriate_model_path_and_closest_beta(beta: str) -> Optional[Tuple[str, float]]:
    beta = float(beta)

    # Load the appropriate model based on the provided beta value
    targetBetas = [0.375, 0.50, 0.75]
    closestBeta = min(targetBetas, key=lambda x: abs(x - beta))
    print(f'CLOSEST BETA IS {closestBeta}')

    # Navigate to /Application/models/ and prepare the model with the appropriate beta value
    cwd = os.path.dirname(os.path.abspath(__file__))
    modelsDir = os.path.join(os.path.dirname(cwd), 'models')
    betaFolderName = f"b{closestBeta:.3f}"  # Use 3 decimal places to avoid rounding if necessary
    betaFolderName = betaFolderName.rstrip('0').rstrip('.') if '.' in betaFolderName else betaFolderName  # Remove trailing zeros and dot if no decimal part
    print(f'Attempting to load model with {betaFolderName}')

    modelFolder = glob.glob(os.path.join(modelsDir, betaFolderName + "*"))
    if not modelFolder:
        return None
    inputModelPath = modelFolder[0]
    
    return inputModelPath, closestBeta
```
- The target betas are all the beta of the current pre-trained models
- Returns the closest beta value model path and the closest beta value

**get_metrics**

The get_metrics function calculates the performance metrics and loss function. This method prints the PSNR (Peak Signal-to-Noise Ratio), the SSIM (Structural Similarity Index), and MSE (Mean Squared Error). This metrics are calculated between the coverImage, secretImage, stegoImage, and the extractedImage:
```python
def get_metrics(coverImage, secretImage, stegoImage, model: StegoModel):
```
- Prints to the Flask server the SSIM, PSNR, MSE of the coverimage and secretImage, and MSE of the secretImage and the extractedImage.
- The data calculated is then prepared and written to a .csv file for record documentation
- Returns the calculated metric_SSIM and PSNR

**preprocess_image**
```python
def preprocess_image(image_data: np.ndarray) -> np.ndarray:
    img = Image.fromarray(image_data)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))

    return img_as_float(img)
```
- Converts a numpy array contraining image data in the format RGBA into the 3 channels RGB
- If the image is a transparent image, the image is converted to a regular RBH image.
- Image is sized to 224x244 (for the model).
- Returns the preprocessed image as a float.

**base64_to_image**
The base64_to_image method coverts a base64 string representation to an image:
```python
def base64_to_image(base64String: str, getBeta: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Union[str, None]]]:
    imgData = base64.b64decode(base64String)
    image = Image.open(io.BytesIO(imgData))
    if getBeta:
        beta = image.info.get("beta")
        print(f'Found beta value of: {beta}')
        return np.array(image), beta
    else:
        return np.array(image)
```
- Decodes a base64 string (A created stegoImage is encoded as a base64 string) into as a NumPy array.
- If there a beta vaue attached within the metadata header, it will print to the server the beta value and then return the decoded string as a NumPy array and the beta value.
- Otherwise, returns the decoded string as a NumPy array.

**image_to_base64**
The image_to_base64 method converts an image to a base64 string representation:
```python
    imageBase64 = base64.b64encode(imageArray)
    
    return imageBase64
```
- Converts a NumPy array that contains image data into an image encoded in base64.
- Returns the image encoded in base64