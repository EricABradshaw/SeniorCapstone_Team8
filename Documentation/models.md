# Deploying CNN-Driven Image Steganography Techniques in a Web Application

## Models Folder
This markup document is to describe the models folder located in `./Application` of the capstone project. This folder is designated to hold the pre-trained SteGuz models that are all trained on different betas values. 

**Beta Value**
A beta value in the context of image-to-image steganography refers to if you want to have a better stego-image (Image created after hiding process of hiding a secret image in a cover image) or a better extracted image. Beta will be between 0 and 1.
- Higher beta value = Better extracted image (and in turn, worse stegoimage)
- Lower Beta value = Better stegoimage (and in turn, worse extracted image)

The pre-trained models are typically trained over a day and a half, then loaded into the application.

### Beta Slider
The beta slider allows for user's to define their own beta during the hiding process. The closest pre-trained model will then be sleected based on where the beta slider currently is. 

The beta value also be saved during the hiding process into the metadata of the created stegoimage. When the user extracts a secret image from a stegoimage, the correct models wil be selected based on the metadata.

# Properties of Models
There are currently three pre-trained models:
- b0.375_StegoModel
  - beta of 0.375
- b0.50_stegoModel
  - beta of 0.50
- b0.75_StegoModel
  - beta of 0.75

# Choosing the Correct Model
A model chossen closest to the current beta value is done within the `serverUtils.py` program with the `./python` folder.

**serverUtils.py:**
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
- Once closest beta is evaluted from beta slider, the model is then prepared that has the most appropriate beta value
- Returns the closest beta value model path and the closest beta value