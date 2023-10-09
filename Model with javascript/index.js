const tf = require('@tensorflow/tfjs');
const sharp = require('sharp');
const fs = require('fs');
const fetch = require('node-fetch'); // Import the 'node-fetch' library for making HTTP requests in Node.js

const express = require('express');
const app = express();
const port = 1234; // Choose a port number

// Serve your model files from a directory
app.use(express.static('js_model'));

// Load the TensorFlow.js model
async function loadModel() {
  try {
    const model = await tf.loadLayersModel(`http://localhost:3000/model.json`);
    return model;
  } catch (error) {
    console.error('Error loading the model:', error);
    throw error;
  }
}

// Preprocess an image using sharp and convert to a TensorFlow tensor
async function preprocessImage(imagePath, height, width) {
  try {
    console.log('Image path:', imagePath);

    // Use 'sharp' to resize and convert the image to grayscale
    const imageBuffer = await sharp(imagePath)
      .resize({ height, width })
      .greyscale()
      .toBuffer();

    // Convert the imageBuffer to a TensorFlow tensor
    const imageTensor = tf.node.decodeImage(imageBuffer, 1); // '1' represents grayscale
    const reshapedImage = imageTensor.expandDims(0); // Add batch dimension
    return reshapedImage;
  } catch (error) {
    console.error('Error preprocessing image:', error);
    throw error;
  }
}

// Make predictions using the loaded model and preprocessed image
async function makePredictions(model, preprocessedImage) {
  try {
    const predictions = model.predict(preprocessedImage);
    return predictions;
  } catch (error) {
    console.error('Error making predictions:', error);
    throw error;
  }
}

// Example usage
const imagePath = './t_upper.png';
const height = 28;
const width = 28;

(async () => {
  try {
    // Load the model
    const model = await loadModel();

    // Preprocess the image
    const preprocessedImage = await preprocessImage(imagePath, height, width);

    // Make predictions
    const predictions = await makePredictions(model, preprocessedImage);

    // Print predictions
    predictions.print();
  } catch (error) {
    console.error('An error occurred:', error);
  }
})();

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
