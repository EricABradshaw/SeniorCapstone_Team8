const tf = require('@tensorflow/tfjs');

async function loadModel(){
    const modelPath = "./js_model/model.json"
    const model = await tf.loadLayersModel(modelPath)
    console.log("model loaded")
}