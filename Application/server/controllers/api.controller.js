/* Communicate with Flask server */
const axios = require('axios')
const sharp = require('sharp') // image processing
const FLASK_SERVER_URL = 'http://127.0.0.1:5000' // put in env

const sendRequestsController = {
  // Sending a post request to Flask server for creating stego 
  hideSend: async (req, res) => {
    try {
      const coverImageSource = req.body.coverImageData
      console.log(coverImageSource)
      const secretImageSource = req.body.secretImageData
      console.log(secretImageSource)
      const modelType = req.body.sliderValue
      let base64Strings = await helperFunctions.fetchAndConvert(coverImageSource, secretImageSource)
      let stego64String = await helperFunctions.sendToFlask(base64Strings, modelType)
      res.status(200).json({ stegoImage: stego64String })
    } catch (err) {
      console.log(err)
    }
  },
  extractSend: async (base64) => {
    try {
      const response = await axios.post(FLASK_SERVER_URL + '/extract_hidden_image', base64, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      console.log("From Flask: " + response.status)
      return response.data.hiddenImage
    } catch (error) {
      console.log('Error: ' + error)
    }
  },
  hideTextSend: async (req, res) => {
    try {
      const coverImageSource = req.body.coverImageData
      const secretBase64 = req.body.textDataUrl
      let strings = await helperFunctions.fetchAndConvertOne(coverImageSource)
      strings.secretString = secretBase64
      const modelType = req.body.sliderValue
      let stego64String = await helperFunctions.sendToFlask(strings, modelType)
      res.status(200).json({ stegoImage: stego64String})
    } catch (error) {
      console.log(error)
    }
  }
}


const helperFunctions = {
  // Takes two image URLs and fetches the images
  // then converts each image to a png
  // returns an object { coverBuffer: ..., secretBuffer: ... }
  fetchAndConvert: async (coverImageUrl, secretImageUrl) => {
    let pngStrings = {}
    try {
      const axiosResponseCover = await axios.get(coverImageUrl, {responseType: 'arraybuffer'})
      const axiosResponseSecret = await axios.get(secretImageUrl, {responseType: 'arraybuffer'})
      pngStrings.coverString = await (await sharp(axiosResponseCover.data).toFormat('png').toBuffer()).toString('base64')
      pngStrings.secretString = await (await sharp(axiosResponseSecret.data).toFormat('png').toBuffer()).toString('base64')
    } catch (error) {
      console.error('Error:', error)
    }
    return pngStrings
  },
  fetchAndConvertOne: async (imageUrl) => {
    let pngStrings = {}
    try {
      const axiosResponseCover = await axios.get(imageUrl, {responseType: 'arraybuffer'})
      pngStrings.coverString = await (await sharp(axiosResponseCover.data).toFormat('png').toBuffer()).toString('base64')
    } catch (error) {
      console.error('Error:', error)
    }
    return pngStrings
  },
  // Sends the object containing two png buffers to Flask server
  sendToFlask: async (data, model) => {
    try {
      data.beta = model;
      let imageData
      await axios.post(FLASK_SERVER_URL + '/create_stego_image_b64', data, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
        .then(async response => {
          console.log('Success:',response.status)
          returnedData = await response.data
          imageData = returnedData.stegoImage
          ssim = returnedData.ssim
          psnr = returnedData.psnr
          console.log(`${returnedData.ssim} ${returnedData.psnr}`)
          resData = {
            "imageData": imageData,
            "ssim": ssim,
            "psnr": psnr
          }
        })
        return resData

    } catch (err) {
      console.error(err)
    }
  }

}

module.exports = {
  sendRequestsController,
  helperFunctions
}