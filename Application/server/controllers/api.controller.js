/* Communicate with Flask server */
const axios = require('axios')
const sharp = require('sharp') // image processing
const FormData = require('form-data')
const FLASK_SERVER_URL = 'http://127.0.0.1:5000' // put in env

const sendRequestsController = {
  // Sending a post request to Flask server for creating stego 
  hideSend: async (req, res) => {
    try {
      const coverImageSource = req.body.coverImageData
      const secretImageSource = req.body.secretImageData
      console.log(`Request received from ${req.get('origin')}`)
      let base64Strings = await helperFunctions.fetchAndConvert(coverImageSource, secretImageSource)
      helperFunctions.sendToFlask(base64Strings)
      res.send('Response')
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
      console.log(pngStrings.coverString.slice(85, 105))
      console.log(pngStrings.secretString.slice(85, 105))
    } catch (error) {
      console.error('Error:', error)
    }
    return pngStrings
  },
  // Sends the object containing two png buffers to Flask server
  sendToFlask: async (data) => {
    try {
      const response = await axios.post(FLASK_SERVER_URL + '/create_stego_image', data, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      console.log("From Flask: " + response.status + response.message)
    } catch (err) {
      console.error(err)
    }

  }

}

// TODO : Create controller for user model
const userController = {
  // create stego image (call flask server /create_stego_image)
  // createStegoImage: async (req, res) => {
  //   try {
  //     const formData = new FormData()

  //     // prepare data for sending to flask server
  //     formData.append('coverImage', req.files.coverImage[0].buffer, {
  //       filename: 'coverImage.png',
  //       contentType: 'image/png',
  //     })
  //     formData.append('secretImage', req.files.secretImage[0].buffer, {
  //       filename: 'secretImage.png',
  //       contentType: 'image/png',
  //     })
  //     formData.append('index', req.body.index)

  //     // make request to flask server
  //     const flaskResponse = await axios.post(`${FLASK_SERVER_URL}/create_stego_image`, formData, {
  //       headers: {
  //         ...formData.getHeaders(),
  //       },
  //       responseType: 'arraybuffer',
  //     })

  //     // send response back to front-end
  //     res.set('Content-Type', 'image/png')
  //     res.send(flaskResponse.data)
  //   }
  //   catch (error) {
  //     console.error('Error calling Flask server:', error)
  //     res.status(500).send('Internal Server Error')
  //   }


  // },

  // TODO: extract hidden image


  // TODO: get image metrics



}

module.exports = {
  sendRequestsController,
  userController,
  helperFunctions
}