// TODO : Set up BD model import
const path = require('path')

// TODO : Create controllers for different models
const dataController = {
  getData: async (req, res) => {
    try {
      // TODO : Retrieve data from database
    } catch (err) {
      console.error(err)
    }
  },

  createData: async (req, res) => {
    try {
      // TODO : Create document and save to DB
    } catch (err) {
      console.error(err)
    }
  },

  updateData: async (req, res) => {
    try {
      // TODO : Update document and save to DB
    } catch (err) {
      console.error(err)
    }
  }, 

  deleteData: async (req, res) => {
    try {
      // TODO : Delete document from DB
    } catch (err) {
      console.error(err)
    }
  }
}


/* Communicate with Flask server */
const axios = require('axios')
const FormData = require('form-data')
const FLASK_SERVER_URL = 'http://localhost:5000' // put in env

// TODO : Create controller for user model
const userController = {
  // create stego image (call flask server /create_stego_image)
  createStegoImage: async (req, res) => {
    try {
      const formData = new FormData()

      // prepare data for sending to flask server
      formData.append('coverImage', req.files.coverImage[0].buffer, {
        filename: 'coverImage.png',
        contentType: 'image/png',
      })
      formData.append('secretImage', req.files.secretImage[0].buffer, {
        filename: 'secretImage.png',
        contentType: 'image/png',
      })
      formData.append('index', req.body.index)

      // make request to flask server
      const flaskResponse = await axios.post(`${FLASK_SERVER_URL}/create_stego_image`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        responseType: 'arraybuffer',
      })

      // send response back to front-end
      res.set('Content-Type', 'image/png')
      res.send(flaskResponse.data)
    }
    catch (error) {
      console.error('Error calling Flask server:', error)
      res.status(500).send('Internal Server Error')
    }


  },

  // TODO: extract hidden image


  // TODO: get image metrics



}

module.exports = {
  dataController,
  userController
}