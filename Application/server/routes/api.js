const express = require('express')
const router = express.Router()

const controller = require('../controllers/api.controller')
/* Everything here is prepended with /api for the route
      for example: /hide is the route .../api/hide */

// /api/hide POST: receive images from body, convert to png, send png to Flask server
router.post('/hide', controller.sendRequestsController.hideSend)

// /api/extract POST: receive image from body, if not png, convert to png, send png to Flask server
router.post('/extract', (req, res) => {

})

module.exports = router