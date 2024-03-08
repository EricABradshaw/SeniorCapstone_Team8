const express = require('express')
const multer = require('multer');
const sharp = require('sharp')
/* Communication from Node -> Flask */

/* end */

const controller = require('../controllers/api.controller')
const router = express.Router()
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

/* Everything here is prepended with /api for the route
      for example: /hide is the route .../api/hide */

// /api/hide POST: receive images from body, convert to png, send png to Flask server
router.post('/hide', controller.sendRequestsController.hideSend)

router.post('/hideText', controller.sendRequestsController.hideTextSend)

// /api/extract POST: receive image from body, if not png, convert to png, send png to Flask server
router.post('/extract', upload.single('stegoImage'), async (req, res) => {
  // Access the uploaded file using req.file.buffer
  const stegoImageData = req.file.buffer
  const resizedImage = await sharp(stegoImageData)
    .resize(224, 224)
    .toBuffer()
  const base64Image = resizedImage.toString('base64')
  let answer = await controller.sendRequestsController.extractSend(base64Image)

  // Send a response
  res.json({ stegoImageData: answer });
})

// URI: /api/recommendation
router.post('/recommendation', controller.sendRequestsController.recommendation)

module.exports = router