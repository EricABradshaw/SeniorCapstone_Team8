/* api.js is primarily used for handling data interactions with the database */
/* This will include business logic and endpoints for database interactions */
const express = require('express')
/* Communication from Node -> Flask */
const multer = require('multer');
/* end */
const controller = require('../controllers/api.controller')
const router = express.Router()

/* multer configuration */ 
const storage = multer.memoryStorage()
const upload = multer({storage: storage})

/* handle the post request that includes our data */
router.post('/create_stego_image', upload.fields([
    { name: 'coverImage', maxCount: 1 },
    { name: 'secretImage', maxCount: 1 }
]), controller.createStegoImage)


module.exports = router