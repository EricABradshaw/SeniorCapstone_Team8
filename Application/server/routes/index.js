const express = require('express')
const router = express.Router()

const controller = require('../controllers/main.controller')

/* GET home page. */
router.route('/')
  .get('/', controller.mainController.sendReactApp)
  .post()

module.exports = router
