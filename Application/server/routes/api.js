/* api.js is primarily used for handling data interactions with the database */
/* This will include business logic and endpoints for database interactions */
const express = require('express')
const controller = require('../controllers/api.controller')
const router = express.Router()

module.exports = router