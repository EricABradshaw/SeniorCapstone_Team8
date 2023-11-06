const express = require('express')
const bodyParser = require('body-parser')
const cors = require('cors')
const path = require('path')
const fs = require('fs')
require('./models/db')
require('dotenv').config()

const app = express()
const PORT = process.env.PORT || 3000

// Middleware
app.use(express.static(path.join(__dirname, '../client/build')))
app.use(bodyParser.json())
app.use(cors()) // cross origin resource sharing

// Route definition
const apiRoutes = require('./routes/api')
const mainRoutes = require('./routes/index')
app.use('/api', apiRoutes)
app.use('/', mainRoutes)

// Server start
app.listen(PORT, () => {
  console.log(`Server is listening on port ${PORT}`)
})