/* Controller for serving React files for frontend */
const path = require('path')

const mainController = {
  sendReactApp: (req, res) => {
    try {
      res.sendFile(path.join(__dirname, '../../client/build/index.html'))
    } catch (err) {}
  }
}

module.exports = {
  mainController
}