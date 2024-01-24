/* Controller for serving React files for frontend */
const path = require('path')
let frontEndPath = path.join(__dirname, '../../client/')

const mainController = {
  sendReactApp: (req, res) => {
    try {
      res.setHeader("Access-Control-Allow-Origin", "*")
      res.setHeader("Access-Control-Allow-Credentials", "true");
      res.setHeader("Access-Control-Max-Age", "1800");
      res.setHeader("Access-Control-Allow-Headers", "content-type");
      res.setHeader("Access-Control-Allow-Methods", "PUT, POST, GET, HEAD, DELETE, PATCH, OPTIONS" ); 
      res.sendFile(path.join(__dirname, `${frontEndPath}/build/index.html`))
    } catch (err) {}
  }
}

module.exports = {
  mainController
}