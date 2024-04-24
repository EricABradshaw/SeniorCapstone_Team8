# Deploying CNN-Driven Image Steganography Techniques in a Web Application

## Server Folder

**Flask Server**
This markup document is to describe the server folder located in `./Application` of the capstone project. The server folder is broken down into 3 folders:

1. controllers
2. models
3. routes

## API Controllers

**Flask Server Communication**

### Dependencies

The following dependencies are required:

- `axios`: Creating HTTP requests.
- `sharp`: Image Processing.
- `base64-img`: Base64 Images (as Images are sebt to server as Base64).
- `form-data`: for working with form data.
- `fs`: File system operations.

### Constants

- `FLASK_SERVER_URL`: `http://127.0.0.1:5000` (Base URL of Flask Server)
- `path`: Controller for serving React files for frontend.

#### 1. Controllers

# **api.controller.js**

The API controller establishes a connection with a Flask server that uses Axios to send and receive data.

#### sendRequestsController

**hideSend**

```javascript
hideSend: async (req, res) => {
    try {
      const coverImageSource = req.body.coverImageData
      const secretImageSource = req.body.secretImageData
      console.log(`Request received from ${req.get('origin')}`)
      let base64Strings = await helperFunctions.fetchAndConvert(coverImageSource, secretImageSource)
      let stego64String = await helperFunctions.sendToFlask(base64Strings)
      res.status(200).json({ stegoImage: stego64String })
    } catch (err) {
      console.log(err)
    }
  }
```

- fetchAndConverts secret and cover images into base64 encoded images.
- Sends a POST request to the Flask server for hiding process.
- Creates base64 string stegoimage from base64 strings.

**extractSend**

```javascript
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
```

- Input is a base64 encoded image.
- Sends a POST request to the Flask server for extraction process (and waits for response).
- Returns the extracted hidden image.

#### helperFunctions

**fetchAndConvert**

```javascript
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
  }
```

- Take the cover image URL and the secret image URL and fetch both images.
- After fetching, convert each image into a png.
- Returns an object {coverBuffer: ..., secretBuffer: ... } containing base64 encoded strings for cover image and secret image.

**sendToFlask**

```javascript
sendToFlask: async (data) => {
    try {
      let resData
      await axios.post(FLASK_SERVER_URL + '/create_stego_image_b64', data, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
        .then(async response => {
          console.log('Success:',response.status)
          resData = await response.data.stegoImage
        })

        return resData

    } catch (err) {
      console.error(err)
    }
  }
```

- Sends a POST request to the Flask server with object containing base64 cover and base64 secret image.
- Expects an object containing base64 encoded strings for cover image and secret image.
- Returns the base64 encoded stegoimage (resData).

# **main.controller.js**

The Main controller serves React files for the frontend of the project.

#### sendReactApp

```javascript
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
```

- CORS (Cross-Origin Resource Sharing) Headers allow for requests from any origin (*), to enable credentials, and specifies allowed Headers/Methods.
- Sends React application's HTML file to client's frontend.

##### 2. Models

# **db.js**

Javascript program to connecting to the MongoDB database. The database connection was implemented for future versions of the project.

**Constants**
`mongoose` = require('mongoose')

```javascript
try {
  console.log("Attempting to connect to the database...")
  const attemptConnect = async () => {
    await mongoose.connect(process.env.MONGO_URI || 'mongodb://127.0.0.1/capstone')
  }
  attemptConnect().then( console.log('Connected to the database'))
} catch (err) {
  console.log(`Could not connect to the database:\n${err}`)
}
```

###### 3. Routes

###### Dependencies

The following dependencies are required:
api.js -> `controller`: Imports the controller module from `api.controller.js`.
index.js -> `controller`: Imports the controller module from `main.controller.js`.

###### Constants

`express`: Web framework for Node.js.
`multer`: Middleware for handling multipart/form-data (used for file uploads).
`sharp`: Image processing library.
`router`: Initializes an instance of the Express Router.
`storage`: Multer memory storage.
`upload`: Handles all file uploads and stores into memory.

# **api.js**

Handels hiding and extracting funticonalities of application.
Note: Pepended with /api for the route.

```javascript
router.post('/hide', controller.sendRequestsController.hideSend)

router.post('/extract', upload.single('stegoImage'), async (req, res) => {
  const stegoImageData = req.file.buffer
  const resizedImage = await sharp(stegoImageData)
    .resize(224, 224)
    .toBuffer()
  const base64Image = resizedImage.toString('base64')
  let answer = await controller.sendRequestsController.extractSend(base64Image)
  res.json({ stegoImageData: answer });
});
```

- /hide: Handles POST requests for hiding process, invoking `sendRequestsController.hideSend`.
- /extract: Handles POST requests for extraction process, invoking `sendRequestsController.extractSend`. Resizes the image to 224x224 before sending request.

# **index.js**

Serves the main page.

```javascript
router.route('/')
  .get(controller.mainController.sendReactApp)
  .post()
```

- GET: Handles GET requests to the root endpoint, invoking `mainController.sendReactApp`
- POST: Handles POST requests to the root endpoint.

# **app.js**

Setup and configuration of Express.js server to handle HTTP requests.

```javascript
const app = express()
const PORT = process.env.PORT || 9000

app.use(bodyParser.json())
app.use(cors({
  origin: 'http://localhost:3000',
  credentials:true
})) // cross origin resource sharing
app.options('*', cors())

// Route definition
const apiRoutes = require('./routes/api')
const mainRoutes = require('./routes/index')
app.use('/api', apiRoutes)
app.use('/', mainRoutes)

// Middleware
app.use(express.static(path.join(__dirname, '../client/build')))

// Server start
app.listen(PORT, () => {
  console.log(`Server is listening on port ${PORT}`)
})
```

- Server listens on PORT 9000 (default).
- Configures the server to parse incoming request bodies in JSON format.
- Handles preflight OPTIONS requests for all routes (*), necessary for CORS to work properly with certain request methods or custom headers.
- apiRoutes mounts at the API path.
- mainRoutes mounts at the root path.
- Configures Express to serve static files (Serves frontend assets).
- Once routes and middleware are set, startup the server
