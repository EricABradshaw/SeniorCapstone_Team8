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

// TODO : Create controller for user model
const userController = {
  // test running the model
  runModel: async(req, res) => {
    // assumptions: images sent via POST request & stored locally
    // need to validate & save files before running the model?
    const coverImagePath = 'path/to/cover/image.png'
    const secretImagePath = 'path/to/hidden/image.png'

    const runModelPath = path.join(__dirname, '../../python/run_model.py'); 

    // command to run python script
    const command = `python ${runModelPath} ${coverImagePath} ${secretImagePath}`
    // use --index --etc

    // exec - one-time execution with a callback
    // real-time output: spawn? possible option
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`exec error: ${error}`)
        return res.status(500).send('Error running model')
      }
      console.log(`stdout: ${stdout}`);
      if (stderr) {
        console.error(`stderr: ${stderr}`);
      }
      return res.status(200).send(stdout); // or do something with the output
    });
  }

  // other functionalities TODO



}

module.exports = {
  dataController,
  userController
}