const mongoose = require('mongoose')

try {
  console.log("Attempting to connect to the database...")
  const attemptConnect = async () => {
    await mongoose.connect(process.env.MONGO_URI || 'mongodb://127.0.0.1/capstone')
  }
  attemptConnect().then( console.log('Connected to the database'))
} catch (err) {
  console.log(`Could not connect to the database:\n${err}`)
}
