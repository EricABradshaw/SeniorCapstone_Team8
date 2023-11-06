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
const userController = { }

module.exports = {
  dataController,
  userController
}