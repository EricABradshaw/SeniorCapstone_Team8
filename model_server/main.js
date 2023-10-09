const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Define the directory containing the files to serve
const directoryToServe = 'js_model';

// Middleware to serve static files from the directory
app.use(express.static(directoryToServe));

// Handle GET requests
app.get('/', (req, res) => {
  res.send('Hello, this is a simple file server!');
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});