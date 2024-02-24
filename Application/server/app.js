const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const path = require("path");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 9000;

app.use(bodyParser.json());
app.use(
  cors({
    origin: "http://localhost:3000",
    credentials: true,
  })
); // cross origin resource sharing
app.options("*", cors());

// Route definition
const apiRoutes = require("./routes/api");
const mainRoutes = require("./routes/index");
app.use("/api", apiRoutes);
app.use("/", mainRoutes);

// Middleware
app.use(express.static(path.join(__dirname, "../client/build")));

// Server start
app.listen(PORT, () => {
  console.log(`Server is listening on port ${PORT}`);
});
