# StegoSource

Purdue Fort Wayne Senior Capstone Project AY 2023-24, Team 8

Utilizing a CNN-Based Image-in-Image Steganography Model in a Web-Based Application

This project utilizes a CNN-based machine learning model developed by Dr. Amal Khalifa and her team at Purdue University Fort Wayne. 

## Dependency Installation

### Python

The image-in-image steganography models used for this project are CNN models utilizing the SteGuz method. You will need to install a version of Python that supports TensorFlow version 2.12.0.

We have been developing and running this application under Python [Python 3.10.11](https://www.python.org/downloads/release/python-31011/).

*While not strictly required it is best to have only one version of Python installed on your machine.*

While installing Python:
- Be sure to check the box that says "Add Python 3.x to PATH"
- After installation, add `%appdata%\Python\Python3x\Scripts` to PATH (if it's not already there).

### node.js

node.js gives us the package manager `npm` and is also the main back-end for our application. `npm` is also necessary to install the required front-end packages, namely React and its dependencies.

Download and install it from [here](https://nodejs.org/en/download/current).

## Quick Setup

Install the dependencies as described above.

If you are running Bash on a Windows machine:
```
git clone https://github.com/EricABradshaw/SeniorCapstone_Team8.git
cd SeniorCapstone_Team8
./setup-Win.sh
cd Application
npm run all
```

If you are running Bash on a Linux machine:
```
git clone https://github.com/EricABradshaw/SeniorCapstone_Team8.git
cd SeniorCapstone_Team8
./setup-Linux.sh
cd Application
npm run all
```

You should also create a file called `.env` under `/Application` with the following contents:
```
PORT=9000
FLASK_SERVER_URI="localhost:5000"
```

The server should automatically run in the Python virtual environment. Your terminal does not need to be in the virtual environment when calling `npm run all`.

## Manual Setup

### Clone this repository

In any terminal:
```
git clone https://github.com/EricABradshaw/SeniorCapstone_Team8.git
cd SeniorCapstone_Team8/Application/python
```

### Activate the virtual environment

If you are on Windows using Bash: 
```
py -3 -m venv .venv
. .venv/Scripts/activate
```

If you are on Linux using Bash or equivalent:
```
python3 -m venv .venv
source ./.venv/bin/activate
```

If you are on Windows using PowerShell or Command Prompt:
```
py -3 -m venv .venv
.venv\Scripts\activate
```
You will know you're in the virtual environment when all commands output by your shell end with `(.venv)`

### Install npm Packages and Python Dependencies

In any terminal with the Python virtual environment active:
```
pip install -r requirements.txt
cd ..
cd client
npm i react-scripts -E
cd ..
npm i -g concurrently
npm i
```

You will also need to create a file called `.env` in `/Application/` with the following contents:
```
PORT=9000
MONGO_URI="mongodb://127.0.0.1/<databasename>"
FLASK_SERVER_URI="localhost:5000"
```

And a file called `.env` in `/Application/client` with the following contents:
```env
REACT_APP_NODE_SERVER_URI="http://localhost:9000"
```

### Start the web app locally

Finally, to start the web server on your local machine:
```
npm run all
```