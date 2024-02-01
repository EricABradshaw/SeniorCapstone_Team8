#!/bin/bash

cd Application/python
py -3 -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
cd ..
cd client
npm i react-scripts -E
cd ..
npm i -g concurrently
npm i

# Create .env file
ENV_FILE="Application/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "PORT=7000" > "$ENV_FILE"
    echo 'MONGO_URI="mongodb://127.0.0.1/capstone"' >> "$ENV_FILE"
    echo 'FLASK_SERVER_URI="localhost:5000"' >> "$ENV_FILE"

    echo ".env file created in /Application"
else
    echo ".env file already exists in /Application"
fi  