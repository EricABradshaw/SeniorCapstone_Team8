#!/bin/bash

cd Application/python
python3 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
cd client
npm i react-scripts -E
cd ..
npm i -g concurrently
npm i

# Create .env file
ENV_FILE="Application/.env"

echo "Don't forget to create a .env file in /Application/"