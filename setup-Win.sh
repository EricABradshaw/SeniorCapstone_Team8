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
echo "Don't forget to create a .env file in /Application/"