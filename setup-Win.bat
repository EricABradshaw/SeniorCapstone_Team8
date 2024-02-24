@echo off

cd Application/python
py -3 -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
cd ..\client
npm i react-scripts -E
cd ..\..
npm i -g concurrently
npm i

REM create a .env file
SET ENV_FILE=Application\.env

echo Don't forget to create a .env file in /Application/