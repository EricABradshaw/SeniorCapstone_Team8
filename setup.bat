@echo off

cd Application/python
py -3 -m venv .venv
.\Application\python\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
cd ..\client
npm i react-scripts -E
cd ..\..
npm i -g concurrently
npm i

REM create a .env file
SET ENV_FILE=Application\.env

IF NOT EXIST "%ENV_FILE%" (
    echo PORT=9000 > "%ENV_FILE%"
    echo MONGO_URI="mongodb://127.0.0.1/capstone" >> "%ENV_FILE%"
    echo FLASK_SERVER_URI="localhost:5000" >> "%ENV_FILE%"

    echo .env file created in Application
) ELSE (
    echo .env file already exists in Application
)