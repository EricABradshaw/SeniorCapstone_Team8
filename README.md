# SeniorCapstone_Team8

Senior Capstone project Team 8 AY 2023-24

## Python Setup

The models were developed using an older version of TensorFlow; 1.xx compatibility has decreased in later releases and full-blown conversion to 2.xx is not within our scope, so to properly use the models we need a version of Python that is compatible with TensorFlow 2.8.0, the latest of which is Python 3.10.11.

While not strictly required it's probably best to uninstall any other versions of Python.

Install [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)

- Be sure to check the box that says "Add Python 3.x to PATH"
- After installation, add `%appdata%\Python\Python3x\Scripts` to your PATH.

## Quick Setup

Running Bash on a Windows machine

```
git clone https://github.com/EricABradshaw/SeniorCapstone_Team8.git
cd SeniorCapstone_Team8
./setup-Win.sh
cd Application
npm run all
```

Running Bash on a Linux machine

```

```

The server should automatically run in the python virtual environment.

## Development Setup Instructions

```
git clone https://github.com/EricABradshaw/SeniorCapstone_Team8.git
cd SeniorCapstone_Team8/Application/python
py -3 -m venv .venv
```

If you are using Bash or an equivalent shell:

```
. .venv/Scripts/activate
```

If you are using command prompt or PowerShell:

```
.venv\Scripts\activate
```

In VS Code, open the command palette (Ctrl+Shift+P) and type "Python: select interpreter...". Set it to the python.exe found in your .venv folder.

```
pip install -r requirements.txt
cd ..
cd client
npm i react-scripts -E
cd ..
npm i -g concurrently
npm i
```

The above setup can be done via the setup shell script from either

- Bash or Bash-like:
  `source setup.sh`
- PowerShell:
  `.\setup.bat`

These scripts will also create a .env file for you in /Application/.

To run the environment, enter the Application directory, then:

```
npm run all
```
