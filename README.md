# SeniorCapstone_Team8

Senior Capstone project Team 8 AY 2023-24

## Python Setup

The models were developed using an older version of TensorFlow; 1.xx compatibility has decreased in later releases and full-blown conversion to 2.xx is not within our scope, so to properly use the models we need a version of Python that is compatible with TensorFlow 2.8.0, the latest of which is Python 3.10.11.

While not strictly required it's probably best to uninstall any other versions of Python.

Install [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)

- Be sure to check the box that says "Add Python 3.x to PATH"
- After installation, add `%appdata%\Python\Python3x\Scripts` to your PATH.

## Development Setup Instructions

The following instructions assume you are using Bash or an equivalent shell within VS Code.

```
git clone https://github.com/EricABradshaw/SeniorCapstone_Team8.git
cd SeniorCapstone_Team8/Application/python
py -3 -m venv .venv
. .venv/Scripts/activate
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
npm run all
```
