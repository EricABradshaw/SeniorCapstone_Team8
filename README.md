# SeniorCapstone_Team8

Senior Capstone project Team 8 AY 2023-24

## Python Setup

Download and Install [Python 3.11.6](https://www.python.org/downloads/release/python-3116/) if you don't already have it.

- Be sure to check the box that says "Add Python 3.11 to PATH"
- After installation, verify that `C:\Program Files\Python311\Scripts` has been added to your PATH.

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
