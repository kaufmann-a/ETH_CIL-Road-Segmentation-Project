# INSTALLATION GUIDE FOR Road Segmentation Main Project

Windows 10 - 64-Bit (similar on other os)

Virtual environment disk space: ~

## 1 Install Anaconda and Python
Download python version 3.9.2 and install: https://www.python.org/downloads/

### 1.1 Set destination Folder

- Better do not use spaces
- Example: C:\Programs\Anaconda3

### 1.2 Enable set PATH environment variable (optional)

### 1.3 Install or Update Anaconda3
If Anaconda is not installed yet on your machine, do:
Go to https://repo.anaconda.com/archive/ and download 2020.11 anaconda version (in my case Anaconda3-2020.11-Windows-x86_64.exe)

If Anaconda is installed on your machine already, update to version 2020.11:
- Open Anaconda Prompt (mac open terminal)
- run `conda update conda`
- run `conda install anaconda=2020.11`
Procede with 2.0

### 1.3 Check installation

- Open Windows CMD
- `python --version`
- \> Python 3.9.2

## 2 Setup
Important: 
- Just use `pip` commands and no `conda` commands!
- Use **CMD** (not Windows PowerShell) to work with virtual environments.

### 2.1 Setup of virtual environement

#### 2.1.1 Install pip
Run the following command (also if pip is already installed in the base environment, run the command):
- `python -m pip install --upgrade pip`
- Check for pip version: `python -m pip --version`, make sure the latest version is installed

#### 2.1.2 Install virtualenv
If virtualenv is not installed yet run the following command:
- `python -m pip install --user virtualenv`

#### 2.1.3 Create environement
- In cmd navigate to the folder you want to set up the environment (Could be `C:\Users\Username\Anaconda3\envs`) 
- Create virtualenv: `python -m venv name_virtualenv`
- Deactivate current environment: `deactivate`
- Activate new environment: `activate name_virtualenv`
- Check if python is pointing to the new environment: `where python` (macOS would be `which python`)
	- Path to new environment should be listed

### 2.2 Install requirements
- Make shure you cd to the location of your project, then run following command: `pip install -r requirements.txt`

## 3 Add new Project in PyCharm
- Check for latest version of PyCharm Professional(2020.3.4) 
- Open Pycharm: File > New Project
- Set Location to GitRepo
- Project Interpreter: Existing Interpreter, Browse for your virtualenv i.e. `C:\Users\Username\Anaconda3\envs\name_virtualenv\Scripts\python.exe`
- Create
