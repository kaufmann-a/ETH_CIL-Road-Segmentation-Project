# INSTALLATION GUIDE FOR TASK 0

Windows 10 - 64-Bit

Requirments: **Patience & nerves**
Virtual environment disk space: ~2.5 GB

## 1 Install Anaconda

 <https://repo.anaconda.com/archive/Anaconda3-2020.07-Windows-x86_64.exe>

### 1.1 Set destination Folder

- Better do not use spaces
- Example: C:\Programs\Anaconda3

### 1.2 Enable set PATH environment variable (optional)

### 1.3 Check installation

- Open Windows CMD
- `python --version`
- \> Python 3.8.5

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
- Open Pycharm: File > New Project
- Set Location to GitRepo
- Project Interpreter: Existing Interpreter, Browse for your virtualenv i.e. `C:\Users\Username\Anaconda3\envs\name_virtualenv\Scripts\python.exe`
- Create
