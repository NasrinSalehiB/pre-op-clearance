@echo off
REM Batch script to set up virtual environment for pre-op-clearance project

echo Setting up virtual environment...

REM Create virtual environment
python -m venv venv

echo Virtual environment created!

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete! Virtual environment is ready.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat

pause

