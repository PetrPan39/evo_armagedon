
@echo off
echo [CML + LLaMA Integration Launcher]
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt
echo Ready. Run main.py to start your CML system.
pause
