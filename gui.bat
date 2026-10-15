@echo off
cd /d %~dp0
set HF_HOME=%~dp0models
set PATH=%~dp0.venv\Scripts;%PATH%
set PYTHONIOENCODING=utf-8
set GRADIO_ANALYTICS_ENABLED=False
python gui.py %*
