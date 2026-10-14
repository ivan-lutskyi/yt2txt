@echo off
cd /d %~dp0
set HF_HOME=%~dp0models
set PATH=%~dp0.venv\Scripts;%PATH%
python yt2txt.py %*
