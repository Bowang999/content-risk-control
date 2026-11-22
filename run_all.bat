@echo off
chcp 65001 >nul
cd /d %~dp0
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
call conda activate riskctl
python run_all.py %*
pause
