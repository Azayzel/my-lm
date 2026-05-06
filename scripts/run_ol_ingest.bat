@echo off
cd /d "D:\repos\Lavely-LLM"
if not exist "logs" mkdir logs
"D:\repos\Lavely-LLM\.venv\Scripts\python.exe" "D:\repos\Lavely-LLM\scripts\ol_ingest.py" >> "D:\repos\Lavely-LLM\logs\ol_ingest.log" 2>&1
