# Windows/PowerShell environment setup for rag-emotion-chatbot
# Run from the project root: .\scripts\setup_env.ps1

py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Environment ready. Activate with: . .\.venv\Scripts\Activate.ps1"
