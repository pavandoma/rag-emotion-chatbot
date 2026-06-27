# Installs Ollama (if missing) and pulls the model used by this project.
# Run from the project root: .\scripts\pull_ollama_model.ps1

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found, installing via winget..."
    winget install Ollama.Ollama
} else {
    Write-Host "Ollama already installed."
}

ollama pull llama3.2:3b
Write-Host "Model llama3.2:3b ready. Test with: ollama run llama3.2:3b `"hello`""
