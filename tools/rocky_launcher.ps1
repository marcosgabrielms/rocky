$ErrorActionPreference = "Stop"

$rockyDeviceId = "VID_1A86&PID_55D3"
$serverPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\server"))
$pythonPath = Join-Path $serverPath ".venv\Scripts\python.exe"

try {
    $rockyDevice = Get-PnpDevice -PresentOnly |
        Where-Object { $_.InstanceId -like "*$rockyDeviceId*" } |
        Select-Object -First 1
}
catch {
    Write-Host "Não foi possível verificar os dispositivos USB do Rocky."
    exit 1
}

if ($null -eq $rockyDevice) {
    Write-Host "Rocky não conectado. Conecte o USB-Enhanced-SERIAL CH343 e tente novamente."
    exit 0
}

do {
    $answer = Read-Host "Rocky conectado. Deseja iniciar a API do Rocky? [S/N]"
    switch ($answer.Trim().ToUpperInvariant()) {
        "S" { $startApi = $true }
        "N" { $startApi = $false }
        default { Write-Host "Resposta inválida. Digite S ou N." }
    }
} while ($null -eq $startApi)

if (-not $startApi) {
    Write-Host "Inicialização cancelada."
    exit 0
}

do {
    $answer = Read-Host "Usar o modelo gratuito openrouter/free? [S/N]"
    switch ($answer.Trim().ToUpperInvariant()) {
        "S" { $useRockyModel = $true }
        "N" { $useRockyModel = $false }
        default { Write-Host "Resposta inválida. Digite S ou N." }
    }
} while ($null -eq $useRockyModel)

if (-not $useRockyModel) {
    Write-Host "Inicialização cancelada."
    exit 0
}

$env:OPENROUTER_MODEL = "openrouter/free"
$secureApiKey = Read-Host "Cole sua chave OpenRouter" -AsSecureString
$apiKeyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureApiKey)
try {
    $env:OPENROUTER_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($apiKeyPointer)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($apiKeyPointer)
}

if ([string]::IsNullOrWhiteSpace($env:OPENROUTER_API_KEY)) {
    Write-Host "A chave OpenRouter não pode ficar vazia."
    exit 1
}

$portInUse = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
    Where-Object { $_.Port -eq 8000 } |
    Select-Object -First 1

if ($null -ne $portInUse) {
    Write-Host "A porta TCP 8000 já está em uso. A API não foi iniciada."
    exit 0
}

if (-not (Test-Path -LiteralPath $serverPath)) {
    Write-Host "Diretório do backend não encontrado: $serverPath"
    exit 1
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host "Python do ambiente virtual não encontrado: $pythonPath"
    exit 1
}

Write-Host "Iniciando a API do Rocky em http://0.0.0.0:8000"
Push-Location $serverPath
try {
    & $pythonPath -m uvicorn app:app --host 0.0.0.0 --port 8000
}
finally {
    Pop-Location
}
