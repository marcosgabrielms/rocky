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
    $answer = Read-Host "Rocky conectado. Deseja iniciar a API do Koda? [S/N]"
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

if (-not (Test-Path -LiteralPath $serverPath)) {
    Write-Host "Diretório do backend não encontrado: $serverPath"
    exit 1
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host "Python do ambiente virtual não encontrado: $pythonPath"
    exit 1
}

Push-Location $serverPath
try {
    & $pythonPath -B (Join-Path $serverPath "tools\chatgpt_auth.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[KODA] A API não foi iniciada: sessão ChatGPT indisponível."
        exit 1
    }
    Write-Host "[KODA] Iniciando backend..."
    & $pythonPath -m uvicorn app:app --host 0.0.0.0 --port 8000
}
finally {
    Pop-Location
}
