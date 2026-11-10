$ErrorActionPreference = "Stop"

$rockyDeviceId = "VID_1A86&PID_55D3"
$serverPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\server"))
$pythonPath = Join-Path $serverPath ".venv\Scripts\python.exe"
$verboseValue = ([string]$env:KODA_VERBOSE).Trim().ToLowerInvariant()
$verboseEnabled = @("1", "true", "yes", "on").Contains($verboseValue)

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

$envFilePath = Join-Path $serverPath ".env"

function Get-KodaSetting([string]$name) {
    $processValue = [Environment]::GetEnvironmentVariable($name)
    if (-not [string]::IsNullOrWhiteSpace($processValue)) {
        return $processValue.Trim()
    }
    if (-not (Test-Path -LiteralPath $envFilePath)) {
        return ""
    }
    foreach ($line in Get-Content -LiteralPath $envFilePath -Encoding UTF8) {
        if ($line -match "^\s*$name\s*=\s*(.*)$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return ""
}

$llmProvider = (Get-KodaSetting "KODA_LLM_PROVIDER").ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($llmProvider)) {
    $llmProvider = "gemini"
}

Push-Location $serverPath
try {
    if ($llmProvider -eq "gemini") {
        if ([string]::IsNullOrWhiteSpace((Get-KodaSetting "GEMINI_API_KEY"))) {
            Write-Host "[KODA] A API não foi iniciada: GEMINI_API_KEY ausente em $envFilePath"
            exit 1
        }
        Write-Host "[KODA] Cérebro: Gemini."
    }
    elseif ($llmProvider -eq "chatgpt_plan") {
        & $pythonPath -B (Join-Path $serverPath "tools\chatgpt_auth.py")
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[KODA] A API não foi iniciada: sessão ChatGPT indisponível."
            exit 1
        }
    }
    else {
        Write-Host "[KODA] A API não foi iniciada: KODA_LLM_PROVIDER inválido ($llmProvider)."
        exit 1
    }
    Write-Host "[KODA] Iniciando backend..."
    $uvicornArguments = @("-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000")
    if (-not $verboseEnabled) {
        $uvicornArguments += "--no-access-log"
    }
    & $pythonPath @uvicornArguments
}
finally {
    Pop-Location
}
