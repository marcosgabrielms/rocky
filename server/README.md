# Servidor local de STT

O servidor recebe um WAV em memória, valida PCM16 mono a 16 kHz e transcreve
o buffer temporário com `faster-whisper`, modelo `small`, em CPU/int8. Nenhum
WAV é salvo em disco e não há API key ou serviço pago.

## Preparação no Windows PowerShell

```powershell
cd server
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Iniciar

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Testar saúde

Em outro terminal PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

## Testar transcrição

Use uma amostra WAV real, mono, PCM16 e 16 kHz. Por exemplo, a amostra já
existente no projeto vizinho `rocky-audio-lab`:

```powershell
curl.exe -X POST http://127.0.0.1:8000/transcribe -F "file=@..\rocky-audio-lab\dataset\oi\oi_20260809_231535_001_001.wav;type=audio/wav"
```
