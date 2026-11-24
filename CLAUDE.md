# Koda

Agente pessoal local-first. O robô ESP32-S3 (`src/`, PlatformIO) é só um "corpo";
o cérebro é o backend Python (`server/`). "Rocky"/"Saga" são nomes legados que
ainda aparecem em pastas, classes e headers: não renomear em massa.

## Como trabalhar

- Em toda tarefa de código, usar a skill `ponytail`: a menor mudança que resolve
  o problema por completo, sem abstrações ou refatorações fora do escopo.
- Fases pequenas e com escopo explícito. Preservar o que já funciona.
- Perguntar antes de decisões de produto, arquitetura ou experiência.
- Nunca fazer commit ou push sem o usuário pedir.
- Commits no padrão https://github.com/iuricode/padroes-de-commits, em português:
  `✨ feat: adiciona cérebro Gemini`, `🐛 fix: ...`, `➕ build: ...`, `🔧 chore: ...`.
  Primeira linha curta; detalhes no corpo; um commit por assunto.
- Antes de mudar código, identificar a camada exata do problema
  (captura, VAD, STT, parser, Core, LLM, TTS, firmware) pelos logs reais.

## Regras do domínio

- Custo recorrente de IA = US$ 0. Sem APIs pagas nem fallback pago automático.
- Comandos de estado, volume, hora, memória e lembretes são determinísticos:
  nunca dependem do LLM.
- O firmware é dono do timeout da janela de comando; o backend não cria um
  segundo relógio concorrente.
- Não alterar VAD, pre-roll ou pipeline de áudio sem evidência nos logs.
- Servo/movimento e Android ficam para depois do Core físico fechado.

## Comandos

Backend (em `server/`, Python 3.12 obrigatório por causa do `audioop`):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
```

Firmware (na raiz):

```powershell
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run
```

Subir a API com o robô conectado (`KODA_VERBOSE=1` para diagnóstico):

```powershell
$env:KODA_VERBOSE=1; .\tools\rocky_launcher.ps1
```

## Ambiente

- `server/.env` (ignorado pelo Git): `KODA_LLM_PROVIDER=gemini` e `GEMINI_API_KEY`.
  Provider alternativo: `chatgpt_plan`.
- LLM no plano gratuito do Gemini, com reserva:
  `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → `gemma-4-31b-it`.
- `src/config/WifiConfig.h` é local (copiar de `WifiConfig.example.h`).
- Voz do Piper em `server/local/tts/voices/pt_BR-jeff-medium.onnx` (+ `.onnx.json`),
  baixada de `rhasspy/piper-voices`; não versionada.
- Firmware usa a plataforma pioarduino (Arduino-ESP32 3.x, exigido por `ESP_I2S.h`).
  No VS Code, manter só a extensão pioarduino; a PlatformIO IDE oficial reescreve
  `.vscode/extensions.json`. Windows precisa de `LongPathsEnabled=1`.
