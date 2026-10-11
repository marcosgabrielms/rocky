# Laboratório offline de voz do Koda

Este laboratório cria quatro WAVs para escolha humana, sem chamar o backend,
o ESP32, OAuth ou qualquer serviço externo.

Use o ambiente do backend e execute, a partir de `server/`:

```powershell
.\.venv\Scripts\python.exe .\tools\voice_lab.py
```

Pré-requisitos locais: Piper funcional, o modelo
`local/tts/voices/pt_BR-jeff-medium.onnx` e FFmpeg disponível no `PATH`.
O script não instala nem baixa nenhum deles.

`--help` funciona sem validar os pré-requisitos. Na validação, um único
sample exatamente no limite PCM é preservado e reportado como `WARNING`; dois
ou mais samples no limite são tratados como `ERROR`.

As saídas ficam em `voice_lab/output/` e são deliberadamente ignoradas pelo
Git. Todos os perfis usam a mesma frase e um alvo de normalização comum.

| Arquivo | Perfil |
| --- | --- |
| `koda_baseline.wav` | Piper puro. |
| `koda_natural.wav` | EQ de clareza e compressão muito leve. |
| `koda_balanced.wav` | Pitch de +0,75 semitom, clareza e assinatura discreta. |
| `koda_synthetic.wav` | Pitch de +1,25 semitom e doubling eletrônico suave. |

O perfil recomendado para ouvir primeiro é `koda_balanced.wav`.
