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

## Koda — baseline de latência (4.9F.1)

O wake oficial agora é somente `Koda`, com a normalização existente de caixa,
pontuação e espaços externos. `Ok`, `Ok Koda`, palavras parecidas e frases
contendo Koda não abrem sessão. Os nomes internos/paths Rocky permanecem.

Esta etapa mede antes de otimizar: não houve redução do release, troca de
modelo, alteração de beam/best_of ou adição de initial_prompt. A comparação
física é necessária antes das próximas subetapas da 4.9F.

Baseline historico da medicao 4.9F.1 (nao representa a configuracao atual;
VAD atual: 20000/12000, politica LLM atual descrita ao final):

- ESP32: start=10000, end=7000 (raw PCM24), attack=100 ms, release=450 ms,
  pre-roll=280 ms, blocos de 64 amostras (~4 ms), captura máxima=4000 ms.
- Whisper 1.2.1: small/cpu/int8, language=pt, beam_size=5, best_of=5,
  condition_on_previous_text=True, word_timestamps=False, vad_filter=True,
  initial_prompt=None, temperatura padrão [0, 0.2, 0.4, 0.6, 0.8, 1.0].
- Hotword no modo wake: Koda; no modo comando: nenhuma.
- Filtro de confiança, Piper/voz, PCM16/32000 Hz, volume 30%, fila,
  confirmações de consumo, personalidade e estados visuais preservados.
- OpenRouter: openrouter/free; allowlist anterior preservada, sem fallback
  pago/retry/segunda geração. Testes automatizados usam mocks, nunca o serviço real.

### Como interpretar os logs

`[KODA LATENCY]` identifica o relógio e a interação/requisição.
`kind=event` representa um instante; `kind=duration` representa um intervalo.
Não subtrair instantes do ESP32 dos instantes do PC: os relógios não são
sincronizados. IDs de requisição do backend não são o contador do firmware;
correlacionar testes sequenciais pela ordem e pelo interaction_id realtime já
existente. Nenhum campo/header novo foi adicionado ao protocolo HTTP.

No firmware, `speech_start_ms` é o trigger confirmado, após o attack.
`speech_end_ms` é uma **estimativa pelo threshold**: início do último período
abaixo de end que terminou com silêncio confirmado. `speech_end_confirmed_ms`
é o evento após o release. Não se trata de medir acusticamente o último fonema.
`vad_release_observed_ms` ajuda a separar essa espera dos custos seguintes.
Uma captura por limite, sem silêncio confirmado, informa `speech_end_ms=unavailable`
e usa métricas `capture_done_to_*`, sem inventar um fim de fala.

| Ponto | Métricas |
| --- | --- |
| ESP32/VAD | speech_start, speech_end, speech_end_confirmed, capture_done; end_reason=silence/max_duration; contadores por boot |
| ESP32/HTTP | stt_request_start, http_response, http_roundtrip_ms, capture_done_to_request_ms |
| Wake físico | speech_end_to_attention_ms (inclui release e caminho HTTP) |
| Backend/STT | stt_start, stt_done, stt_ms; model_load_ms separado para distinguir primeira chamada fria |
| Classificação | classify_done, classify_ms; sem LLM em wake/horas/Idle |
| LLM | llm_request_start, llm_first_token, first_sentence; intervalos até primeiro fragmento textual e frase |
| TTS/fila | tts_start, tts_ready, tts_first_sentence_ms, audio_available, audio_ready_ms |
| ESP32/áudio | playback_start, speech_end_to_playback_start_ms; logs anteriores de download/playback/gap mantidos |

O tempo HTTP **não é stt_ms**: horas inclui TTS e um caminho síncrono pode
incluir LLM. `speech_end_to_http_response_ms` é um limite superior para o fim
do STT no firmware, não um `speech_end_to_stt_done_ms` exato entre relógios.
O instante real de stt_done fica nos logs do backend; não foi falsamente
deduzido da resposta HTTP. O timestamp do primeiro áudio é emitido no enqueue
do primeiro WAV, não ao final da geração inteira.

`services/latency.py: LATENCY_DIAGNOSTICS` permite silenciar os novos logs do
backend. Não há log por amostra ou por frame no firmware.

### Teste físico comparativo

1. Usar o mesmo ambiente, ventilador desligado e distância fixa (40–60 cm).
   Fechar o Serial Monitor antes do flash **manual** do firmware compilado.
2. Reiniciar a API com o ambiente Python válido e o launcher existente, mantendo
   google/gemma-4-31b-it:free. Não expor a API fora da LAN nem colocar chaves nos logs.
3. Abrir COM3 em 115200 baud e guardar tanto o Serial quanto os logs do backend.
4. Cenário A: dizer apenas **Koda** cinco vezes; esperar Attention e o timeout
   de cada sessão antes de repetir. Registrar reconhecimento/falha,
   end_reason e speech_end_to_attention_ms.
5. Cenário B: dizer **Koda**, esperar Attention e falar **Que horas são?**.
   Repetir cinco vezes, registrando stt_ms, tts_first_sentence_ms e
   speech_end_to_playback_start_ms. Confirmar ausência de request LLM.
6. Cenário C: dizer **Koda**, esperar Attention e falar **Por que o céu é azul?**.
   Repetir cinco vezes: stt_ms, llm_first_token_ms, first_sentence_ms,
   tts_first_sentence_ms, speech_end_to_playback_start_ms e routed_model,
   quando informado pelo próprio SSE (sem request extra). Cada pergunta pode
   fazer somente uma chamada ao modelo gratuito.
7. Separar a primeira chamada fria (model_load_ms), falhas e capturas por limite.
   Comparar medianas e amplitude min/max com um baseline nas mesmas condições.
   Não alegar ganho antes dessa coleta, nem comparar novos nomes com áudio
   diferente sem registrar essa diferença.

Testes locais, sem modelo real/reprodução/OpenRouter:

```powershell
cd server
.\.venv\Scripts\python.exe -m compileall -q app.py services test_latency.py
.\.venv\Scripts\python.exe -m unittest test_latency test_conversation test_stt test_realtime_streaming test_sentence_buffer test_realtime_audio test_audio_queue test_realtime_endpoint
```

Se o .venv apontar para um Python removido, restaurar esse ambiente em uma
etapa autorizada separada antes de rodar API/testes HTTP. Os testes com mocks
executados por um Python auxiliar não validam a qualidade acústica do STT real.

### ATTENTION fisico e guard de fala

O firmware atualizado e o backend devem ser utilizados juntos. A wake Koda
retorna `interaction_state=attention`, sem deadline ou countdown do backend.
O corpo mostra os olhos atentos e aguarda o inicio da fala por 6000 ms locais.
Uma candidatura de fala iniciada na janela pode completar os 100 ms de attack;
ao confirmar fala, o timer e cancelado e nao limita captura, STT ou resposta.
Sem inicio de fala, o corpo volta ao Idle e a proxima captura exige nova wake.

Os dois uploads (`/transcribe` e `/transcribe/realtime`) aceitam o header
`X-Rocky-Interaction-Mode: wake|command`. O firmware fixa esse modo no inicio
da captura: Idle usa wake, ATTENTION usa command. Command vai diretamente ao
classificador, mesmo se a captura ou o STT terminarem muito depois dos 6 s.
Wake nao reutiliza uma sessao antiga do backend. Clientes sem o header mantem
apenas um indicador de espera por dispositivo para compatibilidade, sem timer;
eles precisam enviar o header para comunicar corretamente um timeout fisico.
O relogio do backend continua sendo usado para o TTL de contexto, nao ATTENTION.

`SpeechTextSanitizer` limpa localmente sentencas antes do Piper, tanto no
streaming quanto na resposta sincrona. Remove Markdown, URLs, codigo cercado
e labels estruturais, preservando palavras legitimas e o texto original da
resposta/contexto. Nao resume nem trunca o conteudo. SentenceBuffer continua
emitindo a primeira frase pronta imediatamente; links, URLs e fences ficam
integros para limpeza. Frases vazias nao geram WAV e a sequencia de audio
continua contigua, com o mesmo protocolo de ACK e AudioQueue.

O prompt orienta uma frase curta para perguntas simples e no maximo duas para
explicacoes comuns; mais detalhes somente quando pedidos. A politica atual usa
`max_tokens=120`, `temperature=0.2` e Gemma free fixo, sem fallback de modelo,
retry ou chamada extra de resumo. O modelo roteado vem da propria resposta/SSE.
Logs temporarios de texto usam `SPEECH_TEXT_DIAGNOSTICS` e previews limitados
a 240 caracteres; nao incluem chave, tokens de autenticacao ou dados PCM.

Testes deterministas adicionais:

```powershell
cd server
.\.venv\Scripts\python.exe -m unittest test_speech_text test_conversation test_sentence_buffer test_realtime_audio test_realtime_streaming test_realtime_endpoint test_tts_response_speaker
```

Depois de reiniciar o backend atualizado e fazer flash manual, validar:
Koda -> olhos atentos sem numeros -> silencio -> Idle apos ~6 s;
Koda -> pergunta iniciada perto de 5,9 s -> captura command, sem expirar no STT;
Koda -> Que horas sao? -> comando deterministico sem LLM;
Koda -> Quem criou o Linux? -> resposta curta, sem Markdown/metatexto no audio.

### Modelo fixo, custo zero e wake passiva (politica atual)

A unica wake e `Koda` (caixa/pontuacao), pronunciada Kôda. `Ok Koda`, `Ei Koda`,
`Coda`, `Kodal`, `Rocky` e Koda no meio de uma frase nao ativam em Idle.
Durante `capture_mode=wake`, captura/STT sao internos: os olhos permanecem Idle,
sem Listening/Thinking. Apenas uma wake confirmada pelo backend abre Attention.
Sem wake, nao ha LLM nem TTS. Em `capture_mode=command`, os estados visuais
Listening -> Thinking -> Speaking continuam normais. O timeout fisico permanece
6000 ms, sem countdown, e so limita o inicio da fala; nao ha deadline no backend.

`Koda, que horas sao?` extrai o comando e usa o caminho deterministico, sem LLM.
`Koda quem criou o Linux?` extrai a pergunta e faz uma unica geracao. O primeiro
token significativo precisa ser exatamente Koda; nao ha fuzzy/substring nem
whitelist de perguntas. Qualquer pergunta valida pos-wake segue ao LLM se nao
for comando conhecido. O prefixo de wake nao entra no contexto de conversa.

Configuracao manual existente: `OPENROUTER_MODEL`. Allowlist local exata:

- `google/gemma-4-31b-it:free` — padrao;
- `nvidia/nemotron-3-super-120b-a12b:free` — alternativa manual;
- `thinkingmachines/inkling-small:free` — somente selecao manual, nunca automatica.

`openrouter/free`, `openrouter/auto`, modelos pagos, outros `:free` e configuracao
vazia sao bloqueados antes da request. A composicao de app.py usa o mesmo default
do cliente, sem duplicar configuracao. O launcher pergunta pelo Gemma e o define
apenas no processo atual; para avaliar alternativas manualmente, configure a
variavel no PowerShell e inicie Uvicorn diretamente com o ambiente restaurado.

Cada request envia um unico `model`, `temperature=0.2`, `max_tokens=120` e:

```json
{"provider":{"max_price":{"prompt":0,"completion":0},"data_collection":"deny","allow_fallbacks":true}}
```

Sem lista `models`, retries ou segunda geracao. `allow_fallbacks=true` permite
somente providers do mesmo modelo; nenhum provider e fixado por nome. Se nao
houver endpoint gratuito compativel com a privacidade, a falha e local/controlada,
sem relaxar as restricoes. Inkling informa coleta de prompts/outputs, por isso
nao e default e pode ficar indisponivel com `data_collection=deny`.

Reasoning: o Gemma configuravel recebe `enabled=false`; os tres modelos recebem
`exclude=true`. Nao se impõe `effort=none` nas alternativas sem confirmacao de
suporte. O parser usa somente `message.content` / `delta.content`, nunca campos
`reasoning` ou `reasoning_details`. SpeechTextSanitizer permanece nos caminhos
incremental e sincrono, sem resumir semanticamente a resposta.

Referencias oficiais: [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection),
[reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) e
[Inkling Small free](https://openrouter.ai/thinkingmachines/inkling-small:free).

Guardrail externo informado pelo usuario: `Koda - Free Models`, associado apenas
a chave normal do Koda. Segunda barreira recomendada: Only Allow dos tres modelos
acima, sem provider allowlist; Prompt Injection regex = Flag, User messages only;
Sensitive Info com Email address, Phone number, Social Security number, Credit
card number e API keys and secrets. O codigo nao cria/edita Guardrails, nao usa
Management API/key e nao documenta segredos. Flag nao significa bloqueio automatico.

Validacao automatizada usa somente mocks. Testes HTTP requerem .venv funcional;
o Python auxiliar nao substitui o runtime do backend. Teste fisico posterior:
conversa comum em Idle sem reacao; Koda -> Attention -> silencio (~6 s) -> Idle;
Koda -> pergunta durante Attention; Koda + horas e Koda + pergunta na mesma fala.
