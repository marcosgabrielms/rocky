# Servidor local de STT

O servidor recebe um WAV em memória, valida PCM16 mono a 16 kHz e transcreve
o buffer temporário com `faster-whisper`, modelo `small`, em CPU/int8. Nenhum
WAV é salvo em disco e não há API key ou serviço pago.

## Preparação no Windows PowerShell

Runtime oficial: **Python 3.12.10 x64** (`.python-version`). Não usar o Python
interno do PlatformIO nem migrar para 3.13/3.14: o resampling usa `audioop`,
removido em 3.13, e `StrEnum` exige pelo menos 3.11. O playback local usa
`winsound` somente quando habilitado; por padrão fica desligado fora do Windows.
`ROCKY_LOCAL_TTS_PLAYBACK=0` continua desativando a reprodução no PC Windows.

```powershell
cd server
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` fixa as versões diretas e aplica `requirements-lock.txt`
como constraints das dependências transitivas do ambiente validado. Para testes
HTTP, instalar `requirements-dev.txt` (mesmas dependências mais `httpx`).
Somente `.venv` é descartável; modelos locais, `.env` e WAVs não fazem parte dela.
O ambiente virtual continua ignorado pelo Git. Nenhum segredo entra nos manifests.

## Estabilização pós-auditoria

- Cada produtor realtime termina como `completed`, `failed` ou `cancelled`.
  A finalização é idempotente e cobre exceções de rede, inicialização da síntese
  e início da thread; `KeyboardInterrupt`/`SystemExit` não são engolidos.
- `STREAM_DEADLINE_SECONDS=120` é o deadline absoluto da geração SSE, separado
  do timeout de socket/leitura de 20 s. Um timer por stream encerra o socket
  se até uma linha SSE ficar presa; o timer é cancelado ao sair. Zero retry,
  uma chamada OpenRouter por pergunta, mesmos modelos e política gratuita.
- `REALTIME_TOTAL_DEADLINE_MS=180000` limita o polling do ESP32, inclusive
  espera por headers, download e ACK. O HTTP individual permanece em 60000 ms.
  Ao expirar, imprime `[RT] total deadline exceeded`; o caminho já existente
  em `main.cpp` retorna a IDLE. A checagem durante playback acontece entre
  blocos PCM, sem mudar volume, sample rate ou driver do speaker.
- Falha antes de áudio útil sintetiza uma mensagem local de indisponibilidade.
  Se já houve áudio útil, adiciona uma única mensagem de resposta interrompida.
  Esses WAVs ficam disponíveis na fila antes do status `failed`, sem nova LLM.
  Se Piper também falhar, finaliza sem áudio e o firmware retorna a IDLE.
- Nova interação limpa as anteriores sem progresso por 300 s, inclusive
  produtores órfãos e WAVs pendentes, somente no diretório realtime protegido.
  Não há scheduler. WAV parcial de síntese com erro também é removido.
  WAVs órfãos sem referência na fila são limpos na criação de nova interação
  após 300 s de idade; WAVs recentes e modelos são preservados. Isso evita
  apagar um segmento que outro produtor acabou de gerar antes do enqueue.
- `SentenceBuffer` preserva decimais/milhares e `Sr./Sra./Dr./Dra.`. Um ponto
  numérico no final do chunk aguarda lookahead ou fim do stream. `<think>`
  permanece agrupado até poder ser removido pelo sanitizer, inclusive em chunks.
  Respostas numéricas como `Resposta: 42.` e `42.` continuam faláveis.
- Uma instância lazy de `PiperTTS`, protegida por lock, atende os dois caminhos.
  Whisper (`get_model`, cache) e Piper (`_get_voice`) ainda carregam no primeiro
  uso; nenhum warm-up foi adicionado sem medição de cold start.

Validação automatizada (sem LLM/Whisper/Piper reais ou playback):

```powershell
cd server
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
.\.venv\Scripts\python.exe -m compileall -q app.py models services
```

Teste físico posterior, com flash manual: fala comum deve manter IDLE; `Koda`
deve abrir Attention por 6 s; testar `Koda, que horas são?` e `Koda, quem criou
o Linux?`. Em seguida simular backend indisponível e interrupção da rede durante
a resposta. Confirmar mensagem local quando o transporte/Piper permitir e retorno
controlado a IDLE em erro, sem espera infinita. Validação física não concluída.

Pendências fora desta rodada: resiliência Wi-Fi/backend no boot, autenticação,
user_id/device_id, contrato versionado, prova/limpeza de legado, refatoração de
main.cpp, Koda Central, Android, wake local/KWS, LLM local e movimento.

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
2. Reiniciar a API com o ambiente Python válido e o launcher existente, usando
   a politica de custo zero descrita abaixo. Não expor a API fora da LAN nem colocar chaves nos logs.
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
   fazer somente uma chamada HTTP ao OpenRouter, com os modelos free aprovados.
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
explicacoes comuns; mais detalhes somente quando pedidos. O provider padrao e o
ChatGPT Plan autenticado localmente, com `gpt-5.6-terra`, `instructions` para a
personalidade e contexto local enviado em `input`. Cada request usa Responses
com `store=false`, `stream=true`, `max_retries=0` e sem fallback automatico.
Logs temporarios de texto usam `SPEECH_TEXT_DIAGNOSTICS` e previews limitados
a 240 caracteres; nao incluem chave, tokens de autenticacao ou dados PCM.

Testes deterministas adicionais:

```powershell
cd server
.\.venv\Scripts\python.exe -m unittest test_chatgpt_auth test_chatgpt_plan test_speech_text test_conversation test_sentence_buffer test_realtime_audio test_realtime_streaming test_realtime_endpoint test_tts_response_speaker
```

Depois de reiniciar o backend atualizado e fazer flash manual, validar:
Koda -> olhos atentos sem numeros -> silencio -> Idle apos ~6 s;
Koda -> pergunta iniciada perto de 5,9 s -> captura command, sem expirar no STT;
Koda -> Que horas sao? -> comando deterministico sem LLM;
Koda -> Quem criou o Linux? -> resposta curta, sem Markdown/metatexto no audio.

### ChatGPT Plan padrao e OpenRouter legado

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

O caminho padrao usa Continue with ChatGPT e a sessao DPAPI do Windows. O
launcher restaura ou renova a sessao antes de iniciar o backend; nao pede chave
OpenRouter. Se a sessao precisar de novo login durante uma interacao, a chamada
falha de forma controlada: o navegador nao abre no fluxo de voz.

`server/services/llm/openrouter.py` permanece como alternativa legada para
desenvolvimento, mas nao e provider padrao nem fallback automatico do ChatGPT
Plan. As politicas abaixo descrevem somente esse caminho legado.

Ling 3.1 Flash atualmente tem entrada e saida gratuitas, mesmo sem `:free` no
slug. Isso nao garante gratuidade futura: `max_price=0` preserva custo zero e
impede o uso de endpoints que passem a cobrar. A privacidade continua protegida
por `data_collection=deny`.

`openrouter/free`, `openrouter/auto`, modelos pagos, outros `:free` e configuracao
vazia sao bloqueados antes da request. `OPENROUTER_MODEL` continua sendo um
override manual de UM dos dois modelos autorizados, usado apenas ao iniciar o
caminho legado explicitamente no desenvolvimento. Nao ha configuracao
persistente nem leitura de `.env` pelo launcher.

A request legada usa somente o campo singular `model`, nunca `models`, e inclui:

```json
{
  "model": "inclusionai/ling-3.1-flash",
  "stream": true,
  "temperature": 0.2,
  "max_tokens": 120,
  "provider": {
    "max_price": {"prompt": 0, "completion": 0},
    "data_collection": "deny",
    "allow_fallbacks": true
  },
  "reasoning": {"exclude": true}
}
```

Uma pergunta aberta faz exatamente UMA chamada HTTP Python ao OpenRouter.
Nao ha loop de retry, segunda geracao, resumo por outra LLM ou troca automatica
para Gemma. `provider.allow_fallbacks=true` permite somente providers elegiveis
do mesmo modelo, sem fallback automatico entre modelos.
Nenhum provider e fixado por nome. Endpoints incompativeis com preco zero ou
`data_collection=deny` nao sao elegiveis; custo/privacidade nunca sao relaxados.
Se Ling falhar, a mensagem local fixa de indisponibilidade e sintetizada
uma vez, disponibilizada na AudioQueue e entregue antes de o consumidor encerrar
a interacao `failed` e retornar a IDLE, sem nova chamada LLM.

Reasoning: Ling recebe somente `exclude=true`, sem `enabled=false` ou effort
especifico de modelo. O override individual Gemma preserva `enabled=false`.
Excluir reasoning da resposta nao garante desativar o pensamento interno nem
libera seu orcamento de tokens; uma resposta final vazia segue o erro controlado
existente. O parser usa somente `message.content` / `delta.content`, nunca campos
`reasoning` ou `reasoning_details`. SpeechTextSanitizer permanece nos caminhos
incremental e sincrono, sem resumir semanticamente a resposta.

Diagnosticos HTTP (inclusive 429): registram status, `error.code` numerico ou
categoria aprovada, e uma mensagem curta projetada em categorias fixas.
Trechos livres da mensagem remota nunca sao ecoados. O escopo de rate limit so
e indicado quando mencionado no erro; sem evidencia, aparece `unspecified`.
Metadata passa por allowlists: `error_type`, `provider_code`, `provider_name` e
`model`; campos desconhecidos, `raw`, headers e identificadores sao omitidos.
O corpo de erro e lido ate 8193 bytes; acima do limite de 8192 bytes ou com JSON
invalido, so o status e a indisponibilidade dos detalhes sao registrados.
Chave, Authorization, prompt, contexto, audio e dados pessoais nao sao impressos
por esses diagnosticos. `routed_model` vem da resposta/SSE, sem request extra,
e e opcional; slugs conhecidos (tambem sem `:free`) sao registrados uma vez por
stream, e valores desconhecidos sao substituidos por `[unrecognized]`.
Um erro apos conteudo parcial nao reinicia o stream: preserva o aviso local de
interrupcao e as sequencias de audio existentes.

Referencias oficiais: [errors and debugging](https://openrouter.ai/docs/api_reference/errors-and-debugging),
[provider routing](https://openrouter.ai/docs/guides/routing/provider-selection),
[reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) e
[Ling 3.1 Flash](https://openrouter.ai/inclusionai/ling-3.1-flash).

Guardrail externo informado pelo usuario: `Koda - Free Models`, associado apenas
a chave normal do Koda. Segunda barreira recomendada: Only Allow dos dois modelos
autorizados (Ling e Gemma), sem provider allowlist; Prompt Injection regex = Flag, User messages only;
Sensitive Info com Email address, Phone number, Social Security number, Credit
card number e API keys and secrets. O codigo nao cria/edita Guardrails, nao usa
Management API/key e nao documenta segredos. Flag nao significa bloqueio automatico.

Validacao automatizada usa somente mocks. Testes HTTP requerem .venv funcional;
o Python auxiliar nao substitui o runtime do backend. Teste fisico posterior:
conversa comum em Idle sem reacao; Koda -> Attention -> silencio (~6 s) -> Idle;
Koda -> pergunta durante Attention; Koda + horas e Koda + pergunta na mesma fala.
