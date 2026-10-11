"""Coordena LLM streaming e síntese local por sentença, sem transporte físico."""

from dataclasses import dataclass

from models.actions import BackendResponse
from services.audio_queue import RealtimeAudioQueue
from services.conversation import (
    ConversationManager, InteractionDecision, StreamedConversationResponse,
    LLM_UNAVAILABLE_MESSAGE, LLM_INTERRUPTED_MESSAGE,
)
from services.tts.realtime import RealtimeAudioSegment, RealtimeTTS, RealtimeTTSSession
from services.latency import latency_metric, mark_latency, report_latency_summary
from services.runtime_log import error as log_error, info, technical
from services.tts.response_speaker import get_spoken_text
from services.tts.speech_text import SpeechTextSanitizer, log_speech_text


@dataclass(frozen=True)
class StreamedAudioResponse:
    """Resultado lógico de uma resposta realtime e seus WAVs locais válidos."""

    conversation: StreamedConversationResponse
    audio_segments: tuple[RealtimeAudioSegment, ...]
    tts_failed: bool
    interaction_id: int | None


class RealtimeAudioService:
    """Dispara Piper em ordem assim que cada sentença do LLM fica disponível."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        realtime_tts: RealtimeTTS,
        audio_queue: RealtimeAudioQueue | None = None,
    ) -> None:
        self._conversation_manager = conversation_manager
        self._realtime_tts = realtime_tts
        self._audio_queue = audio_queue or RealtimeAudioQueue(realtime_tts.output_directory)

    @property
    def audio_queue(self) -> RealtimeAudioQueue:
        return self._audio_queue

    def create_interaction(self) -> int:
        interaction_id = self._audio_queue.create_interaction()
        try:
            self._realtime_tts.cleanup_orphaned_audio(self._audio_queue.ready_audio_paths())
        except Exception as error:
            log_error("Falha ao limpar áudio realtime.")
            technical(f"[RT] cleanup_error type={type(error).__name__}")
        return interaction_id

    def queue_direct_response(self, response: BackendResponse, request_text: str) -> int | None:
        """Enfileira uma resposta deterministica sem usar playback local do Windows."""
        spoken_text = get_spoken_text(response, request_text)
        if spoken_text is None:
            return None

        spoken_text = SpeechTextSanitizer().sanitize(spoken_text)
        if not spoken_text:
            return None

        interaction_id = self.create_interaction()
        try:
            session = self._realtime_tts.begin_interaction(
                interaction_id, self._audio_queue.ready_audio_paths(),
            )
            segment = session.synthesize(spoken_text, 0)
            self._audio_queue.enqueue(interaction_id, segment)
            self._audio_queue.finish_if_active(interaction_id, failed=False)
            technical(f"[RT] deterministic_audio_ready id={interaction_id}")
            return interaction_id
        except Exception as error:
            self._audio_queue.finish_if_active(interaction_id, failed=True)
            log_error("Falha ao sintetizar resposta determinística.")
            technical(f"[RT] deterministic_tts_error type={type(error).__name__}")
            return None

    def build_stream_audio_response(
        self, text: str, device_id: str, interaction_id: int | None = None
    ) -> StreamedAudioResponse:
        decision = self._conversation_manager.classify(text, device_id)
        return self.build_stream_audio_response_from_decision(decision, interaction_id)

    def build_stream_audio_response_from_decision(
        self, decision: InteractionDecision, interaction_id: int | None = None
    ) -> StreamedAudioResponse:
        session: RealtimeTTSSession | None = None
        segments: list[RealtimeAudioSegment] = []
        conversation: StreamedConversationResponse | None = None
        tts_failed = False
        production_failed = True
        sanitizer = SpeechTextSanitizer()

        def synthesize_sentence(sentence: str, sequence: int, ready_ms: int) -> None:
            nonlocal interaction_id, session, tts_failed
            if tts_failed:
                return

            log_speech_text("LLM", "raw_sentence", sequence, sentence)
            spoken_text = sanitizer.sanitize(sentence)
            log_speech_text("TTS", "sanitized_sentence", sequence, spoken_text)
            if not spoken_text:
                return

            technical(f"[RT] sentence_ready seq={sequence} ms={ready_ms}")
            try:
                if interaction_id is None:
                    interaction_id = self.create_interaction()
                if session is None:
                    session = self._realtime_tts.begin_interaction(
                        interaction_id, self._audio_queue.ready_audio_paths(),
                    )
                audio_sequence = len(segments)
                segment = session.synthesize(spoken_text, audio_sequence)
                self._audio_queue.enqueue(interaction_id, segment)
                if audio_sequence == 0:
                    mark_latency("audio_available")
                    latency_metric("audio_ready_ms", "stt_start", "audio_available")
                    latency_metric("first_audio_available_ms", "llm_request_start", "audio_available")
                segments.append(segment)
            except Exception as error:
                tts_failed = True
                log_error("Falha ao sintetizar áudio realtime.")
                technical(f"[RT] tts_error seq={sequence} type={type(error).__name__}")

        try:
            try:
                conversation = self._conversation_manager.build_stream_response_from_decision(
                    decision, on_sentence=synthesize_sentence,
                )
            except Exception as error:
                log_error("Falha ao processar resposta realtime.")
                technical(f"[RT] producer_error type={type(error).__name__}")
                message = LLM_INTERRUPTED_MESSAGE if segments else LLM_UNAVAILABLE_MESSAGE
                conversation = StreamedConversationResponse(
                    response={"text": message, "interaction_state": "idle", "actions": []},
                    sentences=(), stream_failed=True,
                )
            if conversation.stream_failed:
                message = LLM_INTERRUPTED_MESSAGE if segments else LLM_UNAVAILABLE_MESSAGE
                synthesize_sentence(message, len(segments), 0)
            production_failed = tts_failed or conversation.stream_failed
        finally:
            # BaseException continua propagando, mas nao deixa uma fila ACTIVE.
            if interaction_id is not None:
                self._audio_queue.finish_if_active(interaction_id, failed=production_failed)
            if conversation is not None and conversation.response.get("text"):
                info(f'Resposta: "{conversation.response["text"]}"')
            report_latency_summary()
        if conversation is None:
            raise RuntimeError("A resposta realtime não foi concluída.")
        return StreamedAudioResponse(
            conversation=conversation,
            audio_segments=tuple(segments),
            tts_failed=tts_failed,
            interaction_id=interaction_id,
        )
