"""Coordena LLM streaming e síntese local por sentença, sem transporte físico."""

from dataclasses import dataclass

from services.audio_queue import RealtimeAudioQueue
from services.conversation import ConversationManager, InteractionDecision, StreamedConversationResponse
from services.tts.realtime import RealtimeAudioSegment, RealtimeTTS, RealtimeTTSSession


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
        return self._audio_queue.create_interaction()

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
        tts_failed = False

        def synthesize_sentence(sentence: str, sequence: int, ready_ms: int) -> None:
            nonlocal interaction_id, session, tts_failed
            if tts_failed:
                return

            print(f"[RT] sentence_ready seq={sequence} ms={ready_ms}")
            if session is None:
                if interaction_id is None:
                    interaction_id = self._audio_queue.create_interaction()
                session = self._realtime_tts.begin_interaction(
                    interaction_id,
                    self._audio_queue.ready_audio_paths(),
                )
            try:
                segment = session.synthesize(sentence, sequence)
                self._audio_queue.enqueue(interaction_id, segment)
                segments.append(segment)
            except Exception as error:
                tts_failed = True
                print(f"[RT] tts_error seq={sequence} error={error}")

        conversation = self._conversation_manager.build_stream_response_from_decision(
            decision,
            on_sentence=synthesize_sentence,
        )
        if interaction_id is not None:
            if tts_failed or conversation.stream_failed:
                self._audio_queue.mark_production_failed(interaction_id)
            else:
                self._audio_queue.mark_production_done(interaction_id)
        return StreamedAudioResponse(
            conversation=conversation,
            audio_segments=tuple(segments),
            tts_failed=tts_failed,
            interaction_id=interaction_id,
        )
