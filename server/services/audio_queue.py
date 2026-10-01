"""Fila local e ordenada de segmentos de áudio por interação."""

import time
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from threading import Lock
from collections.abc import Callable

from services.tts.realtime import RealtimeAudioSegment


class AudioItemStatus(StrEnum):
    READY = "ready"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


class AudioProductionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class QueuedAudioSegment:
    interaction_id: int
    sequence: int
    text: str
    audio_path: Path
    sample_rate: int
    channels: int
    bits_per_sample: int
    duration_ms: int
    audio_bytes: int
    status: AudioItemStatus


@dataclass(frozen=True)
class AudioInteractionState:
    interaction_id: int
    production_status: AudioProductionStatus
    next_sequence: int
    ready_count: int
    item_count: int


@dataclass
class _Interaction:
    items: dict[int, QueuedAudioSegment]
    next_sequence: int
    next_to_consume: int
    production_status: AudioProductionStatus
    updated_at: float


class RealtimeAudioQueue:
    """Armazena referências de WAVs por interação e impõe consumo estritamente ordenado."""

    def __init__(self, audio_directory: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self._audio_directory = audio_directory.resolve()
        self._clock = clock
        self._lock = Lock()
        self._next_interaction_id = 0
        self._interactions: dict[int, _Interaction] = {}

    def create_interaction(self) -> int:
        with self._lock:
            interaction_id = self._next_interaction_id
            self._next_interaction_id += 1
            self._interactions[interaction_id] = _Interaction(
                items={},
                next_sequence=0,
                next_to_consume=0,
                production_status=AudioProductionStatus.ACTIVE,
                updated_at=self._clock(),
            )
        print(f"[RTQ] create interaction={interaction_id}")
        return interaction_id

    def has_interaction(self, interaction_id: int) -> bool:
        with self._lock:
            return interaction_id in self._interactions

    def enqueue(self, interaction_id: int, segment: RealtimeAudioSegment) -> QueuedAudioSegment:
        with self._lock:
            interaction = self._get_active_interaction(interaction_id)
            if segment.sequence != interaction.next_sequence:
                raise ValueError(
                    f"Expected sequence {interaction.next_sequence}, received {segment.sequence}."
                )
            if segment.sequence in interaction.items:
                raise ValueError(f"Duplicate sequence {segment.sequence}.")

            item = QueuedAudioSegment(
                interaction_id=interaction_id,
                sequence=segment.sequence,
                text=segment.text,
                audio_path=segment.audio_path,
                sample_rate=segment.sample_rate,
                channels=segment.channels,
                bits_per_sample=segment.bits_per_sample,
                duration_ms=segment.duration_ms,
                audio_bytes=segment.audio_bytes,
                status=AudioItemStatus.READY,
            )
            interaction.items[item.sequence] = item
            interaction.next_sequence += 1
            interaction.updated_at = self._clock()
        print(f"[RTQ] enqueue interaction={interaction_id} seq={item.sequence}")
        return item

    def next_item(self, interaction_id: int) -> QueuedAudioSegment | None:
        with self._lock:
            interaction = self._get_interaction(interaction_id)
            item = interaction.items.get(interaction.next_to_consume)
            if item is None or item.status is not AudioItemStatus.READY:
                return None
        print(f"[RTQ] dequeue interaction={interaction_id} seq={item.sequence}")
        return item

    def mark_consumed(self, interaction_id: int, sequence: int) -> QueuedAudioSegment:
        with self._lock:
            interaction = self._get_interaction(interaction_id)
            if sequence != interaction.next_to_consume:
                raise ValueError(f"Expected sequence {interaction.next_to_consume}, received {sequence}.")
            item = interaction.items.get(sequence)
            if item is None or item.status is not AudioItemStatus.READY:
                raise ValueError(f"Sequence {sequence} is not ready for consumption.")

            consumed = replace(item, status=AudioItemStatus.CONSUMED)
            interaction.items[sequence] = consumed
            interaction.next_to_consume += 1
            interaction.updated_at = self._clock()
            self._remove_audio_file(consumed.audio_path)
        print(f"[RTQ] consumed interaction={interaction_id} seq={sequence}")
        return consumed

    def mark_production_done(self, interaction_id: int) -> None:
        self._mark_production(interaction_id, AudioProductionStatus.COMPLETED)

    def mark_production_failed(self, interaction_id: int) -> None:
        self._mark_production(interaction_id, AudioProductionStatus.FAILED)

    def cancel(self, interaction_id: int) -> None:
        with self._lock:
            interaction = self._get_interaction(interaction_id)
            if interaction.production_status is AudioProductionStatus.CANCELLED:
                return
            interaction.production_status = AudioProductionStatus.CANCELLED
            interaction.updated_at = self._clock()
            for sequence, item in tuple(interaction.items.items()):
                if item.status is AudioItemStatus.READY:
                    cancelled = replace(item, status=AudioItemStatus.CANCELLED)
                    interaction.items[sequence] = cancelled
                    self._remove_audio_file(cancelled.audio_path)
        print(f"[RTQ] cancelled interaction={interaction_id}")

    def state(self, interaction_id: int) -> AudioInteractionState:
        with self._lock:
            interaction = self._get_interaction(interaction_id)
            ready_count = sum(item.status is AudioItemStatus.READY for item in interaction.items.values())
            return AudioInteractionState(
                interaction_id=interaction_id,
                production_status=interaction.production_status,
                next_sequence=interaction.next_to_consume,
                ready_count=ready_count,
                item_count=len(interaction.items),
            )

    def is_empty(self, interaction_id: int) -> bool:
        return self.state(interaction_id).ready_count == 0

    def ready_audio_paths(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(
                item.audio_path
                for interaction in self._interactions.values()
                for item in interaction.items.values()
                if item.status is AudioItemStatus.READY
            )

    def cleanup_interaction(self, interaction_id: int) -> bool:
        with self._lock:
            interaction = self._get_interaction(interaction_id)
            if interaction.production_status is AudioProductionStatus.ACTIVE or any(
                item.status is AudioItemStatus.READY for item in interaction.items.values()
            ):
                return False
            self._interactions.pop(interaction_id)
        print(f"[RTQ] cleanup interaction={interaction_id}")
        return True

    def cleanup_expired(self, maximum_age_seconds: float) -> tuple[int, ...]:
        """Descarta interações terminais antigas, inclusive WAVs ainda pendentes."""
        now = self._clock()
        removed_ids: list[int] = []
        with self._lock:
            for interaction_id, interaction in tuple(self._interactions.items()):
                if (
                    interaction.production_status is AudioProductionStatus.ACTIVE
                    or now - interaction.updated_at < maximum_age_seconds
                ):
                    continue
                for sequence, item in tuple(interaction.items.items()):
                    if item.status is AudioItemStatus.READY:
                        interaction.items[sequence] = replace(item, status=AudioItemStatus.CANCELLED)
                        self._remove_audio_file(item.audio_path)
                self._interactions.pop(interaction_id)
                removed_ids.append(interaction_id)
        for interaction_id in removed_ids:
            print(f"[RTQ] cleanup_expired interaction={interaction_id}")
        return tuple(removed_ids)

    def _mark_production(self, interaction_id: int, status: AudioProductionStatus) -> None:
        with self._lock:
            interaction = self._get_active_interaction(interaction_id)
            interaction.production_status = status
            interaction.updated_at = self._clock()
        log_status = "production_done" if status is AudioProductionStatus.COMPLETED else "production_failed"
        print(f"[RTQ] {log_status} interaction={interaction_id}")

    def _get_active_interaction(self, interaction_id: int) -> _Interaction:
        interaction = self._get_interaction(interaction_id)
        if interaction.production_status is not AudioProductionStatus.ACTIVE:
            raise ValueError(f"Interaction {interaction_id} is not active.")
        return interaction

    def _get_interaction(self, interaction_id: int) -> _Interaction:
        try:
            return self._interactions[interaction_id]
        except KeyError as error:
            raise ValueError(f"Unknown interaction {interaction_id}.") from error

    def _remove_audio_file(self, audio_path: Path) -> None:
        resolved_path = audio_path.resolve()
        if resolved_path.parent != self._audio_directory:
            raise ValueError("Audio path is outside the realtime audio directory.")
        if resolved_path.is_file():
            resolved_path.unlink()
