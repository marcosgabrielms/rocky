"""Testes determinísticos da fila local de áudio por interação."""

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

if "piper" not in sys.modules:
    piper_module = types.ModuleType("piper")
    piper_module.PiperVoice = object
    sys.modules["piper"] = piper_module

from services.audio_queue import AudioItemStatus, AudioProductionStatus, RealtimeAudioQueue
from services.tts.realtime import RealtimeAudioSegment


class AudioQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.directory = Path(self._temporary_directory.name)
        self.now = 0.0
        self.queue = RealtimeAudioQueue(self.directory, clock=lambda: self.now)
        self._segment_index = 0

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def segment(self, sequence: int, text: str = "Resposta.") -> RealtimeAudioSegment:
        audio_path = self.directory / f"rocky_rt_{self._segment_index:04d}_{sequence:03d}.wav"
        self._segment_index += 1
        audio_path.write_bytes(b"wav")
        return RealtimeAudioSegment(
            sequence=sequence,
            text=text,
            audio_path=audio_path,
            sample_rate=32000,
            channels=1,
            bits_per_sample=16,
            duration_ms=100,
            audio_bytes=audio_path.stat().st_size,
        )

    def test_new_interaction_starts_empty(self) -> None:
        interaction_id = self.queue.create_interaction()

        self.assertTrue(self.queue.is_empty(interaction_id))
        self.assertIsNone(self.queue.next_item(interaction_id))
        self.assertEqual(self.queue.state(interaction_id).production_status, AudioProductionStatus.ACTIVE)

    def test_enqueue_and_consume_one_item(self) -> None:
        interaction_id = self.queue.create_interaction()
        item = self.queue.enqueue(interaction_id, self.segment(0))

        self.assertEqual(self.queue.next_item(interaction_id), item)
        consumed = self.queue.mark_consumed(interaction_id, 0)

        self.assertEqual(consumed.status, AudioItemStatus.CONSUMED)
        self.assertTrue(self.queue.is_empty(interaction_id))
        self.assertFalse(consumed.audio_path.exists())

    def test_consumption_is_strictly_ordered(self) -> None:
        interaction_id = self.queue.create_interaction()
        for sequence in range(3):
            self.queue.enqueue(interaction_id, self.segment(sequence, f"Resposta {sequence}."))

        for sequence in range(3):
            self.assertEqual(self.queue.next_item(interaction_id).sequence, sequence)
            self.queue.mark_consumed(interaction_id, sequence)

        self.assertIsNone(self.queue.next_item(interaction_id))

    def test_interactions_are_isolated(self) -> None:
        first = self.queue.create_interaction()
        second = self.queue.create_interaction()
        self.queue.enqueue(first, self.segment(0, "Primeira."))
        self.queue.enqueue(second, self.segment(0, "Segunda."))

        self.queue.mark_consumed(first, 0)

        self.assertTrue(self.queue.is_empty(first))
        self.assertEqual(self.queue.next_item(second).text, "Segunda.")

    def test_production_done_keeps_pending_audio_available(self) -> None:
        interaction_id = self.queue.create_interaction()
        self.queue.enqueue(interaction_id, self.segment(0))
        self.queue.mark_production_done(interaction_id)

        self.assertEqual(self.queue.state(interaction_id).production_status, AudioProductionStatus.COMPLETED)
        self.assertEqual(self.queue.next_item(interaction_id).sequence, 0)

    def test_invalid_or_duplicate_sequence_is_rejected(self) -> None:
        interaction_id = self.queue.create_interaction()
        with self.assertRaises(ValueError):
            self.queue.enqueue(interaction_id, self.segment(1))

        self.queue.enqueue(interaction_id, self.segment(0))
        with self.assertRaises(ValueError):
            self.queue.enqueue(interaction_id, self.segment(0))

    def test_cancel_preserves_other_interactions_and_rejects_new_items(self) -> None:
        cancelled = self.queue.create_interaction()
        unaffected = self.queue.create_interaction()
        first = self.queue.enqueue(cancelled, self.segment(0))
        self.queue.enqueue(unaffected, self.segment(0, "Outra."))

        self.queue.cancel(cancelled)

        self.assertEqual(self.queue.state(cancelled).production_status, AudioProductionStatus.CANCELLED)
        self.assertFalse(first.audio_path.exists())
        self.assertIsNone(self.queue.next_item(cancelled))
        self.assertEqual(self.queue.next_item(unaffected).text, "Outra.")
        with self.assertRaises(ValueError):
            self.queue.enqueue(cancelled, self.segment(1))

    def test_cleanup_removes_only_terminal_and_empty_interaction(self) -> None:
        interaction_id = self.queue.create_interaction()
        self.queue.enqueue(interaction_id, self.segment(0))
        self.queue.mark_production_done(interaction_id)

        self.assertFalse(self.queue.cleanup_interaction(interaction_id))
        self.queue.mark_consumed(interaction_id, 0)
        self.assertTrue(self.queue.cleanup_interaction(interaction_id))
        with self.assertRaises(ValueError):
            self.queue.state(interaction_id)

    def test_expired_terminal_interaction_removes_pending_audio(self) -> None:
        interaction_id = self.queue.create_interaction()
        item = self.queue.enqueue(interaction_id, self.segment(0))
        self.queue.mark_production_done(interaction_id)
        self.now += 61

        self.assertEqual(self.queue.cleanup_expired(60), (interaction_id,))
        self.assertFalse(item.audio_path.exists())
        with self.assertRaises(ValueError):
            self.queue.state(interaction_id)


if __name__ == "__main__":
    unittest.main()
