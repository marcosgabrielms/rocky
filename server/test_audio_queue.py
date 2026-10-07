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

from services.audio_queue import AudioItemStatus, AudioProductionStatus, RealtimeAudioQueue, INTERACTION_MAXIMUM_AGE_SECONDS
from services.tts.realtime import RealtimeAudioSegment


class AudioQueueTest(unittest.TestCase):
    def test_new_interaction_cleans_expired_terminal_and_orphaned_producer(self) -> None:
        terminal = self.queue.create_interaction()
        orphan = self.queue.create_interaction()
        terminal_item = self.queue.enqueue(terminal, self.segment(0))
        orphan_item = self.queue.enqueue(orphan, self.segment(0))
        self.queue.mark_production_failed(terminal)
        self.now += INTERACTION_MAXIMUM_AGE_SECONDS + 1
        new = self.queue.create_interaction()
        self.assertTrue(self.queue.has_interaction(new))
        self.assertFalse(self.queue.has_interaction(terminal))
        self.assertFalse(self.queue.has_interaction(orphan))
        self.assertFalse(terminal_item.audio_path.exists())
        self.assertFalse(orphan_item.audio_path.exists())
        self.queue.finish_if_active(orphan, failed=True)

    def test_finalization_is_idempotent_and_preserves_cancelled(self) -> None:
        cancelled = self.queue.create_interaction()
        done = self.queue.create_interaction()
        self.queue.cancel(cancelled)
        self.queue.finish_if_active(cancelled, failed=True)
        self.queue.finish_if_active(done, failed=False)
        self.queue.finish_if_active(done, failed=True)
        self.assertEqual(self.queue.state(cancelled).production_status, AudioProductionStatus.CANCELLED)
        self.assertEqual(self.queue.state(done).production_status, AudioProductionStatus.COMPLETED)

    def test_cleanup_never_deletes_outside_controlled_directory(self) -> None:
        from dataclasses import replace
        with TemporaryDirectory() as outside:
            path = Path(outside) / "preserve.wav"
            path.write_bytes(b"preserve")
            interaction = self.queue.create_interaction()
            self.queue.enqueue(interaction, replace(self.segment(0), audio_path=path))
            self.queue.mark_production_done(interaction)
            self.now += 301
            with self.assertRaises(ValueError):
                self.queue.cleanup_expired(300)
            self.assertEqual(path.read_bytes(), b"preserve")

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
