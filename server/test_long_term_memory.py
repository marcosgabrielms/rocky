"""Testes da memoria local explicita e separada do contexto recente."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.conversation import ConversationManager, InteractionMode
from services.conversation_context import ConversationContextStore
from services.llm.base import LLMResult
from services.long_term_memory import LongTermMemoryStore


class _LLMDouble:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def ask(self, message: str, context=None) -> LLMResult:
        self.calls.append((message, tuple(context or ())))
        return LLMResult(text=f"Resposta para: {message}")


class LongTermMemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.database_path = Path(self._temporary_directory.name) / "memory.sqlite3"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_memories_persist_across_restart_and_are_isolated_by_user(self) -> None:
        first_store = LongTermMemoryStore(self.database_path)
        first_memory = first_store.add("user-a", "preference", "Prefiro respostas curtas.", now=1.0)
        first_store.add("user-b", "project", "Estou no projeto B.", now=2.0)
        first_store.close()

        second_store = LongTermMemoryStore(self.database_path)
        memories = second_store.list("user-a")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, first_memory.id)
        self.assertEqual(memories[0].content, "Prefiro respostas curtas.")
        self.assertEqual(second_store.list("user-b")[0].category, "project")
        second_store.close()

    def test_list_delete_and_clear_are_scoped_to_the_user(self) -> None:
        store = LongTermMemoryStore(self.database_path)
        first = store.add("user-a", "note", "Nota A.", now=1.0)
        store.add("user-a", "taste", "Gosto de cafe.", now=2.0)
        other = store.add("user-b", "note", "Nota B.", now=3.0)

        self.assertTrue(store.delete("user-a", first.id))
        self.assertFalse(store.delete("user-a", other.id))
        self.assertEqual(tuple(memory.content for memory in store.list("user-a")), ("Gosto de cafe.",))
        self.assertEqual(store.clear("user-a"), 1)
        self.assertEqual(store.list("user-a"), ())
        self.assertEqual(tuple(memory.content for memory in store.list("user-b")), ("Nota B.",))
        store.close()

    def test_search_is_lexical_limited_and_isolated_by_user(self) -> None:
        store = LongTermMemoryStore(self.database_path)
        project_memory = store.add("user-a", "project", "Koda usa ponteiros em C.", now=1.0)
        store.add("user-a", "note", "Ponteiros em C precisam de cuidado.", now=2.0)
        store.add("user-a", "note", "Exemplo de ponteiros em C.", now=3.0)
        store.add("user-a", "note", "Outro uso de ponteiros em C.", now=4.0)
        store.add("user-a", "project", "Projeto de jardinagem.", now=5.0)
        store.add("user-b", "note", "Ponteiros privados de outro usuario.", now=6.0)

        found = store.search("user-a", "me explica ponteiros em C", limit=10)

        self.assertEqual(len(found), 3)
        self.assertTrue(all("ponteiros" in memory.content.casefold() for memory in found))
        self.assertIn(project_memory, store.search("user-a", "projeto"))
        self.assertEqual(store.search("user-b", "ponteiros"), (store.list("user-b")[0],))
        store.close()

    def test_explicit_request_saves_memory_without_touching_recent_context(self) -> None:
        store = LongTermMemoryStore(self.database_path)
        recent_context = ConversationContextStore()
        llm = _LLMDouble()
        manager = ConversationManager(
            0,
            llm,
            context_store=recent_context,
            memory_store=store,
            memory_clock=lambda: 10.0,
        )

        response = manager.build_response(
            "lembra que eu prefiro respostas curtas",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )

        self.assertEqual(response["text"], "Memoria salva.")
        self.assertEqual(llm.calls, [])
        self.assertEqual(recent_context.get_context("local-user", now=10.0), ())
        memory = store.list("local-user")[0]
        self.assertEqual(memory.category, "preference")
        self.assertEqual(memory.content, "eu prefiro respostas curtas")
        store.close()

    def test_reminder_sensitive_and_ordinary_conversation_are_not_saved(self) -> None:
        store = LongTermMemoryStore(self.database_path)
        llm = _LLMDouble()
        manager = ConversationManager(0, llm, memory_store=store, memory_clock=lambda: 10.0)

        reminder = manager.build_response(
            "me lembra de comprar cafe as 18h",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )
        sensitive = manager.build_response(
            "guarda que minha senha e segredo",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )
        manager.build_response(
            "eu gosto de cafe",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )

        self.assertEqual(reminder["text"], "Nao entendi o horario do lembrete.")
        self.assertEqual(sensitive["text"], "Nao posso guardar informacoes sensiveis.")
        self.assertEqual(store.list("local-user"), ())
        self.assertEqual(len(llm.calls), 1)
        store.close()

    def test_llm_receives_selected_memory_once_as_system_context(self) -> None:
        store = LongTermMemoryStore(self.database_path)
        store.add("local-user", "preference", "Prefiro respostas curtas.", now=1.0)
        store.add("local-user", "note", "Estou estudando ponteiros em C.", now=2.0)
        store.add("local-user", "project", "Tenho um projeto de jardinagem.", now=3.0)
        llm = _LLMDouble()
        manager = ConversationManager(0, llm, memory_store=store)

        manager.build_response(
            "Me explica ponteiros em C.",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )

        self.assertEqual(len(llm.calls), 1)
        context = llm.calls[0][1]
        memory_messages = [message for message in context if getattr(message, "role", None) == "system"]
        self.assertEqual(len(memory_messages), 1)
        self.assertIn("Prefiro respostas curtas.", memory_messages[0].content)
        self.assertIn("ponteiros em C", memory_messages[0].content)
        self.assertNotIn("jardinagem", memory_messages[0].content)
        store.close()

    def test_memory_query_and_exact_forget_are_deterministic_and_safe(self) -> None:
        store = LongTermMemoryStore(self.database_path)
        stored = store.add("local-user", "preference", "Eu prefiro respostas curtas.", now=1.0)
        llm = _LLMDouble()
        manager = ConversationManager(0, llm, memory_store=store)

        listed = manager.build_response(
            "o que voce lembra sobre mim",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )
        forgotten = manager.build_response(
            "esquece que eu prefiro respostas curtas",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )

        self.assertIn(stored.content, listed["text"])
        self.assertEqual(forgotten["text"], "Memoria esquecida.")
        self.assertEqual(store.list("local-user"), ())
        self.assertEqual(llm.calls, [])
        store.close()

    def test_forget_does_not_delete_ambiguous_or_partial_memory(self) -> None:
        store = LongTermMemoryStore(self.database_path)
        store.add("local-user", "note", "Estou trabalhando no projeto Koda.", now=1.0)
        store.add("local-user", "note", "Estou trabalhando no projeto Koda.", now=2.0)
        llm = _LLMDouble()
        manager = ConversationManager(0, llm, memory_store=store)

        response = manager.build_response(
            "esquece que estou trabalhando no projeto Koda",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )

        self.assertIn("nao apaguei", response["text"])
        self.assertEqual(len(store.list("local-user")), 2)
        self.assertEqual(llm.calls, [])
        store.close()


if __name__ == "__main__":
    unittest.main()
