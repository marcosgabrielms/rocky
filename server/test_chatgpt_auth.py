"""Auth offline: tokens fictícios, DPAPI real local e rede/browser bloqueados."""

import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from services.auth.chatgpt_session import ChatGPTSession, Credentials, epoch
from services.auth.oauth import (AuthError, Discovery, DYNAMIC_CLIENT, ISSUER, LoginResult,
                                 OAuthClient, OAuthHTTP, OAuthHTTPError, RESOURCE, SCOPES,
                                 Transaction, identity_fingerprint)
from services.auth.windows_store import SessionStore, WindowsDPAPI, WindowsSessionLock, validate_profile

CLIENT = "oaiapp_offline_koda"
HOST = "urn:uuid:65a32f50-8e21-4f8e-9291-0497eb514b0c"
NOW = 1800000000.0
PROFILE = {"ext_agent_host_id": HOST, "issued_client_id": CLIENT, "identity_fingerprint": "a" * 64}
DISCOVERY_DOCUMENT = {"issuer": ISSUER, "authorization_endpoint": ISSUER + "/api/accounts/authorize",
                      "token_endpoint": ISSUER + "/api/accounts/oauth/token", "jwks_uri": ISSUER + "/.well-known/jwks.json",
                      "id_token_signing_alg_values_supported": ["RS256"], "response_types_supported": ["code"],
                      "code_challenge_methods_supported": ["S256"]}


def lock_worker(directory, entered):
    with WindowsSessionLock(Path(directory)):
        entered.set()


@unittest.skipUnless(os.name == "nt", "Persistência DPAPI específica do Windows")
class ChatGPTAuthTest(unittest.TestCase):
    def setUp(self):
        for target in ("urllib.request.OpenerDirector.open", "webbrowser.open"):
            guard = patch(target, side_effect=AssertionError("No live network/browser in auth tests"))
            guard.start()
            self.addCleanup(guard.stop)
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="auth-offline-")
        self.addCleanup(temporary.cleanup)
        self.store = SessionStore(Path(temporary.name))
        self.store.save_profile(PROFILE)
        self.oauth = Mock(spec=OAuthClient)
        self.oauth.http = Mock()
        self.oauth.refresh.side_effect = lambda *_: (self.tokens(), Discovery.parse(DISCOVERY_DOCUMENT))
        self.oauth.login.side_effect = lambda *_, **__: LoginResult(self.tokens(), CLIENT, "a" * 64, "offline-nonce")
        self.messages = []
        self.session = ChatGPTSession(self.store, self.oauth, lambda: NOW, self.messages.append)

    @staticmethod
    def tokens(**overrides):
        return {"access_token": "access-new-secret", "refresh_token": "refresh-new-secret",
                "token_type": "Bearer", "expires_in": 3600, "scope": SCOPES,
                "earliest_refresh_at": NOW + 10, **overrides}

    def save(self, expires=NOW + 3600, earliest=NOW - 60):
        credentials = Credentials("access-old-secret", "refresh-old-secret", "id-old-secret",
                                  expires, earliest, SCOPES, "offline-nonce", **PROFILE)
        self.store.save_credentials(asdict(credentials))
        return credentials

    def test_dpapi_roundtrip_current_user(self):
        plaintext = b"fake-token-for-offline-dpapi"
        blob = self.store.protector.protect(plaintext)
        self.assertNotIn(plaintext, blob)
        self.assertEqual(self.store.protector.unprotect(blob), plaintext)

    def test_credentials_file_has_no_plaintext_token(self):
        self.save()
        raw = self.store.credentials_path.read_bytes()
        for value in (b"access-old-secret", b"refresh-old-secret", b"id-old-secret"):
            self.assertNotIn(value, raw)
        with self.assertRaises((ValueError, UnicodeError)):
            json.loads(raw)

    def test_profile_never_contains_tokens(self):
        self.save()
        document = json.loads(self.store.profile_path.read_bytes())
        self.assertEqual(document, PROFILE)
        for field in ("access_token", "refresh_token", "id_token", "scope", "nonce"):
            with self.subTest(field=field), self.assertRaises(AuthError):
                self.store.save_profile({**PROFILE, field: "secret"})

    def test_access_valid_restores_without_network_or_browser(self):
        self.save()
        self.assertEqual(self.session.ensure(), "restored")
        self.oauth.refresh.assert_not_called()
        self.oauth.login.assert_not_called()
        self.assertIn("Sessao ChatGPT restaurada.", self.messages)

    def test_expired_access_refreshes_once(self):
        self.save(expires=NOW - 1)
        self.assertEqual(self.session.ensure(), "refreshed")
        self.oauth.refresh.assert_called_once_with(CLIENT, "refresh-old-secret")
        self.oauth.login.assert_not_called()
        self.assertEqual(self.session.ensure(), "restored")
        self.oauth.refresh.assert_called_once()

    def test_inference_access_refreshes_silently_without_login(self):
        self.save(expires=NOW - 1)
        self.assertEqual(self.session.access_token_for_inference(), "access-new-secret")
        self.oauth.refresh.assert_called_once_with(CLIENT, "refresh-old-secret")
        self.oauth.login.assert_not_called()
        self.assertIn("Sessao ChatGPT renovada.", self.messages)

    def test_inference_access_never_opens_interactive_login(self):
        with self.assertRaisesRegex(AuthError, "authentication_required"):
            self.session.access_token_for_inference()
        self.oauth.login.assert_not_called()

    def test_pending_rotation_reports_a_safe_reauthentication_reason(self):
        self.save()
        self.session.pending_path.write_text("rotating\n", encoding="ascii")

        with self.assertRaisesRegex(AuthError, "refresh_rotation_pending"):
            self.session.access_token_for_inference()

        self.oauth.refresh.assert_not_called()
        self.oauth.login.assert_not_called()

    def test_near_expiry_refreshes(self):
        self.save(expires=NOW + 30)
        self.assertEqual(self.session.ensure(), "refreshed")

    def test_earliest_refresh_at_prevents_early_refresh(self):
        self.save(expires=NOW + 30, earliest=NOW + 20)
        self.assertEqual(self.session.ensure(), "restored")
        self.oauth.refresh.assert_not_called()

    def test_expired_before_earliest_refresh_stops_without_login(self):
        self.save(expires=NOW - 1, earliest=NOW + 20)
        with self.assertRaisesRegex(AuthError, "refresh_not_yet_allowed"):
            self.session.ensure()
        self.oauth.refresh.assert_not_called()
        self.oauth.login.assert_not_called()

    def test_rotated_token_replaces_old_only_after_successful_save(self):
        self.save(expires=NOW - 1)
        self.session.ensure()
        stored = self.store.read_credentials()
        self.assertEqual(stored["refresh_token"], "refresh-new-secret")
        self.assertEqual(stored["access_token"], "access-new-secret")
        self.assertEqual(stored["id_token"], "id-old-secret")
        self.assertEqual(stored["expires_at"], NOW + 3600)
        self.assertFalse(self.session.pending_path.exists())

    def test_write_failure_preserves_previous_ciphertext_and_blocks_old_refresh_reuse(self):
        self.save(expires=NOW - 1)
        before = self.store.credentials_path.read_bytes()
        original_replace = os.replace

        def fail_credentials(source, destination):
            if Path(destination) == self.store.credentials_path:
                raise OSError("access-new-secret")
            original_replace(source, destination)

        with patch("services.auth.windows_store.os.replace", side_effect=fail_credentials), \
                self.assertRaisesRegex(AuthError, "auth_storage_write_failed"):
            self.session.ensure()
        self.assertEqual(before, self.store.credentials_path.read_bytes())
        self.assertTrue(self.session.pending_path.exists())
        self.oauth.login.side_effect = lambda *_, **__: LoginResult(
            self.tokens(id_token="id-relogin-secret"), CLIENT, "a" * 64, "offline-nonce")
        self.assertEqual(self.session.ensure(), "authenticated")
        self.oauth.refresh.assert_called_once()

    def test_concurrent_threads_perform_exactly_one_refresh(self):
        self.save(expires=NOW - 1)
        sessions = [ChatGPTSession(self.store, self.oauth, lambda: NOW, lambda _: None) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda session: session.ensure(), sessions))
        self.assertEqual(results.count("refreshed"), 1)
        self.assertEqual(results.count("restored"), 3)
        self.oauth.refresh.assert_called_once()

    def test_mutex_serializes_processes(self):
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        process = context.Process(target=lock_worker, args=(str(self.store.directory), entered))
        with self.store.lock():
            process.start()
            self.assertFalse(entered.wait(0.3))
        self.assertTrue(entered.wait(10))
        process.join(10)
        self.assertEqual(process.exitcode, 0)

    def test_revoked_refresh_requests_login_with_saved_profile(self):
        self.save(expires=NOW - 1)
        self.oauth.refresh.side_effect = OAuthHTTPError(400, "invalid_grant")
        self.oauth.login.side_effect = lambda *_, **__: LoginResult(
            self.tokens(id_token="id-relogin-secret"), CLIENT, "a" * 64, "offline-nonce")
        self.assertEqual(self.session.ensure(), "authenticated")
        call = self.oauth.login.call_args
        self.assertEqual(call.args[0], PROFILE)
        self.assertIsNone(call.kwargs["id_token_hint"])

    def test_transient_http_error_preserves_credentials_and_does_not_open_browser(self):
        self.save(expires=NOW - 1)
        before = self.store.credentials_path.read_bytes()
        self.oauth.refresh.side_effect = OAuthHTTPError(503, "server_error")
        with self.assertRaises(OAuthHTTPError):
            self.session.ensure()
        self.assertEqual(before, self.store.credentials_path.read_bytes())
        self.oauth.login.assert_not_called()

    def test_invalid_client_does_not_register_dynamic_client(self):
        self.save(expires=NOW - 1)
        self.oauth.refresh.side_effect = OAuthHTTPError(400, "invalid_client")
        with self.assertRaisesRegex(AuthError, "invalid_client"):
            self.session.ensure()
        self.oauth.login.assert_not_called()

    def test_missing_credentials_triggers_login_only_when_needed(self):
        self.oauth.login.side_effect = lambda *_, **__: LoginResult(
            self.tokens(id_token="id-login-secret"), CLIENT, "a" * 64, "offline-nonce")
        self.assertEqual(self.session.ensure(), "authenticated")
        self.assertEqual(self.session.ensure(), "restored")
        self.oauth.login.assert_called_once()

    def test_login_different_identity_never_replaces_existing_credentials(self):
        self.save()
        before = self.store.credentials_path.read_bytes()
        self.session.pending_path.write_bytes(b"rotating")
        self.oauth.login.side_effect = lambda *_, **__: LoginResult(
            self.tokens(id_token="id-login-secret"), CLIENT, "b" * 64, "offline-nonce")
        with self.assertRaisesRegex(AuthError, "login_identity_mismatch"):
            self.session.ensure()
        self.assertEqual(before, self.store.credentials_path.read_bytes())
        self.assertEqual(self.store.profile(), PROFILE)

    def test_missing_direct_scope_is_rejected(self):
        self.save(expires=NOW - 1)
        self.oauth.refresh.side_effect = lambda *_: (self.tokens(scope="openid offline_access"),
                                                     Discovery.parse(DISCOVERY_DOCUMENT))
        with self.assertRaisesRegex(AuthError, "plan_scope_missing"):
            self.session.ensure()

    def test_refresh_omitted_scope_preserves_original_grant(self):
        self.save(expires=NOW - 1)
        tokens = self.tokens()
        del tokens["scope"]
        self.oauth.refresh.side_effect = lambda *_: (tokens, Discovery.parse(DISCOVERY_DOCUMENT))
        self.assertEqual(self.session.ensure(), "refreshed")
        self.assertEqual(self.store.read_credentials()["scope"], SCOPES)

    def test_reused_refresh_token_is_rejected(self):
        self.save(expires=NOW - 1)
        self.oauth.refresh.side_effect = lambda *_: (self.tokens(refresh_token="refresh-old-secret"),
                                                     Discovery.parse(DISCOVERY_DOCUMENT))
        with self.assertRaisesRegex(AuthError, "refresh_not_rotated"):
            self.session.ensure()

    def test_profile_binding_mismatch_is_rejected(self):
        self.save()
        self.store.save_profile({**PROFILE, "issued_client_id": "oaiapp_other"})
        with self.assertRaisesRegex(AuthError, "credentials_profile_mismatch"):
            self.session.ensure()
        self.oauth.login.assert_not_called()

    def test_profile_import_is_explicit_public_only_and_does_not_modify_source(self):
        source = self.store.directory / "poc-host.json"
        source.write_text(json.dumps(PROFILE), encoding="utf-8")
        before = source.read_bytes()
        self.session.import_public_profile(source)
        self.assertEqual(before, source.read_bytes())
        self.session.ensure = Mock()
        source.write_text(json.dumps({**PROFILE, "access_token": "secret"}), encoding="utf-8")
        with self.assertRaises(AuthError):
            self.session.import_public_profile(source)

    def test_unknown_error_body_is_not_logged(self):
        error = OAuthHTTPError(400, "access-old-secret")
        self.assertEqual(str(error), "oauth_http_failed")

    def test_session_reports_never_contain_tokens(self):
        self.save(expires=NOW - 1)
        self.session.ensure()
        logged = "\n".join(self.messages)
        for secret in ("access-old-secret", "refresh-old-secret", "id-old-secret", "access-new-secret",
                       "refresh-new-secret", "offline-nonce"):
            self.assertNotIn(secret, logged)

    def test_dpapi_tampered_blob_fails_closed(self):
        blob = bytearray(self.store.protector.protect(b"offline-secret"))
        blob[-1] ^= 1
        with self.assertRaises(AuthError):
            self.store.protector.unprotect(bytes(blob))

    def test_earliest_timestamp_formats_and_invalid_values(self):
        self.assertEqual(epoch(1800000000), 1800000000)
        self.assertGreater(epoch("2026-10-08T12:00:00Z"), 0)
        for value in (True, None, -1, float("nan"), float("inf"), "2026-10-08T12:00:00"):
            with self.subTest(value=value), self.assertRaises(AuthError):
                epoch(value)

    def test_first_login_without_plan_scope_preserves_old_credentials(self):
        self.save()
        self.session.pending_path.write_bytes(b"rotating")
        before = self.store.credentials_path.read_bytes()
        self.oauth.login.side_effect = lambda *_, **__: LoginResult(
            self.tokens(id_token="id-login-secret", scope="openid offline_access"), CLIENT, "a" * 64, "offline-nonce")
        with self.assertRaisesRegex(AuthError, "plan_scope_missing"):
            self.session.ensure()
        self.assertEqual(before, self.store.credentials_path.read_bytes())

    def test_refresh_encryption_failure_preserves_previous_credentials(self):
        self.save(expires=NOW - 1)
        before = self.store.credentials_path.read_bytes()
        with patch.object(self.store.protector, "protect", side_effect=AuthError("dpapi_operation_failed")), \
                self.assertRaisesRegex(AuthError, "dpapi_operation_failed"):
            self.session.ensure()
        self.assertEqual(before, self.store.credentials_path.read_bytes())
        self.assertTrue(self.session.pending_path.exists())

    def test_network_uncertain_rotation_preserves_credentials_and_does_not_retry_old_token(self):
        self.save(expires=NOW - 1)
        before = self.store.credentials_path.read_bytes()
        self.oauth.refresh.side_effect = AuthError("oauth_network_failed")
        with self.assertRaisesRegex(AuthError, "oauth_network_failed"):
            self.session.ensure()
        self.assertEqual(before, self.store.credentials_path.read_bytes())
        self.assertTrue(self.session.pending_path.exists())
        self.oauth.login.assert_not_called()

    def test_no_plaintext_temporary_files_remain_after_write_failure(self):
        self.save()
        with patch("services.auth.windows_store.os.replace", side_effect=OSError("secret")), self.assertRaises(AuthError):
            self.store.save_credentials({"access_token": "access-new-secret"})
        self.assertEqual(list(self.store.directory.glob(".auth-*")), [])

    def test_corrupted_credentials_trigger_reauthorization_not_plaintext_fallback(self):
        self.store.credentials_path.write_bytes(b"unusable ciphertext")
        self.oauth.login.side_effect = lambda *_, **__: LoginResult(
            self.tokens(id_token="id-login-secret"), CLIENT, "a" * 64, "offline-nonce")
        self.assertEqual(self.session.ensure(), "authenticated")
        self.assertNotIn(b"access-new-secret", self.store.credentials_path.read_bytes())


class OAuthValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key(), as_dict=True)
        jwk.update(kid="offline-key", use="sig", alg="RS256")
        cls.jwks = {"keys": [jwk]}

    def setUp(self):
        self.discovery = Discovery.parse(DISCOVERY_DOCUMENT)
        self.now = int(time.time())

    def token(self, **overrides):
        claims = {"iss": ISSUER, "aud": CLIENT, "iat": self.now, "exp": self.now + 300,
                  "sub": "offline-subject", "nonce": "offline-nonce", **overrides}
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": "offline-key"})

    def validate(self, token):
        return identity_fingerprint(token, self.jwks, self.discovery, CLIENT, "offline-nonce")

    def test_valid_id_token_returns_only_nonreversible_fingerprint(self):
        expected = hashlib.sha256("\0".join((ISSUER, CLIENT, "offline-subject")).encode()).hexdigest()
        self.assertEqual(self.validate(self.token()), expected)

    def test_signature_is_required(self):
        wrong = jwt.encode({"iss": ISSUER, "aud": CLIENT, "exp": self.now + 300, "iat": self.now,
                            "sub": "offline-subject", "nonce": "offline-nonce"},
                           self.wrong_key, algorithm="RS256", headers={"kid": "offline-key"})
        with self.assertRaises(AuthError):
            self.validate(wrong)

    def test_issuer_audience_expiration_nonce_and_sub_are_checked(self):
        for overrides in ({"iss": "https://other.invalid"}, {"aud": "oaiapp_other"}, {"exp": self.now - 60},
                          {"nonce": "wrong"}, {"sub": ""}, {"azp": "oaiapp_other"}):
            with self.subTest(overrides=overrides), self.assertRaises(AuthError):
                self.validate(self.token(**overrides))

    def test_none_algorithm_is_not_accepted(self):
        token = jwt.encode({"nonce": "offline-nonce"}, None, algorithm="none")
        with self.assertRaises(AuthError):
            self.validate(token)

    def test_refresh_form_uses_issued_client_resource_and_omits_scope(self):
        http = Mock()
        http.json.side_effect = [DISCOVERY_DOCUMENT, {"access_token": "offline"}]
        OAuthClient(http=http).refresh(CLIENT, "offline-refresh")
        form = http.json.call_args.kwargs["form"]
        self.assertEqual(form, {"grant_type": "refresh_token", "client_id": CLIENT,
                                "refresh_token": "offline-refresh", "resource": RESOURCE})
        self.assertNotIn("scope", form)

    def test_refresh_rejects_dynamic_client_before_network(self):
        http = Mock()
        with self.assertRaises(AuthError):
            OAuthClient(http=http).refresh(DYNAMIC_CLIENT, "offline-refresh")
        http.json.assert_not_called()

    def test_returning_authorization_uses_saved_host_client_and_id_hint(self):
        transaction = Transaction.create(CLIENT)
        transaction.redirect_uri = "http://127.0.0.1:54321/auth/callback"
        url = OAuthClient(browser=Mock()).authorization_url(self.discovery, transaction, PROFILE, "offline-id-token")
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["client_id"], [CLIENT])
        self.assertEqual(query["ext_agent_host_id"], [HOST])
        self.assertEqual(query["id_token_hint"], ["offline-id-token"])
        self.assertNotIn("agent_name_hint", query)
        self.assertEqual(query["code_challenge_method"], ["S256"])

    def test_transaction_state_nonce_and_pkce_are_fresh(self):
        first, second = Transaction.create(CLIENT), Transaction.create(CLIENT)
        for field in ("state", "nonce", "verifier"):
            self.assertNotEqual(getattr(first, field), getattr(second, field))

    def test_callback_checks_state_client_and_one_time_consumption(self):
        transaction = Transaction.create(CLIENT)
        query = urlencode({"state": transaction.state, "code": "offline-code", "client_id": CLIENT})
        self.assertEqual(transaction.parse_callback(query), ("offline-code", CLIENT))
        with self.assertRaises(AuthError):
            transaction.parse_callback(query)
        wrong = Transaction.create(CLIENT)
        with self.assertRaises(AuthError):
            wrong.parse_callback(urlencode({"state": wrong.state, "code": "offline", "client_id": "oaiapp_other"}))

    def test_discovery_rejects_nonofficial_endpoints(self):
        for field in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            with self.subTest(field=field), self.assertRaises(AuthError):
                Discovery.parse({**DISCOVERY_DOCUMENT, field: "https://evil.invalid/token"})

    def test_http_refresh_error_code_is_sanitized_without_body(self):
        transport = OAuthHTTP()
        transport.opener = Mock()
        transport.opener.open.side_effect = HTTPError(self.discovery.token_endpoint, 400, "SECRET", {},
            io.BytesIO(b'{"error":"invalid_grant","error_description":"SECRET"}'))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(OAuthHTTPError, "invalid_grant"):
            transport.json(self.discovery.token_endpoint, form={"refresh_token": "SECRET"})
        self.assertEqual(output.getvalue(), "")

    def test_login_uses_validated_identity_and_authorization_code_form(self):
        http = Mock()
        client = OAuthClient(http=http, browser=Mock())
        captured = {}

        def authorize(discovery, transaction, profile, hint):
            transaction.redirect_uri = "http://127.0.0.1:54321/auth/callback"
            captured["transaction"] = transaction
            return transaction.parse_callback(urlencode({"state": transaction.state, "code": "offline-code",
                                                          "client_id": CLIENT}))

        def request(url, *, form=None):
            if url.endswith("openid-configuration"):
                return DISCOVERY_DOCUMENT
            if form:
                captured["form"] = form
                return {"id_token": self.token(nonce=captured["transaction"].nonce)}
            return self.jwks

        http.json.side_effect = request
        with patch.object(client, "_authorize", side_effect=authorize):
            result = client.login(PROFILE)
        self.assertEqual(captured["form"], {"grant_type": "authorization_code", "client_id": CLIENT,
            "code": "offline-code", "code_verifier": captured["form"]["code_verifier"],
            "redirect_uri": "http://127.0.0.1:54321/auth/callback", "resource": RESOURCE})
        self.assertTrue(captured["form"]["code_verifier"])
        self.assertEqual(result.fingerprint, self.validate(self.token()))
        self.assertEqual(captured["transaction"].nonce, "")

    def test_initial_registration_retains_issued_client_after_exchange_failure(self):
        http = Mock()
        http.json.side_effect = [DISCOVERY_DOCUMENT, AuthError("oauth_network_failed")]
        client = OAuthClient(http=http)
        registered = Mock()
        with patch.object(client, "_authorize", return_value=("offline-code", CLIENT)), self.assertRaises(AuthError):
            client.login({"ext_agent_host_id": HOST}, on_registered=registered)
        registered.assert_called_once_with({"ext_agent_host_id": HOST, "issued_client_id": CLIENT})

    def test_refreshed_id_token_validation_and_identity_binding(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="auth-offline-") as temporary:
            store = SessionStore(Path(temporary))
            profile = {**PROFILE, "identity_fingerprint": self.validate(self.token())}
            store.save_profile(profile)
            credentials = Credentials("access-old", "refresh-old", self.token(), NOW - 1, NOW - 60,
                                      SCOPES, "offline-nonce", **profile)
            store.save_credentials(asdict(credentials))
            oauth = Mock(spec=OAuthClient)
            oauth.http = Mock()
            oauth.http.json.return_value = self.jwks
            oauth.refresh.return_value = (ChatGPTAuthTest.tokens(id_token=self.token()), self.discovery)
            session = ChatGPTSession(store, oauth, lambda: NOW, lambda _: None)
            self.assertEqual(session.ensure(), "refreshed")
            stored = store.read_credentials()
            self.assertEqual(stored["identity_fingerprint"], profile["identity_fingerprint"])
            self.assertEqual(stored["refresh_token"], "refresh-new-secret")
            expired = {**stored, "expires_at": NOW - 1, "earliest_refresh_at": NOW - 60}
            store.save_credentials(expired)
            oauth.refresh.return_value = (ChatGPTAuthTest.tokens(id_token=self.token(sub="different-subject")), self.discovery)
            with self.assertRaisesRegex(AuthError, "refresh_identity_mismatch"):
                session.ensure()
            self.assertEqual(store.read_credentials()["refresh_token"], "refresh-new-secret")


class LauncherAuthTest(unittest.TestCase):
    def test_cli_valid_session_does_not_login_and_errors_do_not_leak(self):
        spec = importlib.util.spec_from_file_location("koda_auth_cli_test", Path(__file__).parent / "tools/chatgpt_auth.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        session = Mock()
        output = io.StringIO()
        with patch.object(cli, "ChatGPTSession", return_value=session), contextlib.redirect_stdout(output):
            self.assertEqual(cli.main([]), 0)
            session.ensure.assert_called_once()
            session.import_public_profile.assert_not_called()
            session.ensure.side_effect = RuntimeError("secret-token")
            self.assertEqual(cli.main([]), 1)
        self.assertNotIn("secret-token", output.getvalue())

    def test_launcher_checks_auth_before_starting_the_chatgpt_plan_backend(self):
        script = (Path(__file__).parent.parent / "tools/rocky_launcher.ps1").read_text(encoding="utf-8")
        self.assertIn("chatgpt_auth.py", script)
        self.assertIn("$LASTEXITCODE -ne 0", script)
        self.assertNotIn("Start-Process", script)
        self.assertNotIn("OPENROUTER_MODEL", script)
        self.assertNotIn("OPENAI_API_KEY", script)
        self.assertNotIn("OPENROUTER_API_KEY", script)
        self.assertNotIn("Ling 3.1 Flash", script)
        self.assertLess(script.index("chatgpt_auth.py"), script.index("-m uvicorn"))
        self.assertIn("Deseja iniciar a API do Koda?", script)
        self.assertIn("[KODA] Iniciando backend...", script)
        self.assertNotIn("Fase 2B pendente", script)


if __name__ == "__main__":
    unittest.main()
