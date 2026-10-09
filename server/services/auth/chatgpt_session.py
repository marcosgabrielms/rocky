"""Restore/refresh/login de uma sessão Windows; nenhuma chamada de inferência."""

import hmac
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .oauth import (AuthError, LoginResult, OAuthClient, OAuthHTTPError, PLAN_SCOPE,
                    TERMINAL_REFRESH_ERRORS, identity_fingerprint)
from .windows_store import SessionStore, atomic_write, validate_profile

REFRESH_MARGIN_SECONDS = 120


def epoch(value: object) -> float:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError
            value = parsed.timestamp()
        except ValueError:
            raise AuthError("token_timestamp_invalid") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise AuthError("token_timestamp_invalid")
    return float(value)


def require_secret(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 65536:
        raise AuthError("token_response_invalid")
    return value


def require_scope(value: object) -> str:
    if not isinstance(value, str) or PLAN_SCOPE not in value.split() or "offline_access" not in value.split():
        raise AuthError("plan_scope_missing")
    return value


@dataclass(frozen=True, repr=False)
class Credentials:
    access_token: str
    refresh_token: str
    id_token: str
    expires_at: float
    earliest_refresh_at: float
    scope: str
    nonce: str
    issued_client_id: str
    ext_agent_host_id: str
    identity_fingerprint: str

    @classmethod
    def from_tokens(cls, tokens: dict, profile: dict, nonce: str, now: float, previous=None):
        if not isinstance(tokens.get("token_type"), str) or tokens["token_type"].lower() != "bearer":
            raise AuthError("token_type_invalid")
        lifetime = tokens.get("expires_in")
        if isinstance(lifetime, bool) or not isinstance(lifetime, (int, float)) or not math.isfinite(lifetime) or lifetime <= 0:
            raise AuthError("token_lifetime_invalid")
        scope = require_scope(tokens.get("scope", previous.scope if previous else None))
        refresh = require_secret(tokens.get("refresh_token"))
        if previous and hmac.compare_digest(refresh, previous.refresh_token):
            raise AuthError("refresh_not_rotated")
        return cls(require_secret(tokens.get("access_token")), refresh,
                   require_secret(tokens.get("id_token", previous.id_token if previous else None)),
                   now + lifetime, epoch(tokens.get("earliest_refresh_at", now)), scope,
                   require_secret(nonce), **profile)

    @classmethod
    def restore(cls, record: dict, profile: dict):
        if set(record) != set(cls.__dataclass_fields__):
            raise AuthError("credentials_invalid")
        for name in ("issued_client_id", "ext_agent_host_id", "identity_fingerprint"):
            if not isinstance(record[name], str) or not hmac.compare_digest(record[name], profile.get(name, "")):
                raise AuthError("credentials_profile_mismatch")
        for name in ("access_token", "refresh_token", "id_token", "nonce"):
            require_secret(record[name])
        require_scope(record["scope"])
        return cls(**{**record, "expires_at": epoch(record["expires_at"]),
                      "earliest_refresh_at": epoch(record["earliest_refresh_at"])})


class ChatGPTSession:
    def __init__(self, store=None, oauth=None, clock=None, report=None):
        self.store = store or SessionStore()
        self.oauth = oauth or OAuthClient()
        self.clock = clock or time.time
        self.report = report or (lambda message: print("[KODA] " + message, flush=True))
        self.pending_path = self.store.directory / "rotation.pending"

    def import_public_profile(self, source: Path) -> None:
        """Bootstrap explícito e único; o runtime nunca procura o diretório da POC."""
        import json

        try:
            profile = validate_profile(json.loads(source.read_bytes()))
        except (OSError, ValueError, UnicodeError):
            raise AuthError("profile_import_invalid") from None
        if "issued_client_id" not in profile or "identity_fingerprint" not in profile:
            raise AuthError("profile_import_incomplete")
        with self.store.lock():
            if self.store.profile_path.exists():
                if self.store.profile() != profile:
                    raise AuthError("profile_import_conflict")
            else:
                self.store.save_profile(profile)

    def ensure(self) -> str:
        self.report("Verificando sessao ChatGPT...")
        with self.store.lock():
            profile = self.store.profile()
            try:
                record = self.store.read_credentials()
            except AuthError as error:
                if str(error) != "credentials_unreadable":
                    raise
                record = None
            credentials = self._restore(record, profile)
            if credentials and not self.pending_path.exists():
                outcome = self._resume(credentials, profile)
                if outcome is not None:
                    return outcome
                credentials = None  # Confirmed revocation: do not reuse ID-token hint.
            self.report("Autenticacao ChatGPT necessaria.")
            self.report("Abrindo Continue with ChatGPT...")
            self._login(profile, credentials.id_token if credentials else None)
            self.report("Sessao ChatGPT pronta.")
            return "authenticated"

    def access_token_for_inference(self) -> str:
        """Devolve um token utilizavel sem iniciar autenticacao interativa."""
        with self.store.lock():
            profile = self.store.profile()
            try:
                record = self.store.read_credentials()
            except AuthError as error:
                if str(error) != "credentials_unreadable":
                    raise
                raise AuthError("authentication_required") from None

            credentials = self._restore(record, profile)
            if credentials is None:
                raise AuthError("authentication_required")
            if self.pending_path.exists():
                raise AuthError("refresh_rotation_pending")

            now = self.clock()
            if now < credentials.expires_at - REFRESH_MARGIN_SECONDS:
                return credentials.access_token
            if now < credentials.earliest_refresh_at:
                if now >= credentials.expires_at:
                    raise AuthError("refresh_not_yet_allowed")
                return credentials.access_token

            try:
                self._refresh(credentials, profile)
            except OAuthHTTPError as error:
                self.pending_path.unlink(missing_ok=True)
                if error.status in (400, 401) and error.code in TERMINAL_REFRESH_ERRORS:
                    self.store.invalidate_credentials()
                    raise AuthError("authentication_required") from None
                raise

            refreshed = self._restore(self.store.read_credentials(), profile)
            if refreshed is None:
                raise AuthError("authentication_required")
            self.report("Sessao ChatGPT renovada.")
            return refreshed.access_token

    def _restore(self, record, profile):
        if record is None or record == {"reauthorization_required": True}:
            return None
        return Credentials.restore(record, profile)

    def _resume(self, credentials: Credentials, profile: dict) -> str | None:
        now = self.clock()
        if now < credentials.expires_at - REFRESH_MARGIN_SECONDS:
            self.report("Sessao ChatGPT restaurada.")
            return "restored"
        if now < credentials.earliest_refresh_at:
            if now >= credentials.expires_at:
                raise AuthError("refresh_not_yet_allowed")
            self.report("Sessao ChatGPT restaurada.")
            return "restored"
        try:
            self._refresh(credentials, profile)
        except OAuthHTTPError as error:
            # A received HTTP error did not produce a replacement token set.
            self.pending_path.unlink(missing_ok=True)
            if error.status in (400, 401) and error.code in TERMINAL_REFRESH_ERRORS:
                self.store.invalidate_credentials()
                return None
            raise
        self.report("Sessao ChatGPT renovada.")
        return "refreshed"

    def _refresh(self, credentials: Credentials, profile: dict) -> None:
        # Durable, non-secret marker prevents reuse after uncertain rotation/write failure.
        atomic_write(self.pending_path, b"rotating\n")
        tokens, discovery = self.oauth.refresh(credentials.issued_client_id, credentials.refresh_token)
        try:
            if "id_token" in tokens:
                fingerprint = identity_fingerprint(tokens["id_token"], self.oauth.http.json(discovery.jwks_uri),
                                                   discovery, credentials.issued_client_id, credentials.nonce)
                if not hmac.compare_digest(fingerprint, credentials.identity_fingerprint):
                    raise AuthError("refresh_identity_mismatch")
            replacement = Credentials.from_tokens(tokens, profile, credentials.nonce, self.clock(), credentials)
            self.store.save_credentials(asdict(replacement))
            self.pending_path.unlink(missing_ok=True)
        finally:
            tokens.clear()

    def _login(self, profile: dict, id_token_hint: str | None) -> None:
        result: LoginResult = self.oauth.login(profile, id_token_hint=id_token_hint,
                                               on_registered=self.store.save_profile)
        try:
            previous = profile.get("identity_fingerprint")
            if previous and not hmac.compare_digest(previous, result.fingerprint):
                raise AuthError("login_identity_mismatch")
            registered = validate_profile({**profile, "issued_client_id": result.client_id,
                                           "identity_fingerprint": result.fingerprint})
            credentials = Credentials.from_tokens(result.tokens, registered, result.nonce, self.clock())
            self.store.save_profile(registered)
            self.store.save_credentials(asdict(credentials))
            self.pending_path.unlink(missing_ok=True)
        finally:
            result.tokens.clear()
