"""OAuth público + PKCE e validação OIDC extraídos da POC, sem inferência."""

import base64
import hashlib
import hmac
import json
import secrets
import ssl
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPSHandler, HTTPRedirectHandler, ProxyHandler, Request, build_opener

import jwt

from .windows_store import AuthError, issued_client_valid

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
DYNAMIC_CLIENT = "dynamic_agent_client"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
ASYMMETRIC_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512",
                                   "ES256", "ES384", "ES512", "EdDSA"})
TERMINAL_REFRESH_ERRORS = frozenset({"invalid_grant", "invalid_refresh_token", "token_expired",
                                    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused"})


class OAuthHTTPError(AuthError):
    def __init__(self, status: int, code: object):
        self.status = status
        allowed = TERMINAL_REFRESH_ERRORS | {"invalid_client", "temporarily_unavailable", "server_error"}
        self.code = code if isinstance(code, str) and code in allowed else "oauth_http_failed"
        super().__init__(self.code)


def validate_endpoint(value: object) -> str:
    if not isinstance(value, str):
        raise AuthError("discovery_endpoint_invalid")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.netloc not in {"auth.openai.com", "auth.openai.com:443"}
            or parsed.username is not None or not parsed.path.startswith("/") or parsed.query or parsed.fragment):
        raise AuthError("discovery_endpoint_invalid")
    return value


@dataclass(frozen=True)
class Discovery:
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    algorithms: tuple[str, ...]

    @classmethod
    def parse(cls, document: dict):
        if not isinstance(document, dict) or document.get("issuer") != ISSUER:
            raise AuthError("discovery_issuer_invalid")
        advertised = document.get("id_token_signing_alg_values_supported")
        if not isinstance(advertised, list) or not all(isinstance(item, str) for item in advertised):
            raise AuthError("discovery_algorithms_invalid")
        algorithms = tuple(item for item in advertised if item in ASYMMETRIC_ALGORITHMS)
        if not algorithms:
            raise AuthError("discovery_algorithms_invalid")
        for field, required in (("response_types_supported", "code"), ("code_challenge_methods_supported", "S256")):
            if field in document and (not isinstance(document[field], list) or required not in document[field]):
                raise AuthError("discovery_flow_invalid")
        return cls(*(validate_endpoint(document.get(field)) for field in
                     ("authorization_endpoint", "token_endpoint", "jwks_uri")), algorithms)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class OAuthHTTP:
    def __init__(self):
        self.opener = build_opener(ProxyHandler({}), NoRedirect(), HTTPSHandler(context=ssl.create_default_context()))

    def json(self, url: str, *, form: dict | None = None) -> dict:
        validate_endpoint(url)
        headers = {"Accept": "application/json", "User-Agent": "Koda-ChatGPT-Auth/2A"}
        body = urlencode(form).encode("ascii") if form is not None else None
        if form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = Request(url, data=body, headers=headers, method="POST" if form is not None else "GET")
        try:
            with self.opener.open(request, timeout=30) as response:
                if response.status != 200:
                    raise OAuthHTTPError(response.status, None)
                raw = response.read(2 * 1024 * 1024 + 1)
        except HTTPError as error:
            with error:
                code = None
                try:
                    document = json.loads(error.read(65536))
                    value = document.get("error") if isinstance(document, dict) else None
                    code = value.get("code") if isinstance(value, dict) else value
                except (ValueError, UnicodeError):
                    pass
                raise OAuthHTTPError(error.code, code) from None
        except (URLError, OSError, TimeoutError):
            raise AuthError("oauth_network_failed") from None
        if len(raw) > 2 * 1024 * 1024:
            raise AuthError("oauth_response_too_large")
        try:
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError
            return document
        except (ValueError, UnicodeError):
            raise AuthError("oauth_json_invalid") from None


def identity_fingerprint(id_token: str, jwks: dict, discovery: Discovery, client_id: str, nonce: str) -> str:
    if not issued_client_valid(client_id) or not isinstance(id_token, str) or not id_token or not nonce:
        raise AuthError("identity_input_invalid")
    try:
        header = jwt.get_unverified_header(id_token)
        algorithm, kid = header.get("alg"), header.get("kid")
        if algorithm not in discovery.algorithms or algorithm not in ASYMMETRIC_ALGORITHMS or "crit" in header:
            raise AuthError("identity_algorithm_invalid")
        if not isinstance(kid, str) or not kid or not isinstance(jwks.get("keys"), list):
            raise AuthError("identity_key_invalid")
        keys = [key for key in jwks["keys"] if isinstance(key, dict) and key.get("kid") == kid]
        if len(keys) != 1:
            raise AuthError("identity_key_invalid")
        key = keys[0]
        if key.get("use", "sig") != "sig" or key.get("alg", algorithm) != algorithm:
            raise AuthError("identity_key_invalid")
        if "key_ops" in key and (not isinstance(key["key_ops"], list) or "verify" not in key["key_ops"]):
            raise AuthError("identity_key_invalid")
        claims = jwt.decode(id_token, jwt.PyJWK.from_dict(key, algorithm=algorithm).key,
                            algorithms=[algorithm], audience=client_id, issuer=ISSUER, leeway=5,
                            options={"require": ["iss", "aud", "exp", "iat", "nonce", "sub"]})
    except (jwt.PyJWTError, ValueError, TypeError, KeyError, AttributeError):
        raise AuthError("identity_validation_failed") from None
    if claims.get("aud") not in (client_id, [client_id]) or claims.get("azp", client_id) != client_id:
        raise AuthError("identity_audience_invalid")
    subject, actual_nonce = claims.get("sub"), claims.get("nonce")
    if not isinstance(subject, str) or not subject.strip():
        raise AuthError("identity_subject_invalid")
    if not isinstance(actual_nonce, str) or not hmac.compare_digest(actual_nonce.encode(), nonce.encode()):
        raise AuthError("identity_nonce_invalid")
    return hashlib.sha256("\0".join((ISSUER, client_id, subject)).encode()).hexdigest()


@dataclass(repr=False)
class Transaction:
    client_id: str
    state: str
    nonce: str
    verifier: str
    redirect_uri: str = ""
    deadline: float = 0
    consumed: bool = False

    @classmethod
    def create(cls, client_id: str):
        return cls(client_id, secrets.token_urlsafe(32), secrets.token_urlsafe(32),
                   secrets.token_urlsafe(64), deadline=time.monotonic() + 600)

    def parse_callback(self, query: str) -> tuple[str, str]:
        if self.consumed or time.monotonic() >= self.deadline:
            raise AuthError("callback_expired")
        self.consumed = True
        try:
            values = parse_qs(query, keep_blank_values=True, max_num_fields=20)
        except ValueError:
            raise AuthError("callback_invalid") from None
        if any(len(value) != 1 for value in values.values()):
            raise AuthError("callback_duplicate")
        state = values.get("state", [""])[0]
        if not state or not hmac.compare_digest(state.encode(), self.state.encode()):
            raise AuthError("callback_state_invalid")
        if "error" in values:
            raise AuthError("callback_consent_denied")
        code = values.get("code", [""])[0]
        client = values.get("client_id", [self.client_id])[0]
        if not code.strip() or not issued_client_valid(client):
            raise AuthError("callback_registration_incomplete")
        if self.client_id != DYNAMIC_CLIENT and client != self.client_id:
            raise AuthError("callback_client_mismatch")
        return code, client


class LoopbackServer(HTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(2)
        return connection, address

    def handle_error(self, request, client_address):
        pass  # Default tracebacks may include callback query parameters.


def callback_handler(transaction: Transaction, result: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_error(self, code, message=None, explain=None):
            self.send_page(code)

        def send_page(self, status):
            body = b"Continue with ChatGPT: return to the Koda terminal."
            self.send_response(status)
            for name, value in (("Content-Type", "text/plain"), ("Cache-Control", "no-store"),
                                ("Referrer-Policy", "no-referrer"), ("Connection", "close"),
                                ("Content-Security-Policy", "default-src 'none'"), ("Content-Length", str(len(body)))):
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if (len(self.path) > 8192 or parsed.path != "/auth/callback" or parsed.netloc or parsed.scheme
                    or parsed.fragment or self.headers.get("Host") != urlsplit(transaction.redirect_uri).netloc):
                self.send_page(404)
                return
            if result:
                self.send_page(409)
                return
            try:
                result["authorization"] = transaction.parse_callback(parsed.query)
            except AuthError as error:
                result["error"] = error
                self.send_page(400)
            else:
                self.send_page(200)

    return Handler


@dataclass(frozen=True, repr=False)
class LoginResult:
    tokens: dict
    client_id: str
    fingerprint: str
    nonce: str


class OAuthClient:
    def __init__(self, http=None, browser=None):
        self.http = http or OAuthHTTP()
        self.browser = browser or webbrowser.open

    def discover(self) -> Discovery:
        return Discovery.parse(self.http.json(DISCOVERY_URL))

    def authorization_url(self, discovery, transaction, profile, id_token_hint):
        challenge = base64.urlsafe_b64encode(hashlib.sha256(transaction.verifier.encode()).digest()).rstrip(b"=").decode()
        values = {"client_id": transaction.client_id, "ext_agent_host_id": profile["ext_agent_host_id"],
                  "resource": RESOURCE, "response_type": "code", "scope": SCOPES,
                  "redirect_uri": transaction.redirect_uri, "state": transaction.state, "nonce": transaction.nonce,
                  "code_challenge_method": "S256", "code_challenge": challenge}
        if transaction.client_id == DYNAMIC_CLIENT:
            values["agent_name_hint"] = "Koda"
        elif id_token_hint:
            values["id_token_hint"] = id_token_hint
        return discovery.authorization_endpoint + "?" + urlencode(values)

    def _authorize(self, discovery, transaction, profile, id_token_hint):
        result = {}
        with LoopbackServer(("127.0.0.1", 0), callback_handler(transaction, result)) as server:
            transaction.redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
            try:
                opened = self.browser(self.authorization_url(discovery, transaction, profile, id_token_hint), new=1)
            except Exception:
                raise AuthError("browser_failed") from None
            if not opened:
                raise AuthError("browser_failed")
            server.timeout = 0.5
            while not result and time.monotonic() < transaction.deadline:
                server.handle_request()
        if "error" in result:
            raise result["error"]
        if "authorization" not in result:
            raise AuthError("callback_timeout")
        return result["authorization"]

    def login(self, profile: dict, *, id_token_hint: str | None = None, on_registered=None) -> LoginResult:
        discovery = self.discover()
        transaction = Transaction.create(profile.get("issued_client_id", DYNAMIC_CLIENT))
        try:
            code, client_id = self._authorize(discovery, transaction, profile, id_token_hint)
            if on_registered and "issued_client_id" not in profile:
                on_registered({**profile, "issued_client_id": client_id})
            tokens = self.http.json(discovery.token_endpoint, form={
                "grant_type": "authorization_code", "client_id": client_id, "code": code,
                "code_verifier": transaction.verifier, "redirect_uri": transaction.redirect_uri, "resource": RESOURCE})
            fingerprint = identity_fingerprint(tokens.get("id_token"), self.http.json(discovery.jwks_uri),
                                               discovery, client_id, transaction.nonce)
            return LoginResult(tokens, client_id, fingerprint, transaction.nonce)
        finally:
            transaction.state = transaction.nonce = transaction.verifier = ""
            transaction.consumed = True

    def refresh(self, client_id: str, refresh_token: str) -> tuple[dict, Discovery]:
        if not issued_client_valid(client_id):
            raise AuthError("refresh_client_invalid")
        discovery = self.discover()
        tokens = self.http.json(discovery.token_endpoint, form={"grant_type": "refresh_token",
            "client_id": client_id, "refresh_token": refresh_token, "resource": RESOURCE})
        return tokens, discovery
