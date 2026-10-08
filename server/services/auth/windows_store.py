"""DPAPI CurrentUser, persistência atômica e exclusão mútua no Windows."""

import ctypes
import hashlib
import json
import os
import re
import tempfile
import uuid
from ctypes import wintypes
from pathlib import Path


class AuthError(Exception):
    """Somente códigos internos constantes; nunca transportar segredos em mensagens."""


def validate_profile(record: object) -> dict[str, str]:
    if not isinstance(record, dict) or set(record) - {
        "ext_agent_host_id", "issued_client_id", "identity_fingerprint"
    }:
        raise AuthError("profile_invalid")
    host = record.get("ext_agent_host_id")
    try:
        value = uuid.UUID(host.removeprefix("urn:uuid:"))
        if value.version != 4 or host != "urn:uuid:" + str(value):
            raise ValueError
    except (ValueError, AttributeError, TypeError):
        raise AuthError("profile_host_invalid") from None
    client = record.get("issued_client_id")
    if client is not None and not issued_client_valid(client):
        raise AuthError("profile_client_invalid")
    fingerprint = record.get("identity_fingerprint")
    if fingerprint is not None and (not client or not isinstance(fingerprint, str)
                                   or not re.fullmatch(r"[a-f0-9]{64}", fingerprint)):
        raise AuthError("profile_identity_invalid")
    return dict(record)


def issued_client_valid(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"oaiapp_[A-Za-z0-9_-]{1,250}", value) is not None


def atomic_write(path: Path, contents: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".auth-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        raise AuthError("auth_storage_write_failed") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Any leftover temporary file contains ciphertext, never tokens.


class DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


class WindowsDPAPI:
    def __init__(self):
        if os.name != "nt":
            raise AuthError("windows_required")
        self._crypt = ctypes.WinDLL("crypt32", use_last_error=True)
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name in ("CryptProtectData", "CryptUnprotectData"):
            function = getattr(self._crypt, name)
            function.argtypes = [ctypes.POINTER(DataBlob), ctypes.c_void_p,
                                ctypes.POINTER(DataBlob), ctypes.c_void_p, ctypes.c_void_p,
                                wintypes.DWORD, ctypes.POINTER(DataBlob)]
            function.restype = wintypes.BOOL
        self._kernel.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel.LocalFree.restype = ctypes.c_void_p

    def _transform(self, data: bytes, name: str) -> bytes:
        if not data:
            raise AuthError("dpapi_empty_data")
        buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        source = DataBlob(len(data), buffer)
        target = DataBlob()
        # UI_FORBIDDEN only: never CRYPTPROTECT_LOCAL_MACHINE (CurrentUser scope).
        try:
            ok = getattr(self._crypt, name)(ctypes.byref(source), None, None, None, None, 1,
                                           ctypes.byref(target))
            if not ok:
                raise AuthError("dpapi_operation_failed")
            return ctypes.string_at(target.pbData, target.cbData)
        finally:
            if target.pbData:
                ctypes.memset(target.pbData, 0, target.cbData)
                self._kernel.LocalFree(target.pbData)
            ctypes.memset(buffer, 0, len(data))

    def protect(self, data: bytes) -> bytes:
        return self._transform(data, "CryptProtectData")

    def unprotect(self, data: bytes) -> bytes:
        return self._transform(data, "CryptUnprotectData")


class WindowsSessionLock:
    """Named mutex: threads e processos do mesmo usuário/caminho compartilham lock."""

    def __init__(self, directory: Path):
        if os.name != "nt":
            raise AuthError("windows_required")
        key = hashlib.sha256(str(directory.resolve()).casefold().encode()).hexdigest()
        self._name = "Local\\KodaChatGPTAuth_" + key
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self._kernel.CreateMutexW.restype = wintypes.HANDLE
        self._kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._kernel.WaitForSingleObject.restype = wintypes.DWORD
        self._kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self._kernel.ReleaseMutex.restype = wintypes.BOOL
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._handle = None

    def __enter__(self):
        handle = self._kernel.CreateMutexW(None, False, self._name)
        if not handle:
            raise AuthError("session_lock_failed")
        result = self._kernel.WaitForSingleObject(handle, 60000)
        if result not in (0, 0x80):  # WAIT_OBJECT_0 / WAIT_ABANDONED: caller owns lock.
            self._kernel.CloseHandle(handle)
            raise AuthError("session_lock_timeout")
        self._handle = handle
        return self

    def __exit__(self, *_):
        try:
            self._kernel.ReleaseMutex(self._handle)
        finally:
            self._kernel.CloseHandle(self._handle)
            self._handle = None


class SessionStore:
    def __init__(self, directory: Path | None = None, protector=None):
        if directory is None:
            local = os.environ.get("LOCALAPPDATA")
            if not local:
                raise AuthError("localappdata_missing")
            directory = Path(local) / "Koda" / "auth"
        self.directory = directory.resolve()
        self.protector = protector or WindowsDPAPI()
        self.profile_path = self.directory / "profile.json"
        self.credentials_path = self.directory / "credentials.bin"

    def lock(self):
        return WindowsSessionLock(self.directory)

    def profile(self) -> dict[str, str]:
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            self.save_profile({"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())})
        try:
            return validate_profile(json.loads(self.profile_path.read_bytes()))
        except (OSError, ValueError, UnicodeError):
            raise AuthError("profile_read_failed") from None

    def save_profile(self, record: dict) -> None:
        validated = validate_profile(record)
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write(self.profile_path, json.dumps(validated).encode())

    def read_credentials(self) -> dict | None:
        if not self.credentials_path.exists():
            return None
        try:
            raw = self.credentials_path.read_bytes()
            record = json.loads(self.protector.unprotect(raw))
            if not isinstance(record, dict):
                raise ValueError
            return record
        except (OSError, ValueError, UnicodeError, AuthError):
            raise AuthError("credentials_unreadable") from None

    def save_credentials(self, record: dict) -> None:
        # Only ciphertext ever reaches a temporary file or the final file.
        encrypted = self.protector.protect(json.dumps(record).encode())
        atomic_write(self.credentials_path, encrypted)

    def invalidate_credentials(self) -> None:
        # Atomic tombstone; preserves registration and cannot restore a revoked set.
        self.save_credentials({"reauthorization_required": True})
