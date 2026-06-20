import json
import os
import secrets
import threading
import time
from typing import Optional

from app.config import API_KEYS_PATH


class AccessKeyStore:
    """Small local store for optional OpenAI-compatible API keys."""

    def __init__(self, path: str = API_KEYS_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._keys: list[dict] = []
        self.reload()

    def reload(self):
        with self._lock:
            if not os.path.exists(self.path):
                self._keys = []
                return
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._keys = data if isinstance(data, list) else []

    def _save_locked(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._keys, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _preview(key: str) -> str:
        if len(key) <= 10:
            return "*" * len(key)
        return f"{key[:6]}...{key[-4:]}"

    def has_active_keys(self) -> bool:
        with self._lock:
            return any(item.get("is_active", True) and item.get("key") for item in self._keys)

    def is_valid(self, key: str) -> bool:
        with self._lock:
            return any(
                item.get("is_active", True)
                and item.get("key")
                and secrets.compare_digest(str(item["key"]), key)
                for item in self._keys
            )

    def list_public(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": item.get("id", ""),
                    "name": item.get("name", "API key"),
                    "key_preview": self._preview(str(item.get("key", ""))),
                    "is_active": item.get("is_active", True),
                    "created_at": item.get("created_at", 0),
                    "last_used_at": item.get("last_used_at", 0),
                }
                for item in self._keys
            ]

    def add(self, name: str, key: Optional[str] = None) -> dict:
        key = key.strip() if key else f"sk-mimo-{secrets.token_urlsafe(32)}"
        entry = {
            "id": secrets.token_hex(8),
            "name": name.strip() or "API key",
            "key": key,
            "is_active": True,
            "created_at": int(time.time()),
            "last_used_at": 0,
        }
        with self._lock:
            self._keys.append(entry)
            self._save_locked()
        return {
            "id": entry["id"],
            "name": entry["name"],
            "key": entry["key"],
            "key_preview": self._preview(entry["key"]),
            "is_active": entry["is_active"],
            "created_at": entry["created_at"],
        }

    def delete(self, key_id: str) -> bool:
        with self._lock:
            original_count = len(self._keys)
            self._keys = [item for item in self._keys if item.get("id") != key_id]
            changed = len(self._keys) != original_count
            if changed:
                self._save_locked()
            return changed

    def set_active(self, key_id: str, is_active: bool) -> bool:
        with self._lock:
            for item in self._keys:
                if item.get("id") == key_id:
                    item["is_active"] = is_active
                    self._save_locked()
                    return True
        return False

    def mark_used(self, key: str):
        now = int(time.time())
        with self._lock:
            changed = False
            for item in self._keys:
                if item.get("key") == key:
                    item["last_used_at"] = now
                    changed = True
                    break
            if changed:
                self._save_locked()


access_key_store = AccessKeyStore()
