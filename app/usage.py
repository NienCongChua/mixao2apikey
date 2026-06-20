import json
import os
import threading
import time
from copy import deepcopy

from app.config import USAGE_PATH


TOKEN_KEYS = (
    "requests",
    "input_tokens",
    "output_tokens",
    "cache_tokens",
    "total_tokens",
)


def _empty_totals() -> dict:
    return {key: 0 for key in TOKEN_KEYS}


def _to_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_usage(usage: dict | None) -> dict:
    usage = usage or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    input_tokens = _to_int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("promptTokens")
    )
    output_tokens = _to_int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("completionTokens")
    )
    cache_tokens = _to_int(
        usage.get("cache_tokens")
        or usage.get("cached_tokens")
        or usage.get("cachedTokens")
        or usage.get("cacheTokens")
        or prompt_details.get("cached_tokens")
        or prompt_details.get("cachedTokens")
    )
    total_tokens = _to_int(usage.get("total_tokens") or usage.get("totalTokens"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    return {
        "requests": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_tokens": cache_tokens,
        "total_tokens": total_tokens,
    }


class UsageTracker:
    """Persistent token usage counters for local observability."""

    def __init__(self, path: str = USAGE_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._data = self._empty_data()
        self.reload()

    @staticmethod
    def _empty_data() -> dict:
        return {
            "version": 1,
            "created_at": int(time.time()),
            "updated_at": 0,
            "totals": _empty_totals(),
            "accounts": {},
            "recent": [],
        }

    def reload(self):
        with self._lock:
            if not os.path.exists(self.path):
                self._data = self._empty_data()
                return
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._data = data if isinstance(data, dict) else self._empty_data()
            self._ensure_shape_locked()

    def _save_locked(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _ensure_shape_locked(self):
        self._data.setdefault("version", 1)
        self._data.setdefault("created_at", int(time.time()))
        self._data.setdefault("updated_at", 0)
        self._data.setdefault("totals", _empty_totals())
        self._data.setdefault("accounts", {})
        self._data.setdefault("recent", [])
        for key, value in _empty_totals().items():
            self._data["totals"].setdefault(key, value)

    @staticmethod
    def _merge_totals(target: dict, increment: dict):
        for key in TOKEN_KEYS:
            target[key] = _to_int(target.get(key)) + _to_int(increment.get(key))

    def record(self, account: str, model: str, usage: dict | None, stream: bool = False):
        increment = normalize_usage(usage)
        now = int(time.time())
        account_name = account or "default"
        model_name = model or "unknown"

        with self._lock:
            self._ensure_shape_locked()
            self._merge_totals(self._data["totals"], increment)

            account_entry = self._data["accounts"].setdefault(
                account_name,
                {
                    "name": account_name,
                    "totals": _empty_totals(),
                    "models": {},
                    "recent": [],
                    "updated_at": 0,
                },
            )
            account_entry.setdefault("totals", _empty_totals())
            account_entry.setdefault("models", {})
            account_entry.setdefault("recent", [])
            self._merge_totals(account_entry["totals"], increment)

            model_entry = account_entry["models"].setdefault(
                model_name,
                {"model": model_name, "totals": _empty_totals(), "updated_at": 0},
            )
            model_entry.setdefault("totals", _empty_totals())
            self._merge_totals(model_entry["totals"], increment)
            model_entry["updated_at"] = now

            recent_item = {
                "timestamp": now,
                "account": account_name,
                "model": model_name,
                "stream": bool(stream),
                **increment,
            }
            account_entry["recent"].insert(0, recent_item)
            del account_entry["recent"][50:]

            self._data["recent"].insert(0, recent_item)
            del self._data["recent"][100:]

            account_entry["updated_at"] = now
            self._data["updated_at"] = now
            self._save_locked()

    def snapshot(self) -> dict:
        with self._lock:
            self._ensure_shape_locked()
            data = deepcopy(self._data)

        accounts = []
        for account in data.get("accounts", {}).values():
            models = list(account.get("models", {}).values())
            models.sort(key=lambda item: item.get("totals", {}).get("total_tokens", 0), reverse=True)
            account["models"] = models
            accounts.append(account)
        accounts.sort(key=lambda item: item.get("totals", {}).get("total_tokens", 0), reverse=True)
        data["accounts"] = accounts
        return data

    def reset(self):
        with self._lock:
            self._data = self._empty_data()
            self._save_locked()


usage_tracker = UsageTracker()
