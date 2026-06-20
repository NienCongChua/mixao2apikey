from pydantic_settings import BaseSettings
from typing import Optional
import json
import os


class Settings(BaseSettings):
    port: int = 8080
    host: str = "0.0.0.0"
    api_key: Optional[str] = None  # Nếu None thì không yêu cầu auth
    default_model: str = "mimo-v2.5-pro"
    credentials_file: str = "credentials.json"
    api_keys_file: str = "api_keys.json"
    usage_file: str = "usage_stats.json"
    proxy_url: Optional[str] = None  # Proxy cho requests ra ngoài

    class Config:
        env_prefix = "MIMO_"


settings = Settings()

# Đường dẫn lưu credentials
CREDENTIALS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.credentials_file)
API_KEYS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.api_keys_file)
USAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.usage_file)


def load_credentials() -> list[dict]:
    if os.path.exists(CREDENTIALS_PATH):
        with open(CREDENTIALS_PATH, "r") as f:
            return json.load(f)
    return []


def save_credentials(creds: list[dict]):
    with open(CREDENTIALS_PATH, "w") as f:
        json.dump(creds, f, indent=2, ensure_ascii=False)
