from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Environment-driven configuration for the backend proxy service."""

    ai_service_url: str = os.getenv("AI_SERVICE_URL", "http://127.0.0.1:8001")
    request_timeout: float = float(os.getenv("AI_REQUEST_TIMEOUT", "30.0"))
    request_retries: int = int(os.getenv("AI_REQUEST_RETRIES", "2"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
