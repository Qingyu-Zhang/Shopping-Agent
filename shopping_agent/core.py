from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Fault(Exception):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code, self.status = code, status


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Line(Input):
    offer_id: str = Field(min_length=1, max_length=160)
    quantity: int = Field(default=1, ge=1, le=20)


def normalize_lines(lines: list[dict]) -> list[dict]:
    if not 1 <= len(lines) <= 20:
        raise Fault("invalid_lines", "Select between 1 and 20 lines.", 422)
    values = [Line.model_validate(line).model_dump() for line in lines]
    if len({line["offer_id"] for line in values}) != len(values):
        raise Fault("duplicate_offer", "An offer may appear only once.", 422)
    return sorted(values, key=lambda line: line["offer_id"])


def new_id() -> str:
    return uuid4().hex


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def timestamp() -> float:
    return datetime.now(UTC).timestamp()


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 3)


def iso(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat()


def cash(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


@dataclass(frozen=True)
class Settings:
    database: str = "sqlite:///.data/agent.db"
    platform_database: str = "sqlite:///.data/platform.db"
    catalog_database: str = ""
    platform_url: str = ""
    platform_transport: str = "http"
    platform_token: str = "local-platform-token"
    offer_event_url: str = ""
    demo: bool = True
    embedded_worker: bool = True
    checkout_enabled: bool = True
    autobuy_enabled: bool = False
    elastic_url: str = ""
    elastic_key: str = ""
    index: str = "shopping-v2"
    review_recall: bool = True
    redis_url: str = ""
    model_url: str = ""
    model_key: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    embedding_url: str = ""
    embedding_key: str = ""
    rerank_url: str = ""
    rerank_model: str = ""
    rerank_key: str = ""
    clip_model: str = ""
    catalog_images: str = ""
    vision_chat_model: str = ""
    auth_jwt_secret: str = ""
    auth_issuer: str = "shopping-identity"
    auth_audience: str = "shopping-agent"

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(**{
            name: (
                os.environ.get(name.upper(), "1" if value else "0").lower() in {"1", "true", "yes"}
                if isinstance(value, bool) else os.environ.get(name.upper(), value)
            )
            for name, value in vars(defaults).items()
        })
