from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from typing import Iterable


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


@dataclass(frozen=True)
class AppConfig:
    categories: list[str]
    timezone: str
    daily_time: str
    data_dir: str
    retention_days: int
    openai_base_url: str | None
    openai_chunk_model: str
    openai_overall_model: str
    openai_api_key: str | None
    smtp_host: str | None
    smtp_port: int
    smtp_user: str | None
    smtp_password: str | None
    smtp_from: str | None
    smtp_to: list[str]

    @staticmethod
    def from_env() -> "AppConfig":
        default_categories = ["cs.SE", "cs.CV", "cs.AI", "cs.CR", "cs.LG", "cs.RO"]
        categories = _split_csv(os.getenv("ARXIV_CATEGORIES"), default_categories)
        timezone = os.getenv("APP_TIMEZONE", "Asia/Shanghai")
        daily_time = os.getenv("APP_DAILY_TIME", "09:00")
        data_dir = os.getenv("APP_DATA_DIR", os.path.abspath("data"))
        retention_days = int(os.getenv("APP_RETENTION_DAYS", "30"))
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_chunk_model = os.getenv("OPENAI_CHUNK_MODEL", "gpt-4.1-mini")
        openai_overall_model = os.getenv(
            "OPENAI_OVERALL_MODEL", "claude-sonnet-4-5-20250929"
        )
        openai_api_key = os.getenv("OPENAI_API_KEY")

        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "465"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_from = os.getenv("SMTP_FROM")
        smtp_to = _split_csv(os.getenv("SMTP_TO"), [])

        return AppConfig(
            categories=categories,
            timezone=timezone,
            daily_time=daily_time,
            data_dir=data_dir,
            retention_days=retention_days,
            openai_base_url=openai_base_url,
            openai_chunk_model=openai_chunk_model,
            openai_overall_model=openai_overall_model,
            openai_api_key=openai_api_key,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            smtp_from=smtp_from,
            smtp_to=smtp_to,
        )


def parse_target_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def normalize_categories(items: Iterable[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]
