from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from typing import Iterable

from openai import OpenAI


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
    max_workers: int
    openai_timeout: float
    openai_overall_timeout: float
    openai_max_retries: int

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
        max_workers = int(os.getenv("MAX_WORKERS", "4"))
        openai_timeout = float(os.getenv("OPENAI_TIMEOUT", "180"))
        openai_overall_timeout = float(os.getenv("OPENAI_OVERALL_TIMEOUT", "300"))
        openai_max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "0"))

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
            max_workers=max_workers,
            openai_timeout=openai_timeout,
            openai_overall_timeout=openai_overall_timeout,
            openai_max_retries=openai_max_retries,
        )


def make_openai_client(
    config: AppConfig, *, timeout: float | None = None
) -> OpenAI:
    return OpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        timeout=timeout if timeout is not None else config.openai_timeout,
        max_retries=config.openai_max_retries,
    )


def parse_target_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def normalize_categories(items: Iterable[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]
