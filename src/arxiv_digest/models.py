from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    link: str
    category: str
    published: datetime
    updated: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["published"] = self.published.isoformat()
        data["updated"] = self.updated.isoformat()
        return data


@dataclass(frozen=True)
class SummaryChunk:
    date: str
    chunk_index: int
    content: dict[str, Any]
