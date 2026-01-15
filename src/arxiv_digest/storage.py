from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .models import Paper, SummaryChunk


STATE_FILE = "state.json"
DATE_DIR_FORMAT = "%Y-%m-%d"
LEGACY_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, data: str) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(data, encoding="utf-8")
    temp_path.replace(path)


def load_state(data_dir: str) -> dict[str, Any]:
    path = Path(data_dir) / "state" / STATE_FILE
    if not path.exists():
        return {"seen_by_date": {}, "last_run_date": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(data_dir: str, state: dict[str, Any]) -> None:
    state_dir = Path(data_dir) / "state"
    _ensure_dir(state_dir)
    _atomic_write(state_dir / STATE_FILE, json.dumps(state, ensure_ascii=False, indent=2))


def build_seen_set(state: dict[str, Any], retention_days: int) -> set[str]:
    seen_by_date = state.get("seen_by_date", {})
    cutoff = date.today() - timedelta(days=retention_days)
    valid_dates = {
        day for day in seen_by_date.keys() if date.fromisoformat(day) >= cutoff
    }
    seen: set[str] = set()
    for day in valid_dates:
        seen.update(seen_by_date.get(day, []))
    return seen


def update_state_with_papers(
    state: dict[str, Any],
    target_date: date,
    papers: Iterable[Paper],
) -> dict[str, Any]:
    day_key = target_date.isoformat()
    seen_by_date = state.get("seen_by_date", {})
    seen_by_date.setdefault(day_key, [])
    for paper in papers:
        seen_by_date[day_key].append(paper.paper_id)
    return {
        "seen_by_date": seen_by_date,
        "last_run_date": datetime.utcnow().isoformat(),
    }


def _date_dir(data_dir: str, target_date: date) -> Path:
    return Path(data_dir) / target_date.strftime(DATE_DIR_FORMAT)


def save_raw_papers(data_dir: str, target_date: date, papers: list[Paper]) -> Path:
    raw_dir = _date_dir(data_dir, target_date) / "raw"
    _ensure_dir(raw_dir)
    path = raw_dir / "papers.jsonl"
    lines = [json.dumps(paper.to_dict(), ensure_ascii=False) for paper in papers]
    _atomic_write(path, "\n".join(lines) + ("\n" if lines else ""))
    return path


def save_summary_chunk(
    data_dir: str,
    chunk: SummaryChunk,
) -> Path:
    summaries_dir = _date_dir(data_dir, date.fromisoformat(chunk.date)) / "summaries"
    _ensure_dir(summaries_dir)
    filename = f"summary_part{chunk.chunk_index:02d}.json"
    path = summaries_dir / filename
    payload = {
        "date": chunk.date,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
    }
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def read_raw_papers(data_dir: str, target_date: date) -> list[dict[str, Any]]:
    path = _date_dir(data_dir, target_date) / "raw" / "papers.jsonl"
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def has_summary_for_date(data_dir: str, target_date: date) -> bool:
    summaries_dir = _date_dir(data_dir, target_date) / "summaries"
    if not summaries_dir.exists():
        return False
    prefix = "summary_part"
    return any(path.name.startswith(prefix) for path in summaries_dir.iterdir())


def load_summary_chunks(data_dir: str, target_date: date) -> dict[int, dict[str, Any]]:
    summaries_dir = _date_dir(data_dir, target_date) / "summaries"
    if not summaries_dir.exists():
        return {}
    prefix = "summary_part"
    chunks: dict[int, dict[str, Any]] = {}
    for path in summaries_dir.iterdir():
        if not path.name.startswith(prefix):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunk_index = int(payload.get("chunk_index", 0))
        if chunk_index > 0:
            chunks[chunk_index] = payload.get("content", {})
    return chunks


def save_response_chunk(
    data_dir: str,
    target_date: date,
    chunk_index: int,
    response_payload: dict[str, Any],
) -> Path:
    responses_dir = _date_dir(data_dir, target_date) / "responses"
    _ensure_dir(responses_dir)
    filename = f"response_part{chunk_index:02d}.json"
    path = responses_dir / filename
    _atomic_write(path, json.dumps(response_payload, ensure_ascii=False, indent=2))
    return path


def save_overall_summary(
    data_dir: str,
    target_date: date,
    summary: dict[str, Any],
) -> Path:
    summaries_dir = _date_dir(data_dir, target_date) / "summaries"
    _ensure_dir(summaries_dir)
    path = summaries_dir / "summary_overall.json"
    payload = {
        "date": target_date.isoformat(),
        "type": "overall",
        "content": summary,
    }
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def load_overall_summary(
    data_dir: str,
    target_date: date,
) -> dict[str, Any] | None:
    path = _date_dir(data_dir, target_date) / "summaries" / "summary_overall.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("content", {})


def save_overall_response(
    data_dir: str,
    target_date: date,
    response_payload: dict[str, Any],
) -> Path:
    responses_dir = _date_dir(data_dir, target_date) / "responses"
    _ensure_dir(responses_dir)
    path = responses_dir / "response_overall.json"
    _atomic_write(path, json.dumps(response_payload, ensure_ascii=False, indent=2))
    return path


def migrate_legacy_data(data_dir: str) -> list[tuple[Path, Path]]:
    base_dir = Path(data_dir)
    moved: list[tuple[Path, Path]] = []

    raw_dir = base_dir / "raw"
    if raw_dir.exists():
        for path in raw_dir.glob("papers_*.jsonl"):
            match = LEGACY_DATE_PATTERN.search(path.name)
            if not match:
                continue
            day = date.fromisoformat(match.group(0))
            target_dir = _date_dir(data_dir, day) / "raw"
            _ensure_dir(target_dir)
            target = target_dir / "papers.jsonl"
            if not target.exists():
                path.replace(target)
                moved.append((path, target))

    summaries_dir = base_dir / "summaries"
    if summaries_dir.exists():
        for path in summaries_dir.iterdir():
            if path.name.startswith("summary_") and path.suffix == ".json":
                match = LEGACY_DATE_PATTERN.search(path.name)
                if not match:
                    continue
                day = date.fromisoformat(match.group(0))
                target_dir = _date_dir(data_dir, day) / "summaries"
                _ensure_dir(target_dir)
                if "overall" in path.name:
                    target = target_dir / "summary_overall.json"
                else:
                    part_match = re.search(r"part(\d+)", path.name)
                    if not part_match:
                        continue
                    part = int(part_match.group(1))
                    target = target_dir / f"summary_part{part:02d}.json"
                if not target.exists():
                    path.replace(target)
                    moved.append((path, target))

    responses_dir = base_dir / "responses"
    if responses_dir.exists():
        for path in responses_dir.iterdir():
            if not path.name.startswith("response_"):
                continue
            match = LEGACY_DATE_PATTERN.search(path.name)
            if not match:
                continue
            day = date.fromisoformat(match.group(0))
            target_dir = _date_dir(data_dir, day) / "responses"
            _ensure_dir(target_dir)
            if "overall" in path.name:
                target = target_dir / "response_overall.txt"
            else:
                part_match = re.search(r"part(\d+)", path.name)
                if not part_match:
                    continue
                part = int(part_match.group(1))
                target = target_dir / f"response_part{part:02d}.txt"
            if not target.exists():
                path.replace(target)
                moved.append((path, target))

    prompts_dir = base_dir / "prompts"
    if prompts_dir.exists():
        for path in prompts_dir.iterdir():
            if not path.name.startswith("prompt_"):
                continue
            match = LEGACY_DATE_PATTERN.search(path.name)
            if not match:
                continue
            day = date.fromisoformat(match.group(0))
            target_dir = _date_dir(data_dir, day) / "prompts"
            _ensure_dir(target_dir)
            part_match = re.search(r"part(\d+)", path.name)
            if not part_match:
                continue
            part = int(part_match.group(1))
            target = target_dir / f"prompt_part{part:02d}.txt"
            if not target.exists():
                path.replace(target)
                moved.append((path, target))

    return moved
