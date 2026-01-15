from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta
import math
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from .arxiv_client import fetch_papers
from .config import AppConfig, parse_target_date
from .emailer import send_email
from .models import Paper, SummaryChunk
from .storage import (
    build_seen_set,
    has_summary_for_date,
    load_summary_chunks,
    load_state,
    load_overall_summary,
    migrate_legacy_data,
    read_raw_papers,
    save_raw_papers,
    save_response_chunk,
    save_overall_response,
    save_state,
    save_summary_chunk,
    save_overall_summary,
    update_state_with_papers,
)
from .summarizer import summarize_overall, summarize_papers_stream
from .scheduler import run_daily

def _resolve_target_date(value: date | None) -> date:
    return value or (date.today() - timedelta(days=1))


def _load_papers_from_storage(config: AppConfig, target_date: date) -> list[Paper]:
    raw_items = read_raw_papers(config.data_dir, target_date)
    papers: list[Paper] = []
    for item in raw_items:
        papers.append(
            Paper(
                paper_id=item["paper_id"],
                title=item["title"],
                summary=item["summary"],
                authors=item["authors"],
                link=item["link"],
                category=item["category"],
                published=datetime.fromisoformat(item["published"]),
                updated=datetime.fromisoformat(item["updated"]),
            )
        )
    return papers


def _run_once(config: AppConfig, target_date: date) -> None:
    state = load_state(config.data_dir)
    seen_ids = build_seen_set(state, config.retention_days)

    stored_papers = _load_papers_from_storage(config, target_date)
    has_summary = has_summary_for_date(config.data_dir, target_date)
    if stored_papers:
        papers = stored_papers
        logger.info("Loaded {} papers from storage for {}", len(papers), target_date)
    else:
        papers = fetch_papers(config.categories, target_date)
        save_raw_papers(config.data_dir, target_date, papers)
        logger.info("Fetched {} papers from arXiv", len(papers))

    existing_chunks = load_summary_chunks(config.data_dir, target_date)
    chunk_size = 20
    expected_chunks = math.ceil(len(papers) / chunk_size) if papers else 0

    if stored_papers and (not has_summary or len(existing_chunks) < expected_chunks):
        logger.info(
            "Found {}/{} summary chunks for {}. Reprocessing missing.",
            len(existing_chunks),
            expected_chunks,
            target_date,
        )
        new_papers = papers
    else:
        new_papers = [paper for paper in papers if paper.paper_id not in seen_ids]

    logger.info("New papers after dedupe: {}", len(new_papers))

    save_state(
        config.data_dir,
        update_state_with_papers(state, target_date, new_papers),
    )

    summaries: list[dict[str, Any]] = []
    if not new_papers:
        if not existing_chunks:
            logger.info("No new papers for {}. Skip summarization.", target_date)
            return
        logger.info("No new papers for {}. Reusing existing summaries.", target_date)
        for idx in sorted(existing_chunks):
            summaries.append(existing_chunks[idx])
    else:
        if not config.openai_api_key:
            logger.warning("OPENAI_API_KEY not configured. Skip summarization.")
            return

        client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)
        summary_pairs = summarize_papers_stream(
            client,
            config.openai_chunk_model,
            target_date,
            new_papers,
            chunk_size=chunk_size,
            existing_chunks=existing_chunks,
            on_response=lambda idx, payload: save_response_chunk(
                config.data_dir, target_date, idx, payload
            ),
        )

        for idx, summary in summary_pairs:
            if idx not in existing_chunks:
                save_summary_chunk(
                    config.data_dir,
                    SummaryChunk(
                        date=target_date.isoformat(),
                        chunk_index=idx,
                        content=summary,
                    ),
                )
            summaries.append(summary)

    overall_summary = load_overall_summary(config.data_dir, target_date)
    if not overall_summary:
        if not config.openai_api_key:
            logger.warning("OPENAI_API_KEY not configured. Skip overall summary.")
        else:
            client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)
            overall_summary = summarize_overall(
                client,
                config.openai_overall_model,
                target_date,
                summaries,
                on_response=lambda payload: save_overall_response(
                    config.data_dir, target_date, payload
                ),
            )
            save_overall_summary(config.data_dir, target_date, overall_summary)

    chunk_summaries = summaries
    summaries = chunk_summaries

    if (
        config.smtp_host
        and config.smtp_user
        and config.smtp_password
        and config.smtp_to
    ):
        category_counts = dict(
            sorted(Counter(paper.category for paper in papers).items())
        )
        send_email(
            host=config.smtp_host,
            port=config.smtp_port,
            username=config.smtp_user,
            password=config.smtp_password,
            sender=config.smtp_from or config.smtp_user,
            recipients=config.smtp_to,
            subject=f"arXiv 每日摘要 {target_date.isoformat()}",
            summaries=chunk_summaries,
            overall_summary=overall_summary,
            category_counts=category_counts,
            date_str=target_date.isoformat(),
        )
        logger.info("Email sent to {}", ", ".join(config.smtp_to))
    else:
        logger.warning("SMTP not configured or recipients missing. Skip email.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily arXiv digest crawler.")
    parser.add_argument("--log-path", help="Log file path.")
    parser.add_argument("--date", help="Target date in YYYY-MM-DD.")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")
    parser.add_argument(
        "--env-file",
        help="Load environment variables from a specific .env file.",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate legacy data into date-based folders and exit.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run daily scheduler (default).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logger.add(args.log_path or "arxiv-digest.log")
    if args.env_file:
        load_dotenv(args.env_file)

    config = AppConfig.from_env()

    def task() -> None:
        target_date = _resolve_target_date(parse_target_date(args.date))
        logger.info("Running digest task for {}", target_date.isoformat())
        _run_once(config, target_date)

    if args.migrate:
        moved = migrate_legacy_data(config.data_dir)
        logger.info("Migrated {} legacy files.", len(moved))
        return

    if args.once:
        task()
        return

    run_daily(task, daily_time=config.daily_time, timezone=config.timezone)


if __name__ == "__main__":
    main()
