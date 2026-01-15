from __future__ import annotations
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger


def run_daily(
    task: Callable[[], None],
    *,
    daily_time: str,
    timezone: str,
) -> None:
    hour, minute = daily_time.split(":")
    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(
        task,
        CronTrigger(hour=int(hour), minute=int(minute), timezone=timezone),
        max_instances=1,
        misfire_grace_time=3600,
    )
    logger.info("Scheduler started. Daily time: {} ({})", daily_time, timezone)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
