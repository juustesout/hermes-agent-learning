"""Shared scrape eligibility. One predicate for replay, train, consolidate, shadow."""

from __future__ import annotations

from typing import Any

import features

SCRAPE_TASK_ID = "choose-scrape-strategy"
SKIP_NO_HOST = "no_host"
SKIP_WRONG_TASK = "wrong_task"
SAMPLE_CAP = 5


def _get(row: Any, name: str, default: str = "") -> str:
    if isinstance(row, dict):
        value = row.get(name, default)
    else:
        value = getattr(row, name, default)
    if value is None:
        return default
    return str(value)


def row_task_id(row: Any) -> str:
    return _get(row, "task_id").strip()


def row_target(row: Any) -> str:
    return _get(row, "target")


def skip_reason(row: Any, *, task_id: str = SCRAPE_TASK_ID) -> str | None:
    wanted = str(task_id or SCRAPE_TASK_ID).strip() or SCRAPE_TASK_ID
    if row_task_id(row) != wanted:
        return SKIP_WRONG_TASK
    if not features.host_from_url(row_target(row)):
        return SKIP_NO_HOST
    return None


def eligible_scrape(row: Any, *, task_id: str = SCRAPE_TASK_ID) -> bool:
    return skip_reason(row, task_id=task_id) is None


def classify_rows(
    rows: list[Any],
    *,
    task_id: str = SCRAPE_TASK_ID,
) -> tuple[list[Any], dict]:
    kept: list[Any] = []
    skipped_no_host = 0
    skipped_wrong_task = 0
    samples: list[dict] = []
    for row in rows:
        reason = skip_reason(row, task_id=task_id)
        if reason is None:
            kept.append(row)
            continue
        if reason == SKIP_NO_HOST:
            skipped_no_host += 1
        else:
            skipped_wrong_task += 1
        if len(samples) < SAMPLE_CAP:
            samples.append(
                {
                    "reason": reason,
                    "id": _get(row, "id"),
                    "task_id": row_task_id(row),
                    "target": row_target(row)[:200],
                }
            )
    kept.sort(
        key=lambda r: (_get(r, "created_at"), _get(r, "id")),
    )
    return kept, {
        "skipped_no_host": skipped_no_host,
        "skipped_wrong_task": skipped_wrong_task,
        "skipped_samples": samples,
    }
