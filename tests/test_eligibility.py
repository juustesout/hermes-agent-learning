from __future__ import annotations

import consolidate_ops
import contextual_learner
import eligibility
import replay_advice
import shadow_policy
import train_adaptive
from eligibility import eligible_scrape


def test_consumers_share_eligible_scrape():
    assert replay_advice.eligible_scrape is eligible_scrape
    assert shadow_policy.eligible_scrape is eligible_scrape
    assert consolidate_ops.eligible_scrape is eligible_scrape
    assert contextual_learner.eligible_scrape is eligible_scrape
    assert train_adaptive.eligible_scrape is eligible_scrape


def test_eligible_scrape_requires_task_and_host():
    assert eligible_scrape(
        {"task_id": "choose-scrape-strategy", "target": "https://example.com/x"}
    )
    assert not eligible_scrape(
        {"task_id": "choose-scrape-strategy", "target": "/opt/data/file"}
    )
    assert not eligible_scrape(
        {"task_id": "wp-publish", "target": "https://example.com/x"}
    )
    assert not eligible_scrape(
        {"task_id": "web-research", "target": "https://example.com/x"}
    )
    assert not eligible_scrape(
        {"task_id": "db-query", "target": "https://example.com/x"}
    )


def test_classify_counts_skips():
    rows = [
        {"id": "1", "task_id": "choose-scrape-strategy", "target": "https://a.com"},
        {"id": "2", "task_id": "choose-scrape-strategy", "target": "/opt/data/x"},
        {"id": "3", "task_id": "wp-publish", "target": "https://b.com"},
        {"id": "4", "task_id": "web-research", "target": "memory"},
    ]
    kept, skip = eligibility.classify_rows(rows)
    assert [r["id"] for r in kept] == ["1"]
    assert skip["skipped_no_host"] == 1
    assert skip["skipped_wrong_task"] == 2
    reasons = {s["id"]: s["reason"] for s in skip["skipped_samples"]}
    assert reasons["2"] == "no_host"
    assert reasons["3"] == "wrong_task"


def test_scrape_policy_map_does_not_absorb_other_tasks():
    assert consolidate_ops.TASK_POLICY_MAP["wp-publish"] == "wp-publish"
    assert consolidate_ops.TASK_POLICY_MAP["db-query"] == "db-query"
    assert consolidate_ops.TASK_POLICY_MAP["web-research"] == "web-research"
    ep = consolidate_ops.op_to_episode(
        {
            "id": "wp-1",
            "task_id": "wp-publish",
            "action_id": "wp_rest_api",
            "target": "https://example.com",
            "execution_status": "completed",
        }
    )
    assert ep["policy_id"] == "wp-publish"
    assert ep["metadata"]["eligible_scrape"] is False
