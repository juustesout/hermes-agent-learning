"""Accelerated local simulation of the HN aidevforum 6h posting job.

Uses a loopback WordPress draft mock. No live publish, no remote hosts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

import env_setup  # noqa: F401
import blind_label
import hn_ranker
from sim.client import WPDraftClient
from sim.wp_mock import WPMock

FETCH_SIZE = 30
POSTS_PER_RUN = hn_ranker.POSTS_PER_RUN
CATALOG_N = 6500
RELEVANT_RAG = 0.55


def make_catalog(n: int = CATALOG_N, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    domains = (
        "https://github.com/example/{0}",
        "https://arxiv.org/abs/{0}",
        "https://medium.com/p/{0}",
        "https://news.ycombinator.com/item?id={0}",
        "https://docs.python.org/{0}",
    )
    catalog = []
    for i in range(n):
        item_id = f"hn-{i:04d}"
        if rng.random() < 0.8:
            rag = rng.uniform(RELEVANT_RAG, 1.0)
        else:
            rag = rng.uniform(0.0, max(RELEVANT_RAG - 0.01, 0.0))
        catalog.append(
            {
                "id": item_id,
                "hn_id": item_id,
                "title": f"AI agent tooling {item_id} {'extra words' * (i % 5)}",
                "url": domains[i % len(domains)].format(item_id),
                "summary": f"Summary for {item_id} covering local agent workflows.",
                "rag_score": round(rag, 4),
                "hn_points": rng.randint(1, 400),
                "hn_comments": rng.randint(0, 120),
                "age_hours": round(rng.random() * 36, 2),
            }
        )
    return catalog


def auto_relevant(item: dict) -> bool:
    return float(item.get("rag_score") or 0.0) >= RELEVANT_RAG


def fetch_pool(
    catalog: list[dict],
    posted: set[str],
    fetch_size: int = FETCH_SIZE,
    offset: int = 0,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    del rng
    batch = catalog[offset : offset + fetch_size]
    pool = []
    for rank, row in enumerate(batch):
        if row["id"] in posted:
            continue
        if not auto_relevant(row):
            continue
        item = dict(row)
        item["source_rank"] = rank
        pool.append(item)
    return pool


def posted_ids() -> set[str]:
    return {str(row.get("id")) for row in blind_label.load_items() if row.get("id")}


def post_choice(client: WPDraftClient, item: dict) -> dict:
    title = str(item.get("title") or item["id"])
    content = f"{item.get('summary') or ''}\n{item.get('url') or ''}".strip()
    result = client.create_draft(title, content, meta={"hn_id": item["id"]})
    status = result.get("status")
    try:
        http_status = int(status) if status is not None and status != "network" else 599
    except (TypeError, ValueError):
        http_status = 599
    return {
        **result,
        "http_status": http_status,
        "ok": bool(result.get("ok")),
    }


def simulate_label(item_id: str, relevant: bool, *, agreement: float, rng: random.Random) -> Optional[dict]:
    queued = {str(row.get("id")) for row in blind_label.load_queue()}
    if item_id not in queued:
        return None
    agree = rng.random() < agreement
    human_relevant = relevant if agree else (not relevant)
    return blind_label.record_label(
        item_id,
        relevant=human_relevant,
        duplicate=False,
        needs_correction=False,
        notes="sim",
        rater="sim",
    )


def run_once(
    client: WPDraftClient,
    catalog: list[dict],
    run_id: str,
    *,
    auto_label: bool = False,
    agreement: float = 0.95,
    rng: Optional[random.Random] = None,
    offset: int = 0,
) -> dict:
    dice = rng or random.Random(run_id)
    pool = fetch_pool(catalog, posted_ids(), offset=offset)
    decisions = hn_ranker.select_run(pool, run_id, write=True)
    attempted = 0
    posted = 0
    failed = 0
    outcomes = []
    by_id = {row["id"]: row for row in pool}
    for decision in decisions:
        item = by_id[decision["chosen_id"]]
        result = post_choice(client, item)
        attempted += 1
        if result["ok"]:
            posted += 1
        else:
            failed += 1
        enq = blind_label.enqueue_item(
            {
                **item,
                "http_status": result["http_status"],
                "run_id": run_id,
                "slot": decision["slot"],
                "auto_relevant": auto_relevant(item),
                "auto_duplicate": False,
            }
        )
        labeled = None
        if auto_label:
            labeled = simulate_label(
                item["id"],
                auto_relevant(item),
                agreement=agreement,
                rng=dice,
            )
        outcomes.append(
            {
                "id": item["id"],
                "arm": decision["arm"],
                "http_status": result["http_status"],
                "ok": result["ok"],
                "sampled": enq.get("sampled"),
                "labeled": labeled is not None,
            }
        )
    run = blind_label.record_run(
        attempted=attempted,
        posted=posted,
        failed=failed,
        run_id=run_id,
    )
    return {
        "run_id": run_id,
        "pool": len(pool),
        "decisions": len(decisions),
        "attempted": attempted,
        "posted": posted,
        "failed": failed,
        "outcomes": outcomes,
        "recorded": run["run_id"],
    }


def summarize() -> dict:
    report = blind_label.report(include_sim=True)
    live = blind_label.report(include_sim=False)
    audit = blind_label.audit(expected_posts=max(report["posted"], 1), hours=24 * 365)
    decisions = hn_ranker.load_decisions()
    arms = {"L": 0, "B": 0}
    for row in decisions:
        arm = row.get("arm")
        if arm in arms:
            arms[arm] += 1
    return {
        "posted": report["posted"],
        "sampled": report["sampled"],
        "labeled": report["labeled"],
        "agreement": report["agreement"],
        "live_labeled": live["labeled"],
        "live_agreement": live["agreement"],
        "by_arm": report["by_arm"],
        "decisions": len(decisions),
        "decision_arms": arms,
        "overlap": audit.get("overlap"),
        "arm_balance": audit.get("arm_balance"),
        "audit_ok": audit["ok"],
        "alarms": audit["alarms"],
        "missing_decision": audit.get("posted_without_decision") or [],
    }


def run_sim(
    *,
    runs: int = 200,
    error_rate: float = 0.0,
    auto_label: bool = True,
    agreement: float = 0.95,
    seed: int = 1,
    catalog_n: int = CATALOG_N,
) -> dict:
    os.environ[blind_label.SOURCE_ENV] = blind_label.SOURCE_SIM
    mock = WPMock(error_rate=error_rate, seed=seed)
    base = mock.start()
    t0 = time.time()
    try:
        client = WPDraftClient(base, "crashtest", "secret", retries=3)
        catalog = make_catalog(n=catalog_n, seed=seed)
        rng = random.Random(seed)
        jobs = []
        for i in range(runs):
            run_id = f"sim-{i:04d}"
            jobs.append(
                run_once(
                    client,
                    catalog,
                    run_id,
                    auto_label=auto_label,
                    agreement=agreement,
                    rng=rng,
                    offset=i * FETCH_SIZE,
                )
            )
        elapsed = time.time() - t0
        summary = summarize()
        summary.update(
            {
                "runs": runs,
                "elapsed_s": round(elapsed, 3),
                "wp_posts": len(mock.posts),
                "wp_rejected": len(mock.rejected),
                "jobs_posted": sum(job["posted"] for job in jobs),
                "jobs_failed": sum(job["failed"] for job in jobs),
            }
        )
        return summary
    finally:
        mock.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hn_sim")
    sub = parser.add_subparsers(dest="cmd")
    p_run = sub.add_parser("run", help="Run accelerated posting jobs against a loopback WP mock")
    p_run.add_argument("--runs", type=int, default=200)
    p_run.add_argument("--error-rate", type=float, default=0.0)
    p_run.add_argument("--no-auto-label", action="store_true")
    p_run.add_argument("--agreement", type=float, default=0.95)
    p_run.add_argument("--seed", type=int, default=1)
    p_run.add_argument("--catalog", type=int, default=CATALOG_N)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        if not os.environ.get("AGENT_LEARNING_EVIDENCE_DIR"):
            store = os.environ.get("AGENT_LEARNING_LOCAL_STORE_DIR") or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "store"
            )
            os.environ["AGENT_LEARNING_EVIDENCE_DIR"] = os.path.join(store, "sim-evidence")
        payload = run_sim(
            runs=args.runs,
            error_rate=args.error_rate,
            auto_label=not args.no_auto_label,
            agreement=args.agreement,
            seed=args.seed,
            catalog_n=args.catalog,
        )
        print(json.dumps(payload, indent=2, default=str))
        alarms = [a for a in payload.get("alarms") or [] if a != "sim_data_in_evidence"]
        return 0 if not alarms and not payload.get("missing_decision") else 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
