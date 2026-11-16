"""
consolidate_ops.py — zet ruwe operations.jsonl om naar learning episodes.

Hoe het werkt:
  1. Leest operations.jsonl (append-only log)
  2. Checkt welke operation IDs al verwerkt zijn (consolidated.txt)
  3. Maakt voor elke nieuwe operatie een volwaardig episode JSON bestand aan
     in store/episodes/hermes/{uuid}.json
  4. Optioneel: traint policies bij op basis van nieuwe episodes

Gebruik:
  python3 consolidate_ops.py                    # droge run (--dry-run)
  python3 consolidate_ops.py --run              # echte consolidatie
  python3 consolidate_ops.py --run --train      # + policy update
"""

import json
import os
import uuid
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import env_setup  # noqa: F401 — ensures LocalFileStore backend

STORE_DIR = "/opt/data/profiles/operator/agent-learning/store"
OPSLOG_PATH = os.path.join(STORE_DIR, "operations.jsonl")
CONSOLIDATED_LOG = os.path.join(STORE_DIR, "consolidated.txt")
EPISODES_DIR = os.path.join(STORE_DIR, "episodes", "hermes")

TASK_POLICY_MAP = {
    "choose-scrape-strategy": "choose-scrape-strategy",
    "db-query": "choose-scrape-strategy",  # fallback policy
    "wp-publish": "choose-scrape-strategy",
    "web-research": "choose-scrape-strategy",
    "podcast-research": "podcast-research",
    "podcast-script": "podcast-script",
    "podcast-audio": "podcast-audio",
    "podcast-publish": "podcast-publish",
}

ACTION_TYPE_MAP = {
    "web_scrape": "web_scrape",
    "database_query": "database_query",
    "wp_rest_api": "wp_rest_api",
    "web_search": "web_search",
    "terminal": "terminal",
    "file_read": "file_read",
    "file_write": "file_write",
}


def load_operations() -> list[dict]:
    """Lees alle operaties uit operations.jsonl."""
    if not os.path.exists(OPSLOG_PATH):
        return []
    ops = []
    with open(OPSLOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ops.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ops


def load_consolidated() -> set[str]:
    """Welke operation IDs al verwerkt zijn."""
    if not os.path.exists(CONSOLIDATED_LOG):
        return set()
    with open(CONSOLIDATED_LOG) as f:
        return set(line.strip() for line in f if line.strip())


def mark_consolidated(ids: set[str]):
    """Schrijf verwerkte IDs weg."""
    existing = load_consolidated()
    existing.update(ids)
    with open(CONSOLIDATED_LOG, "w") as f:
        for oid in sorted(existing):
            f.write(oid + "\n")


def op_to_episode(op: dict) -> dict:
    """Converteer ruwe operatie naar volwaardig episode JSON."""
    task_id = op.get("task_id", "unknown")
    action_id = op.get("action_id", "unknown")
    metadata = op.get("metadata", {})
    cf = op.get("context_features", {})

    # Bepaal model_deployment uit context_features of action_id
    model = cf.get("model", None)

    episode = {
        "id": op.get("id", str(uuid.uuid4())),
        "agent_id": "hermes",
        "agent_name": None,
        "task_id": task_id,
        "task_name": None,
        "user_input": "",
        "assistant_output": op.get("result_summary", ""),
        "intent_summary": f"{action_id} → {op.get('target', '')}",
        "action_type": ACTION_TYPE_MAP.get(op.get("action_type", ""), op.get("action_type", "")),
        "action_name": action_id,
        "target": op.get("target", ""),
        "input_summary": f"Auto-logged: {task_id}",
        "expected_outcome": "Operation completed via Hermes tool call",
        "execution_status": op.get("execution_status", "completed"),
        "result_summary": op.get("result_summary", ""),
        "system_message": None,
        "conversation_history": [],
        "tool_calls": [],
        "policy_id": TASK_POLICY_MAP.get(task_id, "choose-scrape-strategy"),
        "policy_version": 0,
        "action_id": action_id,
        "action_logprob": None,
        "context_features": cf,
        "model_deployment": model,
        "correlation_id": None,
        "session_id": None,
        "request_latency_ms": op.get("request_latency_ms"),
        "token_usage": None,
        "metadata": {
            "correct_action_id": action_id if metadata.get("task_completed", True) else "",
            "task_completed": metadata.get("task_completed", True),
            "selection_basis": "auto_logged",
            **(metadata.get("site_type", None) and {"site_type": metadata["site_type"]} or {}),
            **(metadata.get("js_required", None) and {"js_required": metadata["js_required"]} or {}),
            **(metadata.get("cloudflare", None) and {"cloudflare": metadata["cloudflare"]} or {}),
            "content_length": metadata.get("content_length", 0),
            "cost_credits": metadata.get("cost_credits", 0),
            "scrape_success": metadata.get("task_completed", True),
            "content_quality": metadata.get("content_quality", "medium"),
            "latency_ms": op.get("request_latency_ms"),
        },
        "created_at": op.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }
    return episode


def consolidate(dry_run: bool = False) -> dict:
    """Consolideer nieuwe operaties naar episodes."""
    ops = load_operations()
    done = load_consolidated()
    pending = [op for op in ops if op.get("id") not in done]

    if not pending:
        return {"new": 0, "total_ops": len(ops), "total_episodes": len(done)}

    os.makedirs(EPISODES_DIR, exist_ok=True)

    new_ids = set()
    for op in pending:
        ep = op_to_episode(op)
        ep_path = os.path.join(EPISODES_DIR, ep["id"] + ".json")
        new_ids.add(op["id"])
        if dry_run:
            continue
        with open(ep_path, "w") as f:
            json.dump(ep, f, indent=2, default=str)
        # Schrijf ook reward-bestand
        _write_reward(ep)

    if not dry_run:
        mark_consolidated(new_ids)

    return {
        "new": len(pending),
        "total_ops": len(ops),
        "total_episodes": len(done | new_ids),
        "dry_run": dry_run,
    }


def _write_reward(ep: dict):
    """Schrijf reward in het formaat dat agent-learning store verwacht.

    De local store slaat rewards op als JSON-array in
    store/rewards/hermes/{episode_id}.json, waarbij elk element
    voldoet aan Reward.to_dict().
    """
    reward_dir = os.path.join(STORE_DIR, "rewards", "hermes")
    os.makedirs(reward_dir, exist_ok=True)
    success = ep.get("execution_status") == "completed"
    latency = ep.get("request_latency_ms") or 5000
    cost = ep.get("metadata", {}).get("cost_credits", 0)

    completion = 1.0 if success else 0.0
    latency_pen = min(latency / 10000.0, 1.0)
    cost_pen = min(cost / 10.0, 1.0)
    reward_value = completion * 1.0 - latency_pen * 0.3 - cost_pen * 0.3
    reward_value = max(-1.0, min(1.0, reward_value))

    import uuid
    reward_record = {
        "id": str(uuid.uuid4()),
        "episode_id": ep["id"],
        "agent_id": "hermes",
        "source": "aggregate",
        "value": round(reward_value, 4),
        "raw_value": None,
        "metric": None,
        "rubric": None,
        "evaluator": None,
        "metadata": {
            "consolidated_from_op": True,
            "latency_ms": latency,
            "cost_credits": cost,
            "latency_penalty": round(latency_pen, 4),
            "cost_penalty": round(cost_pen, 4),
            "reward_preset": "balanced",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    rpath = os.path.join(reward_dir, ep["id"] + ".json")
    with open(rpath, "w") as f:
        json.dump([reward_record], f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="consolidate_ops")
    parser.add_argument("--run", action="store_true", help="Echt uitvoeren (ipv dry-run)")
    parser.add_argument("--train", action="store_true", help="Na consolidatie ook policy trainen")
    args = parser.parse_args()

    result = consolidate(dry_run=not args.run)

    if result["new"] == 0:
        print(f"Geen nieuwe operaties. Totaal: {result['total_episodes']} episodes uit {result['total_ops']} operaties.")
        sys.exit(0)

    print(f"{'DROGE RUN' if result.get('dry_run') else 'GECONSOLIDEERD'}:")
    print(f"  Nieuwe episodes: {result['new']}")
    print(f"  Totaal episodes: {result['total_episodes']}")
    print(f"  Totaal operaties: {result['total_ops']}")

    if not result.get("dry_run") and args.train:
        print("  Policy training gestart...")
        # Roep train_adaptive.py aan
        os.system(f"cd {STORE_DIR}/.. && python3 train_adaptive.py --agent-id hermes --task-id choose-scrape-strategy 2>&1")