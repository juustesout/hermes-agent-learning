"""
init_contextual_scrape_policy.py — Initialiseer ContextualSoftmaxPolicy
voor de scrape-strategy.

Vervangt de bestaande marginale SoftmaxPolicy door een contextuele
die per site-type leert welke strategie werkt.

Gebruik:
  python3 init_contextual_scrape_policy.py [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import env_setup  # noqa: F401 — ensures LocalFileStore backend

from agent_learning.policy.contextual_softmax import ContextualSoftmaxPolicy
from agent_learning.storage.cosmos import get_default_store
from agent_learning.types import Action, PolicySnapshot

from scrape_features import FEATURE_DIM, FEATURE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description="Init contextual scrape policy")
    parser.add_argument("--force", action="store_true", help="Overschrijf bestaande policy")
    args = parser.parse_args()

    store = get_default_store()
    existing = store.get_active_policy("hermes", "choose-scrape-strategy")

    if existing and not args.force:
        print(f"Bestaande policy gevonden (v{existing.version}).")
        print("Gebruik --force om te overschrijven met een contextuele policy.")
        print(f"Huidige acties: {[a.id for a in existing.actions]}")
        return 1

    actions = [
        Action(
            id="use_webclaw_scrape",
            description="WebClaw MCP (antibot, credits, ~90% success op zware sites)",
            parameters={"engine": "webclaw", "cost_per_call": 1, "max_chars": 1_000_000, "antibot": True},
        ),
        Action(
            id="use_jina_reader",
            description="Jina AI Reader (gratis, snel, faalt op JS-heavy/Cloudflare)",
            parameters={"engine": "jina", "cost_per_call": 0, "max_chars": 20_000},
        ),
        Action(
            id="use_browser",
            description="Hermes browser (universeel, langzaam, interactie mogelijk)",
            parameters={"engine": "browser", "cost_per_call": 0, "interactive": True},
        ),
    ]

    policy = ContextualSoftmaxPolicy.from_actions(
        actions,
        feature_dim=FEATURE_DIM,
        agent_id="hermes",
        task_id="choose-scrape-strategy",
    )

    # Sla feature-namen op in metadata voor inspectie
    policy._snapshot.metadata["feature_names"] = FEATURE_NAMES
    policy._snapshot.metadata["policy_kind"] = "contextual_softmax"
    policy._snapshot.metadata["decision_context"] = (
        "Choose a web scraping strategy based on site characteristics. "
        "Context features: cloudflare, js_heavy, is_wordpress, is_elementor, "
        "content_len_norm, requires_antibot, site_known, has_structured_data."
    )

    # Sla op in store (store_policy saved + activeert)
    store.store_policy(policy._snapshot)

    print(f"ContextualSoftmaxPolicy geinitialiseerd ✅")
    print(f"  feature_dim: {FEATURE_DIM}")
    print(f"  features: {FEATURE_NAMES}")
    print(f"  actions: {[a.id for a in actions]}")
    print(f"  snapshot_id: {policy._snapshot.id}")
    print(f"  policy_kind: contextual_softmax")

    # Print initiele weights
    weights = policy._snapshot.metadata.get("context_weights", {})
    for aid in actions:
        w = weights.get(aid.id, [])
        print(f"  weights[{aid.id}]: {[round(x, 3) for x in w]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())