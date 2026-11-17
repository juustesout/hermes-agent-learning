# Hermes Agent Learning

A self-learning agent system that improves decisions through experience. Built on three levels of learning — from individual edge choices to full graph trajectory optimization.

```
Edge Learning ──→ Path Learning ──→ Graph Evolution
(level 1)         (level 2)         (level 3)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Graph Evolution                          │
│  DagRegistry · MultiDagOrchestrator · GraphOptimizer       │
│  Dashboard (HTML) · RunAnalyzer                            │
├─────────────────────────────────────────────────────────────┤
│                    Path Learning                            │
│  WorkflowDAG · WorkflowRunner · Credit Assignment (γ^d)    │
│  Episode Logging · Reward Propagation                      │
├─────────────────────────────────────────────────────────────┤
│                    Edge Learning                            │
│  SoftmaxPolicy · ContextualSoftmaxPolicy · ThompsonPolicy   │
│  AdaptiveBanditLearner · ContextualAdaptiveBandit          │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                            │
│  agent-learning store (local/Cosmos) · Episodes · Rewards  │
│  Policy snapshots · Metrics                                │
└─────────────────────────────────────────────────────────────┘
```

### Level 1 — Edge Learning
Each decision point has a `TaskPolicy` that learns which action works best. Three policy types:

| Policy | Method | Best for |
|--------|--------|----------|
| `SoftmaxPolicy` | REINFORCE with baseline | Established patterns, many episodes |
| `ContextualSoftmaxPolicy` | W·φ gradient per context | Site-type dependent decisions |
| `ThompsonPolicy` | Beta(α,β) sampling | Cold start, few episodes |

### Level 2 — Path Learning (WorkflowRunner)
Multi-step workflows as DAGs. Each node is a decision point with its own policy. After execution, the global reward is backpropagated along the path:

```
credit[node] = global_reward × γ^distance_from_end
```

Example podcast DAG with γ=0.8:
```
topic_select → research → script → audio → video → publish
dist:   5          4          3        2       1         0
γ^d:  0.33       0.41       0.51     0.64    0.80     1.00
```

### Level 3 — Graph Evolution
- **DagRegistry** — multi-DAG management with meta-policy selection
- **MultiDagOrchestrator** — sequential, conditional, looped, and meta-DAG execution
- **GraphOptimizer** — gamma optimization, bottleneck detection, reorder suggestions
- **Dashboard** — self-contained HTML dashboard with per-node stats and reward trends

---

## Quick Start

```bash
# Install dependencies
pip install agent-learning numpy

# Train a scrape strategy policy
python3 train_adaptive.py --agent-id hermes --task-id choose-scrape-strategy

# Train with Thompson sampling (cold start)
python3 train_thompson.py --agent-id hermes --task-prefix podcast-
```

### Podcast Workflow

```bash
# Show the DAG
python3 podcast_workflow.py --show

# Dry run (simulated)
python3 podcast_workflow.py --dry-run --topic "AI Agents in 2026"

# With forced failure for reward testing
python3 podcast_workflow.py --dry-run --topic "Test" --force-fail publish

# Show credit assignment table
python3 podcast_workflow.py --simulate-rewards --discount 0.8

# Real run with store logging
python3 podcast_workflow.py --run --topic "AI Agents"
```

### Dashboard

```bash
python3 -c "
from graph_evolution import Dashboard
from agent_learning.storage.cosmos import get_default_store
store = get_default_store()
html = Dashboard(store).html_dashboard()
open('dashboard/index.html', 'w').write(html)
print('Dashboard → dashboard/index.html')
"
```

Security note: dashboard output now HTML-escapes all episode metadata fields
before rendering table cells/headings. This prevents script injection when runs
contain untrusted values (for example custom `action_id` strings).

---

## Project Structure

```
├── adaptive_learner.py        # AdaptiveBanditLearner (logit decay, failure det.)
├── contextual_learner.py      # ContextualAdaptiveBanditLearner (W·φ gradients)
├── thompson_policy.py         # ThompsonPolicy + ThompsonLearner (Beta sampling)
├── scrape_features.py         # Context feature schema (8 dims)
├── train_adaptive.py          # Training CLI (marginal + contextual)
├── train_thompson.py          # Training CLI (Thompson sampling)
│
├── workflow_dag.py            # DAG dataclasses (Node, Edge, topo sort)
├── workflow_runner.py         # Sequential executor + credit assignment
├── workflow_visualize.py      # ASCII + Mermaid + run reports
├── podcast_workflow.py        # Concrete 6-node podcast DAG + CLI
│
├── graph_evolution.py         # DagRegistry, MultiDagOrch, GraphOptimizer, Dashboard
├── skill_pack.py              # Export/import shareable workflow.json packs
├── skill_schema/              # Versioned JSON Schema + stdlib validator
│                              #   (workflow, node, episode, reward, case)
├── schema/workflow.schema.json  # compat alias → skill_schema/schemas/skill.schema.json
├── hermes_plugin/             # Hermes observer — standaard sinds 17 Nov 2026
│
├── cron_workflow.sh           # Daily cron: train + run + dashboard
├── train_scrape_policy.sh     # Every-6h cron: scrape policy training
│
├── store/                     # Runtime data (gitignored)
│   ├── episodes/
│   ├── policies/
│   ├── rewards/
│   └── runs/
├── docs/                      # Documentatie voor andere agents/developers
│   └── hermes-skill.md        # SKILL.md snapshot (Hermes skill store is source of truth)
└── pyproject.toml
```

---

## Policy Hierarchy

```
Policy
├── SoftmaxPolicy              # Marginal logits → softmax probabilities
│   └── ContextualSoftmaxPolicy  # W·φ feature-weighted logits
└── ThompsonPolicy             # Beta(α,β) posterior per action
```

## Learner Hierarchy

```
Learner
└── ReinforceLearner
    └── AdaptiveBanditLearner (logit decay + failure det. + dynamic LR)
        └── ContextualAdaptiveBanditLearner (W·φ gradients)
```

## Operation Logging (OpsLog)

Tool-operaties worden automatisch gelogd naar `store/operations.jsonl` via de
**Hermes plugin** (zie sectie hieronder). Geen handmatige logging nodig.

De nachtelijke cron (`opslog-consolidate`, 03:00 UTC) consolideert de ruwe logs
naar volwaardige leer-episodes.

| Component | Locatie |
|-----------|---------|
| **Hermes plugin** | `hermes_plugin/__init__.py` — `post_tool_call` hook → `opslog.log_operation()` |
| **Consolidator** | `consolidate_ops.py` — ruwe log → episode + reward |
| **Live log** | `store/operations.jsonl` (append-only) |
| **Cron** | `opslog-consolidate` — elke nacht 03:00 UTC |

### Task-ID conventies

| Task-ID | Action-Type |
|---------|-------------|
| `choose-scrape-strategy` | web_scrape, web_search |
| `db-query` | database_query |
| `wp-publish` | wp_rest_api |
| `web-research` | web_search |
| `podcast-*` | research, script, audio, publish |

### Cron Schedule

| Job | Schedule | Action |
|-----|----------|--------|
| `train-scrape-policy` | Every 6 hours | Train ContextualAdaptiveBandit on scrape episodes |
| `podcast-workflow-daily` | Daily 13:00 UTC | Train all policies + run podcast workflow + dashboard |
| `opslog-consolidate` | Daily 03:00 UTC | Consolidate operations.jsonl → learning episodes |

Cron scripts zijn nu path-portable: ze gebruiken standaard de repository-map
als basis en accepteren overrides via environment variables:

- `AGENT_LEARNING_LOCAL_STORE_DIR` (default: `<repo>/store`)
- `DASHBOARD_DIR` (default: `<repo>/dashboard`, alleen `cron_workflow.sh`)

## Hermes plugin (standaard sinds 17 Nov 2026)

`hermes_plugin/` is een thin Hermes observer. Hij logt elke Hermes tool-call
automatisch naar `store/operations.jsonl` via `opslog.log_operation()`.
Geen handmatige logging meer nodig.

### Installatie

```bash
# 1. Installeer vanuit GitHub
hermes plugins install juustesout/hermes-agent-learning --enable

# 2. De plugin.yaml en __init__.py zitten in de hermes_plugin/ subdirectory.
#    Hermes verwacht ze in de root van de plugin directory. Leg symlinks aan:
cd /opt/data/profiles/operator/plugins/hermes-agent-learning
ln -sf hermes_plugin/plugin.yaml plugin.yaml
ln -sf hermes_plugin/__init__.py __init__.py

# 3. Enable (indien --enable bij stap 1 niet werkte)
hermes plugins enable agent-learning

# 4. Verifieer
hermes plugins doctor agent-learning

# 5. Herstart de gateway
hermes gateway restart
```

### Werking

`post_tool_call` mapped Hermes tool outcomes naar `operations.jsonl`. De hook roept
`opslog.log_operation()` aan. Houd de repo importeerbaar (`PYTHONPATH=<repo>`,
`AGENT_LEARNING_REPO=<repo>`, of `pip install -e .`) zodat de import werkt.
Als `opslog` niet importeerbaar is, is de hook een no-op.

### CLI

```bash
hermes agent-learning stats
hermes agent-learning count
hermes agent-learning tail -n 20
```

## CI (GitHub Actions)

Deze repo bevat CI in `.github/workflows/ci.yml`.
Bij pushes naar `main` en op pull requests draait de pipeline op Python 3.11 en 3.12:

- `ruff check . --select E9,F63,F7,F82` (kritieke lint checks voor syntax/name errors)
- `bandit -r .` (security static analysis, exclusief `store/`, `dashboard/` en `tests/`)
- `python -m compileall -q .` (syntax check)
- `pytest tests/`
- `python holdout_eval.py --baseline-only --json`
- `python test_adaptive_vs_standard.py` (vaste seeds)

## Dependency Updates & Vulnerability Audit

- Dependabot configuratie: [.github/dependabot.yml](./.github/dependabot.yml)
  - `pip` updates (wekelijks)
  - GitHub Actions updates (wekelijks)
- Fallback audit workflow: [.github/workflows/security-audit.yml](./.github/workflows/security-audit.yml)
  - draait `pip-audit` op pull requests, wekelijks en handmatig (`workflow_dispatch`)

Als Dependabot in een andere repo niet opent, controleer dan in GitHub:
- **Settings → Security & analysis** (Dependabot alerts + security updates)
- **Actions permissions** en of workflows mogen draaien
- of de default branch echt `main` is

---

## Related Work

This project builds on [Microsoft agent-learning](https://github.com/microsoft/agent-learning) (v0.10.0) for the core REINFORCE learner and storage layer. Additional algorithms and architecture are inspired by:

- **Self-Improving Agentic Systems** — LinUCB, Thompson sampling, cost-aware credit
- **AgentForge** — Generic workflow selector pattern
- **OpenJarvis** — Trace storage and skill discovery
- **SAP contextual-bandit-agent-router** — MCP-based bandit service pattern

---

## License

MIT