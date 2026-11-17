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
├── hn_ranker.py               # HN slot coin-flip + item ranker (prior-v1 / logreg)
├── hn_sim.py                  # Accelerated local posting job (loopback WP mock)
├── sim/                       # WordPress draft mock + loopback client
├── blind_label.py             # Blind labels, enqueue, audit for HN evidence
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
| **Hermes plugin** | `hermes_plugin/__init__.py` — `post_tool_call` hook → `opslog.log_operation()` (`source=hermes`) |
| **Automation logger** | `run_log.py` — cron/workflow steps write themselves (`source=automation`) |
| **Consolidator** | `consolidate_ops.py` — ruwe log → episode + reward |
| **Live log** | `store/operations.jsonl` (append-only) |
| **Run heartbeat** | `store/runs_automation.jsonl` — start/end per automation run |
| **Cron** | `opslog-consolidate` — elke nacht 03:00 UTC |

Automations (podcast cron, HN 6h job) do not go through the Hermes hook.
They log with `run_log.run_log(workflow)`: one opslog row per step, one
start/end pair in `runs_automation.jsonl`. A step without `ok()`/`fail()`,
or an exception, is a failure and is re-raised. `run_log.heartbeat()`
flags missing or unfinished runs. Podcast/cron wrappers are a follow-up.

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

Enabling the plugin is the opt-in. Tool-call success (`status == ok`) is not
user-task success; metadata stores `tool_ok` and `task_success=false`.

### Installatie

`plugin.yaml` and `__init__.py` live at the repository root, so a Git install
needs no symlinks. `hermes_plugin/` remains the implementation package.

```bash
hermes plugins install juustesout/hermes-agent-learning --enable
hermes plugins doctor agent-learning --ci
```

From a local checkout:

```bash
hermes plugins install . --enable
hermes plugins doctor . --ci
```

Keep the repo importable (`PYTHONPATH=<repo>`, `AGENT_LEARNING_REPO=<repo>`,
or `pip install -e .`) so the hook can import `opslog`. If `opslog` is missing,
the hook is a no-op.

### Store and privacy

| Variable | Default | Meaning |
|----------|---------|---------|
| `AGENT_LEARNING_OPSLOG_DIR` | `AGENT_LEARNING_LOCAL_STORE_DIR` or `<repo>/store` | Where `operations.jsonl` is written |
| `AGENT_LEARNING_LOCAL_STORE_DIR` | `<repo>/store` | Shared learning store |
| `AGENT_LEARNING_OPSLOG_ENABLED` | `1` | Set `0` to keep the plugin loaded but silent |
| `AGENT_LEARNING_OPSLOG_SESSION_IDS` | `0` | Set `1` to store Hermes session/turn ids |
| `AGENT_LEARNING_OPSLOG_RETENTION_DAYS` | `30` | Drop older operations; `0` disables prune |
| `AGENT_LEARNING_TOOL_TASK_MAP` | built-in web tools | JSON object `{tool_name: task_id}` |
| `AGENT_LEARNING_TOOL_TASK_MAP_FILE` | unset | JSON file with the same mapping |

Logged fields: `task_id`, `action_id`, redacted `target` (query strings stripped),
redacted error summary, latency, tool status. No tool results. Session ids are
off unless enabled. Query parameters and common secret patterns are redacted.
There is no encryption; treat the JSONL file as sensitive.

Check write health with `hermes agent-learning health` or `python3 opslog.py health`.

### CLI

```bash
hermes agent-learning stats
hermes agent-learning count
hermes agent-learning health
hermes agent-learning tail -n 20
```

## Blind labels (HN evidence)

Frozen before measurement: `evidence/success_definition.json`,
`evidence/hn_baseline.json`, and `evidence/hn_ranker.json`. Baseline = first
remaining relevant item in source order after hard dedupe. Learned = `hn_ranker.py`
on the same remaining pool. HTTP 2xx is technical success only.

Randomization is one coin per posting slot (`p_learned=0.5`). The 6h job must
select, then post, then enqueue:

```bash
python3 hn_ranker.py select --run-id 2026-11-17T12 --json '[{"id":"...","title":"...","url":"...","rag_score":0.8,"source_rank":0}]'
python3 blind_label.py enqueue --json '{"id":"hn-1","title":"...","url":"...","summary":"...","http_status":201,"auto_relevant":true}'
python3 blind_label.py record-run --attempted 2 --posted 2 --failed 0 --run-id 2026-11-17T12
python3 blind_label.py next
python3 blind_label.py label --id hn-1 --relevant yes --duplicate no --needs-correction no
python3 blind_label.py report
python3 blind_label.py audit
```

`items.jsonl` is rater-safe. `assignments.jsonl` holds one decision record per
slot (arm L or B, pool, chosen_id, features) and one outcome record per HTTP
attempt. `audit` exits 2 if posts lag, a post has no decision, or the kill
switch is on (`AGENT_LEARNING_HN_RANKER_KILL=1`).

About 20 percent of posts are sampled (stable hash, ~11 labels/week at 8 posts/day).
If rater/auto agreement drops below 90 percent, revise the automatic quality
measure before using it as proof. The ranker stays on `prior-v1` (`rag_score`
descending) until 200 auto-labeled posts and that agreement floor.

### Local mechanical simulation

`hn_sim.py` runs the 6h job against a loopback WordPress mock. Drafts only;
publish is 403. Isolate evidence so production files stay untouched:

```bash
AGENT_LEARNING_EVIDENCE_DIR=/tmp/hn-sim python3 hn_sim.py run --runs 200
pytest -q tests/test_wp_mock.py tests/test_hn_sim.py
```

This checks select → draft POST → enqueue → record-run → audit. It does not
measure whether the content is good. Sim rows are tagged `source=sim`.
`report` and ranker training ignore them. Audit reports L/B overlap,
`n_contrast` (slots where L != B), and a binomial arm-balance check.

The primary lift claim is intention-to-treat on all posted slots. `n_contrast`
and the contrast-subset effect are explanatory: ITT difference is about
`contrast_rate` times the effect where L and B actually differ. Sim contrast
numbers are still not quality evidence.

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

This project builds on [Microsoft agent-learning](https://github.com/microsoft/agent-learning) (MIT License, v0.10.x) for the core REINFORCE learner and storage layer. That package is a runtime dependency (`agent-learning>=0.10.0,<0.11`); this repository does not vendor or re-license it. Additional algorithms and architecture are inspired by:

- **Self-Improving Agentic Systems** — LinUCB, Thompson sampling, cost-aware credit
- **AgentForge** — Generic workflow selector pattern
- **OpenJarvis** — Trace storage and skill discovery
- **SAP contextual-bandit-agent-router** — MCP-based bandit service pattern

Hermes is a product of Nous Research. This repository is an independent MIT-licensed observer/learning kernel (`juustesout/hermes-agent-learning`) and is not an official Nous Research distribution.

---

## License

MIT. Copyright (c) 2026 juustesout.