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
│
├── cron_workflow.sh           # Daily cron: train + run + dashboard
├── train_scrape_policy.sh     # Every-6h cron: scrape policy training
│
├── store/                     # Runtime data (gitignored)
│   ├── episodes/
│   ├── policies/
│   ├── rewards/
│   └── runs/
├── dashboard/                 # Generated HTML (gitignored)
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

## Cron Schedule

| Job | Schedule | Action |
|-----|----------|--------|
| `train-scrape-policy` | Every 6 hours | Train ContextualAdaptiveBandit on scrape episodes |
| `podcast-workflow-daily` | Daily 13:00 UTC | Train all policies + run podcast workflow + dashboard |

## CI (GitHub Actions)

Deze repo bevat CI in [.github/workflows/ci.yml](./.github/workflows/ci.yml).
Bij pushes naar `main` en op pull requests draait de pipeline op Python 3.11 en 3.12:

- `ruff check .` (linting)
- `bandit -r .` (security static analysis, exclusief `store/` en `dashboard/`)
- `python -m compileall -q .` (syntax check)

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