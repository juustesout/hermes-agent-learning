#!/bin/bash
# cron_workflow.sh — Dagelijkse 13:00 cron: train policies, run podcast workflow, dashboard
# 
# Vereist:
#   Hermes venv met agent-learning geïnstalleerd
#   AGENT_LEARNING_STORE_BACKEND=local
#   AGENT_LEARNING_LOCAL_STORE_DIR=<repo>/store (default)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
if [ -d "/opt/hermes/.venv/bin" ]; then
  export PATH="/opt/hermes/.venv/bin:/usr/local/bin:/usr/bin:/bin"
fi

: "${AGENT_LEARNING_STORE_BACKEND:=local}"
: "${AGENT_LEARNING_LOCAL_STORE_DIR:=$REPO_ROOT/store}"
: "${AGENT_LEARNING_ADAPTIVE_LOGIT_DECAY:=0.97}"
: "${AGENT_LEARNING_ADAPTIVE_FAILURE_THRESHOLD:=3}"
: "${AGENT_LEARNING_ADAPTIVE_LR_BOOST:=5.0}"
export AGENT_LEARNING_STORE_BACKEND
export AGENT_LEARNING_LOCAL_STORE_DIR
export AGENT_LEARNING_ADAPTIVE_LOGIT_DECAY
export AGENT_LEARNING_ADAPTIVE_FAILURE_THRESHOLD
export AGENT_LEARNING_ADAPTIVE_LR_BOOST

cd "$REPO_ROOT" || exit 1

echo "=== HERMES WORKFLOW CRON ==="
echo "Started at $(date)"
echo "---"

# ---- Fase 1: Train all existing policies -------------------------------
echo ""
echo "=== Fase 1: Policy training ==="

# Scrape strategy (bestaand)
python3 train_adaptive.py \
  --agent-id hermes \
  --task-id choose-scrape-strategy \
  --min-episodes 1 \
  2>&1 || echo "[scrape] no episodes or error — continuing"

# Podcast task policies — train met Thompson (betere cold start)
echo ""
echo "--- Thompson: podcast tasks ---"
python3 train_thompson.py \
  --agent-id hermes \
  --task-prefix podcast- \
  --min-episodes 1 \
  --prior-alpha 3 --prior-beta 2 \
  2>&1 || echo "[thompson] no tasks or error — continuing"

# ---- Fase 2: Run podcast workflow (simulated, logged to store) ----------
echo ""
echo "=== Fase 2: Podcast workflow run ==="

python3 -c "
import sys, os
sys.path.insert(0, '.')

from podcast_workflow import build_podcast_dag, podcast_decision, podcast_execute, podcast_reward
from workflow_runner import WorkflowRunner
from agent_learning.storage.cosmos import get_default_store

store = get_default_store()

print(f'Store: {type(store).__name__}')
n_eps_before = store.count_episodes('hermes', full_only=False)
print(f'Episodes before: {n_eps_before}')

dag = build_podcast_dag(0.8)
runner = WorkflowRunner(dag, store=store, agent_id='hermes',
                         decision_fn=podcast_decision,
                         execute_fn=podcast_execute,
                         reward_fn=podcast_reward)
run = runner.execute(context={'topic': 'Dagelijkse Cron Run'})

n_eps_after = store.count_episodes('hermes', full_only=False)
new_eps = n_eps_after - n_eps_before
print(f'Episodes after:  {n_eps_after} (+{new_eps})')
print(f'Global reward:   {run.global_reward}')
print(f'Node results:')
for nr in run.node_results:
    print(f'  [{nr.distance_from_end}] {nr.node_name:<20} {nr.action_name:<20} {nr.execution_status:<10} credit={nr.credit_reward or 0:+.4f}')
" 2>&1 || echo "[workflow] run error — continuing"

# ---- Fase 3: Generate dashboard HTML -----------------------------------
echo ""
echo "=== Fase 3: Dashboard ==="

DASHBOARD_DIR="${DASHBOARD_DIR:-$REPO_ROOT/dashboard}"
export DASHBOARD_DIR
mkdir -p "$DASHBOARD_DIR"

python3 -c "
import sys, os
sys.path.insert(0, '.')

from graph_evolution import Dashboard
from agent_learning.storage.cosmos import get_default_store

store = get_default_store()
dash = Dashboard(store)
html = dash.html_dashboard()
path = os.path.join(os.environ.get('DASHBOARD_DIR', 'dashboard'), 'index.html')
with open(path, 'w') as f:
    f.write(html)
print(f'Dashboard written: {path} ({len(html)} chars)')
" 2>&1 || echo "[dashboard] error — continuing"

# ---- Done ----------------------------------------------------------------
echo ""
echo "=== COMPLETE ==="
echo "Finished at $(date)"