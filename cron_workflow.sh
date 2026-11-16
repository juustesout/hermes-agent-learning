#!/bin/bash
# cron_workflow.sh — Dagelijkse 13:00 cron: train policies, run podcast workflow, dashboard
#
# Vereist:
#   Hermes venv met agent-learning geïnstalleerd
#
# Exit code: aantal gefaalde stappen (0 = alles OK).

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
echo "Store: $AGENT_LEARNING_LOCAL_STORE_DIR"

# Track failures — elke stap verhoogt EXIT_CODE bij falen
EXIT_CODE=0

step_ok()   { echo "[OK] $1"; }
step_fail() { echo "[FAIL] $1"; EXIT_CODE=$((EXIT_CODE + 1)); }

# ---- Fase 1: Train all existing policies -------------------------------
echo ""
echo "=== Fase 1: Policy training ==="

python3 train_adaptive.py \
  --agent-id hermes \
  --task-id choose-scrape-strategy \
  --min-episodes 1 \
  2>&1 && step_ok "scrape adaptive" || step_fail "scrape adaptive"

echo ""
echo "--- Thompson: podcast tasks ---"
python3 train_thompson.py \
  --agent-id hermes \
  --task-prefix podcast- \
  --min-episodes 1 \
  2>&1 && step_ok "podcast thompson" || step_fail "podcast thompson"

# ---- Fase 2: Run podcast workflow --------------------------------------
echo ""
echo "=== Fase 2: Podcast workflow run ==="

python3 -c "
import sys, os
sys.path.insert(0, '.')

from podcast_workflow import build_podcast_dag, make_podcast_decision, podcast_execute
from workflow_runner import WorkflowRunner
from cost_aware_rewards import balanced_reward
from agent_learning.storage.cosmos import get_default_store

store = get_default_store()
reward_fn = balanced_reward()

print(f'Store: {type(store).__name__}')
print(f'Reward: balanced (completion=1.0, latency=0.3, cost=0.3, quality=0.2)')
try:
    n_eps_before = store.count_episodes('hermes', full_only=False)
    print(f'Episodes before: {n_eps_before}')
except Exception:
    n_eps_before = None

dag = build_podcast_dag(0.8)
runner = WorkflowRunner(dag, store=store, agent_id='hermes',
                         decision_fn=make_podcast_decision(store),
                         execute_fn=podcast_execute,
                         reward_fn=reward_fn.compute)
run = runner.execute(context={'topic': 'Dagelijkse Cron Run'})

if n_eps_before is not None:
    try:
        n_eps_after = store.count_episodes('hermes', full_only=False)
        print(f'Episodes after:  {n_eps_after} (+{n_eps_after - n_eps_before})')
    except Exception:
        pass
print(f'Global reward:   {run.global_reward}  ({run.global_reward_reason})')
print(f'Node results:')
for nr in run.node_results:
    print(f'  [{nr.distance_from_end}] {nr.node_name:<20} {nr.action_name:<20} {nr.execution_status:<10} credit={nr.credit_reward or 0:+.4f}')
" 2>&1 && step_ok "podcast workflow" || step_fail "podcast workflow"

# ---- Fase 3: Skill Discovery -------------------------------------------
echo ""
echo "=== Fase 3: Skill Discovery ==="

python3 -c "
import sys, os
sys.path.insert(0, '.')

from skill_discovery import SkillDiscovery
from agent_learning.storage.cosmos import get_default_store

store = get_default_store()
disc = SkillDiscovery(store)
report = disc.report(dag_id='podcast', min_support=2)
print(report)

skills = disc.discover(dag_id='podcast', min_support=2)
print(f'\nDiscovered {len(skills)} skills (min_support=2)')
for s in skills[:5]:
    print(f'  [{s.confidence:.2f}] {s.name}  (freq={s.frequency})')
" 2>&1 && step_ok "skill discovery" || step_fail "skill discovery"

# ---- Fase 4: Generate dashboard HTML -----------------------------------
echo ""
echo "=== Fase 4: Dashboard ==="

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
" 2>&1 && step_ok "dashboard" || step_fail "dashboard"

# ---- Done ----------------------------------------------------------------
echo ""
echo "=== COMPLETE ==="
echo "Finished at $(date)"
echo "Exit code: $EXIT_CODE ($EXIT_CODE step(s) failed)"
exit "$EXIT_CODE"