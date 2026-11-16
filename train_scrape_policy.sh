#!/bin/bash
# train_scrape_policy.sh — Cron job wrapper voor AdaptiveBandit training
# Draait in de Hermes venv, scored + traint de scrape-strategy policy.

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

python3 train_adaptive.py \
  --agent-id hermes \
  --task-id choose-scrape-strategy \
  --min-episodes 1 \
  2>&1

echo "---"
echo "Training completed at $(date)"