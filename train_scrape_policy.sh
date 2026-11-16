#!/bin/bash
# train_scrape_policy.sh — Cron job wrapper voor AdaptiveBandit training
# Draait in de Hermes venv, scored + traint de scrape-strategy policy.

export PATH="/opt/hermes/.venv/bin:/usr/local/bin:/usr/bin:/bin"
export AGENT_LEARNING_STORE_BACKEND=local
export AGENT_LEARNING_LOCAL_STORE_DIR=/opt/data/profiles/operator/agent-learning/store
export AGENT_LEARNING_ADAPTIVE_LOGIT_DECAY=0.97
export AGENT_LEARNING_ADAPTIVE_FAILURE_THRESHOLD=3
export AGENT_LEARNING_ADAPTIVE_LR_BOOST=5.0

cd /opt/data/profiles/operator/agent-learning || exit 1

python3 train_adaptive.py \
  --agent-id hermes \
  --task-id choose-scrape-strategy \
  --min-episodes 1 \
  2>&1

echo "---"
echo "Training completed at $(date)"