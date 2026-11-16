#!/bin/bash
set -e

# Prefer the Hermes venv when present
if [ -d "/opt/hermes/.venv/bin" ]; then
  export PATH="/opt/hermes/.venv/bin:$PATH"
fi

cd "$(dirname "$0")/.."

python3 -m pytest tests/ -v --tb=short 2>&1
