#!/bin/bash
# opslog — log een operatie
# Gebruik: opslog task-id action-id target [duration_ms] [cost] [true/false] "summary"
# Voorbeeld: opslog choose-scrape-strategy use_webclaw_scrape agentcursus.nl 3100 1 true "WebClaw scrape 24500 chars"

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="${AGENT_LEARNING_REPO:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}"
OPSLOG_PY="${OPSLOG_PY:-$REPO_ROOT/opslog.py}"
CONSOLIDATE_PY="${CONSOLIDATE_PY:-$REPO_ROOT/consolidate_ops.py}"

case "${1:-}" in
  log)
    shift
    $PYTHON "$OPSLOG_PY" log --task-id "$1" --action-id "$2" --target "$3" \
      ${4:+--duration-ms "$4"} ${5:+--cost "$5"} \
      --success "${6:-true}" --summary "${7:-}" --action-type "${8:-}" \
      ${9:+--logprob "$9"} ${10:+--context-features "${10}"}
    ;;
  stats)
    $PYTHON "$OPSLOG_PY" stats
    ;;
  tail)
    $PYTHON "$OPSLOG_PY" tail --n "${2:-10}"
    ;;
  count)
    $PYTHON "$OPSLOG_PY" count
    ;;
  health)
    $PYTHON "$OPSLOG_PY" health
    ;;
  consolidate)
    $PYTHON "$CONSOLIDATE_PY" --run
    ;;
  consolidate-dry)
    $PYTHON "$CONSOLIDATE_PY"
    ;;
  *)
    echo "Gebruik: opslog {log|stats|tail|count|health|consolidate} [args...]"
    echo ""
    echo "  opslog log <task-id> <action-id> <target> [duration_ms] [cost] [success] [summary] [action-type] [logprob] [context-features-json]"
    echo "  opslog stats"
    echo "  opslog tail [n]"
    echo "  opslog count"
    echo "  opslog health"
    echo "  opslog consolidate"
    echo ""
    echo "Voorbeelden:"
    echo '  opslog log choose-scrape-strategy use_webclaw_scrape site.nl 3100 1 true "WebClaw scrape" web_scrape'
    echo '  opslog log choose-scrape-strategy use_jina_reader site.nl 800 0 false "Jina failed" web_scrape -0.3 '\''{"cloudflare":1.0}'\'
    echo '  opslog log db-query postgres_query pattern_trainer 450 0 true "SELECT companies" database_query'
    echo '  opslog log web-research webclaw_search "AI agents" 2000 0 true "Searched topic" web_search'
    echo '  opslog log wp-publish rest_api_post site.nl/businesses 1200 0 true "Post created" wp_rest_api'
    ;;
esac
