# Backfill Log — 18 november 2026

## Wat

Oude `mcp_webclaw_webclaw_scrape`-operaties met `task_id=hermes-tool` her-labeld
naar `choose-scrape-strategy`, zodat de nieuwe mapping (plugin PR #261118-feat-eligible-scrape)
ook op historische data van toepassing is.

## Waarom

De Hermes-plugin kreeg een mapping `mcp_webclaw_webclaw_scrape → choose-scrape-strategy`.
Zonder backfill blijven historische webclaw-scrapes onder `hermes-tool` hangen en worden
ze niet meegeteld in de replay-gate of training.

## Uitgevoerd

| Veld | Waarde |
|------|--------|
| Datum | 2026-11-18 |
| Script | `scripts/ops/backfill_webclaw_task.py` |
| Auteur | Hermes agent (operator profile) |
| Backup van | `store/operations.jsonl` |
| Backup naar | `store/operations.jsonl.backup-20261118` |
| Gewijzigde rijen | 6 |
| Filter | `action_id == mcp_webclaw_webclaw_scrape AND task_id == hermes-tool` |
| Nieuwe task_id | `choose-scrape-strategy` |
| Metadata tag | `backfill_task_id: hermes-tool->choose-scrape-strategy` |

## Terugdraaien

```bash
cp store/operations.jsonl.backup-20261118 store/operations.jsonl
# Daarna opnieuw consolidatie:
python3 consolidate_ops.py --run
```

## Status na backfill

| Metriek | Voor | Na |
|---------|------|-----|
| choose-scrape-strategy URL-ops | 2 | 8 (operations.jsonl) |
| choose-scrape-strategy URL-episodes | 3 | 9 (incl weeskind 0d846871) |
| Unieke domeinen in historie | 2 | 5 |
| Gate status | exit 3 (n=3) | exit 3 (n=9) |

## Weeskinderen

Episode `0d846871` (agentcursus.nl) bestaat nog in `store/episodes/hermes/` maar de
originele operatie is al opgeruimd uit `operations.jsonl` (retentie). De episode is
valide en telt mee in de replay (episodes-bron), maar niet in een opslog-scan.