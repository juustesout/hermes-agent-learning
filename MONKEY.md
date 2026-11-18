# MONKEY

Status: 18 november 2026. Werkdocument is `MVP.md`.

F (eligibility + skipped-tellers) is het huidige werk. A/D niet starten tot de hook-API op een echte Hermes-installatie is gecontroleerd.

`eligible_scrape(row)` is waar als `task_id == choose-scrape-strategy` en `host_from_url(target)` niet leeg is. Gedeeld door replay, train, consolidate, shadow-health.

Niet doen tot de MVP dat toestaat: plugin-UI, `min_n` verlagen, stap 2/3 (advies injecteren / harde routering), Track B in het advies van dezelfde call.
