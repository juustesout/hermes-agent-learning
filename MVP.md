# MVP — Adaptive Learning for Hermes

Status: 18 november 2026. Opdrachtgever: Juust. Voor: Monkey.
Werkdocument. Elk onderdeel heeft toetsbare criteria.
Naam is voorlopig. Merkcheck op "Hermes" en "Adaptive Learning for Hermes" eerst. Tot die check klaar is: geen publicatie.

Dit document beschrijft wat er in de MVP zit en wat niet. Het is geen intern ontwerp.

## 1. Doel

Een open source Hermes-plugin die bijhoudt welke tool-keuzes en automations goed uitpakken, die een vaste baseline draait, en die pas leert als de learner aantoonbaar beter is. Eerste gebruikers: Juust, en een externe pilot die de plugin installeert.

Geen SaaS. Geen multi-tenant. Geen N8N-integratie.

## 2. Wat erin zit

A. Decision-record. Een record per beslissing: `decision_id`, `join_key`, tijdstip, gevraagde tool, advies (indien aanwezig), gekozen tool, features die op dat moment bekend waren (alleen pre-call), later de uitkomst. Alle andere logs zijn afgeleid van dit record.

B. Shadow mode. `pre_tool_call` retourneert altijd `None`. De policy adviseert en logt. Hermes-gedrag verandert niet.

C. Vaste baseline. Een vastgelegde regel per workflow, in de repo. Die regel draait standaard. Een learner mag pas gebruikt worden als de promotiepoort (E) is gehaald.

D. Outcome-interface. Zie sectie 3.

E. Promotiepoort. Learner alleen actief als op echte uitkomsten, ITT, de learner aantoonbaar beter is dan de baseline met een vastgelegde marge. Zonder dat bewijs blijft de baseline. De poort is een functie met een getest besluit, geen handmatige keuze.

F. Eligibility. Een functie `eligible(row)` gedeeld door training, consolidatie, replay en shadow-health. Rijen die niet voldoen worden geteld, niet stil weggelaten.

G. CLI. `status`, `record-outcome`, `report`, `export-skill`.

H. Installatie voor de pilot. README die de installatie in een keer beschrijft, en een config met workflow-id, baseline-regel en outcome-endpoint.

## 3. Outcome-niveaus

Elke beslissing krijgt maximaal drie uitkomsten, elk met een eigen tijdstip:

1. Technisch: de tool gaf geen fout. Zichtbaar op `post_tool_call`.
2. Geaccepteerd: het systeem gaf een id of bevestiging. Zichtbaar op de hook als het resultaat dat bevat.
3. Bevestigd: het eindresultaat is geverifieerd. Via `record-outcome`, callback, of handmatige invoer.

De succesmaat van de promotiepoort is niveau 3 als die beschikbaar is. Ontbreekt niveau 3, dan blijft de poort gesloten en meldt het rapport dat.

## 4. Criteria

### A. Decision-record

- Elke beslissing heeft precies een record, gekoppeld op `join_key`.
- Een uitkomst zonder bijbehorend record wordt als afwijking gerapporteerd, niet stil toegevoegd.
- Pre-call-velden bevatten geen html of headers.

### B. Shadow mode

- Test met 1000 willekeurige inputs: `pre_tool_call` retourneert altijd `None`.
- Exception in de policy-lookup wordt gelogd en vangt de hook op.
- Hook onder 50 ms, ook met koude cache.
- Kill switch zet advies en log uit.

### C. Baseline

- Baseline-regel als vastgelegde functie in de repo, met een golden test.
- Zonder geldige learner draait de plugin de baseline, en het rapport zegt dat.

### D. Outcome-interface

- `record-outcome` accepteert `decision_id`, niveau en status, en valideert die.
- Een bevestiging kan later komen dan de beslissing, en koppelt correct.
- Een dubbele bevestiging verandert de uitkomst niet.

### E. Promotiepoort

- Pure functie: invoer is het rapport, uitvoer is open of closed met een reden.
- Te weinig data, constant advies, of marge onder de drempel: closed.
- Test met vaste fixtures voor elk van die gevallen.

### F. Eligibility

- Test controleert dat training, consolidatie, replay en shadow-health dezelfde functie gebruiken.
- Rapport toont `skipped_no_host` en `skipped_wrong_task`, met een sample van afgewezen targets.

### G. CLI

- `status`: een menselijke regel, met gate-status, aantal beslissingen, baseline.
- `report`: per workflow succesmaat op elk niveau, overlap, `mean_top_margin`.
- `export-skill`: `workflow.json` en optioneel `cases.jsonl`, zonder operationele gegevens.

### H. Installatie

- Schone installatie op een tweede machine, volgens de README, zonder aanpassingen buiten de config.
- Geen hardcoded paden. Config of omgevingsvariabelen.

## 5. Wat er expliciet niet in zit

- Multi-tenant, accounts, hosted variant.
- TikTok-adapter of andere platformspecifieke integratie. Outcome-interface is generiek.
- N8N-integratie.
- Spamprotectie en platformbeleid. Verantwoordelijkheid van de gebruiker.
- Dashboard of webinterface.
- Harde routering van tool-keuzes. Pas na een gehaalde promotiepoort, buiten de MVP.
- Opgeschoonde, gecertificeerde release. Open source met een pilot, geen productiesoftware.

## 6. Volgorde

1. Eligibility en skipped-tellers (F). Meting moet kloppen.
2. Decision-record (A) en outcome-interface (D).
3. Baseline en promotiepoort (C, E).
4. CLI (G) en installatie (H).
5. Shadow mode (B) afronden, pilot starten.

Elke stap eigen tests. Groen voordat de volgende start.

Open punt 1 blokkeert stap 2: hook-API eerst tegen Hermes-docs of bron op een echte installatie.

## 7. Open punten

1. Hook-API van Hermes. Dat `pre_tool_call`, `post_tool_call` en `pre_llm_call` bestaan en zo werken, is nog niet tegen de Hermes-documentatie of broncode gecontroleerd. Dat moet op een echte installatie, voor stap 2.
2. Naam. Merkcheck. Tot die check klaar is: geen publicatie.
3. Doorlooptijd. Zes tot acht weken is een inschatting. Na stap 1 en 2 bijstellen.
4. Pilot-partij. Wie, welke workflow. Dat bepaalt welke outcome-niveaus beschikbaar zijn.

## 8. Stand in deze repo (18 nov 2026)

Geen intern ontwerp. Alleen of de criteria nu al groen zijn.

| Onderdeel | Stand | Bewijs |
|-----------|-------|--------|
| F Eligibility | grotendeels groen, niet af | `eligibility.eligible_scrape` gedeeld door replay, train, consolidate, shadow-health. Test: `tests/test_eligibility.py`. Replay toont skipped + sample + `mean_top_margin`. Mapping `wp-publish`/`db-query`/`web-research` valt niet meer in scrape. |
| B Shadow | aanwezig, niet afgerond als MVP-stap | `hermes_plugin.on_pre_tool_call` altijd `None`. Tests in `tests/test_shadow_policy.py` / `tests/test_hermes_plugin.py`. Afronden is stap 5, niet nu. |
| Replay-gate | helper, geen MVP-onderdeel | `replay_advice.py`. Exit 0/2/3/4 getest. Op echte store: exit 3, n=3. `min_n` blijft 12. |
| A Decision-record | niet begonnen | Logs liggen nog in `operations.jsonl`, `shadow_decisions.jsonl`, episodes. Geen atoom. |
| C Baseline | niet begonnen | Geen vastgelegde baseline-functie per workflow. |
| D Outcome-interface | niet begonnen | Geen `record-outcome`, geen niveau 3. |
| E Promotiepoort | niet begonnen | Replay-gate is ijk van adviesverdeling, niet ITT versus baseline op niveau 3. |
| G CLI | niet begonnen | `replay_advice.py` is geen `status`/`report`/`export-skill`. |
| H Installatie | niet begonnen | README is ontwikkelaarsdoc, geen pilot-install. |

Stap 1 (F) is het huidige werk. Stap 2 (A, D) start pas na open punt 1.
