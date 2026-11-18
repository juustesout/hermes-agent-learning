# Replay-CLI

Status: 18 november 2026. Spec + CLI `replay_advice.py`. Poortwachter voor de meetperiode.
Track B in geheugen, formeel register, holdout/constraints en het automatische meetperiode-hek blijven buiten deze spec.

## 1. Waarom

Shadow-data verzamelen heeft geen zin zolang het adviesmodel op echte URL-historie dezelfde actie teruggeeft. De meetperiode start pas als replay op de bestaande scrape-episodes minstens twee verschillende `advised_tool`-waarden laat zien.

Dit is niet de live-alarm `constant_advisor` (drempel 95 procent op een venster van 200 shadow-rijen). Die blijft voor later. Replay is een offline gate op een kleine, echte corpus.

## 2. Wat het is, wat het niet is

Het is:

- een read-only CLI die bestaande episodes in tijdvolgorde door Track A en `shadow_policy.advise` haalt
- een rapport: n, verdeling, of de gate open is
- dezelfde `pre_call_features` als shadow en training

Het is niet:

- training, holdout-accuracy, of een tweede `features.py`
- een Hermes-hook, LLM-call, of netwerk
- schrijven naar `store/shadow_decisions.jsonl`, `store/operations.jsonl`, of `store/domain_outcomes.jsonl`
- het automatische meetperiode-hek (punt 6 in FEATURES.md)

## 3. Corpus

Bron, in deze volgorde. Eerste bron die minstens een rij oplevert wint, tenzij `--jsonl` of `--episodes-dir` expliciet is gezet.

1. `--jsonl PAD` — een jsonl met episode-achtige objecten
2. `--episodes-dir PAD` — JSON-bestanden, default `store/episodes/hermes/`
3. `store/episodes/hermes/` als die bestaat
4. `store/operations.jsonl` als fallback, via dezelfde velden als `consolidate_ops.op_to_episode`

Filter:

- `task_id` default `choose-scrape-strategy`
- `target` is verplicht en moet een host opleveren (`features.host_from_url(target)` niet leeg)
- sorteer op `created_at` (ISO-string), dan `id`
- geen HTML, geen headers

Velden die de CLI mag lezen:

| Veld | Gebruik |
|------|---------|
| `id` | rapportregel |
| `target` | Track A URL |
| `created_at` | `at=` voor features; volgorde |
| `action_id` of `action_name` | gevraagde tool; optioneel `action_id=` in features |
| `execution_status` | succes voor in-memory domeinhistorie van latere rijen |
| `task_id` | filter |

Succes voor historie: `execution_status` in `{completed, success, ok}` is True; `failed` / `error` / `timeout` / `blocked` / `cancelled` is False; anders sla de rij over voor geheugen (advies zelf gaat wel door).

Deze worktree heeft geen `store/` (gitignored). De twaalf bestaande episodes zitten op de machine met de echte store, niet in git. Tests leveren een fixture van twaalf URL-rijen. De gate op een echte run faalt als er minder dan twaalf URL-dragende scrape-episodes zijn. Gebruik dan niet stilletjes de testhulp. Zijn er meer dan twaalf, replay alle gefilterde rijen; twaalf is het minimum, niet het maximum.

## 4. Verwerking

Een vast snapshot, eenmaal geladen. Default: `shadow_policy.load_snapshot("choose-scrape-strategy")`. Override: `--snapshot PAD`.

Per episode, in tijdvolgorde:

```text
history = uitkomsten van eerdere replay-rijen (niet van store/domain_outcomes.jsonl)
phi = features.pre_call_features(target, history=history, at=created_at, action_id=action_id)
advice = shadow_policy.advise(snapshot, phi)
```

Daarna, als de huidige rij een bekende succes/fail heeft, append `{domain, success, action_id, timestamp}` aan de in-memory history. Die append mag de phi van de huidige rij niet veranderen.

`FeatureNameMismatch` of een ontbrekend snapshot is een harde fout, geen stille nulvector.

Geen `record_pre` / `record_post`. Geen plugin. Geen `remember_outcome`.

## 5. Gate

Laat `rows` de replay-rijen zijn met niet-lege `advised_tool`.

```text
n = len(rows)
n_distinct = aantal unieke advised_tool
top_action, top_count = modus
top_share = top_count / n   als n > 0, anders 0
gate_open = (n >= min_n) and (n_distinct >= 2)
```

Default `min_n=12`. Geen venster van 200. Geen drempel 95 procent.

Constant advies (bijvoorbeeld 12 keer `use_jina_reader`) is een rapport, geen bug om te verbergen. De CLI print de verdeling en sluit de gate.

Exitcodes:

| Code | Betekenis |
|------|-----------|
| 0 | gate open: `n >= min_n` en minstens twee verschillende adviezen |
| 1 | gebruik / IO / JSON |
| 2 | n genoeg, advies constant (`n_distinct < 2`) |
| 3 | te weinig URL-episodes (`n < min_n`) |
| 4 | snapshot ontbreekt of `FeatureNameMismatch` |

Stdout is het rapport (JSON). Optioneel `--out PAD` schrijft hetzelfde object; default pad indien gezet: `store/replay-report.json`. Dat bestand is geen shadow-log.

Rapportvelden:

```json
{
  "gate_open": false,
  "n": 12,
  "min_n": 12,
  "n_distinct": 1,
  "top_action": "use_jina_reader",
  "top_share": 1.0,
  "mean_top_margin": 0.02,
  "counts": {"use_jina_reader": 12},
  "skipped_no_host": 9,
  "skipped_wrong_task": 0,
  "skipped_samples": [{"reason": "no_host", "id": "ep-x", "task_id": "choose-scrape-strategy", "target": "/opt/data/file"}],
  "task_id": "choose-scrape-strategy",
  "policy_id": "pol-a",
  "policy_version": 1,
  "feature_track": ["url_host_nieuws", "url_host_code", "url_host_social", "url_host_blog", "url_host_overig", "url_path_depth", "url_has_query", "domain_seen_count", "domain_last_success", "domain_prior_rate"],
  "source": "replay",
  "rows": [
    {
      "id": "ep-1",
      "target": "https://github.com/org/repo",
      "created_at": "2026-11-01T00:00:00+00:00",
      "requested_tool": "web_extract",
      "advised_tool": "use_jina_reader",
      "probabilities": {
        "use_webclaw_scrape": 0.047426,
        "use_jina_reader": 0.952574,
        "use_browser": 0.0
      },
      "context_features": {}
    }
  ]
}
```

`policy_id` is `snapshot.id` (het opgeslagen policy-document). `task_id` blijft de taakfilter. Leeg `policy_id` komt alleen voor als het snapshot ontbreekt, en dat is exit 4.

`rows[].probabilities` is altijd `{action_id: prob}` uit `shadow_policy.advise`, nooit `{}`. Daarmee is "bijna constant" (0.51 / 0.49) te onderscheiden van "ruim gewonnen". `mean_top_margin` is het gemiddelde van (hoogste minus tweede) over de replay-rijen. `rows[].context_features` volgt `PRE_CALL_FEATURE_NAMES`. Geen Track B-sleutels.

Rijen zonder host of met een andere `task_id` tellen niet mee voor `n`. Ze verschijnen als `skipped_no_host` / `skipped_wrong_task` plus `skipped_samples`. Filter: `eligibility.eligible_scrape`.

Stderr krijgt altijd een regel: `gate=closed  exit=3  n=3/12  distinct=1  top=...  margin=...`.

## 6. CLI

Bestand: `replay_advice.py`. Geen extra PyPI. Geen wijziging aan `agent_learning`.

```bash
python3 replay_advice.py
python3 replay_advice.py --task-id choose-scrape-strategy --min-n 12
python3 replay_advice.py --jsonl path/to/episodes.jsonl --snapshot path/to/policy.json
python3 replay_advice.py --out store/replay-report.json
```

Flags:

- `--task-id` default `choose-scrape-strategy`
- `--min-n` default 12
- `--jsonl`
- `--episodes-dir`
- `--snapshot`
- `--out`
- `--agent-id` default `hermes` (alleen voor snapshot-lookup)

Geen `--train`, geen `--write-outcomes`, geen `--live`.

## 7. Tests

Nieuw: `tests/test_replay_advice.py`. Geen netwerk. Tijdelijke store via de bestaande fixture.

1. Twaalf URL-episodes over minstens twee host-categorieen, snapshot met niet-vlakke Track A-gewichten: exit 0, `n_distinct >= 2`, `gate_open` true.
2. Dezelfde twaalf, gewichten die een actie altijd winnen: exit 2, `top_share == 1.0`, rapport toont de constante actie.
3. Elf geldige rijen: exit 3, geen gate_open.
4. Oud 8-dim snapshot: exit 4, `FeatureNameMismatch`.
5. Replay schrijft niet naar `domain_outcomes.jsonl` of `shadow_decisions.jsonl`.
6. Uitkomst van episode t+1 lekt niet in phi van episode t (`domain_last_success` identiek aan een aanroep met `at=` van t).
7. `advice_features` in de replay-module is `features.pre_call_features`.
8. Rijen zonder host in `target` tellen niet mee voor n.

De testhulp mag synthetische URLs gebruiken. De productie-CLI mag die hulp niet importeren als corpus.

## 8. Acceptatie voor de meetperiode

Handmatig, na groene tests op deze CLI:

1. Snapshot is Track A (`feature_track=pre_call`, namen gelijk aan het register). Zo niet: `python3 init_contextual_policy.py --force` en hertrain, buiten deze spec.
2. `python3 replay_advice.py` op de echte store eindigt met exit 0.
3. Als exit 2: meting niet starten; rapport bewaren; niet omzeilen met een andere featureset.

Eerst deze CLI. Daarna pas shadow-data voor de meetperiode.
