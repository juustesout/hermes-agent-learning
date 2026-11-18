# Features

Status: 18 november 2026. Bron: `features.py` op branch `261118-feat-precall-features` (PR #18).
Hermes hoeft `features.py` niet opnieuw te schrijven. Dit document beschrijft wat er is, hoe je het gebruikt, en wat er nog niet is.

## 1. Principe

Een featureset per moment. Advies bij `pre_tool_call` mag alleen gebruiken wat op dat moment bekend is.

| Track | Wanneer | Functie | Mag HTML/headers lezen |
|-------|---------|---------|------------------------|
| A, `pre_call` | voor de tool-call | `features.pre_call_features` / alias `advice_features` | nee |
| B, `post_fetch` | na de fetch | `features.post_fetch_features` | ja |

Training van het adviesmodel en shadow importeren dezelfde functie:

```python
from features import pre_call_features, advice_features
assert advice_features is pre_call_features
```

`shadow_policy.advice_features` is diezelfde functie. Een mismatch tussen snapshot-`feature_names` en het register is een fout (`FeatureNameMismatch`), geen stille nul.

## 2. Track A (advies)

Namen, in deze volgorde (`PRE_CALL_FEATURE_NAMES`, dim = 10):

| Naam | Wat | Bereik |
|------|-----|--------|
| `url_host_nieuws` | 1 als host-categorie nieuws | 0/1 |
| `url_host_code` | 1 als code | 0/1 |
| `url_host_social` | 1 als social | 0/1 |
| `url_host_blog` | 1 als blog | 0/1 |
| `url_host_overig` | 1 als overig | 0/1 |
| `url_path_depth` | padsegmenten, cap 8, gedeeld door 8 | 0..1 |
| `url_has_query` | querystring aanwezig | 0/1 |
| `domain_seen_count` | eerdere calls naar dit domein, cap 20, gedeeld door 20 | 0..1 |
| `domain_last_success` | laatste uitkomst dit domein; 0.5 als geen historie | 0 / 0.5 / 1 |
| `domain_prior_rate` | Beta-gemiddelde over eerdere uitkomsten; met `action_id` per actie | 0..1 |

Categorieen (Juust, 18 nov): nieuws, code, social, blog, overig. Matching is suffix-lijst in `_HOST_SUFFIXES`, plus lichte heuristics (`news` in host, `blog` in host, `git.*`, `*.github.io`). Geen netwerk.

Domeinhistorie mag worden opgeslagen (Juust, 18 nov). Bestand: `store/domain_outcomes.jsonl`. Tijdsordening: feature op tijdstip `t` gebruikt alleen rijen met timestamp strikt voor `t`.

```python
from features import pre_call_features

phi = pre_call_features("https://github.com/org/repo")
# optioneel: history=..., at="2026-11-18T12:00:00+00:00", action_id="web_extract"
```

Geen `html=` of `headers=` argument. Een aanroep daarmee moet `TypeError` geven.

## 3. Track B (niet voor advies van de huidige call)

Namen (`POST_FETCH_FEATURE_NAMES`):

`cloudflare`, `requires_antibot`, `js_heavy`, `is_wordpress`, `is_elementor`, `content_len_norm`, `has_structured_data`

Die komen uit `scrape_features.describe_site(url, html, headers)`. Ze horen bij de uitkomst van de fetch. `site_known` (hardcoded 0 in de oude schema) is vervangen door `domain_seen_count`.

Na `post_tool_call` schrijft shadow:

- `post_fetch_features` op het decision-record, als er HTML in `result` zat
- een rij in `domain_outcomes.jsonl` via `remember_outcome` (domein, success, action_id, timestamp)

Dat geheugen voedt Track A van de *volgende* call, niet de huidige.

## 4. Wie wat aanroept

| Plek | Track | Doel |
|------|-------|------|
| `shadow_policy.record_pre` | A | advies + log `feature_track` + waarden |
| `shadow_policy.record_post` | B + geheugen | HTML-features op het record; `remember_outcome` |
| `hermes_plugin.build_operation` | A | `context_features` op de opslog-rij (`source=hermes`) |
| `contextual_learner._extract_phi` | A | training aligned op `PRE_CALL_FEATURE_NAMES`; skipt ineligible scrape-rows |
| `eligibility.eligible_scrape` | filter | één regel voor replay, train, consolidate, shadow-health |
| `init_contextual_policy.py` | A | nieuw snapshot, `feature_track=pre_call` |
| `train_adaptive.py` | A | zero-vector in Track A-volgorde; skipt ineligible scrape-rows |
| `holdout_eval.py` | B (oud) | nog `scrape_features.SITE_PROFILES` |

Oud 8-dim snapshot (cloudflare/js/...) is ongeldig voor advies. Hertrain:

```bash
python3 init_contextual_policy.py --force
python3 train_adaptive.py --agent-id hermes --task-id choose-scrape-strategy
```

## 5. Shadow-record

Velden die bij Track A horen:

- `feature_track`: de namen die gebruikt zijn (gelijk aan het register)
- `context_features`: die waarden
- `advised_tool`, `probabilities`, `requested_tool`
- `source=shadow`

Alarm `constant_advisor`: meer dan 95% van de laatste 200 adviezen dezelfde actie. Geschreven naar `store/shadow-health.json`. Drempel 95% is akkoord van Juust.

## 6. Wat er nog niet is

Dit staat niet in code. Geen extra `features.py` voor nodig totdat er een spec is.

1. **Replay-CLI op echte URL-historie.** Spec: `REPLAY.md`. CLI: `replay_advice.py`. Meetperiode start pas bij exit 0 op minstens twaalf URL-episodes met `n_distinct >= 2`.

2. **Track B als geheugen-features, niet alleen success/fail.** Spec: post-fetch features "voeden het geheugen per domein voor de volgende call". Nu onthouden we alleen `{domain, success, action_id, timestamp}`. Cloudflare/js van call t-1 zitten niet in Track A van call t.

3. **Formeel register-object.** Spec: "elke feature heeft een naam, een beschikbaarheid, en een functie." Nu: twee naamlijsten plus `pre_call_features` / `post_fetch_features`. Geen `FEATURES = {name: {availability, fn}}`.

4. **`url_host_category` als een scalar.** Spec noemt een feature. Implementatie is one-hot (vijf kolommen). Dat is een keuze, geen gat in gedrag, maar de naam in de spec is niet de naam in de vector.

5. **Holdout en constraints op Track A.** `holdout_eval.py` en `constraints.py` werken nog op de oude HTML-features. Advies zelf gebruikt ze niet meer (`shadow_policy` filtert niet meer met `apply_scrape_constraints`).

6. **Meetperiode-hek.** Geen automatische check "replay niet-constant + acht tests groen => meting mag starten."

7. **`domain_prior_rate` per actie in de vector.** Spec: Beta per domein per actie. De vector heeft een scalar. Met `action_id=` wordt die scalar de prior van die actie; zonder `action_id` de domein-prior. `action_priors()` bestaat wel, maar zit niet als aparte kolommen in Track A.

Als een van deze punten nodig is voor de meetperiode, eerst een korte spec, dan pas code. Niet opnieuw `features.py` van scratch.
