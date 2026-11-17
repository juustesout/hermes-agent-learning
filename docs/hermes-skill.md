---
name: hermes-agent-learning
description: Hermes zelflerend — TaskPolicies, AdaptiveBanditLearner, ContextualSoftmaxPolicy + WorkflowRunner
---

# Hermes + agent-learning: self-learning architectuur

## ⚠️ CRITICAL: WorkflowRunner is een LEER-FRAMEWORK, geen productie-pipeline

De `podcast_workflow.py` in deze repo is een **leer-demo** met gesimuleerde stappen
(time.sleep + hardcoded resultaten). Het logged episodes, traint policies, en berekent
credit assignment — maar er wordt **geen echte tool aangeroepen**.

**De echte podcast productie-pipeline staat in skill `podcast-ops`:**
`bash /opt/data/profiles/operator/podcast/scripts/run_podcast.sh`

Wanneer de gebruiker zegt "maak de podcast" of "draai de podcast" → gebruik `podcast-ops`.
Wanneer de gebruiker zegt "leer van podcast runs" of "train policy" → gebruik `hermes-agent-learning`.

**Gebruikerscorrectie (16 Nov 2026):** werd boos omdat ik een gesimuleerde podcast bouwde
terwijl de echte pipeline al bestond. Altijd eerst checken of er al een productie-pipeline
is (`podcast-ops` skill) voordat je een leer-abstraction bouwt. Nooit simuleren wat de
gebruiker als echt verwacht.

## Status
- ✅ Level 1: **Edge learning** — scrape-strategy TaskPolicy leert per keuze
- ✅ Level 1.5: **Contextuele bandit** — ContextualSoftmaxPolicy leert per site-type
- ✅ Level 2: **Path learning** — WorkflowRunner met credit assignment (γ=0.8)
- ✅ Level 3: **Graph evolution** — DagRegistry, MultiDagOrchestrator, GraphOptimizer, Dashboard
- ✅ **Cost-aware reward** — 4 presets (balanced/speed/cost/quality), latency+cost+quality meegenomen in reward
- ✅ **SkillDiscovery** — n-gram mining over store action_sequences, clustert runs, confidence scoring
- 🔲 **LinUCB** — derde policy-type voor high-dimensional context

## Bestandsoverzicht

| Component | Bestanden |
|-----------|-----------|
| Level 1 — Learners | `adaptive_learner.py`, `contextual_learner.py`, `scrape_features.py` |
| Level 2 — Workflow | `workflow_dag.py`, `workflow_runner.py`, `workflow_visualize.py`, `podcast_workflow.py` |
| Level 3 — Evolution | `graph_evolution.py` |
| Thompson — Cold start | `thompson_policy.py`, `train_thompson.py` |
| Cost-aware reward | `cost_aware_rewards.py` (4 presets, latency/cost/quality reward) |
| Skill Discovery | `skill_discovery.py` (n-gram mining, clustering, confidence) |
| Scripts — Training | `train_adaptive.py`, `init_contextual_policy.py`, `test_adaptive_vs_standard.py` |
| Scripts — Cron | `train_scrape_policy.sh`, `cron_workflow.sh` |
| Bootstrap | `env_setup.py` — auto-configureert LocalFileStore backend |
| Holdout eval | `holdout_eval.py` — learned vs baseline comparison |
| Constraints | `constraints.py` — harde actie-filters obv context |
| Workflow synth | `workflow_synthesizer.py` — Prompt-to-DAG interface |
| Verificatie | `scripts/verify_roundtrip.py` — round-trip test (ops→consolidate→query→train→decide) |

## Prompt-to-DAG — Hoe Hermes workflows maakt (sinds 17 Nov 2026)

**Kernprincipe:** Hermes (LLM) is de intelligentie, `workflow_synthesizer.py`
is het gereedschap. Geen keyword-matching voor template-selectie — Hermes
analyseert de taak met eigen begrip.

### Flow
```
Gebruiker: "Onderzoek AI trends en publiceer een blog"

Hermes (LLM):
  1. Analyseert taak met eigen reasoning
     → "research + create + validate + publish nodig"
  2. Roept synthesizer aan met expliciete parameters:
     syn.run(task="...", template="research_create_validate_publish",
             context={"topic": "AI Agents"})
  3. Synthesizer bouwt DAG, init Thompson policies, registreert in DagRegistry
  4. WorkflowRunner voert uit met constraints + lokale rewards
  5. Episodes + rewards in store, klaar voor learning
```

### Correctie (17 Nov 2026)
De gebruiker wees erop dat de keyword-matching in `suggest_template()` te
rudimentair is voor workflow-creatie. Oplossing: niet de Python-code slimmer
maken, maar Hermes (LLM) de taak laten begrijpen en de synthesizer als tool
gebruiken met expliciete `template=` parameter.

### WorkflowSynthesizer API
```python
from workflow_synthesizer import WorkflowSynthesizer

syn = WorkflowSynthesizer(store)

# Full pipeline: synthesize + execute in one call
run = syn.run(
    task="Onderzoek AI trends en publiceer een blog",
    template="research_create_validate_publish",  # Hermes kiest, geen keyword-match
    context={"topic": "AI Agents", "_is_approved": False},
)
print(f"Reward: {run.global_reward}")
for nr in run.node_results:
    print(f"  {nr.node_name}: {nr.action_name} local={nr.local_reward}")

# Or step by step
dag = syn.synthesize(task="...", template="research_publish")
decision_fn = syn.make_decision(dag)       # constraints ingebouwd
execute_fn = syn.make_execute(dag)         # generieke execute stub
reward_fn = syn.make_reward()              # CostAwareReward
local_reward_fn = syn.make_local_reward(dag)  # per node-type
```

### Beschikbare templates
| Template | Nodes | Gebruik |
|----------|-------|---------|
| `research_publish` | research → create → publish | Blog, artikel, content |
| `research_create_validate_publish` | research → create → validate → publish | Content met kwaliteitscheck |
| `scrape_extract_store` | research → extract → publish | Data-extractie, contactgegevens |
| `monitor_validate_notify` | monitor → validate → notify | Monitoring, alerts |

### Beschikbare node types
| Type | Acties | Doel |
|------|--------|------|
| `research` | web_search, web_scrape, llm_only, manual | Info verzamelen |
| `extract` | llm_extract, regex, manual | Data extraheren |
| `create` | llm_full, llm_outline, template, manual | Content genereren |
| `publish` | wordpress, youtube, spotify, draft | Publiceren |
| `monitor` | poll_api, rss_feed, web_check, manual | Bewaken |
| `validate` | llm_check, schema_check, manual_review, auto_pass | Kwaliteitscheck |
| `notify` | telegram, email, slack, log_only | Notificeren |
| `transform` | llm_transform, template, script, manual | Data transformeren |

## Fixes aangebracht (17 Nov 2026, n.a.v. Claude code review)

### Bug fixes
1. **ThompsonLearner dubbeltelling** — verwijderd: `policy.observe()` werd zowel in de committed code als in `ThompsonLearner.update()` aangeroepen, wat elke observatie dubbel telde. Gefixt door de `policy.observe()` call te verwijderen uit `update()`.
2. **Reward formaat** — oude bare-dict reward bestanden (zonder `id`, `agent_id`, `source`) veroorzaakten `AttributeError` in `query_rewards()`. 5 bestanden geconverteerd naar `[dict]` formaat.
3. **Level 2 learning** — `make_podcast_decision()` initialiseert nu automatisch een ThompsonPolicy uit de node's `action_ids` als er nog geen policy in de store bestaat. Gebruikt Beta(1,1) uniforme prior voor echte cold-start exploratie.
4. **Env vars** — `env_setup.py` zet `AGENT_LEARNING_STORE_BACKEND=local` en `AGENT_LEARNING_LOCAL_STORE_DIR` automatisch, zodat `get_default_store()` een `LocalFileStore` retourneert i.p.v. `InMemoryStore`. Geïmporteerd in `consolidate_ops.py`, `train_thompson.py`.
5. **Drie rewardformules → één source of truth** — `consolidate_ops._write_reward()` gebruikte een eigen inline formule. Vervangen door `CostAwareReward.balanced()` met dezelfde weights. `podcast_reward` gedocumenteerd als domein-specifiek (critical-path bonus i.p.v. cost/latency). `CostAwareReward` is nu de enige plek waar reward-weights gedefinieerd staan.
6. **Cron scripts** — `|| echo "... continuing"` verving door `step_ok/step_fail` met cumulatieve `EXIT_CODE`. Script eindigt met `exit $EXIT_CODE` (0 = alles OK, >0 = aantal gefaalde stappen). `%LOGFILE%` uit git tracking verwijderd + `.gitignore` entry.

### Cost-aware reward — real latency injection
Sinds 17 Nov 2026: `WorkflowRunner.execute()` slaat na elke node-executie de
werkelijke latency (uit timestamps) op in `run.context["_latency:node_id"]`
en optionele cost uit `exec_meta["cost"]`. `CostAwareReward.compute()` leest
deze waarden, zodat de reward functie echte latency gebruikt i.p.v. tabel-schattingen.

### Prior fix
- Beta prior gewijzigd van (2,2) naar (1,1) voor werkelijk uniforme prior (Beta(2,2) is centered at 0.5, niet uniform).

## Automatische operation logging (opslog)

Tool-operaties worden automatisch gelogd via de **Hermes plugin**
(`hermes_plugin/__init__.py`). De `post_tool_call` hook roept `opslog.log_operation()`
aan voor elke Hermes tool-call — geen handmatige logging nodig.

- **Plugin:** repo-root `plugin.yaml` + `__init__.py` (implementation in `hermes_plugin/`)
- **Install:** `hermes plugins install juustesout/hermes-agent-learning --enable`
- **Live log:** `store/operations.jsonl` (append-only JSONL, path via `AGENT_LEARNING_OPSLOG_DIR` or `AGENT_LEARNING_LOCAL_STORE_DIR`)
- **Consolidatie:** `consolidate_ops.py --run` zet ruwe logs om in episodes + rewards
- **Cron:** `opslog-consolidate` elke nacht 03:00 UTC (no_agent script)
- **Opt-out:** `AGENT_LEARNING_OPSLOG_ENABLED=0`
- **Health:** `hermes agent-learning health` / `python3 opslog.py health`

> Local checkout: `hermes plugins install . --enable` then `hermes plugins doctor . --ci`.
> No symlinks. Query strings and common secrets are redacted; session ids are off by default.

### Task-ID conventies
| Prefix | Voorbeelden |
|--------|-------------|
| choose-scrape-strategy | web_scrape, web_search |
| db-query | database_query |
| wp-publish | wp_rest_api |
| web-research | web_search |
| podcast-* | research, script, audio, publish |

## Blind labels (HN evidence, sinds 17 Nov 2026)

Bevroren vóór meting: `evidence/success_definition.json`, `evidence/hn_baseline.json`.
Baseline = eerste twee overgebleven relevante items in bronvolgorde na harde dedupe.
HTTP 2xx is technisch succes; kwaliteit = blinde labels.

```bash
python3 blind_label.py assign --id ...
python3 blind_label.py enqueue --json '{"id":"...","title":"...","url":"...","summary":"...","http_status":201}'
python3 blind_label.py record-run --attempted 2 --posted 2 --failed 0
python3 blind_label.py next
python3 blind_label.py label --id ... --relevant yes --duplicate no --needs-correction no
python3 blind_label.py report
python3 blind_label.py audit
```

Rater ziet geen arm/policy/rank/auto-score. Arm staat in `assignments.jsonl`.
Munt-worp per item, p=0.5. Sample = 20 procent, stabiele hash.

## Sessie-start

```bash
export AGENT_LEARNING_STORE_BACKEND=local
export AGENT_LEARNING_LOCAL_STORE_DIR="${AGENT_LEARNING_LOCAL_STORE_DIR:-$PWD/store}"
```

---

## Level 1 — Edge Learning (scrape strategies)

### Context features (8 dims)
Alfabetisch: cloudflare, content_len_norm, has_structured_data, is_elementor, is_wordpress, js_heavy, requires_antibot, site_known

### Episode schema (scrape)
```json
{
  "context_features": {"cloudflare": 1.0, "js_heavy": 0.5, ...},
  "metadata": {
    "correct_action_id": "use_webclaw_scrape",
    "task_completed": true,
    "latency_ms": 3100, "cost_credits": 1, "content_quality": "high",
    "content_length": 24500, "scrape_success": true
  }
}
```

### Policy hierarchy
```
SoftmaxPolicy (marginal, geen context)
  ├── ContextualSoftmaxPolicy (W·phi, leert per context)
  └── ThompsonPolicy (Beta sampling, beste cold start)
```

### Learner hierarchy
```
Learner
  └── ReinforceLearner (vanilla REINFORCE + baseline)
       └── AdaptiveBanditLearner (logit decay + failure det. + dynamic LR)
            └── ContextualAdaptiveBanditLearner (W·phi gradients)
```

### Training
- `train_adaptive.py --agent-id hermes --task-id choose-scrape-strategy` — AdaptiveBandit (REINFORCE)
- `train_thompson.py --agent-id hermes --task-prefix podcast-` — Thompson sampling (cold start)
- Detecteert automatisch of policy contextual is en routeert naar juiste learner.
- Cron scrape: elke 6 uur via `train_scrape_policy.sh`.
- **Podcast cron: GEPAUZEERD (16 nov 2026).** Windows Task Scheduler neemt over — zie 'Deployment' hieronder. De Hermes cron `podcast-workflow-daily` stond op 13:00 maar draaide gesimuleerde stappen i.p.v. de echte productie-pipeline (`podcast-ops`). Gebruik altijd `podcast-ops` voor productie, `hermes-agent-learning` voor analyse.

---

## Thompson Sampling (cold start)

`thompson_policy.py` — alternatief voor Softmax, gebaseerd op Beta-distributies.

### Waarom Thompson?
- Softmax begint uniform en convergeert langzaam met REINFORCE gradients
- Thompson samplet uit Beta(α,β) per action — exploreert natuurlijk (hoge variance bij weinig data)
- Na 100 positieve observaties convergeert naar >95% voor de beste action
- In 50 trials bij cold start worden alle 3 actions verkend (geen "dode" actions)

### Werking
- Prior: Beta(1,1) — uniform (flat) prior
- Reward ∈ [-1,1] → Beta observation: success=(r+1)/2, failure=1−success
- `choose()`: sample Beta per action, pick max sample
- `probabilities()`: alpha/(alpha+beta) per action
- `uncertainty()`: stddev van Beta — daalt van ~0.29 naar 0.10 na 10 observaties

### Auto-creation in podcast workflow
Sinds 17 Nov 2026: `make_podcast_decision()` maakt automatisch een ThompsonPolicy
aan uit de node's action_ids als er nog geen policy in de store bestaat.
Eerste run in een schone omgeving exploreert dus direct via Thompson sampling
i.p.v. altijd de eerste action te kiezen.

### Gebruik
```python
from thompson_policy import ThompsonPolicy, train_thompson
policy = ThompsonPolicy.from_actions(actions, agent_id="hermes", task_id="podcast-research")
decision = policy.choose()
policy.observe("webclaw", 0.8)  # Beta update
```

### CLI
```bash
python3 train_thompson.py --agent-id hermes --task-prefix podcast- --prior-alpha 3 --prior-beta 2
```

---

## Level 2 — Workflow Runner (path learning)

### Concept
Elke workflow = DAG van `{context, actionset, TaskPolicy}` nodes.
Per stap: beslissing → execute → episode. Eindbeloning wordt teruggepropageerd.

### Stappen in een run
1. **Topological sort** — nodes in afhankelijkheidsvolgorde
2. **Policy decision** — via TaskPolicy.decide() of decision callback
3. **Execution** — via node.execute() of execute callback
4. **Episode logging** — elke beslissing naar agent-learning store
5. **Global reward** — reward callback na laatste node
6. **Credit assignment** — global_reward × γ^distance_from_end per node

### Credit assignment
```
credit[node] = global_reward × γ^distance_from_end

distance = aantal edges van node naar verste leaf node
γ = 0.8 (default)

Voorbeeld podcast DAG (γ=0.8):
  publish:       dist=0  γ^0 = 1.0000  krijgt 100% van global reward
  video:         dist=1  γ^1 = 0.8000
  audio:         dist=2  γ^2 = 0.6400
  script:        dist=3  γ^3 = 0.5120
  research:      dist=4  γ^4 = 0.4096
  topic_select:  dist=5  γ^5 = 0.3277
```

### Lokale rewards per node (sinds 17 Nov 2026)

Naast de globale credit-assignment (γ^distance) krijgt elke node een **lokaal
reward-signaal** onafhankelijk van de globale uitkomst. Dit lost het probleem
op dat een node die het goed deed gestraft wordt omdat een latere node faalde.

**Type:** `LocalRewardFn = (node, action_id, status, summary, exec_meta, context) → (reward, reason)`

**Default:** `_default_local_reward()`:
- Completed: 0.5 base + 0.2 (content > 20 chars) + 0.1 (clean output, geen error-patroon) + 0.1 (bevat URLs) + 0.1 (structured data) = max 1.0
- Error-inhoud (403, 404, timeout, cloudflare, etc.): -0.3 penalty
- Lege output: -0.2 penalty
- Failed: -0.5

**Kwaliteitschecks in local rewards (sinds 17 Nov 2026):**
De local reward checkt of output daadwerkelijk zinvol is, niet alleen of
de status "completed" is. Dit is een lichte vorm van LLM-as-a-Judge zonder
LLM-kosten:
- Error-patroon detectie met word boundaries (" 403 ", " 500 ", etc.)
  zodat "15000 chars" niet matcht op "500"
- Minimale content-lengte (penalty bij < 20 chars)
- URL detectie voor research-nodes
- Structured data markers ({, [, <html, class=) voor data-extractie-nodes

**Opslag:** Als `RewardSource.METRIC` in de store, gelinkt aan de episode.
Hierdoor kunnen learners zowel globale credit-rewards als lokale METRIC-rewards
gebruiken tijdens training.

**Gebruik:**
```python
from workflow_runner import WorkflowRunner, _default_local_reward

runner = WorkflowRunner(dag, store=store,
    decision_fn=my_decide, execute_fn=my_execute,
    reward_fn=my_reward,
    local_reward_fn=_default_local_reward)  # ← optioneel
run = runner.execute(context={"topic": "..."})
# Summary toont nu: "local=+1.000" per node
```

**Waarom lokaal i.p.v. alleen globaal?**
- Een scrape die junk returned kan als "completed" tellen → globale reward ziet het niet
- Een publish die faalt na een perfecte script → alle nodes krijgen minder credit
- Lokale rewards geven direct feedback per node, zonder positional bias van γ^distance

### Podcast workflow DAG
```
topic_select → research → script → audio → video → publish
```

Elke node heeft eigen `task_id` en `action_ids`, zodat elke stap onafhankelijk leert.
Zie `podcast_workflow.py` voor de volledige configuratie per node.

### Werkende callbacks in podcast_workflow.py
- `podcast_decision(node, context)` — kiest actie (override / correct_path / fallback)
- `podcast_execute(node, action_id, context)` — simuleert uitvoering met latency
- `podcast_reward(context, node_results)` — weegt critical path, publish bonus, fail penalty

### CLI gebruik
```bash
cd /opt/data/profiles/operator/agent-learning

# Toon DAG
python3 podcast_workflow.py --show

# Droge run
python3 podcast_workflow.py --dry-run --topic "AI Agents"

# Met geforceerde failure
python3 podcast_workflow.py --dry-run --topic "Test" --force-fail publish

# Toon credit assignment tabel
python3 podcast_workflow.py --simulate-rewards

# Met store voor echte episode logging
python3 podcast_workflow.py --run --topic "AI Agents"
```

### WorkflowRunner API
```python
from workflow_dag import WorkflowDAG, WorkflowNode
from workflow_runner import WorkflowRunner

dag = WorkflowDAG(id="podcast", nodes=[...], edges=[...])
runner = WorkflowRunner(dag, store=store, agent_id="hermes",
                         decision_fn=my_decide, execute_fn=my_execute,
                         reward_fn=my_reward)
run = runner.execute(context={"topic": "..."})
print(runner.summary(run))
```

---

## Level 3 — Graph Evolution (multi-DAG + optimalisatie)

### Bestanden
| Bestand | Beschrijving |
|---|---|
| `graph_evolution.py` | DagRegistry, RunAnalyzer, GraphOptimizer, MultiDagOrchestrator, Dashboard |

### DagRegistry + Meta-policy
- `register(dag, priority=1, tags=[...])` — voeg DAG toe
- `select_dag(context, preference=...)` — kies DAG obv gemiddelde reward, prioriteit, run count
- Werkt als injectie voor MultiDagOrchestrator

### MultiDagOrchestrator
| Mode | Beschrijving |
|---|---|
| `run_sequential([A, B, C])` | A → B → C met chained context |
| `run_conditional(fn, {"ok": A, "fail": B})` | if X then A else B |
| `run_looped("podcast", max=5)` | Herhaal tot reward > threshold |
| `run_meta_dag(meta_dag, {node: dag_id})` | DAG van DAGs |

### GraphOptimizer
- `optimize_gamma(dag_id)` — bepaalt gamma obv reward volatility (hoog → lagere γ)
- `find_bottlenecks(dag_id)` — nodes met >30% failure rate
- `suggest_reorder(dag_id)` — detecteert of volgorde optimaler kan
- `full_report(dag_id)` — complete analyse

### Dashboard
- `terminal_report(dag_id)` — per-run overzicht in terminal
- `html_dashboard(dag_ids)` — self-contained HTML (dark theme, charts)

---

## Cost-Aware Rewards

`cost_aware_rewards.py` — reward functie die completion, latency, cost en quality combineert.

### Formule
```
reward = w_c × completion - w_l × latency_penalty - w_k × cost_penalty + w_q × quality
Alles genormaliseerd naar [0,1], eindresultaat clamped naar [-1,1]
```

### Presets
| Preset | w_c | w_l | w_k | w_q | Gebruik |
|--------|-----|-----|-----|-----|---------|
| `balanced_reward()` | 1.0 | 0.3 | 0.3 | 0.2 | Standaard (cron) |
| `speed_focused()` | 1.0 | 0.8 | 0.2 | 0.1 | Snelheid prioriteit |
| `cost_focused()` | 1.0 | 0.2 | 0.8 | 0.1 | Budget modus |
| `quality_focused()` | 1.0 | 0.1 | 0.1 | 0.6 | Kwaliteit eerst |

### Tool cost estimates
Bekende toolkosten en latency zitten in `TOOL_COST_ESTIMATES` / `TOOL_LATENCY_ESTIMATES`. Fallback: 1.0 credit / 5s. Annotatie via `CostAwareReward.annotate_execution(ctx, node_id, action_id, latency_s, cost)`.

**Let op:** sinds 17 Nov 2026 injecteert `WorkflowRunner.execute()` automatisch
de werkelijke latency (uit timestamps) in `context["_latency:node_id"]` na elke
node-executie. `CostAwareReward.compute()` leest deze waarden, dus bij gebruik
via WorkflowRunner zijn latency-waarden real-time, niet geschat.
Voor standalone gebruik (zonder WorkflowRunner) of voor cost: als `_cost:node_id`
niet gezet is, valt `compute()` terug op `estimate_cost()`.

### Gebruik in WorkflowRunner
```python
from cost_aware_rewards import balanced_reward
reward_fn = balanced_reward()
runner = WorkflowRunner(dag, reward_fn=reward_fn.compute, ...)
```

---

## Skill Discovery

`skill_discovery.py` — mine action-sequence patronen uit de episode store.

### Pipeline
1. **Run reconstruction** — groepeer episodes per workflow_run, sorteer op distance_from_end
2. **N-gram mining** — vind alle action-subsequences van lengte 2+ die in ≥2 runs voorkomen
3. **Clustering** — groepeer runs met identieke action-sequences
4. **Confidence scoring** — frequency_ratio × avg_reward, gecapt op 1.0

### Output
```python
skills = disc.discover(dag_id="podcast", min_support=2)
# [DiscoveredSkill(name, sequence, frequency, confidence, avg_reward), ...]
```

### CLI / Cron
SkillDiscovery draait in `cron_workflow.sh`. Resultaten uit 3 runs: 15 skills gevonden (alle confidence 1.00 — runs waren identiek, dus patronen zijn 100% voorspelbaar). Naarmate de echte productie-pipeline varieert, worden de confidence-scores realistischer.

---

## Holdout-Evaluatie

`holdout_eval.py` — vergelijkt het geleerde scrape-beleid met een vaste
baseline op een gelabelde test set. Dit is de enige manier om te meten of
het systeem daadwerkelijk beter wordt.

### Test set
10 gelabelde scenarios met bekende correcte actie:
- 6 uit `SITE_PROFILES` (wordpress_astra, wordpress_elementor, cloudflare_wp,
  cloudflare_js_heavy, simple_html, api_endpoint)
- 4 extra variaties (wp_zonder_cf, js_app, zware_cf, simpele_api)

### Baselines
| Baseline | Strategie |
|----------|-----------|
| `always_first` | Kies eerste action (default fallback) |
| `always_webclaw` | Kies altijd WebClaw (veilige default) |
| `cheapest` | Kies Jina (gratis) |
| `fastest` | Kies Jina (snelste) |

### Usage
```bash
python3 holdout_eval.py                           # vs always_webclaw
python3 holdout_eval.py --baseline cheapest        # vs goedkoopste
python3 holdout_eval.py --json                     # machine-readable
python3 holdout_eval.py --store-results            # bewaar voor trend
```

### Output
- Accuracy (correct/total)
- Total cost in credits
- Average latency in seconds
- Action distribution
- Per-category breakdown (wordpress, cloudflare, simple, structured, js_app)
- Delta vs baseline

Resultaten worden opgeslagen in `store/eval/holdout_{timestamp}.json`
voor trend-analyse over tijd (zie `--store-results`).

---

## Harde Constraints (sinds 17 Nov 2026)

`constraints.py` — blokkeert acties op basis van context, onafhankelijk van reward.
Een learner die op completion optimaliseert kan leren om site-voorwaarden te schenden.
Constraints voorkomen dat door de actie te blokkeren VOORDAT de policy kiest.

### Scrape constraints

| Situatie | Geblokkeerd | Reden |
|----------|-------------|-------|
| Cloudflare/antibot site | `use_jina_reader` | Jina faalt op CF-sites |
| JS-heavy site (≥0.8) | Alles behalve WebClaw/Browser | Alleen deze kunnen JS aan |
| Simpele HTML/API | `use_browser` | Browser is overkill (5 credits, 10s) |
| Bekende simpele WP-site | `use_browser` | WebClaw of Jina volstaat |

### Podcast constraints

| Node | Conditie | Geblokkeerd | Vervanger |
|------|----------|-------------|-----------|
| `publish` | `_is_approved=False` of `_is_dry_run=True` | `wordpress`, `youtube`, `spotify`, `all` | `draft` |

### Hoe het werkt

In `make_podcast_decision()` (en in de toekomst ook in scrape-beslissingen):

1. Policy kiest een actie via Thompson/Softmax sampling
2. `apply_podcast_constraints()` filtert de toegestane acties obv context
3. Als de gekozen actie niet in de toegestane lijst zit → fallback naar eerste toegestane
4. Decision source krijgt `_constrained` suffix zodat het zichtbaar is in logs

```python
from constraints import apply_scrape_constraints, apply_podcast_constraints

# Scrape: filter actions before passing to policy
allowed = apply_scrape_constraints(context_features, all_actions)

# Podcast: filter after policy choice (ingeplugd in make_podcast_decision)
allowed = apply_podcast_constraints("publish", actions, {"_is_approved": False})
# Result: ["draft"]
```

### Design principe

Constraints zijn HARD — ze geven geen penalty, ze blokkeren. Dit is bewust:
- Een penalty in de reward heeft leer-episodes nodig om effect te hebben
- Een constraint werkt meteen, zonder training
- Constraints zijn deterministisch en transparant
- Ze voorkomen dure fouten (publish naar live site, rate limiting door CF-scrapes)

---

## Gebruik (elke sessie)

De scripts zetten zelf de store backend correct via `import env_setup`. Voor
direct gebruik in Python (bijv. execute_code):

```bash
export PATH="/opt/hermes/.venv/bin:$PATH"
```

Of importeer `env_setup` in je Python code:
```python
import env_setup  # noqa: F401 — zorgt voor LocalFileStore backend
from agent_learning.storage.cosmos import get_default_store
store = get_default_store()  # nu LocalFileStore i.p.v. InMemoryStore
```

## Testen

**Ad-hoc verificatie** (geen formele test suite — schrijf per component):

```bash
cd /opt/data/profiles/operator/agent-learning
export PATH="/opt/hermes/.venv/bin:$PATH"

# Drielaags verificatie-patroon:
# 1. syntax: py_compile elk gewijzigd bestand
# 2. ad-hoc: tijdelijk script onder /tmp/hermes-verify-*/ dat gedrag test
# 3. round-trip: volledige pipeline (ops → consolidate → query → train → run)

python3 -c "import py_compile; py_compile.compile('workflow_runner.py', doraise=True)"
python3 -c "
import sys, os, tempfile
sys.path.insert(0, '.')
import env_setup
t = tempfile.mkdtemp(prefix='hv_')
os.environ['AGENT_LEARNING_STORE_BACKEND'] = 'local'
os.environ['AGENT_LEARNING_LOCAL_STORE_DIR'] = t
from agent_learning.storage.cosmos import get_default_store
s = get_default_store()
# ... test gedrag ...
"
```

## Deployment — Windows Task Scheduler

De cron job `podcast-workflow-daily` (13:00 UTC) draait via Windows Task Scheduler voor betrouwbaarheid.

**Batch file:** `/opt/data/profiles/operator/hermes-podcast-workflow.bat`  
*(NIET in de Python repo — cross-platform infra hoort in profiles/, niet in de codebase)*

**Windows UNC pad:** `\\wsl.localhost\Ubuntu\opt\data\profiles\operator\hermes-podcast-workflow.bat`

**Task Scheduler instellingen:**
- Program: `C:\Windows\System32\cmd.exe`
- Arguments: `/c C:\profiles\operator\hermes-podcast-workflow.bat`
- Start in: `C:\profiles\operator\agent-learning`
- Trigger: daily 13:00
- Log: `%USERPROFILE%\hermes-podcast-workflow.log`

### Waarom Windows Task Scheduler i.p.v. Hermes cron?
- Windows Task Scheduler overleeft WSL herstarts, slaap/hibernate, en reboot
- Hermes cron draait alleen als Hermes proces actief is
- Logging naar Windows-bestand is eenvoudiger te controleren

## Referenties
- `references/episode-schema.md` — Level 1: scrape episode schema + context features
- `references/workflow-episode-schema.md` — Level 2: workflow run episode schema + rewards
- `references/credit-assignment.md` — Level 2/3: credit assignment math + gamma-optimalisatie
- `references/architecture.md` — System architecture overview (all levels)
- `references/graph-evolution-api.md` — Level 3: full API reference (5 classes)
- `references/design-philosophy.md` — Meetbare skills als programmeerbare workflows
- `references/next-steps.md` — Roadmap: cost-aware reward, SkillDiscovery, LinUCB, MCP
- `references/benchmark-results.md` — Benchmark vergelijkingen
- `references/ecosystem-research.md` — ChatGPT research scan (Nov 2026): 15+ projecten
- `references/opslog-schema.md` — Opslog JSONL schema, task→policy mapping, reward formule
- **`references/code-review-findings-20261117.md`** — Externe code review: 8 gevonden problemen (3 bugs, 5 ontwerpfouten), checklist voor nieuwe componenten
- **`scripts/verify_roundtrip.py`** — Ad-hoc round-trip test: ops→consolidate→query→train→decide
- **`references/prompt-to-dag-pattern.md`** — Prompt-to-DAG pattern: Hermes als LLM die workflow synthetiseert uit taakomschrijving
- **`holdout_eval.py`** — Holdout-evaluatie: learned policy vs baseline op gelabelde test set