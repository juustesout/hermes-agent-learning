# WP crash-test harness (local)

Mechanical tests against a mock of the WordPress REST API. Loopback only.
Drafts only. `status=publish` returns 403.

```bash
pytest -q tests/test_wp_mock.py
AGENT_LEARNING_EVIDENCE_DIR=/tmp/hn-sim python3 hn_sim.py run --runs 200
```

Covered:

- Client refuses every host outside 127.0.0.1/localhost.
- Mock accepts only `status=draft`; publish is 403.
- Auth failure (401) is not retried.
- 300 draft creates in a row: unique ids.
- Injected 5xx: retries work, no duplicate posts.
- 200 accelerated ranker jobs: every post has a decision record.

Sim rows are tagged `source=sim`. Training, live `report`, and agreement
ignore them. Audit still reports L/B overlap, `n_contrast`, and arm balance.
Primary claim remains ITT on all posted slots; `n_contrast` is explanatory.
Sim contrast-subset effects are not quality evidence.

Not covered: whether the content is effective. That needs real labels on real posts.
