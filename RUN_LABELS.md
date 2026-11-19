# Run labels

One rater (Juust) labels the HN candidate pool. The UI shows one blind card
at a time and writes through `record_label()`. It does not fetch HN, rank,
or post.

If the UI says the queue is empty, that is expected until a run has called
`enqueue_pool`. Do not invent items into the live store.

## Walkthrough

From the repo root, same shell, same store:

```bash
python3 blind_label.py report
python3 blind_label.py next
python3 label_ui.py
```

Open `http://127.0.0.1:8765`. Title opens in a new tab. Keys: 1/2 relevant,
3/4 duplicate, 5/6 correction, Enter save. An unlabeled item stays in the
queue. Relabel appends; the latest row counts.

If `next` prints `No unlabeled items in the blind sample`, the UI will also
be empty. Fill the queue first (below), then start the UI again.

## How the queue fills

Live path: the 6h job runs `hn_ranker.py select` on the RAG-filtered pool
(~30 items). `select_run(..., write=True)` posts two slots and then calls
`blind_label.enqueue_pool` on the whole pool.

Manual path, same store as the UI:

```bash
python3 hn_ranker.py select --run-id 2026-11-19T12 --jsonl pool.jsonl
```

Or enqueue without selecting:

```bash
python3 blind_label.py enqueue-pool --run-id 2026-11-19T12 --jsonl pool.jsonl
```

There is no `python3 -m <module> enqueue-pool`. The CLI is `blind_label.py`.
`enqueue-pool` needs `--run-id` and either `--json` or `--jsonl`. Each item
needs `id`, `title`, and `url`. Missing those, or an id already in
`candidates.jsonl` / `items.jsonl`, is skipped.

Posted items can also enter via `blind_label.py enqueue` (20 percent sample,
only if not already queued from the pool). That is not how you fill a
30-item labelling session.

## Store

Default: `store/evidence/hn-aidevforum/` (`queue.jsonl`, `candidates.jsonl`,
`labels.jsonl`, `items.jsonl`). Override with `AGENT_LEARNING_EVIDENCE_DIR`.
`label_ui.py` and `blind_label.py` must see the same value.

Do not write test or dummy rows into the live store. For a dry run:

```bash
export AGENT_LEARNING_EVIDENCE_DIR=/tmp/hn-labels
python3 blind_label.py enqueue-pool --run-id dry --jsonl pool.jsonl
python3 label_ui.py --port 18765
```

## For Hermes

Empty queue means `unlabeled_queue()` is empty: no rows in `queue.jsonl`,
or every queued id already has a row in `labels.jsonl`. `label_ui.py` only
reads `next_payload()` -> `unlabeled_queue()` -> `blind_view()`. It cannot
fill the queue.

Wrong:

```bash
python3 -m <module> enqueue-pool
python3 label_ui.py --port 18765
```

Right, after you have a real pool file (id/title/url per row):

```bash
python3 blind_label.py enqueue-pool --run-id <run-id> --jsonl pool.jsonl
python3 blind_label.py report
python3 label_ui.py --port 18765
```

`select` also queues the pool. Use that when the job already has the
candidate list. Do not call `enqueue-pool` with made-up HN items on the
live evidence dir.

`label_ui.py` binds `127.0.0.1` only. Default port is 8765; `--port` is
fine. Rater is fixed `juust`. Cards expose only `VISIBLE_FIELDS`: id,
title, url, summary, posted_at, hn_id, workflow.
