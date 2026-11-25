# The filings evaluation (passage level)

Document-level recall saturates on a corpus of about twenty 10-Ks (every search finds the right
filing), and chunk ids differ from one chunker to the next. This evaluation therefore grades
**passages**: a question carries short verbatim `expected_spans`, and a retrieved passage is a hit
when it contains one of them.

The question set itself (`filings_v1.jsonl`, about 40 questions on the 10-Ks of five issuers,
FY2022-FY2025, at least four of them abstentions) is not in this folder yet: it needs the filings as
Markdown, which cannot be fetched from here. It is written and checked as described below, then
validated by the owner before it decides the eigenmind default.

## Question format

JSON Lines, the format of `README.md` (version 2) plus one optional field:

| Field | Meaning |
|---|---|
| `expected_spans` | A list of 1 to 3 short, distinct, verbatim texts of the filing that answer the question. |

```json
{"id": "msft-fy24-cash", "question": "How much cash, cash equivalents and short-term investments did Microsoft hold at the end of fiscal 2024?", "expected_doc_ids": ["msft-10k-fy2024"], "expected_spans": ["Cash, cash equivalents, and short-term investments totaled $75.5 billion"], "kind": "factual", "tags": ["issuer:msft", "fy:2024"], "filters": {"doc_ids": ["msft-10k-fy2024"]}}
{"id": "abs-01", "question": "What dividend per share did Oracle declare for fiscal 2031?", "expected_behavior": "abstain", "expected_doc_ids": []}
```

- `expected_doc_ids` names the filing(s) that hold the spans (the `<doc_id>` of the Markdown file).
  `filters` may restrict the search to them; an unfiltered question is harder and says more.
- A question that abstains (`expected_behavior: "abstain"`) carries no span. It is searched, never
  scored, and reported apart with the score of its first passage.
- Files of the older versions still load and score exactly as before; `expected_spans` is ignored by
  `rag-engine eval` (documents) and read by `rag_engine.eval.evaluate_passages` (passages).

## How a question is checked

A span is accepted only if it was **found, verbatim, in the Markdown of the filing** the question
names, not recalled: the checking script (a case- and whitespace-insensitive search, the same as
`rag_engine.eval.span_hit`) must find each span exactly once or twice in that filing, and the
passage around it must answer the question without the question's own words being the span. A span
that appears in several filings (boilerplate) is refused: it would be a hit for the wrong filing.
Spans are copied from the Markdown, not from the SEC web page: tables and line breaks differ. The
whole set is then read once by the owner.

## Scoring

`span_hit(text, spans)`: both sides are case-folded and every run of white space becomes one space;
the passage is a hit if it contains any span. For each question the *rank* is the position of the
first hit among the `k` passages returned (`k = 10`). Then:

- hit@5, hit@10: share of questions with a hit within the first 5 / 10 passages;
- MRR@10: mean of 1/rank (0 past 10 or on a miss);
- each with a 95 % percentile interval by resampling the questions (1000 draws, fixed seed), given
  from 20 questions up (both are parameters of `evaluate_passages`);
- abstentions: searched and reported apart, never in a metric.

## Measurement protocol (`scripts/filings_eval.py`)

```
python scripts/filings_eval.py FILINGS_DIR eval/filings_v1.jsonl --backend onnx \
    --sweep-threshold 0.5,0.65,0.8 --sweep-keep-top 0,1,2 --sweep-solver milp,lp_greedy --json out.json
```

`FILINGS_DIR` holds `<doc_id>.md` and `<doc_id>.json` (`{"title", "form", "date"}`). The script
builds one local index per chunking profile in a temporary directory and scores the matrix

| chunking | retrieval |
|---|---|
| v1, v2 | dense; hybrid (dense + BM25, RRF); hybrid + eigenmind (`select_diverse`) |

and prints hit@5, hit@10 and MRR@10 with intervals, the **seconds per 10-K** (chunk, embed and write one
filing, averaged) and the milliseconds per search. `--backend fake` runs the same code with
meaningless vectors: a smoke test only.

**Decision rules**, printed under the table:

1. **v2 is kept if v2 >= v1 on hit@10** (hybrid retrieval).
2. **eigenmind is on by default only if hybrid+eigenmind > hybrid on hit@10 with a non-overlapping
   interval** (the lower bound of eigenmind above the upper bound of hybrid), on the same chunking.
   Too few questions for an interval, an overlap, or no gain: it stays off.

**Parameters to sweep** (nothing is hard-coded; each is an option of the script):

| Parameter | Option | Why |
|---|---|---|
| `select_diverse` `threshold` (default 0.65) | `--sweep-threshold` | Real embeddings are anisotropic: at 0.65 most candidates may form one clique and the selection collapses to the fused order. |
| mean-centring / removal of top principal directions before the graph | not exposed by `select_diverse` yet | The other cure for anisotropy; needs `select_diverse` to pass `center` / `remove_top` to `eigenmind.normalize_embeddings` before it can be swept. |
| `time_limit` (default 2 s), `solver` (`milp`, `lp_greedy`) | `--time-limit`, `--sweep-solver` | Accuracy against latency of the search. |
| `keep_top` (default 0) | `--sweep-keep-top` | Supported Diversity never selects an isolate: the best fused hit can be lost; `keep_top=1` keeps it. |
| `candidates`, `rrf_k` | `--candidates`, `--rrf-k` | Depth of each side of the fusion and the RRF constant. |

The table is recorded in the task ledger with the questions file hash (`questions_sha256` in the
JSON output).

**A swept winner needs confirmation.** The sweep picks the best of many configurations on the very
questions that score them, so its gain is optimistic. A threshold / `keep_top` / solver chosen this way
is confirmed on questions it was not chosen on (a held-out part of the set, or new questions) before
eigenmind becomes the default.

A passage counts only if it comes from one of the question's `expected_doc_ids` (a sentence repeated
by another year's 10-K is no hit), and the span comparison is done in NFKD with curly quotes and
dashes folded to ASCII and zero-width characters dropped, as the index stores text in NFKD.
