# The filings evaluation (passage level)

Document-level recall saturates on a corpus of about twenty 10-Ks (every search finds the right
filing), and chunk ids differ from one chunker to the next. This evaluation therefore grades
**passages**: a question carries short verbatim `expected_spans`, and a retrieved passage is a hit
when it contains one of them.

The question set is `filings_v1.jsonl`: 48 questions (43 answering, 5 abstentions) on the 10-Ks
of Microsoft, Alphabet, Amazon, Meta and Oracle for fiscal years 2022 to 2025 (20 filings), on capital
expenditures, AI and data-centre investment, financing, debt, leases and liquidity. Every span was
copied from the Markdown and passes `scripts/check_filings_eval.py` (below).

> **Status: pending the owner's validation.** The set has not been read by the owner yet. Until it
> is, the numbers it gives are provisional and it does not decide the eigenmind default.

The filings are not in the repository (full 10-K texts). They are fetched with company-corpus's live
code (`company_corpus.live.filings.fetch_submissions` / `list_filings`, form `10-K`, report dates
2022-01-01 to 2025-12-31, then `company_corpus.live.documents.filing_document`) and written as
`<doc_id>.md` (`FilingDocument.markdown`) with a `<doc_id>.json` sidecar: `company` (the SEC name
as EDGAR gives it, e.g. `AMAZON COM INC`), `form`, `date` (filing date), `report_date`, `doc_id`,
`url`, `cik` and `sections` (`[{"name", "start", "end"}]`, the `FilingDocument.sections`). The
script titles each filing as the no-vault MCP titles its chunks (`filing_title`, ruling R40):
`<company> <form> for the period ended <report_date>, filed <date>`, e.g. `AMAZON COM INC 10-K for
the period ended 2022-12-31, filed 2023-02-03`. Under v2 the title is the chunk context (embedded
and indexed for BM25), so another title would measure another configuration than the one that
ships; a sidecar `title` that differs from the built one is refused. The doc ids are `<ticker>-10k-fy<YYYY>`, the fiscal year being the year
of the report date (Microsoft's ends on 30 June, Oracle's on 31 May, the others' on 31 December).

## Results (provisional, pending the owner's validation of the set)

ONNX `intfloat/multilingual-e5-base`, 4 CPU, `k = 10`, 43 answering questions, 1000 bootstrap draws;
filings titled as the MCP titles them (`filing_title`, above). v1 does not put the title in its
chunks (no context header), so its rows do not depend on it.

| chunking | retrieval | hit@5 | hit@10 | MRR@10 | s/10-K |
|---|---|---|---|---|---|
| v1 | dense | 0.209 [0.09,0.33] | 0.256 [0.14,0.40] | 0.125 [0.05,0.21] | 46.7 |
| v1 | hybrid | 0.326 [0.19,0.47] | 0.419 [0.28,0.56] | 0.241 [0.13,0.34] | 46.7 |
| v1/section | dense | 0.186 [0.07,0.30] | 0.209 [0.09,0.33] | 0.120 [0.05,0.21] | 34.4 |
| v1/section | hybrid | 0.279 [0.16,0.42] | 0.349 [0.21,0.49] | 0.238 [0.13,0.35] | 34.4 |
| v2 | dense | 0.558 [0.40,0.70] | 0.698 [0.56,0.81] | 0.397 [0.27,0.52] | 126.2 |
| v2 | hybrid | 0.651 [0.51,0.79] | 0.767 [0.63,0.88] | 0.474 [0.35,0.59] | 126.2 |
| v2/section | dense | 0.558 [0.40,0.70] | 0.651 [0.49,0.79] | 0.387 [0.26,0.51] | 120.3 |
| **v2/section** (what the MCP ships) | **hybrid** | **0.698 [0.56,0.81]** | **0.767 [0.63,0.88]** | **0.452 [0.33,0.57]** | 120.3 |

- **v2 >= v1** on hit@10 (hybrid): 0.419 -> 0.767 whole-document, 0.349 -> 0.767 per section
  (per question: v2/section wins 18, v1/section wins 0).
- **eigenmind stays off**: at its default threshold (0.65) hybrid+eigenmind gives the same metrics as
  hybrid on every row (e5 cosines between candidates sit above the threshold, so the graph is one
  clique); higher thresholds, swept on an earlier run with other titles, start selecting and lose.
- Most of the 10 misses of v2/section hybrid are year confusions: adjacent 10-Ks carry near-identical
  sentences with other figures. The five abstentions' top fused scores (0.029 to 0.033) are not
  calibrated: no abstention threshold follows from them.

### Caveats: what these numbers do not show

- **The v2 context is the title only.** The company-corpus Markdown of a filing has no `#` headings,
  so the context of a v2 chunk is the filing title alone: the section path of the context (title ›
  h1 › h2 …) is not exercised by this set. The no-vault MCP passes the title only as well, so the
  rows match what ships, not what v2 does on a document with headings.
- **The intervals are optimistic.** They resample the 43 questions as if independent, but the
  questions cluster on 19 to 20 filings of 5 issuers: questions on one filing or one issuer are
  correlated, and an interval that resampled filings or issuers would be wider. Only large
  differences are decidable.
- **Spans come from prose only.** No span is a table cell, so how v1 and v2 split tables (v2 repeats
  the header row in every piece) is not measured.
- **The v2 gain is not split between its causes.** v2 changes two things at once, the token cap
  (480 tokens, so no chunk is truncated by the 512-token window of e5-base) and the title in the
  context (embedded and indexed for BM25). No run isolates one from the other.
- **Year confusion penalises v1 by construction.** The questions carry no `filters`, so the issuer
  and the fiscal year must be found from the text. A v1 chunk carries neither (no context), while a
  v2 chunk carries the title, which names the company and the period. Part of the v1 -> v2 gap is
  therefore this, not the quality of the cut.

## Question format

JSON Lines, the format of `README.md` (version 2) plus one optional field:

| Field | Meaning |
|---|---|
| `expected_spans` | A list of 1 to 3 short, distinct, verbatim texts of the filing that answer the question. |

```json
{"id": "msft-fy24-cash", "question": "How much cash, cash equivalents and short-term investments did Microsoft hold at the end of fiscal 2024?", "expected_behavior": "answer", "expected_doc_ids": ["msft-10k-fy2024"], "expected_spans": ["Cash, cash equivalents, and short-term investments totaled $75.5 billion"], "kind": "liquidity", "tags": ["issuer:msft", "fy:2024", "section:item7"], "language": "en", "rationale": "Item 7, Liquidity and Capital Resources, Cash, Cash Equivalents, and Investments."}
{"id": "abs-01", "question": "What dividend per share did Oracle declare for fiscal 2031?", "expected_behavior": "abstain", "expected_doc_ids": [], "kind": "abstention", "tags": ["abstain"], "language": "en", "rationale": "Future fiscal year: no 10-K covers it."}
```

- `expected_doc_ids` names the filing(s) that hold the spans (the `<doc_id>` of the Markdown file).
  No question of `filings_v1.jsonl` has `filters`: the search must find the issuer and the year
  itself (the README format allows `filters`, which would make a question easier and say less).
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

```
python scripts/check_filings_eval.py eval/filings_v1.jsonl FILINGS_DIR [--min-chars 20] [--max-per-filing 2]
```

prints one line per failed rule and exits 1 if there is any: an expected filing missing from the
directory; a span shorter than `--min-chars` once normalised or without a letter (no dash-only or
tiny span); a span found in none of the question's filings, or more than `--max-per-filing` times
in one; a span also found in a filing the question does not expect. Spans are taken from prose
sentences, not from tables (whose cells the chunker may split), and name a figure or a fact that
only that year's filing states (a 10-K repeats the previous year's figures, so a span such as
"$75.5 billion" alone would also be in the next filing).

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

`FILINGS_DIR` holds `<doc_id>.md` and `<doc_id>.json` (`{"company", "form", "date", "report_date"}`,
or a `title` instead of `company`, plus `sections` for the per-section cut). The script builds one local index per chunking profile and cut
in a temporary directory and scores the matrix

| chunking | retrieval |
|---|---|
| v1, v2 | dense; hybrid (dense + BM25, RRF); hybrid + eigenmind (`select_diverse`) |

`--cuts document,section` adds the cut: `document` chunks the whole Markdown at once; `section`
cuts it at the Item sections of the sidecar first and chunks each piece alone (the text outside the
sections being pieces too), as the no-vault MCP indexes (`document_chunks` of
`myopenfund_mcp.filings.indexer`, ruling R35). Its rows are labelled `v1/section`, `v2/section`;
**`v2/section` is the configuration the MCP ships** (its default `MOF_CHUNK_MAX_TOKENS=480`,
`MOF_CHUNK_OVERLAP_TOKENS=32` give exactly the v2 profile). The script
prints hit@5, hit@10 and MRR@10 with intervals, the **seconds per 10-K** (chunk, embed and write one
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
