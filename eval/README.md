# Evaluating the search

Question files for `rag-engine eval`, the script that builds the automatic one, and a small fixture for the tests.

| File | What it is |
|---|---|
| `known_item_v1.jsonl` | 833 automatic questions on the policy core of the central-bank corpus, built from the manifests by `make_known_item.py`. |
| `analytical_v1.jsonl` | Questions on the content of the documents, written and reviewed by hand. A separate file with the same format; it is added once its questions are validated. |
| `make_known_item.py` | Builds a known-item file from manifest files. Standard library only; not part of the installed package. |
| `fixtures/corpus.json`, `fixtures/questions.jsonl` | 12 small documents (30 chunks) and 10 questions with known answers. The test suite loads them into a collection and checks the exact metrics. |

## Question file format

JSON Lines, UTF-8, one question per line:

```json
{"id": "ki-described-filters-en-00dae99322d77d3d", "question": "Federal Reserve FOMC press conference transcript, 14 June 2023", "expected_doc_ids": ["00dae99322d77d3d"], "filters": {"source_codes": ["us"], "doc_types": ["B1"], "date_from": "2023-04-30", "date_to": "2023-07-29"}, "kind": "described+filters", "language": "en", "tags": ["bank:us", "doc_type:B1", "date:day"]}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | A string, unique in the file. |
| `question` | yes | The text given to the search. |
| `expected_doc_ids` | yes | The `doc_id` of every document that answers the question: a non-empty list. |
| `filters` | no | Filters applied to the search: an object with the field names of `rag_engine.search.SearchFilters` (`source_codes`, `doc_types`, `languages`, `doc_ids`, `date_from`, `date_to`, `corpus`) and its rules (lists, never a bare string; dates written `YYYY-MM-DD`). |
| `kind` | no | A label; the report gives the metrics per kind. |
| `tags` | no | A list of labels; the report gives the metrics per tag. |
| `language`, `rationale` | no | Free text, kept and not used. |

The file is checked strictly when it is read: a duplicate `id`, an empty `expected_doc_ids`, an unknown field or filter, a bad date are errors that name the line. No other field is accepted, so that a misspelt `filters` is never ignored.

## Running an evaluation

```bash
export QDRANT_HOST=localhost QDRANT_PORT=6333      # and QDRANT_API_KEY, QDRANT_HTTPS if needed
export RAG_ENGINE_BACKEND=onnx                     # or torch; the model the collection was embedded with
rag-engine eval --questions eval/known_item_v1.jsonl --collection documents \
    --k 10 --chunk-limit 50 --output report.json
```

The command prints a report and, with `--output`, writes every detail as JSON. `--fail-under-hit-at-k 0.9` makes it exit with code 1 when the overall hit@k is below 0.9. Exit code 2 means the evaluation could not run (arguments, settings, question file, server, collection, embedding backend).

How a question is scored:

- The search returns chunks; an answer is a document. For each question, `--chunk-limit` chunks are asked for (default 50, at most `RAG_ENGINE_SEARCH_MAX_LIMIT`), and collapsed into documents: a document takes the place of its best chunk. The metrics look at the first `--k` documents (default 10).
- If the chunks returned hold fewer than `k` documents although the whole chunk budget was used, the report says for how many questions: a larger `--chunk-limit` could have ranked more documents.
- Before anything is searched, the expected documents are looked up in the collection. A question none of whose expected documents is there is listed under **not in the collection** and left out of every metric: the search cannot find a document that was never ingested. When only some are absent the question is scored, and the absent documents are listed.

The metrics, each a mean over the scored questions (overall, per `kind` and per tag):

| Metric | Meaning |
|---|---|
| hit@1 | Share of questions whose first document is an expected one. |
| hit@k | Share of questions with an expected document among the first `k`. |
| MRR | Mean of `1 / rank` of the best-placed expected document, 0 when it is beyond `k`. |
| recall@k | Mean share of the expected documents (those present in the collection) found among the first `k`. |

The **misses** (no expected document among the first `k`) are listed with the three best documents returned and their titles.

## The known-item batch

A known-item question asks for one document known to exist, described by its metadata. Nothing is written by hand, so the batch is large and covers every bank and doc type; it measures whether a document a user can describe is found, not whether a question on the content is answered. That is the job of the analytical file.

`known_item_v1.jsonl` covers the scope of the production collection: the PDF documents of types A1 (rate-decision press release), A2 (policy statement), A3 (minutes), B1 (press-conference transcript), E1 (monetary policy report) and F1 (staff projections), 2,566 documents of six central banks (`au`, `ca`, `ecb`, `gb`, `jp`, `us`). At most 12 documents are drawn per (bank, doc type), 214 in all.

To build it again:

```bash
python eval/make_known_item.py \
    --manifests '/path/to/central-bank-corpus/data/manifest/*.jsonl' \
    --doc-types A1 A2 A3 B1 E1 F1 --mime application/pdf \
    --per-cell 12 --seed 1 --output eval/known_item_v1.jsonl
```

The committed file was built with seed 1 from the manifests of `central-bank-corpus` at commit `3f08e930afe9c71e87973ee0e5fe4de861244205`. The same manifests and seed give the same file, byte for byte; the order of a cell depends only on a hash of the seed and of each `doc_id`. The script prints a summary of what it built and skipped. Other parameters: `--banks`, `--date-window-days` (default 45), `--max-expected` (default 3).

### Kinds of question

| `kind` | Question | Example |
|---|---|---|
| `title` | The title of the document as the manifest has it. | `FOMC press conference transcript 2023-06-14` |
| `described` | The bank, the doc type and the date in words, in English and in French (`language`). | `Federal Reserve FOMC minutes, 20 June 2012` / `Compte rendu de la réunion de politique monétaire du FOMC de la Réserve fédérale, 20 juin 2012` |
| `described+filters` | The English description, with the filters a caller who knows these facts would set: `source_codes`, `doc_types`, and a window of 45 days on each side of the date. | |

Comparing `described` with `described+filters` shows what the filters bring. Every question is tagged with its bank (`bank:us`), its doc type (`doc_type:A3`) and the precision of its date (`date:day`, `date:month`, `date:year`, `date:none`), so the report breaks the metrics down along each.

| | Questions |
|---|---|
| By kind | `title` 194, `described` 426 (213 English, 213 French), `described+filters` 213 |
| By bank | `au` 132, `ca` 85, `ecb` 236, `gb` 96, `jp` 48, `us` 236 |
| By doc type | A1 96, A2 95, A3 280, B1 95, E1 219, F1 48 |
| By date precision | day 548, month 165, year 108, none 12 |

### Dates the manifests do not really know

Some dates are stored more precisely than they are known. In the scope, every Bank of England document (305 minutes, 24 reports) is dated 1 January of its year, and so are 14 Bank of Canada reports; every Reserve Bank of Australia report (84), every ECB document of type E1 (53) and every Bank of Canada summary of deliberations (10) is dated the first of a month. Sixteen documents have no date.

A question never states more than the manifest knows:

- A date is read as known to the **day** unless it falls on the first of a month, or when the manifest says so (`date_precision: day`). It is written in full: `20 June 2012`.
- A date on the first of a month other than January is read as known to the **month**: `Reserve Bank of Australia monetary policy report, May 2015`. The question expects every document of that bank and type dated in that month, and its date window covers the whole month plus 45 days on each side.
- A date on 1 January is read as known to the **year** only, and is not written at all. Like a document without a date, it is described with the wording of its title: `Bank of England MPC minutes: Minutes of the MPC Meeting held on 9 and 10 April 2008`. Its filters carry no date.
- A title that itself spells out a placeholder date (`RBA Statement on Monetary Policy 2015-05-01`, for a document known to the month) is not used as a `title` question.
- A document without a title, or whose title is a file name (`may-2018.pdf`), gives no `title` question; without a date known to the month either, it gives no question at all.

The date is written without saying what it is the date of: in the manifests it is the day of the meeting for some banks and the day of publication for others.

### Ambiguous questions

A question is ambiguous when several documents of the scope fit it: the same title, or the same bank, doc type and date. In the scope, 112 documents share their bank, type and day with another (two ECB decisions published the same day, the separately stored sections of recent Bank of Canada reports), and 12 share their title.

- When at most three documents fit, all of them are expected: `expected_doc_ids` lists them and the question is tagged `several-expected`. The batch has 3 such questions.
- When more fit, the question is not written: with ten or eighteen sections of one report sharing a date, "the report of that date" has no single answer. The 21 documents met this way while drawing the sample (20 sections of Bank of Canada reports, and one Federal Reserve statement known to the month only, in a month with four statements) gave no description, and other documents of their cell were drawn instead.

Among the 237 documents looked at to draw 214, the script skipped: 22 `title` questions for a missing title, 1 for a title that is a file name, 20 for a title spelling out a placeholder date; the descriptions of 21 documents whose date is shared by more than three documents; the descriptions of 3 documents with neither a usable title nor a date known to the month.

## The fixture

`fixtures/corpus.json` describes 12 documents with their metadata and the text of their chunks, and the model the fixture is meant for: the deterministic `FakeEmbedder` of the engine, with empty prefixes. That embedder has no semantics: two similar texts are not close. The questions of `fixtures/questions.jsonl` are therefore built so that their answer does not depend on meaning: the text of a question is the text of a chunk (same vector, first place), or its filters leave only the expected document. One question expects a document that is not in the corpus, one a document that a date filter leaves out, and one is ranked second on purpose, so that every part of the report has a known value. `tests/eval_fixture.py` loads the corpus into a collection; `tests/test_eval.py` checks the metrics.
