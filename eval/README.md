# Evaluating the search

Question files for `rag-engine eval`, the script that builds the automatic one, and a small fixture for the tests. `rag-engine eval-check` checks a question file without any collection.

| File | What it is |
|---|---|
| `known_item_v1.jsonl` | 797 automatic questions on the policy core of the central-bank corpus, built from the manifests by `make_known_item.py`. |
| `make_known_item.py` | Builds a known-item file from manifest files. Standard library only; not part of the installed package. |
| `fixtures/corpus.json`, `fixtures/questions.jsonl` | 13 small documents (32 chunks) and 13 questions with known answers, in the first version of the format. The test suite loads them into a collection and checks the exact metrics. |
| `analytical_v2.jsonl` | 91 questions on the content of the policy core, written by hand and checked against the documents: 72 that the corpus answers, 4 whose premise is false, 15 that the corpus cannot answer. |
| `filings_v1.jsonl` | 48 questions (43 answering with verbatim `expected_spans`, 5 abstentions) on the 10-Ks of five issuers, FY2022-FY2025, checked by `scripts/check_filings_eval.py`; pending the owner's validation (see `filings_README.md`). |
| `filings_v2.jsonl` | 92 questions (85 answering, 7 abstentions) on the same 20 filings, each with a retrieval `difficulty` (17 easy, 54 medium, 21 hard), 10 in French (4 French/English twins), 12 table questions whose spans carry the year row (`year row [...] answer row`) and 18 more with table rows as alternatives, 7 questions that need every fact (`span_match: "all"`, with groups of alternative spans), 2 `chunk-loss` and 2 `chunk-split` questions v2 cannot answer by construction; audit of v1 and the v2.1 and v2.2 fixes in `filings_v2_AUDIT.md`; measured: v2/section hybrid hit@10 0.741 (results in `filings_README.md`). |
| `filings_v2_AUDIT.md` | The audit of every v1 question (kept, rewritten or dropped, with the reason) behind `filings_v2.jsonl`, and the v2.1 and v2.2 changes after its two reviews (per question: alternatives added, difficulty factors). |
| `filings_README.md` | The passage-level evaluation on SEC filings: the `expected_spans` field, how questions are checked, the measurement protocol of `scripts/filings_eval.py`. |
| `fixtures/questions_v2.jsonl` | 13 questions on the same 13 documents that use every field of the second version of the format: essential and acceptable documents, either-of groups, expected behaviour, a second run without filters, questions that abstain. |

`analytical_v2.jsonl` holds 91 questions on the content of the documents (see "The analytical batch" below). Unlike the known-item questions, which measure whether a document a user can describe is found, they ask what an analyst would ask, and they use the second version of the format.

## Question file format

JSON Lines, UTF-8, one question per line:

```json
{"id": "ki-described-filters-en-eaa2a0eb62cb290d", "question": "European Central Bank monetary policy decisions of 14 April 2022", "expected_doc_ids": ["eaa2a0eb62cb290d", "102c44e911f246ff"], "filters": {"source_codes": ["ecb"], "doc_types": ["A1", "B1"], "date_from": "2022-02-28", "date_to": "2022-05-29"}, "kind": "described+filters", "language": "en", "tags": ["bank:ecb", "doc_type:A1", "date:day", "date-meaning:meeting", "filter-candidates:4-10", "several-expected", "equivalent-documents"]}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | A string, unique in the file. |
| `question` | yes | The text given to the search. |
| `expected_doc_ids` | yes | The `doc_id` of every document that answers the question: a non-empty list. |
| `filters` | no | Filters applied to the search: an object with the field names of `rag_engine.search.SearchFilters` (`source_codes`, `doc_types`, `languages`, `doc_ids`, `date_from`, `date_to`, `corpus`, `include_undated`) and its rules (lists, never a bare string; dates written `YYYY-MM-DD`). |
| `kind` | no | A label; the report gives the metrics per kind. A kind with at least one question that carries filters is a *filtered* kind (see the report). |
| `tags` | no | A list of labels; the report gives the metrics per tag, and a question counts for each of its tags. |
| `language`, `rationale` | no | Free text, kept and not used. |

The file is checked strictly when it is read: a duplicate `id`, an empty `expected_doc_ids`, an unknown field or filter, a bad date are errors that name the line. No other field is accepted, so that a misspelt `filters` is never ignored. A question must hold something to search for as it is written and once put in Unicode NFKD, the form the search embeds it in: a question made of an isolated accent (`´`) is refused when the file is read, with its line, instead of stopping the run at that question.

### Format v2: what a question may say more

Seven optional fields were added. A file that uses none of them is read and scored exactly as before, and gives the same report. A complete line:

```json
{"id": "ex-01", "question": "How did the Monetary Policy Committee vote on Bank Rate in August 2016?", "expected_behavior": "answer", "essential_ids": ["fx-gb-minutes-2016-08"], "essential_any_of": [["fx-ecb-decision-2016-03", "fx-ecb-combined-2016-03"]], "acceptable_ids": ["fx-gb-report-2016-08"], "expected_doc_ids": ["fx-gb-minutes-2016-08", "fx-ecb-decision-2016-03", "fx-ecb-combined-2016-03", "fx-gb-report-2016-08"], "filters": {"source_codes": ["gb", "ecb"]}, "run_unfiltered": true, "must_include": ["unanimously", "0.25 percent"], "must_not_include": ["6-3"], "kind": "analytical+filters", "tags": ["factual", "votes"], "language": "en", "rationale": "The minutes hold the vote; the report of the same month is relevant without being needed."}
```

| Field | Meaning | Rules |
|---|---|---|
| `expected_behavior` | What the right response does: `"answer"` (the documents answer the question; the default), `"correct_premise"` (the premise of the question is false, and the documents show it), `"abstain"` (the collection does not hold the information). | One of the three. |
| `essential_ids` | The documents without which the question cannot be answered. | A list of `doc_id`, none twice. May be empty; must be empty on a question that abstains. |
| `essential_any_of` | Groups of documents that carry the same fact: the answer needs **any one** document of each group. | A list of groups, each a non-empty list of distinct `doc_id`. A document is in one group at most, and in neither `essential_ids` nor `acceptable_ids`. Not allowed on a question that abstains. |
| `acceptable_ids` | The documents that are relevant without being necessary: a neighbouring meeting that restates the fact, a report on the same subject. On a question that abstains: its context documents (below). | A list of `doc_id`, none twice, and none that is also essential or in a group. May be empty. |
| `expected_doc_ids` | The essential documents, the members of the groups and the acceptable documents together. | When one of the three fields above is given, even as an empty list, it may be left out; if it is given it must name exactly the same documents, in any order. It may be empty only when `expected_behavior` is `"abstain"`. |
| `must_include` | Short facts a right answer holds (`"5-4"`, `"Shirai"`). | A list of non-blank strings, none twice. May be empty. |
| `must_not_include` | Typical false statements (`"unanimous"`). | The same; a text may not be in both lists. |
| `run_unfiltered` | `true`: score the question a second time without its `filters`. | `true` or `false`; `true` only on a question whose `filters` restrict something. |

Every rule that is broken is an error naming the line, as for the first fields: an `expected_doc_ids` that lacks or adds a document, a document both essential and acceptable, or in two groups, a question that expects no document and does not abstain, an essential document on a question that abstains, an unknown behaviour, `run_unfiltered` without filters, a misspelt field (`essential`, `must_includes`, `expected_behaviour`). The loader also refuses a key written twice in one object (JSON would silently keep the last one), a document id with a space before or after it (it would never match), and a `kind` that ends with `#unfiltered`, the name the report gives to a second run.

**Requirements: essential documents and either-of groups.** What an answer needs is a list of *requirements*: one per document of `essential_ids`, and one per group of `essential_any_of`. A requirement is satisfied at `k` when one of its documents is among the first `k` documents. Use a group when several documents carry the same fact and any of them lets a system answer: the decision release, the statement and the combined document of one meeting day; the minutes and a later report that restates a total; twenty-four consecutive minutes that all record the same dissent. Naming one of them essential and the others acceptable would punish a correct system that returns the sibling.

```json
{"id": "ex-02", "question": "By how much did the ECB lower the deposit facility rate in March 2016?", "essential_ids": [], "essential_any_of": [["fx-ecb-decision-2016-03", "fx-ecb-combined-2016-03"]], "acceptable_ids": ["fx-ecb-account-2016-04"]}
```

Here the question has one requirement, satisfied by the decision or by the combined document; the account is relevant without being needed. A question that compares two meetings would have two groups, one per meeting. The members of a group are expected documents: hit@1, hit@k, MRR and recall@k count them like the others.

**Questions that abstain.** `"abstain"` says that the corpus does not answer the question. Nothing is then the right document, so **such a question is never scored**, whether it lists documents or not: it is searched, reported apart (the Abstention section of the report), and left out of every mean. The documents it lists are its *context documents*: those a system may reasonably show while it says that the corpus does not hold the answer (the document that shows where the information stops, the nearest one on the subject). They go in `acceptable_ids` (or in `expected_doc_ids` alone); none is essential, because a document that answered would make the question an ordinary one. The report says, for information, how many of them came back among the first `k`: returning one is neither a hit nor a fault.

**What is scored now, and what is carried for later.** The engine returns documents; it writes no answer (a client does). The harness therefore scores retrieval only:

- **Scored now**, on documents: where the expected documents stand (hit@1, hit@k, MRR, recall@k, as before), and how many of the **requirements** are satisfied among the first `k` (`essential_recall@k`, `all_essential@k`). `expected_behavior` groups the results, and decides one thing: a question that abstains is searched and reported apart, not scored. `run_unfiltered` adds a second run.
- **Carried, not scored**: `must_include` and `must_not_include`. They are validated, kept on the question (`Question.must_include`) and written for each question in the JSON report, so that one file serves this harness and, later, a scorer of the answers a client generates from the documents. Nothing here reads them. Whether a response really abstains or corrects a premise is likewise a property of the answer, and is not judged here.

**When a question states its essential documents.** As soon as one of `essential_ids`, `essential_any_of` and `acceptable_ids` is present in the line, **even as an empty list**: `"essential_any_of": []` alone means what `"essential_ids": []` alone means (no requirement, and `expected_doc_ids`, if given, must then name the documents of the three fields and no other). `null` counts as a field left out.

A question written the old way, with `expected_doc_ids` alone, does not say which of its documents are essential: it is scored as before, and the two essential metrics leave it out (`null` in the JSON report, `n/a` for a line of the report without any such question). It is never counted as a zero.

For compatibility the JSON report still carries `essential.per_document.*` (with `essential.intervals` and `essential.few_documents`) with its former meaning, means over the documents of the requirements; the text report does not print it, and the numbers to read are the per-question ones (`essential.all_essential_at_k`, `essential.essential_recall_at_k`, `essential.pooled_essential_recall_at_k`).

**Kinds and filters.** One question with filters makes its `kind` a filtered kind, kept whole out of the headline (see the report). In a file where most questions carry no filter, give the few that do a kind of their own (`analytical+filters`), as the known-item batch does; otherwise every question of the kind leaves the headline. `rag-engine eval-check` points at such kinds.

## Checking a question file

```bash
rag-engine eval-check --questions eval/analytical_v2.jsonl \
    --manifests '/path/to/central-bank-corpus/data/manifest/*.jsonl' \
    --doc-types A1 A2 A3 B1 E1 F1 --mime application/pdf
```

The command reads the file with the rules above and nothing else: no Qdrant, no embedding model, no setting. It prints the number of questions per kind, tag, expected behaviour and language, how many carry filters and how many of these are scored again without them, how many abstain (with and without context documents), and, for the questions that are scored, how many have how many requirements (essential documents and either-of groups) and how many expected documents. Then what is allowed and probably not meant: question texts used twice (compared without regard to case and spacing; their number and the first ten), kinds that mix questions with and without filters, questions with more requirements than `--k` (default 10: they can never have all they need) and questions with more expected documents than `--k` (their recall@k cannot reach 1). The first line names the file and is cut to 110 characters, the end of the path being kept.

With `--manifests` (the catalogue of the corpus: JSON Lines files with one document per line, its `doc_id`, `mime_type` and `doc_type`; paths or glob patterns), every document a question names is looked up, and those that are not in the catalogue are listed with the questions that name them and how (`essential`, `either-of`, `acceptable`, `expected`, `context`). `--doc-types` and `--mime` give the scope that was ingested: a document found in the catalogue with another doc type or MIME type is listed too, since it is not in the collection. The documents named are also counted per doc type and MIME type.

Exit code 0 when the file is valid and every document it names was found (and is in the scope); 2 otherwise: a line that breaks a rule, a document that is missing, a manifest that cannot be read. From Python: `rag_engine.eval_check.check_questions`, `load_catalogue`, `format_check`.

## Running an evaluation

```bash
export QDRANT_HOST=localhost QDRANT_PORT=6333      # and QDRANT_API_KEY, QDRANT_HTTPS if needed
export RAG_ENGINE_BACKEND=onnx                     # or torch; the model the collection was embedded with
rag-engine eval --questions eval/known_item_v1.jsonl --collection documents \
    --k 10 --chunk-limit 50 --output report.json
```

The command prints a report and, with `--output`, writes every detail as JSON. Exit code 2 means the evaluation could not run (arguments, settings, question file, server, collection, embedding backend). What is cheap is checked before the embedding model is loaded:

- the collection exists and has one unnamed vector of the dimension the settings announce (`RAG_ENGINE_DIM`); a collection with named vectors is refused with a message that says so;
- at least one expected document is in the collection. A collection that holds none of them is almost always the wrong collection: the command stops with exit code 2. `--allow-empty` runs the evaluation all the same; nothing is scored, and a warning says so.

`--fail-under-hit-at-k 0.9` makes the command exit with code 1 when the **headline** hit@k is below 0.9 (the headline is defined below; it is not the pooled mean of all questions). `--bootstrap-resamples N` sets the number of resamples of the intervals (default 1000). `--score-levels X ...` sets the scores the questions that abstain are reported against (default `0.7 0.75 0.8 0.85 0.9`).

### How a question is scored

- The search returns chunks; an answer is a document. For each question, `--chunk-limit` chunks are asked for (default 50, at most `RAG_ENGINE_SEARCH_MAX_LIMIT`), and collapsed into documents: a document takes the place of its best chunk. The metrics look at the first `--k` documents (default 10).
- If the chunks returned hold fewer than `k` documents although the whole chunk budget was used, the report says for how many questions: a larger `--chunk-limit` could have ranked more documents.
- Before anything is searched, the expected documents are looked up in the collection. A question none of whose expected documents is there is listed under **not in the collection** and left out of every metric: the search cannot find a document that was never ingested. When only some are absent the question is scored, and the absent documents are listed.

| Metric | Meaning |
|---|---|
| hit@1 | Share of questions whose first document is an expected one. |
| hit@k | Share of questions with an expected document among the first `k`. |
| MRR | Mean of `1 / rank` of the best-placed expected document, 0 when it is beyond `k`. |
| recall@k | Mean share of the expected documents (those present in the collection) found among the first `k`. |
| all_essential@k | **Share of questions** with all their requirements satisfied among the first `k`. Printed with its counts: `0.75 (3/4)`. |
| essential_recall@k | For a question, the share of its requirements that are satisfied. Printed two ways: the **mean over the questions**, and **pooled** over the requirements (satisfied / total, with the counts). |

A requirement is one essential document or one either-of group; it is satisfied when one of its documents is among the first `k`, and left out when none of its documents is in the collection. The expected documents are the essential ones, the members of the groups and the acceptable ones together, and hit@1, hit@k and MRR look at the best-placed of them: **an acceptable document at rank 1 is a hit**. recall@k counts them all alike. Only the last two metrics look at what the answer needs, and they are computed on the questions that state their essential documents and have at least one requirement to look for. A question that abstains is in none of these.

### Choosing `k` for chain and aggregation questions

A question that needs six meetings has six requirements; one that follows a figure over three years may have twelve. With more requirements than `k`, `all_essential@k` is 0 and `essential_recall@k` is below 1 whatever the search returns. Such questions are not hidden in the mean: the report counts them (`unreachable`, per line and in the JSON `summary.essential_unreachable`), lists them, and marks each (`essential_unreachable`). `rag-engine eval-check` lists them before any run. An either-of group is one requirement whatever the number of its documents: twenty-four minutes that all carry the same fact take one place, not twenty-four.

The remedy is a larger `--k`, at least the largest number of requirements. `k` documents cannot come out of fewer chunks, so `--k` is bounded by `--chunk-limit`, itself bounded by `RAG_ENGINE_SEARCH_MAX_LIMIT` (default 50): with the defaults `k` goes up to 50. A long document can take several of the chunks, so leave room: `--k 20` with the default `--chunk-limit 50`, or raise both limits. The other metrics move with `k` too (hit@k and recall@k can only rise), so compare two runs at the same `k`.

Long lists of *expected* documents are another matter. A question with 44 expected documents (most of them acceptable) cannot have a recall@10 above 10/44, and that says nothing against the search: the report counts these questions (`More expected documents than k`, JSON `summary.expected_beyond_k`), and recall@k should not be read for them.

### Equal scores

Two chunks can have exactly the same score: the same text stored in two documents does. Qdrant returns such chunks in an order of its own, which can change from one request to the next and with the limit, so the rank of a document could change between two runs. The harness removes what it can of that: the chunks returned are put in one order (decreasing score, then `doc_id`, then `chunk_number`) before they are collapsed into documents. The same chunks therefore always give the same ranking.

The order among equal scores is still arbitrary. A question is marked `tied_at_decision` when its hit@1 or its hit@k hangs on it: an expected document has the same score as documents that are not expected, across rank 1 or rank `k`, and another order of the tied documents would change the answer. The report counts these questions; the JSON report marks each. The mark concerns hit@1 and hit@k only: recall@k and the essential metrics are not checked for ties.

What stays outside the harness: when chunks are tied at the edge of the chunk budget, which of them Qdrant returns is its choice, and the harness can only order what it was given.

### Reading the report

```
                  questions  documents  hit@1              hit@10             MRR                recall@10
kind: filtered *          5          6  0.83 [0.50, 1.00]  0.83 [0.50, 1.00]  0.83 [0.50, 1.00]  0.67
kind: second *            1          1  0.00               1.00               0.50               1.00       (few documents)
kind: verbatim            6          4  0.92 [0.75, 1.00]  1.00 [1.00, 1.00]  0.96 [0.88, 1.00]  1.00
headline                  6          4  0.92 [0.75, 1.00]  1.00 [1.00, 1.00]  0.96 [0.88, 1.00]  1.00
overall (pooled)         12         11  0.75               0.92               0.83               0.88
```

(The fixture, with the minimum number of documents lowered to 4 so that intervals are shown.)

- **One line per kind first, then the headline, then the pooled line.** Every line gives the number of questions and the number of **distinct documents** these questions expect.
- **Per-document means.** The lines of the kinds, of the headline and of the tags give, for each metric, the mean over the documents of the mean over the questions that expect the document. The questions of one document are correlated: a document is easy or hard to find however it is asked for, and the known-item batch asks for each document three or four times. Four questions on one document are not four independent observations. A question that expects several documents counts for each of them.
- **Interval.** `[low, high]` is a 95% interval of the per-document mean of hit@1, hit@k and MRR: the documents are drawn again with replacement 1000 times (`--bootstrap-resamples`), with a fixed seed, and the interval goes from the 2.5th to the 97.5th percentile of the means. The same results always give the same interval. A line with fewer than 20 documents gets no interval and the mark `(few documents)`: an interval from a handful of documents would look more certain than it is. Intervals are given for the kinds and the headline, not for the tags.
- **`headline`**: the kinds asked without filters, together, per document. It is the number to compare between two versions of the search, and the one `--fail-under-hit-at-k` looks at. On the known-item batch it is `title`, `label` and `described`; it is `n/a`, and a threshold is not met, when every kind carries filters.
- **`*`: a kind asked with filters** is kept out of the headline. Filters narrow the candidates: when they leave fewer than `k` documents, an expected document is among the first `k` whatever the search does. For such a kind **only hit@1 and MRR are informative**; hit@k and recall@k are true by construction.
- **`overall (pooled)`**: the plain mean over all scored questions, every kind, filtered or not, each question counted once. It is kept for reference and labelled as pooled, also in the JSON (`metrics.overall.aggregation`); it is not the headline.
- **Precision.** Rates are printed to two decimals: with two hundred documents an interval is several points wide, and a third decimal would be noise. The JSON report holds the exact values (`metrics.headline`, `metrics.overall`, `metrics.by_kind`, `metrics.by_tag`: the means over the questions, `per_document`, `intervals`, `few_documents`).

Then come the questions left out, the questions with equal scores, and the **misses** (no expected document among the first `k`), each with the three best documents returned and their titles.

### The parts of the report for format v2

They are printed only when the questions call for them (a question states its essential documents or another behaviour than `answer`, abstains, or is scored again without its filters); the report of a file of the first version is unchanged, to the character. The tables below are the report of `fixtures/questions_v2.jsonl`.

```
                                 questions  documents  hit@1  hit@10  MRR   recall@10
kind: graded                             5         13  1.00   1.00    1.00  0.82       (few documents)
kind: graded+filters *                   5          7  1.00   1.00    1.00  0.56       (few documents)
kind: graded+filters#unfiltered          2          3  0.67   1.00    0.83  1.00       (few documents)
headline                                 7         13  0.97   1.00    0.99  0.82       (few documents)
overall (pooled)                        10         13  1.00   1.00    1.00  0.78
```

In such a report, a first table wider than 110 characters (long kind names with intervals) puts the interval of each line on a line of its own, under it.

**A second run without filters** (`run_unfiltered`) is one more result, under the id of the question followed by `#unfiltered`. It is counted in a kind of its own, `<kind>#unfiltered`, and carries the tags of its question and `unfiltered-rerun`. That kind is asked without filters, so it is **in the headline**, next to the kinds that never had filters; the kind of the question, with its filters, stays a filtered kind marked `*`. `overall (pooled)` and the lines per expected behaviour count each question once, as it is written: the second runs are not in them. In the JSON report they are under `unfiltered_reruns`, apart from `results`.

```
Per question (the same lines as means over the questions: each question counts once):
                                 questions  hit@1  hit@10  MRR   recall@10
kind: graded                             5  1.00   1.00    1.00  0.95
kind: graded+filters *                   5  1.00   1.00    1.00  0.60
kind: graded+filters#unfiltered          2  0.50   1.00    0.75  1.00
headline                                 7  0.86   1.00    0.93  0.97
Which to read. Per document (the first table) when several questions ask for one document, as in
  a known-item batch: a document asked for four times is not four observations. Per question
  (this table, and overall (pooled) above) when questions expect many documents each: in a
  per-document mean a question that expects 44 documents weighs 44 times one that expects one.
More expected documents than k=10 (1 question: recall@10 cannot reach 1 for it): fx2-q05
```

**Per question.** The first table gives per-document means: right for a batch that asks for each document several times, wrong for one whose questions expect many documents each, where a question with 44 expected documents weighs 44 times a question with one. This table gives the same lines as plain means over the questions (JSON: `hit_at_1`, `hit_at_k`, `mrr`, `recall_at_k` of each aggregate, next to `per_document`). Here `fx2-q05`, with its thirteen documents, pulls the per-document recall of `graded` down to 0.82; per question it is 0.95. The last line counts the questions whose expected documents exceed `k`: their recall@k cannot reach 1.

```
What the filter brings (2 questions scored with their filters, then without):
                    questions  hit@1  hit@10  MRR   essential_recall@10
with their filters          2  1.00   1.00    1.00  0.75
without filters             2  0.50   1.00    0.75  1.00
Means over these questions. The second runs are the kinds named #unfiltered above; they are
  in the headline and not in overall (pooled).
```

**What the filter brings** compares, on the questions scored both ways and only on them, the means with and without the filters (JSON: `filter_effect`, with each pair). Here the filters put the right document first for both questions, where the search alone does it for one; and they cost a quarter of the requirements, because a filter on the doc type left out a document that was needed. The essential mean is over the pairs that state requirements.

```
Per expected behaviour (per-document means; each question once, as it is written):
                                    questions  documents  hit@1  hit@10  MRR   recall@10
expected_behavior: answer                   8         13  1.00   1.00    1.00  0.75
expected_behavior: correct_premise          2          3  1.00   1.00    1.00  0.67
expected_behavior: abstain                  3  see the Abstention section
correct_premise: scored on the documents that show the premise is false, like any question.
  Whether the response corrects the premise is a property of the answer: it is not judged here.
abstain: never scored, whether the question lists documents or not; in no line above.
```

**Per expected behaviour.** A `correct_premise` question is scored on the documents that show the premise is false, like any question. The lines tell whether these questions are harder to retrieve for; they say nothing on the response itself. The questions that abstain have a count and no metric: they are in the Abstention section.

```
Essential documents (what an answer needs, on the questions that state it):
                                    all_essential@10  essential_recall@10  pooled        unreachable
kind: graded                        0.75 (3/4)        0.94                 0.81 (13/16)            1
kind: graded+filters *              0.20 (1/5)        0.50                 0.50 (4/8)              0
kind: graded+filters#unfiltered     1.00 (2/2)        1.00                 1.00 (3/3)              0
headline                            0.83 (5/6)        0.96                 0.84 (16/19)            1
overall (pooled)                    0.44 (4/9)        0.70                 0.71 (17/24)            1
expected_behavior: answer           0.43 (3/7)        0.75                 0.73 (16/22)            1
expected_behavior: correct_premise  0.50 (1/2)        0.50                 0.50 (1/2)              0
No interval: every line has fewer than 20 questions that state requirements.
requirement: one document of essential_ids, or one group of essential_any_of (documents that
  carry the same fact): satisfied when one of its documents is among the first 10. One whose
  documents are all absent from the collection is left out.
all_essential@10: share of the questions with every requirement satisfied (their number / the
  questions of the line that state requirements; the others are left out, n/a without any).
essential_recall@10: mean over these questions of the share of their requirements satisfied.
pooled: requirements satisfied / requirements, these questions together (a long one weighs more).
unreachable: questions with more requirements than k=10: they can never have them all.
  For chain and aggregation questions raise --k (at most --chunk-limit, itself bounded by
  RAG_ENGINE_SEARCH_MAX_LIMIT).
hit@1, hit@10, MRR and recall@10 look at every expected document: essential, either-of or
  acceptable.

More requirements than k=10 (1):
  fx2-q05: 13 requirements
```

**Essential documents.** Each line is computed on its questions that state requirements (the others are left out):

- `all_essential@k` is **always a share of questions**: those with every requirement satisfied among the first `k`, with the counts. In `kind: graded` three of the four questions have all they need: 0.75.
- `essential_recall@k` is the **mean over the questions** of the share of their requirements that are satisfied: (1 + 1 + 10/13 + 1) / 4 = 0.94.
- `pooled` is the same share **over the requirements**, all the questions of the line together: 13 of 16. One question with thirteen requirements weighs thirteen times there, and once in the mean.

The kinds and the headline get a 95% interval for each of the three, from resampling the **questions** with replacement (`--bootstrap-resamples`, fixed seed), when they have at least 20 questions that state requirements (`interval_min_questions=` in Python); the intervals are printed under the table, one line per kind. The JSON report has, for every group (`essential`): `questions`, `all_essential_at_k`, `all_essential_questions`, `essential_recall_at_k`, `requirements`, `requirements_satisfied`, `pooled_essential_recall_at_k`, `question_intervals`, `few_questions`, `unreachable`, and for every question `requirements_present`, `requirements_satisfied`, `essential_recall_at_k`, `all_essential_at_k`. The keys `per_document`, `intervals` and `few_documents` of `essential` are kept with their first meaning (means over the documents of the requirements, and an interval from resampling these documents): a question counts there once for each of its documents, so they are not shares of questions, and the text report does not print them.

```
Abstention (3 questions whose expected behaviour is to abstain: searched, in no mean, not judged):
         top score  >=0.7  >=0.75  >=0.8  >=0.85  >=0.9  context@10  top document
fx2-q04      1.000      1       1      1       1      1  1/1         fx-gb-report-2016-08  Inflation Report - Au…
fx2-q10      1.000      1       1      1       1      1  -           fx-jp-minutes  Minutes of the Moneta…
fx2-q11        n/a      0       0      0       0      0  -           nothing was returned
>=x: documents ranked with a score of at least x, among all those returned for the question.
context@10: the context documents of the question among the first 10, over those that are in
  the collection. A context document is one the question lists: a system may show it while it
  says that the corpus does not answer. Returning one is neither a hit nor a fault; - without any.
For comparison, top score of the 10 scored questions: min 0.264, median 1.000, max 1.000.
No threshold is applied: whether to answer is decided by the caller of the search.
```

**Abstention.** A question whose `expected_behavior` is `"abstain"` has no right document: there is no rank, no hit and no miss, and it is **left out of every mean and every count of scored questions**, whether it lists documents or not. It is searched all the same, with its filters, and the section shows what came back: the first document and its score, the number of documents ranked at or above each score level (among all the documents returned for the question, not the first `k` only), and how many of its context documents are among the first `k` (`fx2-q04` lists one, and it comes first: that is not a hit). The last lines give, for comparison, the lowest, the median and the highest score of the first document over the scored questions of the same run; a scored question whose search returned nothing has no top score, and the line then says how many have one.

What the section is for: seeing whether the scores separate the questions the corpus cannot answer from those it can, so that a threshold below which a client should answer "this is not in the corpus" can be studied on real data. What it is not: a verdict. **No threshold is applied and nothing passes or fails**; the engine returns documents, and abstaining is a decision of the client that writes the answer. The levels are a parameter (`--score-levels`, `score_levels=`); the defaults suit cosine scores of the multilingual E5 model and mean nothing for the test embedder of the fixture. JSON: `abstentions` (`score_levels`, `top_score_of_scored_questions`, and one entry per question with its first documents, `context_doc_ids`, `absent_context_doc_ids` and `context_documents_at_k`).

## The known-item batch

A known-item question asks for one document known to exist, described by its metadata. Nothing is written by hand, so the batch is large and covers every bank and doc type; it measures whether a document a user can describe is found, not whether a question on the content is answered.

`known_item_v1.jsonl` covers the scope of the production collection: the PDF documents of the doc types A1, A2, A3, B1, E1 and F1 of six central banks (`au`, `ca`, `ecb`, `gb`, `jp`, `us`), 2,566 documents in 18 (bank, doc type) cells. At most 12 documents are drawn per cell, 214 in all; with the equivalent documents (below) the questions expect 248 distinct documents.

What the numbers of this batch are, and are not:

- **A per-cell average of a stratified sample, not a corpus-weighted one.** Every cell gives 12 documents (10 for the ten Bank of Canada summaries), whether it holds 33 documents or 323. The headline and the `overall` line weigh the 18 cells equally; they do not say how often a document drawn at random from the collection is found. The per-tag lines (`bank:`, `doc_type:`) give each cell its own numbers.
- **All documents are English** (`language: en` for the 2,566 documents of the scope). The French questions therefore measure cross-lingual retrieval, a French query against English text, not a French search.
- The questions of one document are correlated, hence the per-document means of the report.

To build it again:

```bash
python eval/make_known_item.py \
    --manifests '/path/to/central-bank-corpus/data/manifest/*.jsonl' \
    --doc-types A1 A2 A3 B1 E1 F1 --mime application/pdf \
    --per-cell 12 --seed 1 --output eval/known_item_v1.jsonl
```

The committed file was built with seed 1 from the manifests of `central-bank-corpus` at commit `3f08e930afe9c71e87973ee0e5fe4de861244205`. The same manifests and seed give the same file, byte for byte; the order of a cell depends only on a hash of the seed and of each `doc_id`. Its SHA-256 is `4b6678842d7d9c763704dce98cf179126a85ba290913c44dc2904174639e236e`. The script prints a summary of what it built and skipped. Other parameters: `--banks`, `--date-window-days` (default 45), `--max-expected` (default 3).

| | Questions |
|---|---|
| By kind | `title` 36, `label` 158, `described` 393 (210 English, 183 French), `described+filters` 210 |
| By bank | `au` 132, `ca` 76, `ecb` 236, `gb` 72, `jp` 45, `us` 236 |
| By doc type | A1 96, A2 95, A3 265, B1 95, E1 198, F1 48 |
| By date precision | day 548, month 165, year 75, none 9 |
| By date meaning | meeting 409, publication 182, period 116, not stated 90 |

### Kinds of question

| `kind` | Question | Example |
|---|---|---|
| `title` | A real title: the wording the bank gives the document. | `Minutes of the Monetary Policy Committee Meeting held on 7 and 8 July 2010` |
| `label` | A catalogue label: a name the scraper made from the type and the date of a document it found no title for. | `FOMC minutes 2019-03-20` |
| `described` | The bank, the doc type and the date in words, in English and in French (`language`). | `Minutes of the Federal Reserve FOMC meeting of 20 March 2019` / `Compte rendu de la réunion du FOMC de la Réserve fédérale du 20 mars 2019` |
| `described+filters` | The English description, with the filters a caller who knows these facts would set: `source_codes`, `doc_types`, and a window of 45 days on each side of the date. | |

Every question is tagged with its bank (`bank:us`), its doc type (`doc_type:A3`), the precision of its date (`date:day`, `date:month`, `date:year`, `date:none`) and what its date is the date of (`date-meaning:`, below), so the report breaks the metrics down along each.

### Titles and labels

In the manifests the `title` of 2,089 of the 2,566 documents is not a title: it is a label the scraper wrote, the name of the series followed by the date (`FOMC minutes 1997-11-12`, `BoC Monetary Policy Report 2022-04`). 367 documents carry the wording of the bank, and 110 have no title.

The rule that tells them, in `is_catalogue_label`: **a title that ends, after a space, with a date written the ISO way (`YYYY-MM-DD` or `YYYY-MM`) is a label.** The scraper always writes its dates that way and no title given by a bank does. The wordings it uses are listed per bank in `LABEL_PREFIXES`; all 2,089 labels of the scope are one of them, and the summary of the script counts the labels that are not, so that a new wording is noticed:

| Bank | Label wordings, each followed by the date |
|---|---|
| `au` | `RBA monetary policy decision`, `RBA board minutes`, `RBA Statement on Monetary Policy` |
| `ca` | `BoC governing council deliberations`, `BoC Monetary Policy Report` |
| `ecb` | `Monetary policy decision`, `Monetary policy statement`, `Monetary policy account`, `ECB press conference`, `ECB staff projections`, `ECB E1` |
| `us` | `FOMC statement`, `FOMC minutes`, `FOMC press conference transcript`, `FOMC Monetary Policy Report`, `Summary of Economic Projections` |
| `gb`, `jp` | none: the titles are those of the bank (`Minutes of the MPC Meeting held on 9 and 10 April 2008`, `Monetary Policy Report - August 2021`, `Meeting on January 20 and 21, 2015`) |

What each kind can and cannot tell. The engine embeds the text of the chunks only; the title is a payload field and is not searched.

- **`title`**: the wording of the bank probably appears in the text, on the first page (for the Bank of Japan the title of the manifest is the text of the link on the bank's page, `Meeting on April 26 and 27, 2017`, which is the end of the title printed on the minutes). The kind measures whether a user who types the title of a document finds it. It cannot be compared across banks: only `gb` and `jp` have real titles.
- **`label`**: the label is printed nowhere on the document. Nothing in the indexed text equals it; a hit comes from the words of the label that describe the document (`minutes`, `FOMC`) and from the date, which the document writes in another form. The kind measures how well a terse catalogue-style query with an ISO date works. It says nothing on title matching, and it must not be read as "the title is found": 158 of the 194 questions that were of kind `title` before the split are labels.

A label that spells out a placeholder date (`RBA Statement on Monetary Policy 2015-05-01`, for a document known to the month) is not used as a question. A document without a title, or whose title is a file name (`may-2018.pdf`), gives neither.

### The words for a doc type, per bank

A doc type is not the same document at every bank. The words of a description come from one table per (bank, doc type), `CELL_WORDING` in the script, established from the titles and the addresses of the manifests:

| Bank | Type | English | French |
|---|---|---|---|
| `au` | A1 | Reserve Bank of Australia monetary policy decision media release | Communiqué de décision de politique monétaire de la Banque de réserve d'Australie |
| `au` | A3 | Minutes of the Reserve Bank of Australia Board monetary policy meeting | Compte rendu de la réunion de politique monétaire du Conseil de la Banque de réserve d'Australie |
| `au` | E1 | Reserve Bank of Australia Statement on Monetary Policy | Rapport trimestriel de politique monétaire (Statement on Monetary Policy) de la Banque de réserve d'Australie |
| `ca` | A3 | Bank of Canada summary of Governing Council deliberations | Résumé des délibérations du Conseil de direction de la Banque du Canada |
| `ca` | E1 | Bank of Canada Monetary Policy Report | Rapport sur la politique monétaire de la Banque du Canada |
| `ecb` | A1 | European Central Bank monetary policy decisions | Décisions de politique monétaire de la Banque centrale européenne |
| `ecb` | A2 | European Central Bank monetary policy statement at the press conference | Déclaration de politique monétaire à la conférence de presse de la Banque centrale européenne |
| `ecb` | A3 | European Central Bank account of the monetary policy meeting | Compte rendu de la réunion de politique monétaire de la Banque centrale européenne |
| `ecb` | B1 | European Central Bank combined monetary policy decisions and statement | Document combiné des décisions et de la déclaration de politique monétaire de la Banque centrale européenne |
| `ecb` | E1 | Eurosystem and ECB staff macroeconomic projections for the euro area | Projections macroéconomiques pour la zone euro des services de l'Eurosystème et de la BCE |
| `gb` | A3 | Bank of England Monetary Policy Committee minutes | Procès-verbal de la réunion du Comité de politique monétaire de la Banque d'Angleterre |
| `gb` | E1 | Bank of England Monetary Policy Report | Rapport sur la politique monétaire de la Banque d'Angleterre |
| `jp` | A3 | Minutes of the Bank of Japan Monetary Policy Meeting | Compte rendu de la réunion de politique monétaire de la Banque du Japon |
| `us` | A2 | Federal Reserve FOMC statement | Communiqué du FOMC de la Réserve fédérale |
| `us` | A3 | Minutes of the Federal Reserve FOMC meeting | Compte rendu de la réunion du FOMC de la Réserve fédérale |
| `us` | B1 | Transcript of the Federal Reserve FOMC press conference | Transcription de la conférence de presse du FOMC de la Réserve fédérale |
| `us` | E1 | Federal Reserve Monetary Policy Report to the Congress | Rapport sur la politique monétaire de la Réserve fédérale au Congrès |
| `us` | F1 | Federal Reserve FOMC Summary of Economic Projections | Résumé des projections économiques du FOMC de la Réserve fédérale |

Where the cell differs from the name of its type:

- **`ecb` B1 is not a press-conference transcript.** All 33 documents are the file `ecb.ds<date>.en.pdf` of the monetary policy statement section: the combined monetary policy decisions and statement of a meeting day, without the questions and answers.
- **`ecb` E1 is not a monetary policy report**: the 53 documents are the staff macroeconomic projections, among them six breakdown tables and one guide to the projection exercises, which the catalogue labels `ECB E1`.
- **`us` E1 is not an FOMC document**: it is the Monetary Policy Report the Board of Governors submits to the Congress twice a year (`<date>_mprfullreport.pdf`); the label `FOMC Monetary Policy Report` is the scraper's.
- **`au` E1** is the quarterly Statement on Monetary Policy, named as the bank names it.
- **`ca` A3** is the summary of Governing Council deliberations, and **`ecb` A3** the account of the monetary policy meeting: neither bank calls them minutes.

A (bank, doc type) the table does not hold gets the default words of its doc type, `TYPE_WORDING`, with the name of the bank: `{bank} rate decision press release` (A1), `{bank} monetary policy statement` (A2), `{bank} monetary policy meeting minutes` (A3), `{bank} press conference transcript` (B1), `{bank} monetary policy report` (E1), `{bank} economic projections` (F1), and their French counterparts. A bank without a name, or a doc type without words, gives no description.

### What the date is the date of

The `date` of a manifest row is the day of the meeting for some cells and the day of publication for others. "Bank of Canada minutes, October 2025" would name the summary of the September meeting, published in October. The table `DATE_MEANING` of the script says, per (bank, doc type), what the date is, and the wording follows it:

| Meaning | The date is | English | French |
|---|---|---|---|
| `meeting` | the day of the policy meeting, its last day when it lasts two; a decision, a statement or a press conference of that day carries it too | `… of 20 June 2012`, `… of June 2012` | `… du 20 juin 2012`, `… de juin 2012`, `… d'août 2012` |
| `publication` | the day, or the month, the document was published | `…, published on 7 April 2016`, `…, published in October 2025` | `…, publié le 7 avril 2016`, `…, publié en octobre 2025` (the participle agrees: `publiée`, `publiées`) |
| `period` | the month an issue of a report is named after | `…, May 2015` | `…, mai 2015` |
| `unknown` | not established (also any cell that is not in the table) | `…, 20 June 2012` | `…, 20 juin 2012` |

A bare date after a comma asserts nothing; no description attaches a publication date to a meeting. Each question is tagged `date-meaning:meeting`, `date-meaning:publication`, `date-meaning:period` or `date-meaning:unknown` after the date written in it, and `date-meaning:not-stated` when its text carries no date written by the script (a real title; a description by the title).

The evidence, from the manifests (scope of 2,566 documents):

| Bank | Type | Date is | Evidence |
|---|---|---|---|
| `au` | A1 | publication | The release of the decision. Of the 213 meeting dates of the board minutes (A3), 198 are also the date of a release: since 2008 it is published on the day of the meeting. In 2006 and 2007 the five releases that follow a meeting are dated the day after it, none the same day. |
| `au` | A3 | meeting | The date is in the address (`…/2006/07112006.html`, `…/2016/2016-07-05.html`); 212 of 213 are Tuesdays; 198 are the day of a decision release, and minutes are not published on the day of the meeting. |
| `au` | E1 | period | All 84 dates are the first of a month, and the address names the issue (`/smp/2006/nov/`). |
| `ca` | A3 | publication | All 10 addresses hold the month of the date and name the announcement the summary reports (`/2025/10/summary-…-fixed-announcement-date-of-september-17-2025`): for 6 of the 10 the announcement is in the month before. |
| `ca` | E1 | period | Label and address name the issue by its month (`mpr-2021-07`, 54 reports dated the first of that month). The 100 pages of recent reports carry the day of release (`mpr-2025-07-30`), which the addresses of the summaries name as an announcement date for 5 of 9 days. |
| `ecb` | A1 | meeting | The date is in the address (`pr011025`, `ecb.mp240125`) for all 323; 307 are Thursdays; 278 of the 317 distinct days are also the day of a press-conference statement (A2). The accounts (A3) are never dated one of these days. It is the day the decision was taken and announced. |
| `ecb` | A2 | meeting | The date is in the address (`is991104`) for all 295; 278 are the day of a decision (A1). |
| `ecb` | A3 | publication | None of the 95 dates is the day of a decision (A1); 84 are exactly 28 or 35 days after the previous one. The date is in the address (`mg150521`). |
| `ecb` | B1 | meeting | All 33 are dated a day with a decision (A1) and a statement (A2); the date is in the address (`ecb.ds220609`). |
| `ecb` | E1 | period | All 53 dates are the first of a month, and the address names the issue (`ecbstaffprojections201609`, `projections202103_ecbstaff`). |
| `gb` | A3 | meeting, the year only | All 305 dates are 1 January. The year is the year of the meeting the title names, for the 302 titles that name one. Never written in a question. |
| `gb` | E1 | period, the year only | All 24 dates are 1 January; the year is the year of the issue the title names. Never written in a question. |
| `jp` | A3 | meeting | For the 32 dated documents the date is the last day the title names (`Meeting on February 17 and 18, 2015`, dated 2015-02-18) and is in the address (`g150218.pdf`). Six documents have no date. |
| `us` | A2 | meeting | The date is in the address for all 237. It is the date of the minutes of the same meeting (219 minutes), which are published three weeks later: the shared date is the meeting's. |
| `us` | A3 | meeting | The date is in the address (`fomcminutes20120620.pdf`) for all 262; 219 share it with a statement (A2), the others are older than the statements of the scope or have none. |
| `us` | B1 | meeting | All 91 are dated the day of a statement (A2). |
| `us` | F1 | meeting | All 76 are dated the day of a statement (A2). |
| `us` | E1 | publication | The day is in the address (`20090224_mprfullreport.pdf`) for all 38; none is the day of a statement; 19 are in February or March and 19 in June or July, the two reports of a year. |

### Dates the manifests do not really know

Some dates are stored more precisely than they are known. In the scope, every Bank of England document (305 minutes, 24 reports) is dated 1 January of its year, and so are 14 Bank of Canada reports; every Reserve Bank of Australia report (84), every ECB document of type E1 (53) and every Bank of Canada summary of deliberations (10) is dated the first of a month. Sixteen documents have no date.

A question never states more than the manifest knows:

- A date is read as known to the **day** unless it falls on the first of a month, or when the manifest says so (`date_precision: day`). It is written in full: `20 June 2012`.
- A date on the first of a month other than January is read as known to the **month**: `Reserve Bank of Australia Statement on Monetary Policy, May 2015`. The question expects every document of that bank and type dated in that month, and its date window covers the whole month plus 45 days on each side.
- A date on 1 January is read as known to the **year** only, and is not written at all. Like a document without a date, it is described with the wording of its title, in English: `Bank of England Monetary Policy Committee minutes: Minutes of the MPC Meeting held on 9 and 10 April 2008`. Its filters carry no date.
- When that title is a catalogue label there is nothing left to describe the document with, and no description is written: it would repeat the `label` question (3 documents of the sample, the January issues of the Bank of Canada report).
- A document without a usable title and without a date known to the month gives no question at all.

### French questions

A French description is written in French from end to end: `Compte rendu de la réunion du FOMC de la Réserve fédérale du 20 juin 2012`, `Résumé des délibérations du Conseil de direction de la Banque du Canada, publié en octobre 2025`. A document that can only be described by its title has an English title: a French question would be a French prefix glued to an English title, which nobody types. **No French question is written for it**: 27 documents of the sample (24 of the Bank of England, 3 of the Bank of Japan without a date) have an English description only, which is why `described` has 210 English questions and 183 French ones.

### Ambiguous questions

A question is ambiguous when several documents of the scope fit it: the same title, or the same bank, doc type and date. In the scope, 112 documents share their bank, type and day with another (two ECB releases published the same day, the separately stored sections of recent Bank of Canada reports), and 12 share their title.

- When at most three documents of the cell fit (`--max-expected`), all of them are expected: `expected_doc_ids` lists them and the question is tagged `several-expected`.
- When more fit, the question is not written: with ten or eighteen sections of one report sharing a date, "the report of that date" has no single answer. The 21 documents met this way while drawing the sample (20 sections of Bank of Canada reports, and one Federal Reserve statement known to the month only, in a month with four statements) gave no description, and other documents of their cell were drawn instead.

### Equivalent documents

The catalogue holds some texts twice, under two doc types. A question about one of them has two right answers, and counting only one would score a correct first result as a miss.

The declared case, `COMBINED_DOCUMENTS` in the script: **the ECB document of type B1 is the combination of the A1 decision and the A2 statement of the same day.** The 33 combined documents (March 2022 to April 2026) are each dated a day that also has a decision and a statement; 104 documents of the scope are concerned (33 combined documents, 38 decisions, 33 statements).

The rule: same bank, same date known to the day, and doc types that the table declares as a combined document and one of its parts.

- A question about a **part** (a decision, a statement) also expects the combined document of that day.
- A question about the **combined document** also expects its parts of that day.
- A decision and a statement are **not** equivalent to each other: they are two different texts, which only the combined document holds together. A question about the decision does not expect the statement.
- A question to the month (a date on the first of a month) expects the equivalent documents of that month.

This holds for every kind. A `title` or `label` question expects the equivalent documents of the documents that carry the title. In `described+filters`, the `doc_types` filter admits the doc type of every expected document (`["A1", "B1"]` for a decision, `["A1", "A2", "B1"]` for a combined document): dropping these questions would have removed the whole `ecb` B1 cell from the kind. Such a question is tagged `equivalent-documents` as well as `several-expected`. Equivalent documents come on top of `--max-expected`: a combined document dated a day with two releases of type A1 expects four documents.

In the batch, 75 questions expect several documents; 72 of them because of equivalent documents (48 about a combined document, 12 about a decision, 12 about a statement). Without the rule each of these 72 expected one document only.

### `described+filters`: how many documents pass the filters

For every question of this kind the script counts the documents of the scope that pass its filters (the bank, the doc types and the date window, as the search applies them: a document without a date does not pass a date bound) and writes the count as a tag:

| Tag | Questions | |
|---|---|---|
| `filter-candidates:1` | 66 | the filters alone name the document |
| `filter-candidates:2-3` | 98 | |
| `filter-candidates:4-10` | 19 | |
| `filter-candidates:>10` | 27 | no date window: the 24 Bank of England documents and 3 Bank of Japan documents without a date |

For 183 of the 210 questions at most ten documents pass: with `k = 10` the expected document is among the first ten whatever the search does, and hit@10 and recall@10 are true by construction. **Only hit@1 and MRR are informative for this kind**, and for `filter-candidates:1` not even those. The kind is kept, because comparing its hit@1 with the one of `described` shows what the filters bring; it is kept out of the headline, the report marks it with `*`, and the per-tag lines `filter-candidates:` give the metrics for each count.

### What was skipped

Among the 237 documents looked at to draw 214: 22 `title` questions for a missing title and 1 for a title that is a file name; 20 `label` questions for a label spelling out a placeholder date; the descriptions of 21 documents whose date is shared by more than three documents of their cell; the descriptions of 3 documents without a usable title and without a date known to the month, and of 3 documents that only a catalogue label describes; the French description of 27 documents that only an English title describes.

## The analytical batch

`analytical_v2.jsonl`: 91 questions on what the documents say, for the same scope as the known-item batch (PDF documents of types A1, A2, A3, B1, E1, F1 of the central-bank corpus, six banks, all in English).

| | |
|---|---|
| Questions | 91. `an-01` to `an-20`: 20 questions and 9 variants (suffix `a`, tag `variant:easy`, for an easier wording: 6; suffix `b`, tag `variant:extra`, for one more question on the same documents: 3). `an-21` to `an-70`: 49 questions from 50 candidates, one dropped because its document is not in the scope, plus one abstention counterpart (`an-64b`, also `variant:extra`). `an-71` to `an-82`: 12 questions that the corpus cannot answer. |
| Expected behaviour | 72 `answer`, 4 `correct_premise`, 15 `abstain` |
| Language | 81 English, 10 French (the documents are English: the French questions measure cross-lingual retrieval) |
| Documents named | 424 distinct, all in the scope |
| Kinds | `analytical` (86, no filter) and `analytical+filters` (5, each also scored without its filters) |
| Requirements | 184 for the 76 scored questions: 90 essential documents and 94 either-of groups (in 37 questions) |

What the questions cover, by tag: chains over several meetings (`chain`), comparisons between banks (`comparative`), a detail in the middle of a long document (`buried-detail`), votes (`votes`), a cell of a projections table (`table`), counts over several documents (`aggregation`), a bank that the question does not name (`unnamed-bank`), a false premise (`false-premise`), a negative answer (`negation`), a document or a bank that the corpus does not hold (`out-of-corpus`; among the questions that abstain, `absent-bank` when the bank is not in the collection and `coverage-gap` when the bank is there and the period or the document is not), a fact about such a bank that another bank's document reports (`indirect-source`), a question with many documents of other banks on its subject (`distractors`), a keyword query (`short-query`) and a vague one (`vague`).

**How they were checked, and what was not.** For every question the documents were found in the corpus manifests, fetched from the catalogue URL and read; the expected answer was corrected wherever a document said otherwise; one verbatim quote per expected document was matched word for word against the extracted text by a script. A document that could not be read (a catalogue URL that returns only a contents page) is never an expected document. Questions whose document is not in the scope were rewritten onto what the scope holds, turned into `abstain` questions, or dropped. Every member of an either-of group was checked by a pattern for the fact of its group in the text of the document.

- **The texts read are those of the live catalogue URLs, not the ingested copies.** Nothing compares them with what the collection holds (no checksum, no look at the stored chunks): a page that changed since the ingestion, or a part of a page the ingestion did not keep (the questions and answers of an ECB press-conference page, on which `an-44` and `an-47` rest, and one of the two documents of the ECB part of `an-31`), would not have been noticed.
- **One verification pass, and an independent second check of 16 questions**, for `an-01` to `an-70`. The other 63 were read once. `an-71` to `an-82` were checked for an absence, which is another matter (below).

**Essential, either-of and acceptable.** `essential_ids` are the documents without which the question cannot be answered. Where several documents carry the same fact, the question has an either-of group instead (`essential_any_of`), and any one of them satisfies the requirement: the press release, the statement and the combined document of one ECB meeting day, when the fact is in the decision text (not when it is only in the questions and answers of the statement page); an FOMC statement and the minutes that repeat it; Bank of England minutes and the Monetary Policy Report of the same month; a projections table and its reprint; the consecutive Bank of Japan minutes that record the same proposal (`an-67`: one of the eleven of 2015 and one of the eight of 2016). `acceptable_ids` are relevant without being needed. A document belongs to one group only, so a requirement set can say neither "these two documents, or that one" nor "this document for two requirements". Four questions are therefore stricter than the documents would allow, and under-credit one kind of retrieval each:

- `an-26`: the count of eight cuts is stated by one account and can also be made from the eight decisions, which are acceptable; a system that returns the eight decisions without the account is not credited for the count.
- `an-48`: a projections table also repeats the row of the previous round; a system that returns one table is credited for its own round only, not for the previous one it also shows.
- `an-38`: the press-conference transcript of 22 March 2023 carries both Federal Reserve requirements (the March wording, and the phrase the Committee dropped: "we no longer state that we anticipate that ongoing rate increases will be appropriate"), and counts for the dropped phrase only; a system that returns it without the statement or the minutes of 22 March gets 3 of 4 where the documents would allow 4. The other expressible structure, the transcript grouped with the statement and the minutes of March, would under-credit every system that returns the March documents without the statement of 1 February, which is the more likely retrieval.
- `an-29`: the July 2019 minutes have the words "mid-cycle adjustment" as the view of most participants and count for the July meeting; the characterisation by the Chair is credited to the two transcripts that attribute it to him (31 July and 18 September 2019), so a system that returns the minutes without either transcript is not credited for it.

On the scored questions `must_not_include` holds only statements the documents show to be false; where the corpus cannot tell (what else the announcement of a bank outside the corpus said, for `an-55`), the list is empty. The rationales say what a document states and which neighbouring document does not; they claim no exclusivity over the whole corpus, which was not read in full.

One question (`an-23`) has 12 requirements: run with `--k` above 10 to give it a chance.

**The questions that abstain** (`an-25`, `an-53`, `an-64b`, and `an-71` to `an-82`) list context documents and no essential one. They are reported apart and judged by nobody here: the harness shows what the search returns for them and at what score; whether a client then declines to answer is not measured. Two questions about a bank the corpus does not hold are *not* abstentions: `an-55` and `an-57` (`indirect-source`) are answered by a sentence of another bank's minutes.

**`an-71` to `an-82`: twelve questions with a true answer that the collection does not carry.** "The bank is not in the collection" is not enough to make an abstention: `an-55` and `an-57` are there because another bank's minutes report the fact. These twelve therefore ask for what a neighbouring document does not restate (a named dissenter, a vote split, the argument of one paragraph, one cell of a table), and each was checked for its absence.

| | Questions | What is missing |
|---|---|---|
| `absent-bank` (5) | `an-71` Riksbank, `an-72` Czech National Bank, `an-73` Bank of Korea, `an-74` Swiss National Bank (French), `an-81` Norges Bank | The bank: reservations and votes of a decision, the level of a technical parameter, the guidance given with a decision. |
| `coverage-gap` (7) | `an-75` and `an-76` (French) Bank of Japan, `an-77` Bank of Canada, `an-78`, `an-79` and `an-82` Bank of England, `an-80` Federal Reserve | A period or a document of a bank that is there: the Bank of Japan minutes of 2018 to September 2025, the Bank of Canada summaries before March 2025, the Monetary Policy Committee minutes of 2003 and of March to December 2005, the Bank of England reports before 2018, the FOMC minutes before February 1994. |

- **Each has a true answer, and the rationale gives it** with the address of its source on the site of the bank that published it, read when the question was written, so that the question can be checked to be fair. The premise of each is true; for `an-76` and `an-78` it is also precise (a decision the collection reports, a vote split the question states), and the right response is still to abstain, not to correct the question. `must_not_include` holds what an answer from memory would say (the names, the split) and, for four questions, a false statement an invented answer would make (a unanimous decision, a Governor in the majority).
- **Seven name a bank of the collection** (`coverage-gap`) and look answerable: the same kind of document is there for the neighbouring months or years. Ten of the fifteen questions that abstain are about a vote.
- **How the absence was checked, and on what.** For each question: (1) the manifests, for a row of the scope whose title, address or date could be the source: none (the scope holds the Monetary Policy Committee minutes of January and February 2005 only for that year, none of 2003, and no Bank of Japan minutes dated 2018 or 2024); (2) a search for the names, figures and wordings of the answer in the extracted text of **773 of the 2,566 documents** of the scope, read from the catalogue addresses: the 689 fetched to check `an-01` to `an-70`, and 84 fetched for these questions; (3) among them, for each question, the minutes, accounts, reports and press-conference documents of the six banks for the month of the fact and the two that follow, and the minutes on each side of a gap: 177 documents in all, between 2 (`an-80`: the FOMC minutes of the collection start in February 1994, so only those of 1994 could recall the vote) and 24 per question. None carries an answer. The 773 texts include all 38 Bank of Japan minutes and all 10 Bank of Canada summaries of the scope, 74 of the 305 Bank of England minutes and 67 of the 262 FOMC minutes.
- **What that does not show.** The other 1,793 documents were not read: a later document that returns to one of these facts (a retrospective box in a report, an answer at a press conference years afterwards) would not have been seen. The catalogue addresses of the Reserve Bank of Australia's Statement on Monetary Policy and of the Bank of Canada reports before 2023 return a contents page, so what the collection holds of them is not what was searched. As for the other questions, the texts are those of the live addresses, not the ingested copies. A question found to be answerable when the batch is run against the collection should become an `indirect-source` question, like `an-55` and `an-57`.

**What is not scored.** `must_include` and `must_not_include` describe a correct answer, in English: they are for a scorer of generated answers, which is not part of this harness, and a literal string comparison would be too strict for them. The `must_include` items are mostly in the form the documents use (`"1/2 percentage point"`, `"6-1/2 percent"`): 337 of the 342 items of the scored questions are found word for word in the extracted text of a required document, and two more are there across a line break. Some are derived all the same: a count the answer must give (`"eight"`, `"five"`), the month of a meeting (`"December 2021"`, `"October 2025"`), the bank a second-hand fact is attributed to. For the fifteen questions that abstain they are a plain statement that the corpus does not hold the information.

**Which numbers to read for this batch.**

- `all_essential@k`, the share of questions with everything they need among the first `k`, and `essential_recall@k`, per question and pooled. They are the numbers about the answer; read them with their intervals, which resample the questions.
- The **per-question** lines of the first tables (the `Per question` table, `overall (pooled)`), not the per-document ones: 18 of the 76 scored questions expect more than ten documents (up to 44), and each of them weighs as many times in a per-document mean. For the same 18 questions recall@10 cannot reach 1.
- hit@1 and hit@k with care: they are satisfied by any expected document, and a question with dozens of acceptable documents (every decision of a three-year cycle, twenty-four sets of minutes) is hit by almost any document of the right bank and period. A high hit@1 on these says little; `all_essential@k` is the demanding number.
- The variants apart, with the tags `variant:easy` and `variant:extra`: a base question and its variant share their documents, so they are not two observations. More generally 65 pairs of scored questions share a required document (8 of them a question and its own variant), and 63 of the 76 questions are in at least one such pair: the results are correlated, and the intervals, which treat the questions as independent, are narrower than they should be.

**Limits of the batch, as measured on the file.**

- **The questions help the search more than a user would.** 87 of the 91 name the bank or its committee (all but the four tagged `unnamed-bank`), and 65 give a year; 64 do both. One question is a keyword query (`an-62`) and one is vague (`an-63`); the other 89 are full sentences. 10 are French.
- **Coverage of the scope.** Of the 18 (bank, doc type) cells of the scope, two never appear in a requirement: the Statement on Monetary Policy of the Reserve Bank of Australia (`au` E1, whose catalogue URL returns a contents page) and the ECB staff projections (`ecb` E1). The requirements of 32 questions include a Federal Reserve document, 27 an ECB one, 17 a Bank of England one, 10 each for the Bank of Japan and the Reserve Bank of Australia, and 3 for the Bank of Canada.
- **Years.** The scope runs from 1994 to 2026. No requirement is dated before 2003, nor in 2004, 2005, 2006, 2007 or 2010; 2017 and 2026 appear in one question each. The most asked years are 2022 (12 questions), 2016 and 2020 (11 each).
- **76 scored questions are few.** An interval on 71 questions is some twenty points wide, and the correlation above makes it optimistic. The fifteen questions that abstain are fewer still, and ten of them are about a vote: enough to look at the scores the search gives them, not to set a threshold.

**Limits of the scope that the questions met.** No Bank of Canada rate-decision release, and its summaries of deliberations only from March 2025; Bank of Japan minutes for 2015–2017 and from October 2025 only; no Bank of England minutes of 2003 and only those of January and February for 2005, and its reports from February 2018 only (those of 2018, 2019 and January 2020 are catalogued with the minutes; the rows typed as reports start in November 2020); FOMC minutes from February 1994; the ECB's combined "decisions and statement" documents carry no questions and answers (the statement pages do); Bank of England documents are dated 1 January of their year in the manifests. Questions that needed a missing document say so in their `rationale`.

## The fixtures

`fixtures/corpus.json` describes 13 documents with their metadata and the text of their chunks, and the model the fixture is meant for: the deterministic `FakeEmbedder` of the engine, with empty prefixes. That embedder has no semantics: two similar texts are not close. The questions of `fixtures/questions.jsonl` are therefore built so that their answer does not depend on meaning: the text of a question is the text of a chunk (same vector, first place), or its filters leave only the expected document.

Every part of the report has a known value there. One question expects a document that is not in the corpus, one a document that a date filter leaves out, one is ranked second on purpose. One document, the combined decisions and statement, repeats a chunk of two others: a question that expects only one of the two documents holding the chunk is second by the order of equal scores (`tied_at_decision`), and the same question expecting both is first. Two questions ask for the same document, so that the per-document means differ from the means over the questions; every question carries several tags; the kind `verbatim` is asked without filters and makes the headline, the kinds `filtered` and `second` are not. `tests/eval_fixture.py` loads the corpus into a collection; `tests/test_eval.py` checks the metrics.

`fixtures/questions_v2.jsonl` asks thirteen more questions of the same corpus, with the fields of the second version, and again every number is known. One question has an acceptable document first and its essential document second, by the order of equal scores: a hit at rank 1. One names all thirteen documents as essential: with `k = 10` ten of them are among the first ten, whichever they are, and the question is out of reach. Filters leave an essential document out for two questions, and the essential one out while the acceptable one is first for a third. Two questions are scored again without their filters, and one of them is second without them. One states only `expected_doc_ids`, so the essential metrics leave it out. Two have an either-of group: one needs the statement or the combined document, which hold the same chunk; the other needs the decision and one of these two, and its filters leave the combined document alone, so the group is satisfied and the decision is not. Two correct a premise. Three abstain: one lists a context document, which holds the text of the question and comes first without being a hit; two list none, one of which is the text of a chunk (its first document and score are known) while the filter of the other leaves nothing. `tests/test_eval_v2.py` checks the values, and `tests/test_eval_check.py` the counts `rag-engine eval-check` prints for both files.
