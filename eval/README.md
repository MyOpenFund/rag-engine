# Evaluating the search

Question files for `rag-engine eval`, the script that builds the automatic one, and a small fixture for the tests.

| File | What it is |
|---|---|
| `known_item_v1.jsonl` | 797 automatic questions on the policy core of the central-bank corpus, built from the manifests by `make_known_item.py`. |
| `make_known_item.py` | Builds a known-item file from manifest files. Standard library only; not part of the installed package. |
| `fixtures/corpus.json`, `fixtures/questions.jsonl` | 13 small documents (32 chunks) and 13 questions with known answers. The test suite loads them into a collection and checks the exact metrics. |

To come: `analytical_v1.jsonl`, questions on the content of the documents, written and checked by hand, in the same format. The file does not exist yet; it is added once its questions are validated. Until then the only committed questions are the known-item ones, and they measure one thing: whether a document that a user can describe is found.

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
| `filters` | no | Filters applied to the search: an object with the field names of `rag_engine.search.SearchFilters` (`source_codes`, `doc_types`, `languages`, `doc_ids`, `date_from`, `date_to`, `corpus`) and its rules (lists, never a bare string; dates written `YYYY-MM-DD`). |
| `kind` | no | A label; the report gives the metrics per kind. A kind with at least one question that carries filters is a *filtered* kind (see the report). |
| `tags` | no | A list of labels; the report gives the metrics per tag, and a question counts for each of its tags. |
| `language`, `rationale` | no | Free text, kept and not used. |

The file is checked strictly when it is read: a duplicate `id`, an empty `expected_doc_ids`, an unknown field or filter, a bad date are errors that name the line. No other field is accepted, so that a misspelt `filters` is never ignored. A question must hold something to search for as it is written and once put in Unicode NFKD, the form the search embeds it in: a question made of an isolated accent (`´`) is refused when the file is read, with its line, instead of stopping the run at that question.

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

`--fail-under-hit-at-k 0.9` makes the command exit with code 1 when the **headline** hit@k is below 0.9 (the headline is defined below; it is not the pooled mean of all questions). `--bootstrap-resamples N` sets the number of resamples of the intervals (default 1000).

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

### Equal scores

Two chunks can have exactly the same score: the same text stored in two documents does. Qdrant returns such chunks in an order of its own, which can change from one request to the next and with the limit, so the rank of a document could change between two runs. The harness removes what it can of that: the chunks returned are put in one order (decreasing score, then `doc_id`, then `chunk_number`) before they are collapsed into documents. The same chunks therefore always give the same ranking.

The order among equal scores is still arbitrary. A question is marked `tied_at_decision` when its hit@1 or its hit@k hangs on it: an expected document has the same score as documents that are not expected, across rank 1 or rank `k`, and another order of the tied documents would change the answer. The report counts these questions; the JSON report marks each.

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

## The fixture

`fixtures/corpus.json` describes 13 documents with their metadata and the text of their chunks, and the model the fixture is meant for: the deterministic `FakeEmbedder` of the engine, with empty prefixes. That embedder has no semantics: two similar texts are not close. The questions of `fixtures/questions.jsonl` are therefore built so that their answer does not depend on meaning: the text of a question is the text of a chunk (same vector, first place), or its filters leave only the expected document.

Every part of the report has a known value there. One question expects a document that is not in the corpus, one a document that a date filter leaves out, one is ranked second on purpose. One document, the combined decisions and statement, repeats a chunk of two others: a question that expects only one of the two documents holding the chunk is second by the order of equal scores (`tied_at_decision`), and the same question expecting both is first. Two questions ask for the same document, so that the per-document means differ from the means over the questions; every question carries several tags; the kind `verbatim` is asked without filters and makes the headline, the kinds `filtered` and `second` are not. `tests/eval_fixture.py` loads the corpus into a collection; `tests/test_eval.py` checks the metrics.
