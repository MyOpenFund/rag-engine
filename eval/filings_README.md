# The filings evaluation (passage level)

Document-level recall saturates on a corpus of about twenty 10-Ks (every search finds the right
filing), and chunk ids differ from one chunker to the next. This evaluation therefore grades
**passages**: a question carries short verbatim `expected_spans`, and a retrieved passage is a hit
when it contains one of them.

The first question set is `filings_v1.jsonl`: 48 questions (43 answering, 5 abstentions) on the 10-Ks
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
- **Spans come from prose only.** No v1 span is a table cell, so how v1 and v2 split tables is not
  measured by v1 (v2 repeats the header row of a table in every piece, but in company-corpus Markdown that
  row is blank: the years are in a data row only the first piece carries; see v2 below).
- **The v2 gain is not split between its causes.** v2 changes two things at once, the token cap
  (480 tokens, so no chunk is truncated by the 512-token window of e5-base) and the title in the
  context (embedded and indexed for BM25). No run isolates one from the other.
- **Year confusion penalises v1 by construction.** The questions carry no `filters`, so the issuer
  and the fiscal year must be found from the text. A v1 chunk carries neither (no context), while a
  v2 chunk carries the title, which names the company and the period. Part of the v1 -> v2 gap is
  therefore this, not the quality of the cut.

## Version 2: `filings_v2.jsonl` (pending measurement)

> **Status: pending measurement.** v2 has not been run with the ONNX model yet; it has no published
> number. v1 stays in the repository for history and its numbers above stay as published: v2 does not
> replace them, it is a different set (a v2 number is never compared with a v1 number as if one set had
> been scored twice).

v1 was challenged question by question (`filings_v2_AUDIT.md`: 36 kept, 9 rewritten, 3 dropped) and
extended with harder and more varied questions; an adversarial review of that first v2 led to v2.1, and a
second review to v2.2 (the changes tables of the audit). **92 questions: 85 answering, 7 abstentions.**

### Why a second set

The reviews of v1 found: spans from prose only (tables not measured); every answering question of
similar difficulty and form (one sentence, one filing, English); year confusion as the main failure;
ten questions missing in every hybrid run. The audit added a defect of the set itself: **four questions
asked for a figure the next 10-K restates** (a balance at year end, last year's capex) and expected only
the first filing, so a right passage from the next filing scored as a miss; one more did not pin a year
the 10-Ks of three years answer. Part of v1's "year confusion" misses can be such false misses. No v1
span is lost by the chunker (every one is held whole by a v1 and a v2 chunk), so the persistent misses
are retrieval misses.

v2.1 went further on the same defect: **every answering question was searched against all 20 filings
for other passages that answer it** (the same figure or fact in another sentence of the same 10-K, or
restated by another year's 10-K), and those that answer were added as spans, with their filing in
`expected_doc_ids`. The audit records, per question, how many were added: 21 questions gained 28
alternative spans (3 of them bringing a filing that was missing); a question with none added has no other
answering passage in the 20 filings, or only partial ones (the audit names those).

v2.2 ran the search again with the table rows in it, under one rule: **a self-labelled table row that
answers a prose question counts as an alternative**. A row is self-labelled when its own label names
what is asked (`Purchases of property and equipment`, `Payments for dividends and dividend equivalents`,
`April 2022 Notes issuance of $12.8 billion`) and its year comes from the table's year row (or from the
label itself, or from the filing's balance date); a `Total` row, a column of a table, or a row that only
includes the figure asked (Meta's `Proceeds from issuance of long-term debt, net` for its May 2023 notes)
does not count. A later 10-K whose three-year statement still shows the year asked answers too, and is
expected. 18 questions gained table rows this way (tag `table-alt`; `amzn-fy22-notes` among them, whose
debt-table rows name the 2022 issuances in all four Amazon 10-Ks), and the prose passages the second
review found were added (Alphabet's first dividend "first paid in June 2024", in both twins; its 2025
infrastructure expectation in the Executive Overview; Meta's $1.34 billion data center charge, now the
primary span; Amazon's 2023 capex in its FY2023 10-K).

### Difficulty: retrieval difficulty

Each question carries `difficulty`: `easy`, `medium` or `hard`. It rates how hard the **passage is to
retrieve**, not how hard the answer is to compute or to read (a negation, an arithmetic step, a note
far in the 10-K are recorded in tags, not in the label). Five factors are measured on each question
(`scripts/check_filings_eval.py --difficulty` measures them, prints them and fails on a label they
contradict):

- **Lexical overlap** (`OV`): the share of the question's content words (stop words, issuer names and
  years left out, light stemming) found within 300 characters of the best answering span. Bands:
  high >= 0.6, mid 0.35 to 0.6, **low < 0.35**. A French question is read the same way, on the words it
  shares with the English text (names, figures, cognates): cross-lingual difficulty shows as low
  overlap, unless a shared name carries the match (*Scale AI*, *OpenAI*).
- **Year-template exposure** (`YT`): the number of other filings of the issuers of the expected filings
  (not expected themselves) that repeat the wording of one of the question's spans with other figures: at
  least half of that span's four-word shingles, figures masked. `YT >= 1` is a factor: a neighbouring
  year's 10-K says the same thing with other numbers. (v2.1 read the first span only; v2.2 reads every
  span, which moved six labels, listed in the audit.)
- **Table** (`TB`): the answer is a table row only (tag `table`). Every 10-K repeats its tables year
  after year, so a table counts as one factor and its year-template exposure is not counted again. A
  question whose prose answers too and whose table rows are alternatives (tag `table-alt`) is not a
  table question.
- **No year** (`NY`): the question names no year ("latest") while the issuer's other 10-Ks carry the
  same template.
- **Multi-filing "all"** (`MF`): `span_match: "all"`, two facts that two filings (or two issuers) give.
- *Far from Item 7* (a note, Item 1A, Oracle's Item 15) is recorded (`section:` tags) but **not
  counted**: the retriever ranks chunks, it does not see where in the filing a chunk sits.

**hard**: `MF`, or two factors or more among low `OV`, `YT`, `TB`, `NY`. **medium**: one factor, or mid
overlap. **easy**: none, and high overlap. Abstentions are rated apart: easy when the company is not in
the corpus, medium when the period or figure is plausible but absent, hard when a real-world fact has
an in-corpus distractor (Amazon's Anthropic notes for Alphabet; SoftBank in Oracle's 10-K for Stargate;
other issuers' dividends for Amazon). The script does not rate abstentions: their seven labels were
checked by hand against that rule (the audit lists them).

v2 labels mixed retrieval with answer difficulty (negation, numeric reasoning, notes); v2.1 re-checked
every label against the criteria above (the audit lists each change and each question's factors).

### Mix

| | easy | medium | hard | all |
|---|---|---|---|---|
| answering | 15 | 52 | 18 | 85 |
| abstention | 2 | 2 | 3 | 7 |
| **all** | **17 (18 %)** | **54 (59 %)** | **21 (23 %)** | **92** |
| of which French | 1 | 3 | 6 | 10 |
| of which a table answer (`table`) | 0 | 8 | 4 | 12 |
| of which table rows as alternatives (`table-alt`) | 3 | 10 | 5 | 18 |
| of which `span_match: "all"` | 0 | 0 | 7 | 7 |
| of which a French/English twin | 1 | 4 | 3 | 8 (4 pairs) |
| of which `chunk-loss` | 2 | 0 | 0 | 2 |
| of which `chunk-split` | 0 | 1 | 1 | 2 |

Medium is above the 45 % aimed at: the v1 questions kept as they are (the owner's rule: keep what is not
defective) are mostly year-template questions, one factor each; reading every span for the year template
(v2.2) moved four more from easy to medium. The labels are as measured, `chunk-loss` and `chunk-split`
questions included (their difficulty is that of retrieving the passage, were it whole).

- **Kinds**: ai-capex 19, capex 14, financing 10, debt 10, liquidity 9, leases 9, buybacks-dividends 6,
  accounting-policy 5 (useful lives and depreciation of servers), risk 2, segment-capex 1, abstention 7.
- **Issuers** (a question may name two): Meta 19, Alphabet 18, Amazon 18, Microsoft 17, Oracle 15.
- **Fiscal years** (tags: the years asked, not the filings that restate them; a question may name
  several): 2022: 12, 2023: 25, 2024: 30, 2025: 46. Every one of the 20 filings is asked about. FY2025
  still weighs most (41 % of the year tags, 46 % in v2): it is what users ask about first and no later
  10-K restates it; v2.1 dropped three FY2025 questions and added mostly 2022 to 2024 ones.
- **Tags** for slicing the results by hand: `year-confusion` 44, `restated` 19, `table-alt` 18, `note` 13,
  `table` 12, `paraphrase` 10, `french` 10, `twin` 8, `numeric` 7, `multi-span` 7, `multi-filing` 4,
  `negation` 3, `risk-factor` 2, `chunk-loss` 2, `chunk-split` 2, `no-year` 1, and
  `section:item1a|item7|item8|outside` from where the first span sits in the sidecar sections
  (`outside`: Oracle's statements and notes, in Item 15, past the sidecar's Item 8 stub).

### Twins: the same fact in both languages

Four facts are asked twice, in French and in English, with the same spans and filings (tag `twin`):
Microsoft's OpenAI funding commitments, Amazon's 2023 Anthropic investment, Alphabet's first dividend,
Meta's servers and network assets at the end of 2023 (a table row). The by-language breakdown compares
two groups that differ in their questions as well as in their language; the twins are the pairs where
only the language differs, so comparing a twin's two ranks says what the language costs.

### Several filings, several spans

- **`expected_doc_ids` lists every filing that answers** (30 answering questions): a restated figure
  (the next 10-K's sentence or table row, each with its own span), an event several 10-Ks describe, a
  risk factor repeated from one year to the next. A passage of another filing is still no hit.
- **Several spans are alternatives** (`span_match: "any"`, the default and v1's behaviour): any one
  span held by a passage of an expected filing is a hit.
- **`span_match: "all"`** (7 questions, all hard): the answer needs two or three facts (2024 and 2025,
  Amazon and Meta). Each fact is a **group** of spans, written as a list inside `expected_spans`; a
  group is found when a passage among the first `k` (from an expected filing) holds one of its spans,
  and the rank is the position where the last group is first found. One group missing is a miss.
  Groups let a fact have alternatives (Meta's 2024 dividends are given in three sentences) without "all"
  then requiring every alternative. A lone string is a group of one, so a v1 or v2 file scores exactly
  as before (checked on random runs against the previous scorer).

### What v2 measures and what it does not

- It measures, besides v1's prose look-ups: retrieval of **table rows** (property and equipment,
  free-cash-flow, segment and cash-flow tables); **two-filing** retrieval within the first `k`; risk
  factors and notes; negations; French questions and their English twins; and, through the tags and the
  breakdown, which kind of question fails.
- **A table span carries its years: `year row [...] answer row`.** In company-corpus Markdown the header
  row of a table is blank (`|  |  |  |`): the years sit in a data row below it. v2 repeats the (blank)
  header row in every piece of a split table, so a piece cut below the year row holds figures without
  years and cannot answer "in 2025". A table span is therefore written as two fragments, the year row
  and the answer row, joined by `[...]`: **one passage must hold both, in this order**, whatever rows lie
  between them (v2.1 spelled out every row in between, verbatim, so a span encoded today's cut; the gap
  span does not). A self-labelled row whose label carries its date (`August 2022 Notes`, `April 2022 Notes
  issuance of $12.8 billion`) or a balance of the filing's date (`Construction commitments`) needs no
  year row. The one row kept without its years is the lone Consolidated row of `amzn-fy25-aws-share`, an
  alternative in the second group next to its gap span: v2 cuts that row from its years, and the first
  group (the AWS row with the years) gives the column order.
- **Which table questions, and what v2 cannot answer.** The ten table questions of v2.1 were chosen among
  rows a v2 chunk holds with their year row: by themselves they would hide how v2 cuts tables. v2.2
  therefore restores the two v2.1 had dropped, tagged `chunk-split` (`msft-fy25-capex-table-fr`, the
  cash-flow statement's additions to property and equipment, and `amzn-fy25-leases-not-commenced`, the
  commitments table): no prose gives their figure and no v2 chunk holds their row with its years, so
  they miss under v2 by construction, as the `chunk-loss` ones do. The table rows added as alternatives
  show the same thing at scale: of the 56 table-row spans of the set, every one is held with its year
  row by a v1 chunk (whose text is longer, though e5 embeds only its first 512 tokens), and 24 by no v2
  chunk, section or document cut (19 of the 36 `table-alt` rows: most rows of the long cash-flow and
  equity statements). Those questions stay reachable in v2 through their prose spans.
- It does not measure the answer: a span in the passages is necessary, not sufficient (the numeric
  questions also need the arithmetic; a negation needs the reader to read it).
- **Abstentions are reported apart, without a metric.** The runner searches the 7 abstention questions
  and writes, for each row, each one's number of passages and top score in the JSON (`report` ›
  `abstentions`); the printed table has no abstention column, and no hit rate or threshold is computed,
  since the fused scores are not calibrated (v1's five spanned 0.029 to 0.033). An abstention metric
  needs a threshold policy first; until then the abstentions only show what a system would be shown.
- **Groups are small** (21 hard, 10 French): the runner prints them without an interval; read them as
  indications, and only large differences as signals. The questions still cluster on 20 filings of 5
  issuers (see the v1 caveats).
- Several questions reuse spans of others (`meta-capex-outlook-compare` and `meta-fy24-guidance`, the
  twins, `amzn-capex-2023-2025-fr` and `amzn-fy23-capex`): they are correlated.
- **Two questions measure text loss, on purpose** (tag `chunk-loss`). With the section cut, some text
  is in no chunk: when a section piece starts with `---` (a page rule of the Markdown), the start of the
  piece is parsed as YAML front matter and dropped, up to the next `---`. Amazon's Item 7 opening (in all
  four Amazon 10-Ks) and Alphabet's FY2025 Item 7 trends and Item 1A leasing risk are lost that way.
  `amzn-fcf-drivers` and `googl-fy25-ai-infra-costs` ask about that text: they miss with the section cut
  until the parsing is fixed, and are reachable with the document cut. Their labels say how hard the
  passage would be to retrieve were it whole (both easy), so they sit in the easy group:
  **`--exclude-tag chunk-loss,chunk-split`** makes the runner print the headline table a second time
  without these four questions (and write it as `excluding` in the JSON), so every row is read with and
  without them. Every other answering question has a span held whole by a chunk of each profile and cut
  (`--reachable`, below).

### Checking and running v2

```
python scripts/check_filings_eval.py eval/filings_v2.jsonl FILINGS_DIR --require-difficulty --difficulty --report
python scripts/check_filings_eval.py eval/filings_v2.jsonl FILINGS_DIR --reachable v1,v2 --reach-cuts section,document --model-cache MODELS
python scripts/filings_eval.py FILINGS_DIR eval/filings_v2.jsonl --backend onnx --cuts document,section \
    --exclude-tag chunk-loss,chunk-split --json out.json
```

`--reachable` chunks the expected filings with each profile and cut, as the runner does, and reports a
question none of whose spans (with `"all"`: none of one group's spans) is held whole by a chunk (a
`[...]` span: all its fragments by one chunk); a `chunk-loss` or `chunk-split` question is exempt, and a
note says when it has become reachable. The runner prints, under
the main table, the same metrics **by difficulty** and **by language** for every row, and writes them as
`by_difficulty` / `by_language` in the JSON.

## Question format

JSON Lines, the format of `README.md` (version 2) plus three optional fields:

| Field | Meaning |
|---|---|
| `expected_spans` | A list of short, distinct, verbatim texts of the filings that answer the question (v1: 1 to 3; v2.2: up to 10, alternatives included). With `"all"`, an item may be a list of strings: a group of alternatives for one fact. A span may hold `[...]`: its fragments (a table's year row, then its answer row) must all be held by one passage, in order. |
| `span_match` | `"any"` (default: one span is enough, several spans are alternatives) or `"all"` (every item, span or group, is needed, at least two; a group is found by any one of its spans). |
| `difficulty` | `"easy"`, `"medium"` or `"hard"` (defined in the v2 section); a label the runner groups by. |

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
directory; an expected filing that holds none of the question's spans; with `--require-difficulty`, a
question without `difficulty`; an abstention tag `absent:<term>` whose term occurs as a whole word in
some filing, or `absent@<issuer>:<term>` in some filing of that issuer; a span shorter than `--min-chars` once normalised or without a letter (no dash-only or
tiny span); a span found in none of the question's filings, or more than `--max-per-filing` times
in one; a span also found in a filing the question does not expect; with `--reachable`, a question no
chunk can answer (below). In v1, spans are taken from prose sentences, not from tables (whose cells the
chunker may split); v2 adds table spans, which since v2.1 run verbatim from the table's year row to the
answer row (the header row is blank in this Markdown); since v2.2 they are gap spans, `year row [...]
answer row`, found in a filing where the answer row follows the year row within `--gap-window`
characters (3000 by default, about a v2 chunk), and held by a chunk when one chunk holds both. A span
names a
figure or a fact that only its filing states (a 10-K repeats the previous year's figures, so a span such
as "$75.5 billion" alone would also be in the next filing); when the next filing restates the answer in
its own words or row, that filing is expected too, with its own span (v2).

## Scoring

`span_hit(text, spans)`: both sides are case-folded and every run of white space becomes one space;
the passage is a hit if it contains any span (a span holding `[...]`: if it contains each fragment, in
order). For each question the *rank* is the position of the
first hit among the `k` passages returned (`k = 10`). Then:

- hit@5, hit@10: share of questions with a hit within the first 5 / 10 passages;
- MRR@10: mean of 1/rank (0 past 10 or on a miss);
- each with a 95 % percentile interval by resampling the questions (1000 draws, fixed seed), given
  from 20 questions up (both are parameters of `evaluate_passages`);
- abstentions: searched and reported apart, never in a metric;
- with `span_match: "all"`, the rank is where the last of the groups is first found (a group is found by
  any one of its spans, from an expected filing; a lone span is a group of one), a miss when one group
  is never found;
- `PassageReport.groups("difficulty" | "language" | "kind")` splits a report; the runner prints the
  difficulty and language breakdowns, without intervals.

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
