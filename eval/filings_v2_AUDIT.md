# Audit of `filings_v1.jsonl` for `filings_v2.jsonl`, and the v2.1 fixes

Every v1 question was read against its filing and checked mechanically (`scripts/check_filings_eval.py
--report`, and a chunking pass that confirms each span is held whole by at least one v1 and one v2
chunk of the section cut). The owner's rule: a question that is complementary stays as it is; only a
real defect leads to a rewrite or a drop.

**Summary: 48 questions — 36 kept, 9 rewritten, 3 dropped.**

| Decision | Count | Reasons |
|---|---|---|
| kept | 36 | sound; 5 of them only had their `rationale` completed (no change to question, filings or spans) |
| rewritten | 9 | 4 *restated next year*, 2 *span not answering*, 2 *lexical copy*, 1 *year not pinned* |
| dropped | 3 | duplicate intent (2 answering, 1 abstention) |

Defects found:

- **Restated next year (4).** The question asks for a year-end balance or a year's flow, and the next
  10-K restates it (`$94.6 billion and $75.5 billion as of June 30, 2025 and 2024`). v1 expected only the
  first filing, so a correct passage of the next filing scored as a **miss**: part of v1's "year
  confusion" misses may be right answers. v2 expects both filings, each with its own span.
- **Year not pinned (1).** `orcl-fy22-bridge` asks about the March 2022 bridge facility without naming a
  filing; the FY2023 and FY2024 10-Ks describe it too. v2 expects all three.
- **Span not answering (2).** Two "why" questions had a span that stopped at the amounts; the reason
  follows in the same sentence. v2 extends the span into the reason.
- **Lexical copy (2).** The question repeated the span's line item or qualifier word for word, which
  only a reader of the 10-K would write. v2 asks it as a user would.
- **Duplicate intent (3).** Three Microsoft "leases not yet commenced" questions of the same template,
  two Oracle RPO questions of the same template, three "company not in the corpus" abstentions.
- **No unreachable span.** Every v1 span is held whole by at least one chunk in both profiles, so v1's
  ten persistent misses are retrieval misses, not text lost by the chunker (see the README on a chunker
  text-loss finding that does affect other paragraphs).

"Wording" is the year-confusion exposure: in how many *other* filings the longest stretch of the span
between figures occurs (the same sentence with other numbers). It drives the medium rating.

| v1 id | Decision | v2 difficulty | Wording in other filings | Reason |
|---|---|---|---|---|
| msft-fy24-cash | rewritten | medium | 3 | Restated next year: the FY2025 10-K gives the June 30, 2024 balance; FY2025 added with its span. |
| msft-fy24-cfo | kept | medium | 2 | Year must be told apart from FY2023/FY2025 sentences of the same shape. |
| msft-fy24-investing | rewritten | medium | 1 | Span not answering: "why" asked, span stopped at the amounts; now runs into "acquisitions". |
| msft-fy25-financing | rewritten | medium | 1 | Span not answering: as above; now runs into "repayments of debt". |
| msft-fy25-capex-ai | kept | medium | 0 | Earlier years say "We expect capital expenditures to increase": near-identical wording. |
| msft-fy22-leases | kept | medium | 1 | Same sentence every year with other figures. |
| msft-fy24-leases | **dropped** | — | 1 | Duplicate intent: third Microsoft question of the same lease template (FY2022 and FY2025 kept; FY2023 asked in French in v2). |
| msft-fy25-leases | kept | medium | 1 | Same template, latest year. |
| msft-fy25-cp | kept | medium | 1 | Negation, year template. |
| googl-fy22-capex | rewritten | medium | 3 | Restated next year: the FY2023 10-K gives 2022 capex ($31.5 billion); FY2023 added. |
| googl-fy24-capex | kept | medium | 3 | The FY2025 10-K restates 2024 spend but not the "relative to 2024" expectation the question also asks: only FY2024 answers both; rationale completed. |
| googl-fy25-capex | kept | easy | 3 | The 2026 plan pins the year ("In 2026, we expect"). |
| googl-fy25-leases | kept | medium | 3 | Paraphrase and template. |
| googl-fy25-ppa | kept | easy | 0 | Unique event. |
| googl-fy25-notes | kept | easy | 0 | Unique wording. |
| googl-fy24-cp | rewritten | medium | 0 | Restated next year: FY2025 Note 6 gives the $2.3 billion of December 31, 2024; FY2025 added. |
| googl-fy23-cash | rewritten | medium | 3 | Lexical copy of the line item ("cash, cash equivalents, and short-term marketable securities"); asked as "cash and short-term investments". |
| googl-fy25-credit | kept | medium | 3 | Paraphrase; the facilities sentence recurs every year. |
| amzn-fy25-capex | kept | medium | 3 | Template "Cash capital expenditures were ..." every year. |
| amzn-fy23-capex | kept | hard | 3 | Numeric reasoning (rise or fall from two figures) plus year; the FY2024 10-K lacks 2022 so cannot answer; rationale completed. |
| amzn-fy25-anthropic | kept | easy | 0 | Anchored by "Anthropic" and the year in the span. |
| amzn-fy22-notes | kept | easy | 0 | Unique wording. |
| amzn-fy23-credit | kept | easy | 3 | The date "In November 2023" is in the span. |
| amzn-fy25-cp | kept | medium | 1 | The FY2024 10-K has the same sentence shape. |
| amzn-fy24-liquidity | rewritten | medium | 0 | Restated next year: the FY2025 10-K gives the December 31, 2024 balance; FY2025 added. |
| amzn-fy25-finance-leases | kept | medium | 3 | Template every year. |
| meta-fy22-capex | rewritten | medium | 3 | Lexical copy: the question carried the span's qualifier ("including principal payments on finance leases"). |
| meta-fy23-guidance | kept | medium | 3 | Every Meta 10-K has a guidance sentence. |
| meta-fy24-guidance | kept | medium | 0 | Same, other wording. |
| meta-fy25-guidance | kept | medium | 0 | Same; paraphrase. |
| meta-fy23-notes | kept | medium | 2 | Template "net proceeds from the issuance of fixed-rate senior unsecured notes". |
| meta-fy25-notes | kept | medium | 2 | Same template. |
| meta-fy24-leases | kept | medium | 1 | Template. |
| meta-fy25-venture | kept | easy | 0 | Unique (Louisiana). |
| meta-fy22-restructuring | kept | easy | 0 | Unique. |
| orcl-fy22-bridge | rewritten | medium | 2 | Year not pinned: the FY2023 and FY2024 10-Ks describe the March 2022 bridge facility too; both added with their span. |
| orcl-fy23-notes | kept | medium | 3 | Template "During fiscal 20xx, we issued ... of senior notes". |
| orcl-fy23-cerner | kept | easy | 0 | Unique. |
| orcl-fy25-capex | kept | easy | 0 | Unique wording. |
| orcl-fy25-notes | kept | easy | 0 | Unique wording. |
| orcl-fy25-rpo | kept | medium | 3 | Template every year. |
| orcl-fy24-rpo | **dropped** | — | 3 | Duplicate intent with orcl-fy25-rpo; its span lives on in the hard two-filing question `orcl-rpo-growth`. |
| orcl-fy25-termloan | kept | easy | 0 | Unique. |
| abs-01 | kept | medium | — | Future year; Oracle does report dividends for other years. Rationale completed, `absent:` tag added. |
| abs-02 | kept | easy | — | Company not in the corpus (`absent:nvidia` checked). |
| abs-03 | kept | easy | — | Company not in the corpus (Apple is named by Meta and Microsoft, never with its capex). Rationale completed. |
| abs-04 | **dropped** | — | — | Duplicate intent: third "company not in the corpus" abstention (and Tesla is named in Oracle's 10-Ks, Ellison's directorship). |
| abs-05 | kept | medium | — | No fiscal 2030 capex plan; "fiscal year 2030" occurs only as a lease commencement year. Rationale completed. |

Every v2 question carries a `difficulty`; v1 questions also gained tags (`year-confusion`,
`restated`, `negation`, `paraphrase`, `multi-filing`, `numeric`) and a `section:` tag derived from where
the span sits in the sidecar sections (Oracle's statements and notes sit in Item 15, past the sidecar's
Item 8 stub: `section:outside`).

## v2.1: fixes of the adversarial review

An adversarial review of v2 (probes against the 20 filings and the v1/v2 chunks of both cuts) asked
for changes. The owner's rule still holds: complementary v1 questions are kept; only a real defect leads
to a rewrite or a drop; harder questions are added. **v2 80 questions -> v2.1 90: 5 dropped, 15 new,
45 changed (spans, wording or label), 30 kept untouched.**

| Review item | What v2.1 does |
|---|---|
| C1 missing answering filings | `amzn-fy25-server-life-fr` + FY2024 (it gives the same reason; the v2 rationale said it did not); `amzn-fy23-anthropic-fr` + FY2024 (two sentences) and the FY2023 Item 8 sentence; `meta-fy22-first-notes` + FY2023 note and the FY2022 Item 7 sentence; `msft-activision-value` pinned to the January 2022 announcement (the FY2024 10-K's $75.4 billion is the closing price) with the Item 1 sentence added. |
| C2 alternatives, systematic | Every answering question was searched against all 20 filings: each figure of its spans, outside the spans, and the key phrases of non-numeric answers (negations, reasons, risks). 21 questions gained 28 alternative spans (count per question below); where nothing was added, the column says 0 and partial passages are named. The reviewer's list is covered (overview bullets of the Alphabet capex questions, Note 6 for `googl-fy24-cp` and `googl-fy25-credit`, the depreciation-effect sentence, Meta's proceeds and buyback sentences, Meta's Item 1A dividend sentence, Oracle's notes (3 and 2), Cerner, bridge note bullet, Amazon's Item 8 Anthropic notes, Meta's other 5.5-year passages). |
| C2 span groups | `expected_spans` items may be lists (groups of alternatives) with `span_match: "all"`: a group is found by any of its spans, "all" means every group. Implemented in `rag_engine.eval` (`Question.span_groups`, `_passage_rank`) with tests; a lone string is a group of one, and the reviewer's differential check (`diff_v1.py`, 40 random runs of v1 against the pre-v2 scorer) and the same check of the v2 file against the v2 scorer give identical ranks and metrics. |
| I1 table spans | The header row of a company-corpus table is blank; the years are a data row only the first piece of a split table carries. Every table span now runs verbatim from the year row to the answer row, and is held whole by a chunk of both profiles (section cut). Dropped: `msft-fy25-capex-table-fr` and `amzn-fy25-leases-not-commenced`, whose row is cut from its years in every v2 chunk (no v2 passage can answer, no prose gives the figure). Kept with a reason: the lone Consolidated row as the second group of `amzn-fy25-aws-share`. The README's "repeats the header row" claim is corrected. |
| I2 difficulty | The five relabels asked for are applied; difficulty is now defined as retrieval difficulty (README: overlap bands, year-template exposure, table, no year, multi-filing "all"; far-from-Item-7 recorded, not counted) and measured by `check_filings_eval.py --difficulty`, which fails on a contradicting label. Every label was re-checked: 21 of the 75 kept questions changed label (table below). Hard is back to 20 % with new questions: paraphrases with low overlap (`msft-fy22-construction-commitments`, `orcl-fy25-lease-commitments`, `orcl-fy23-fcf`), cross-lingual on a table (`meta-fy23-servers-fr`) and on a note across three filings (`msft-fy23-depreciation-fr`), a no-year "latest" question (`googl-latest-capex-outlook`), multi-filing "all" (`googl-vs-meta-fy24-capex`, `meta-dividends-2024-2025`). Test bounds in `tests/test_eval_spans.py` adjusted (easy 18-35 %, medium 45-62 %, hard 19-30 %). |
| I3 twins | Four French/English pairs with the same spans and filings (tag `twin`): OpenAI commitments, Amazon's 2023 Anthropic investment, Alphabet's first dividend, Meta's servers and network assets (table). |
| I4 two-part questions | `googl-fy25-capex` split (amount here; the 2026 plan is `googl-latest-capex-outlook`); `googl-fy24-capex` asks the expectation only; `meta-fy25-venture` asks the maximum exposure only (paraphrased). |
| I5 Oracle risk | `orcl-datacenter-capacity-risk` pinned to FY2025, with the heading, two body sentences and the summary bullet (all new in FY2025). |
| I6 redundancy, year balance | Dropped `meta-fy25-guidance`, `amzn-fy25-capex` (their spans live on in the two-filing questions) and `amzn-fy24-server-retire`. Year tags: 2025 from 46 % to 40 %; new questions are mostly on 2022 to 2024. |
| I7 additions | Oracle's $43.4 billion lease commitments; a Microsoft buyback question; Amazon's purchase obligations (the footnote: the total is a header-less table row in v2); a no-year "latest" question; Microsoft-OpenAI in English; a depreciation total (Microsoft, French); free cash flow of Oracle (paraphrase). |
| I8 abstentions | README: the runner reports abstentions apart (each one's top score in the JSON), no threshold metric yet. `abs-03` and `abs-05` tags made concrete search terms, verified absent. |
| Chunk loss | `amzn-fcf-drivers` (Item 7 opening, all four Amazon 10-Ks) and `googl-fy25-ai-infra-costs` (FY2025 Item 7 trends), tag `chunk-loss`: unreachable with the section cut until the front-matter parsing of a section starting with `---` is fixed, reachable with the document cut. README wording fixed (the cause is that parsing, not the chunker in general). |
| Minors | `meta-fy25-capex-growth` rationale; French calques ("durée d'utilité", "dépenses d'investissement"); the reachability check committed as `check_filings_eval.py --reachable` (spans held whole by a chunk, groups honoured, `chunk-loss` exempt) with tests. |

Checks run on the final file: `check_filings_eval.py --require-difficulty --difficulty` 0 problems;
`--reachable v1,v2 --reach-cuts section,document` 0 problems (the two `chunk-loss` questions are
unreachable with the section cut and reachable with the document cut, as intended; one alternative of
`amzn-fy25-anthropic`, the Q2 + Q4 notes passage, is held whole by v1 chunks only).

Per question. "Alternatives" counts spans added by C1/C2 (not the table spans rewritten by I1). Factors:
`OV` overlap and band, `YT` year-template exposure (not counted for a table), `TB` table, `NY` no year,
`MF` span_match all.

| id | v2.1 | alternatives | difficulty v2 → v2.1 | factors | change |
|---|---|---|---|---|---|
| msft-fy24-cash | kept | 0 | medium → medium | OV 0.83 high, YT 2 |  |
| msft-fy24-cfo | kept | 0 | medium → medium | OV 0.50 mid, YT 3 |  |
| msft-fy24-investing | kept | 0 | medium → medium | OV 0.50 mid, YT 3 |  |
| msft-fy25-financing | kept | 0 | medium → medium | OV 0.80 high, YT 3 |  |
| msft-fy25-capex-ai | kept | 0 | medium → medium | OV 0.62 high, YT 2 |  |
| msft-fy22-leases | kept | 0 | medium → medium | OV 0.82 high, YT 3 |  |
| msft-fy25-leases | kept | 0 | medium → medium | OV 0.89 high, YT 3 |  |
| msft-fy25-cp | kept | 0 | medium → medium | OV 1.00 high, YT 1 |  |
| googl-fy22-capex | changed | 1 | medium → medium | OV 0.67 high, YT 2 | C2: +1 alternative (FY2022 Item 7 overview bullet). |
| googl-fy24-capex | changed | 0 | medium → medium | OV 0.50 mid, YT 2 | I4: two-part question reduced to the expectation (the 2024 amount is asked by googl-vs-meta-fy24-capex); no alternative passage. |
| googl-fy25-capex | changed | 1 | easy → medium | OV 0.67 high, YT 3 | I4: two-part question split: the 2026 plan moved to googl-latest-capex-outlook (no year). C2: +1 alternative (overview bullet). I2: relabelled easy -> medium by the retrieval criteria. |
| googl-fy25-leases | kept | 0 | medium → medium | OV 0.89 high, YT 3 |  |
| googl-fy25-ppa | kept | 0 | easy → easy | OV 0.88 high |  |
| googl-fy25-notes | kept | 0 | easy → easy | OV 0.80 high |  |
| googl-fy24-cp | changed | 1 | medium → easy | OV 1.00 high | C2: +1 alternative (FY2024 Note 6, Debt). I2: relabelled medium -> easy by the retrieval criteria. |
| googl-fy23-cash | kept | 0 | medium → medium | OV 0.60 high, YT 3 |  |
| googl-fy25-credit | changed | 1 | medium → medium | OV 0.67 high, YT 3 | C2: +1 alternative (FY2025 Note 6, Credit Facility). |
| amzn-fy23-capex | changed | 0 | hard → medium | OV 0.60 high, YT 3 | I2: relabelled hard -> medium. C2: no alternative added (the cash-flow rows give purchases of property and equipment, another measure than cash capital expenditures). |
| amzn-fy25-anthropic | changed | 1 | easy → easy | OV 1.00 high | C2: +1 alternative (FY2025 Item 8: the Q2 and Q4 notes, $1.3 + $1.4 billion; the span covers both, one alone is half the answer). |
| amzn-fy22-notes | kept | 0 | easy → easy | OV 1.00 high |  |
| amzn-fy23-credit | changed | 0 | easy → medium | OV 1.00 high, YT 3 | I2: relabelled easy -> medium by the retrieval criteria. |
| amzn-fy25-cp | kept | 0 | medium → medium | OV 0.75 high, YT 1 |  |
| amzn-fy24-liquidity | kept | 0 | medium → medium | OV 1.00 high, YT 2 |  |
| amzn-fy25-finance-leases | kept | 0 | medium → medium | OV 1.00 high, YT 3 |  |
| meta-fy22-capex | changed | 1 | medium → medium | OV 0.67 high, YT 3 | C2: +1 alternative (FY2022 Item 7, Liquidity). |
| meta-fy23-guidance | kept | 0 | medium → medium | OV 1.00 high, YT 3 |  |
| meta-fy24-guidance | kept | 0 | medium → medium | OV 0.50 mid, YT 1 |  |
| meta-fy23-notes | changed | 1 | medium → medium | OV 0.60 high, YT 3 | C2: +1 alternative (FY2023 Item 7, Cash Used in Financing Activities). |
| meta-fy25-notes | changed | 1 | medium → medium | OV 0.67 high, YT 3 | C2: +1 alternative (FY2025 Item 7, Cash Used in Financing Activities). |
| meta-fy24-leases | kept | 0 | medium → medium | OV 0.83 high, YT 3 |  |
| meta-fy25-venture | changed | 0 | easy → medium | OV 0.20 low | I4: two-part question reduced to the exposure (paraphrased: v2 copied "maximum exposure to loss"). I2: relabelled easy -> medium by the retrieval criteria. |
| meta-fy22-restructuring | kept | 0 | easy → easy | OV 0.67 high | C2: no alternative added (partial only: Item 7 gives the $1.34 billion abandonment component alone). |
| orcl-fy22-bridge | changed | 1 | medium → easy | OV 0.86 high | C2: +1 alternative (FY2022 debt note bullet). Question made singular: the March 2022 revolving credit agreement was not for Cerner. I2: relabelled medium -> easy by the retrieval criteria. |
| orcl-fy23-notes | changed | 3 | medium → medium | OV 1.00 high, YT 3 | C2: +3 alternatives (FY2023 Item 7 interest expense, twice, and the debt note footnote). |
| orcl-fy23-cerner | changed | 1 | easy → easy | OV 0.75 high | C2: +1 alternative (FY2023 Item 7, working capital). |
| orcl-fy25-capex | changed | 0 | easy → medium | OV 0.75 high, YT 1 | I2: relabelled easy -> medium by the retrieval criteria. |
| orcl-fy25-notes | changed | 2 | easy → easy | OV 1.00 high | C2: +2 alternatives (FY2025 Item 7 interest expense and the debt note footnote). |
| orcl-fy25-rpo | kept | 0 | medium → medium | OV 1.00 high, YT 3 |  |
| orcl-fy25-termloan | kept | 0 | easy → easy | OV 0.86 high |  |
| amzn-fy22-short-credit | changed | 0 | easy → medium | OV 0.75 high, YT 3 | I2: relabelled easy -> medium by the retrieval criteria. C2: no alternative added (partial only: the FY2023 note names the November 2022 facility without its amount). |
| meta-fy24-buyback | changed | 1 | easy → hard | OV 0.33 low, YT 3 | C2: +1 alternative (FY2024 Item 7 overview bullet). I2: relabelled easy -> hard by the retrieval criteria. |
| orcl-fy25-useful-life | changed | 0 | easy → medium | OV 0.86 high, YT 2 | I2: relabelled easy -> medium by the retrieval criteria. |
| googl-fy24-dividend-fr | changed | 0 | easy → medium | OV 0.20 low | I2: relabelled easy -> medium. C2: no alternative added (none: the FY2025 10-K does not say when the dividend began). |
| msft-fy25-openai-fr | changed | 0 | easy → medium | OV 0.33 low | I2: relabelled easy -> medium by the retrieval criteria. |
| msft-activision-value | changed | 1 | easy → easy | OV 0.60 high | C1: pinned to the announcement (the FY2024 10-K gives the $75.4 billion purchase price at closing, another figure); Item 1 sentence added. |
| meta-fy22-first-notes | changed | 2 | easy → medium | OV 0.50 mid, YT 1 | C1: FY2022 Item 7 sentence and the FY2023 debt note ("including $10.0 billion issued in August 2022") added. I2: relabelled easy -> medium by the retrieval criteria. |
| orcl-fy25-ampere | changed | 1 | easy → easy | OV 0.71 high | C2: +1 alternative (FY2025 Item 1A). |
| amzn-fy23-anthropic-fr | changed | 2 | medium → medium | OV 0.50 mid | C1: the FY2024 10-K restates the Q3 2023 investment (Item 7 and Item 8); FY2023 Item 8 sentence added as an alternative. |
| msft-fy23-computer-equipment | changed | 0 | medium → medium | OV 0.62 high, YT 2, TB | I1: each span now runs from the year header row to the answer row (verbatim), so a hit carries the column years. |
| googl-fy23-cip | changed | 0 | medium → medium | OV 0.67 high, YT 1, TB | I1: span from the year header row to the row. |
| googl-fy25-tech-infra | changed | 0 | medium → medium | OV 0.60 high, YT 1, TB | I1: span from the year header row to the row. |
| orcl-fy24-capex-change | changed | 0 | medium → medium | OV 0.75 high, YT 3, TB | I1: span from the year header row of the free cash flow table to the row. |
| orcl-fy25-cip | changed | 0 | medium → medium | OV 0.62 high, YT 3, TB | I1: span from the year header row to the row. |
| meta-fy23-servers | changed | 0 | medium → medium | OV 0.60 high, YT 2, TB | I1: spans from the year header row to the row. |
| googl-fy23-useful-life | changed | 1 | medium → easy | OV 0.67 high | C2: +1 alternative (FY2023 Note 1, depreciation-effect sentence). I2: relabelled medium -> easy by the retrieval criteria. |
| amzn-fy25-aws-share | changed | 0 | hard → hard | OV 0.33 low, YT 3, TB, MF | I1: AWS span runs from the year header row; the lone Consolidated row stays as the second required span (it is cut from its header in v2 chunks; the first span gives its column order). |
| meta-fy25-capex-growth | changed | 0 | hard → medium | OV 0.50 mid, YT 3, TB | I1: span from the year header row of the free cash flow table to the row (the cash-flow-statement occurrence is split from its header in v2 chunks). I2: relabelled hard -> medium. Minor: rationale fixed (the row is in the Item 7 free cash flow table; the prose gives 2025 only). |
| meta-capex-outlook-compare | kept | 0 | hard → hard | OV 0.33 low, MF |  |
| orcl-rpo-growth | kept | 0 | hard → hard | OV 0.60 high, YT 2, MF |  |
| amzn-capex-2023-2025-fr | changed | 0 | hard → hard | OV 0.00 low, YT 2, MF | Minor: French calque "dépenses d'investissement en trésorerie" replaced. |
| server-life-amzn-vs-meta | changed | 3 | hard → hard | OV 0.33 low, YT 1, MF | C2: spans grouped (one group per issuer); +1 Amazon and +2 Meta FY2024 alternatives (Note 1 and the property and equipment note footnote). |
| amzn-fy25-server-life-fr | changed | 1 | hard → medium | OV 0.00 low | C1: FY2024 10-K gives the same reason for the change effective 1 January 2025; added. French calque fixed (durée d'utilité). I2: relabelled hard -> medium by the retrieval criteria. |
| meta-fy25-scaleai-fr | changed | 0 | hard → easy | OV 0.67 high | I2: relabelled hard -> easy. |
| meta-fy22-dividend | changed | 1 | hard → easy | OV 1.00 high | C2: +1 alternative (FY2022 Item 1A). I2: relabelled hard -> easy by the retrieval criteria. |
| amzn-fy25-no-borrowings | changed | 0 | hard → medium | OV 0.75 high, YT 3 | I2: relabelled hard -> medium. C2: no alternative added (partial only: the debt note gives the credit facilities and the commercial paper in separate sentences). |
| googl-fy25-power-constraint | changed | 0 | hard → medium | OV 0.50 mid | I2: relabelled hard -> medium by the retrieval criteria. |
| orcl-datacenter-capacity-risk | changed | 0 | hard → medium | OV 0.58 mid, YT 1 | I5: pinned to FY2025; body (2) and bullet (1) spans added; FY2024 dropped from the expected filings. I2: relabelled hard -> medium by the retrieval criteria. |
| msft-fy23-leases-fr | kept | 0 | hard → hard | OV 0.22 low, YT 3 |  |
| abs-01 | kept | - | medium → medium | abstention |  |
| abs-02 | kept | - | easy → easy | abstention |  |
| abs-03 | changed | - | easy → easy | abstention | I8: absent tags made concrete search terms. |
| abs-05 | changed | - | medium → medium | abstention | I8: absent tags made concrete search terms. |
| abs-googl-anthropic | kept | - | hard → hard | abstention |  |
| abs-orcl-stargate | kept | - | hard → hard | abstention |  |
| abs-amzn-dividend | kept | - | hard → hard | abstention |  |
| msft-fy25-openai | new | 0 | - → easy | OV 1.00 high | NEW: I3/I7: English twin of msft-fy25-openai-fr. |
| amzn-fy23-anthropic | new | 0 | - → easy | OV 1.00 high | NEW: I3: English twin of amzn-fy23-anthropic-fr. |
| googl-fy24-dividend | new | 0 | - → medium | OV 0.33 low | NEW: I3: English twin of googl-fy24-dividend-fr. |
| meta-fy23-servers-fr | new | 0 | - → hard | OV 0.00 low, YT 2, TB | NEW: I3/I2: French twin of meta-fy23-servers (cross-lingual on table content). |
| orcl-fy25-lease-commitments | new | 0 | - → hard | OV 0.30 low, YT 3 | NEW: I7: Oracle FY2025 additional lease commitments ($43.4 billion), asked in a user's words (rental deals, not started). |
| msft-fy23-buyback | new | 0 | - → medium | OV 0.50 mid, YT 2 | NEW: I7: a buyback question for Microsoft. |
| amzn-fy24-purchase-obligations | new | 0 | - → medium | OV 0.67 high, YT 3 | NEW: I7: Amazon purchase obligations. |
| googl-latest-capex-outlook | new | 0 | - → hard | OV 0.60 high, YT 2, NY | NEW: I7/I2: no-year "latest" question; the 2026 half of v2 googl-fy25-capex. |
| msft-fy23-depreciation-fr | new | 0 | - → hard | OV 0.00 low, YT 1 | NEW: I7: depreciation total (French, three filings). |
| orcl-fy23-fcf | new | 0 | - → hard | OV 0.33 low, YT 2, TB | NEW: I7: free cash flow of a non-Microsoft issuer (paraphrase, table). |
| googl-vs-meta-fy24-capex | new | 0 | - → hard | OV 0.75 high, YT 2, MF | NEW: I2: multi-issuer "all" question with groups of alternatives. |
| amzn-fcf-drivers | new | 0 | - → easy | OV 0.60 high | NEW: chunk-loss: Amazon Item 7 opening. |
| googl-fy25-ai-infra-costs | new | 0 | - → easy | OV 0.67 high | NEW: chunk-loss: Alphabet FY2025 Item 7 trends paragraph. |
| meta-dividends-2024-2025 | new | 0 | - → hard | OV 0.40 mid, MF | NEW: I2: multi-filing "all" question (Meta dividends 2024 vs 2025), groups of alternatives. |
| msft-fy22-construction-commitments | new | 0 | - → hard | OV 0.29 low, YT 3 | NEW: I2: paraphrase question with low overlap and a year template (FY2022, balance of years). |
| meta-fy25-guidance | **dropped** | - | medium → - | - | I6 Meta guidance cluster (4): its span lives on in meta-capex-outlook-compare; FY2025 weighs most. |
| amzn-fy25-capex | **dropped** | - | medium → - | - | I6 Amazon capex sentence cluster (3): its span lives on in amzn-capex-2023-2025-fr; FY2025 weighs most. |
| amzn-fy24-server-retire | **dropped** | - | easy → - | - | I6 server useful-life cluster (5): third Amazon question of the cluster. |
| msft-fy25-capex-table-fr | **dropped** | - | hard → - | - | I1: the cash-flow row is cut from its year header in every v2 chunk (the header-carrying span is unreachable in v2), and no prose gives the total: no v2 passage can answer. |
| amzn-fy25-leases-not-commenced | **dropped** | - | hard → - | - | I1: the commitments row is cut from its year header in every v2 chunk (header span unreachable in v2), and no prose gives the total. |
