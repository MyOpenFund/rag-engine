# Audit of `filings_v1.jsonl` for `filings_v2.jsonl`

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
