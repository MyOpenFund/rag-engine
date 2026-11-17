# Golden fixture: the points of the previous pipeline

`expected.json` holds the Qdrant points (id, payload, vector) that the previous ingestion
pipeline produced for the two documents of this folder: `ingest_item` of the data-orchestrator
(`data_orchestrator/core.py`), driving the chunker of eigenmind. It was captured before that
code moved into `rag_engine.ingest.ingest_document`.

`tests/test_ingest.py` ingests the same two documents with `ingest_document` and asserts that
the points are the same: same number, same order, same ids, same payloads (keys, key order and
values), same vectors. This is what keeps the points already in Qdrant valid.

**Do not regenerate `expected.json` to make a failing test pass.** A difference means the
engine no longer writes what is stored.

## Files

| File | What |
|---|---|
| `monetary-policy-statement.md` | Markdown, six headed sections, French and English text. No pages: 5 points without `page`. |
| `inflation-report-extract.pdf` | PDF, 3 pages (the second one short), written once by PyMuPDF. 3 points, on pages 0, 1 and 2. |
| `expected.json` | The captured points and everything the test needs to reproduce them. |

`expected.json`, per document: `file`, `collection`, `doc_id`, `metadata` (the payload given
by the source, as the vault source of the data-orchestrator builds it), `ingestion_date` (the one of the capture, which the test
gives back to `ingest_document`), and `points`, in the order they were written. At the top:
the versions of the capture, the fake embedder (`rag_engine.embedding.FakeEmbedder`, 8
dimensions, seed 0, prefixes `passage: ` / `query: `) and the OCR mode (`never`, so that the
result does not depend on the OCR setup of the host).

The two documents differ in their metadata on purpose:

- the PDF is the one closest to production. Its metadata holds `doc_id` itself, equal to the
  document's id, as the vault source copies it into the payload it hands over
  (`_PAYLOAD_COLUMNS` in `data_orchestrator/sources/vault.py`), and one key outside the usual
  columns, `series`. The fixture therefore pins that a `doc_id` repeated in the metadata is
  accepted and changes nothing in the points;
- the markdown document has no `doc_id` in its metadata.

The files are stored byte for byte (`.gitattributes`): `metadata.sha256` is the hash of each
document as committed.

Worth knowing when reading the file: ChunkNorris decomposes accented characters (Unicode
NFKD), so the chunk texts hold `e` + U+0301 where the markdown file holds `é`. The metadata
is not touched (`title` keeps `é`).

## How it was captured

From the root of this repository, with the script kept for that purpose,
`tools/capture_golden.py`:

```bash
# First run, once: writes the PDF, then captures.
PYTHONPATH="<path to data-orchestrator>:<path to the fork>" \
  <venv>/bin/python tools/capture_golden.py --make-pdf
# The run that wrote the expected.json committed here (the PDF already existed).
PYTHONPATH="<path to data-orchestrator>:<path to the fork>" \
  <venv>/bin/python tools/capture_golden.py --force
```

`<path to data-orchestrator>` is a checkout of the data-orchestrator, `<path to the fork>` a
checkout of the eigenmind fork it drives, both at the commits of the table below; `<venv>` is
a Python environment with the `ingest` extra of this package.

The script writes `expected.json` in this folder, or the file given with `--output`. It
refuses to overwrite an existing file unless `--force` is given. To compare a new capture
with the committed one without replacing it, use `--output <some other file>`.

`expected.json` was captured twice on the same day, with the same commits and versions: a
first time, then again after `doc_id` was added to the metadata of the PDF. Between the two,
nothing changed but that metadata key and the `ingestion_date` of the capture: same ids, same
vectors, same payloads otherwise.

The script imports the real `data_orchestrator.core.ingest_item` and calls it with a
`FakeEmbedder` and a store that records what it is asked to write. Nothing of the previous
pipeline was replaced or stubbed, and no module had to be stood in for.

| | |
|---|---|
| Date | 2026-11-17 |
| Python | 3.13.18 |
| chunknorris | 1.5.0 |
| PyMuPDF | 1.27.2.2 |
| numpy | 2.5.5 |
| qdrant-client | 1.20.0 |
| data-orchestrator | commit `7196bfc364e8e38d986335cdf9e5ca18722689c2` |
| eigenmind | commit `6e9093a777d059aefac5340de51c2c77fc93dec7` |
| This package | editable install of this repository, for `FakeEmbedder` only |

The PDF is not reproducible byte for byte: PyMuPDF writes a new document id in the trailer
(`/ID`) at each generation. Nothing else differs between two generations, and the PDF has no
date metadata (no creation or modification date). This is why it is committed and
`--make-pdf` refuses to overwrite it.
