"""Parity of the two embedding backends on the real model (marker ``models``).

The collection is filled with vectors from the torch backend (sentence-transformers), and a
search service may embed its queries with the onnx backend, which needs no PyTorch. That only
works if both backends give the same vector for the same text. This file measures it:

* for each of 50 fixed texts, the cosine between the torch vector and the onnx vector is at
  least ``parity_min`` (``RAG_ENGINE_PARITY_MIN``, default 0.999), as a passage and as a query;
* on a fixed corpus of 20 chunks and 5 queries, the five best chunks of each query are the
  same, in the same order, whether the query is embedded with onnx or with torch, the chunks
  being embedded with torch in both cases.

The onnx vectors must come from the files the engine pins (``ONNX_MODELS``: one revision, and
one SHA-256 per file — the ONNX file, the tokenizer, the configuration), or the measure would
say nothing about what production loads. So the files the embedder loaded are hashed again
here, and a copy with one byte changed, in the ONNX file or in ``tokenizer.json``, must be
refused before any inference.

It needs both extras (``rag-engine[torch,onnx]``) and the model files, which are downloaded on
the first run. Run it with ``pytest -m models -s`` to see the measured numbers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import time
import unicodedata

import numpy as np
import pytest

import rag_engine.embedding as embedding_mod
from rag_engine.config import Settings
from rag_engine.embedding import (
    ONNX_MODELS,
    ModelIntegrityError,
    ModelSpec,
    OnnxEmbedder,
    cosine_parity,
    make_embedder,
)

SENTENCES = {
    "en": [
        "The fund invests mainly in investment-grade corporate bonds issued in euros.",
        "Its benchmark is a broad index of European credit with maturities between one and ten years.",
        "The portfolio manager may hold up to ten percent of the assets in high-yield securities.",
        "Currency risk is hedged systematically, so the share class in dollars follows the euro one.",
        "Subscriptions and redemptions are settled two business days after the valuation date.",
        "The ongoing charges amounted to less than one percent of the average net assets last year.",
        "Over five years the annualised volatility stayed below that of the reference index.",
        "A swing pricing mechanism protects long-term holders against the cost of large flows.",
        "The management company publishes the full list of holdings every quarter with a delay.",
        "Sustainability risks are part of the credit analysis, and some sectors are excluded.",
        "Derivatives are used to adjust the duration of the portfolio, not to build leverage.",
        "The depositary checks that the calculation of the net asset value follows the prospectus.",
        "Central banks raised their policy rates quickly, and bond prices fell during that period.",
        "Since then the carry of the portfolio has recovered and coupons are reinvested at higher yields.",
    ],
    "fr": [
        "Le fonds investit principalement dans des obligations d'entreprises de bonne qualité en euros.",
        "Son indicateur de référence est un indice large du crédit européen de maturité un à dix ans.",
        "Le gérant peut détenir jusqu'à dix pour cent de l'actif en titres à haut rendement.",
        "Le risque de change est couvert de façon systématique pour les parts libellées en dollars.",
        "Les souscriptions et les rachats sont réglés deux jours ouvrés après la date de valorisation.",
        "Les frais courants ont représenté moins d'un pour cent de l'actif net moyen l'an dernier.",
        "Sur cinq ans, la volatilité annualisée est restée inférieure à celle de l'indice.",
        "Un mécanisme de swing pricing protège les porteurs de long terme contre le coût des grands flux.",
        "La société de gestion publie chaque trimestre l'inventaire complet du portefeuille, avec un délai.",
        "Les risques de durabilité font partie de l'analyse crédit, et certains secteurs sont exclus.",
        "Les dérivés servent à ajuster la sensibilité du portefeuille, pas à créer de l'effet de levier.",
        "Le dépositaire vérifie que le calcul de la valeur liquidative respecte le prospectus.",
        "Les banques centrales ont relevé rapidement leurs taux directeurs, et les prix ont baissé.",
        "Depuis, le portage s'est reconstitué et les coupons sont réinvestis à des rendements plus élevés.",
    ],
    "de": [
        "Der Fonds investiert überwiegend in auf Euro lautende Unternehmensanleihen guter Bonität.",
        "Als Vergleichsmaßstab dient ein breiter Index europäischer Anleihen mit Laufzeiten bis zehn Jahre.",
        "Der Fondsmanager darf bis zu zehn Prozent des Vermögens in Hochzinsanleihen halten.",
        "Das Währungsrisiko der Dollar-Anteilsklasse wird systematisch abgesichert.",
        "Zeichnungen und Rücknahmen werden zwei Geschäftstage nach dem Bewertungstag abgerechnet.",
        "Die laufenden Kosten lagen im vergangenen Jahr unter einem Prozent des Nettovermögens.",
        "Über fünf Jahre blieb die annualisierte Volatilität niedriger als die des Referenzindex.",
        "Ein Swing-Pricing-Verfahren schützt langfristige Anleger vor den Kosten großer Mittelbewegungen.",
        "Die Verwaltungsgesellschaft veröffentlicht vierteljährlich die vollständige Vermögensaufstellung.",
        "Nachhaltigkeitsrisiken fließen in die Kreditanalyse ein, einige Branchen sind ausgeschlossen.",
        "Derivate dienen der Steuerung der Duration und nicht dem Aufbau einer Hebelwirkung.",
        "Die Verwahrstelle prüft, ob die Berechnung des Anteilwerts dem Verkaufsprospekt entspricht.",
    ],
    "es": [
        "El fondo invierte principalmente en bonos corporativos con grado de inversión emitidos en euros.",
        "Su índice de referencia es un índice amplio de crédito europeo con vencimientos de uno a diez años.",
        "El gestor puede mantener hasta un diez por ciento del patrimonio en valores de alto rendimiento.",
        "El riesgo de divisa de la clase en dólares se cubre de forma sistemática.",
        "Las suscripciones y los reembolsos se liquidan dos días hábiles después de la fecha de valoración.",
        "Los gastos corrientes fueron inferiores al uno por ciento del patrimonio medio el año pasado.",
        "En cinco años la volatilidad anualizada se mantuvo por debajo de la del índice.",
        "Un mecanismo de ajuste de precios protege a los partícipes de largo plazo frente a grandes flujos.",
        "La sociedad gestora publica cada trimestre la cartera completa con cierto retraso.",
        "Los riesgos de sostenibilidad forman parte del análisis de crédito y se excluyen algunos sectores.",
        "Los derivados se utilizan para ajustar la duración de la cartera, no para apalancarla.",
        "El depositario comprueba que el cálculo del valor liquidativo se ajusta al folleto.",
    ],
    "it": [
        "Il fondo investe principalmente in obbligazioni societarie di buona qualità denominate in euro.",
        "Il parametro di riferimento è un indice ampio del credito europeo con scadenze da uno a dieci anni.",
        "Il gestore può detenere fino al dieci per cento del patrimonio in titoli ad alto rendimento.",
        "Il rischio di cambio della classe in dollari è coperto in modo sistematico.",
        "Le sottoscrizioni e i rimborsi sono regolati due giorni lavorativi dopo la data di valorizzazione.",
        "Le spese correnti sono state inferiori all'uno per cento del patrimonio medio lo scorso anno.",
        "In cinque anni la volatilità annualizzata è rimasta al di sotto di quella dell'indice.",
        "Un meccanismo di swing pricing tutela i sottoscrittori di lungo periodo dai costi dei flussi.",
        "La società di gestione pubblica ogni trimestre l'elenco completo dei titoli in portafoglio.",
        "I rischi di sostenibilità rientrano nell'analisi del credito e alcuni settori sono esclusi.",
        "I derivati servono a regolare la durata finanziaria del portafoglio, non a creare leva.",
        "Il depositario verifica che il calcolo del valore della quota rispetti il prospetto.",
    ],
    "ja": [
        "このファンドは主にユーロ建ての投資適格社債に投資します。",
        "ベンチマークは残存期間が一年から十年の欧州クレジット指数です。",
        "運用担当者は資産の十パーセントまでハイイールド債を組み入れることができます。",
        "ドル建てクラスの為替リスクは常にヘッジされています。",
        "購入と解約の代金は基準価額の算出日から二営業日後に決済されます。",
        "昨年の運用管理費用は平均純資産の一パーセント未満でした。",
        "過去五年間の年率ボラティリティは参考指数を下回りました。",
        "スイング・プライシングは大口の資金移動から長期の投資家を保護する仕組みです。",
        "運用会社は四半期ごとに全銘柄の保有明細を公表します。",
        "サステナビリティのリスクは信用分析に含まれ、一部の業種は除外されます。",
        "デリバティブはデュレーションの調整に使われ、レバレッジを目的としません。",
        "受託銀行は基準価額の計算が目論見書に従っているかを確認します。",
    ],
}


def long_text(language: str, words: int, start: int = 0) -> str:
    """A text of about ``words`` words, made of the sentences of ``language`` in rotation.

    Japanese has no spaces between words: there, ``words`` counts sentences.
    """
    sentences = SENTENCES[language]
    parts: list[str] = []
    count, index = 0, start
    while count < words:
        sentence = sentences[index % len(sentences)]
        parts.append(sentence)
        count += 1 if language == "ja" else len(sentence.split())
        index += 1
    return ("" if language == "ja" else " ").join(parts)


ACCENTED = "Société Générale : résumé du rapport d'activité, déjà publié à Noël à Zürich."
VOICED = "ガバナンスとデリバティブ、パフォーマンスのデータを分析しています。"  # voiced and semi-voiced kana

SHORT_TEXTS = [
    # English
    "bond",
    "What is the ongoing charge of the fund?",
    "The share class distributes a dividend every quarter.",
    "Performance fees are calculated with a high-water mark.",
    "Net asset value per share as of the last business day of the month.",
    # French
    "obligation",
    "Quels sont les frais courants du fonds ?",
    "La part distribue un dividende chaque trimestre.",
    "L'OPCVM est conforme à la directive européenne et agréé par l'autorité de tutelle.",
    "Où trouver le document d'informations clés pour l'investisseur ?",
    # German
    "Anleihe",
    "Wie hoch sind die laufenden Kosten des Fonds?",
    "Die Anteilsklasse schüttet vierteljährlich eine Dividende aus.",
    "Rückgabe von Anteilen ist an jedem Bewertungstag möglich; Ausgabeaufschläge können anfallen.",
    # Spanish
    "bono",
    "¿Cuáles son los gastos corrientes del fondo?",
    "La clase reparte un dividendo cada trimestre.",
    "El folleto está disponible en español, inglés y francés.",
    # Italian
    "obbligazione",
    "Quali sono le spese correnti del fondo?",
    "La classe distribuisce un dividendo ogni trimestre.",
    "Il rendimento passato non è indicativo di quello futuro.",
    # Japanese
    "債券",
    "ファンドの運用管理費用はいくらですか。",
    "東京で働いていますが、毎月ドル建ての投資信託を購入しています。",
    VOICED,
    "ぎんこうがはっこうするさいけんのげんぽんはほしょうされていますか。",
    # Short, empty, and blank strings
    "",
    "a",
    " ",
    "?",
    # Numbers, tickers, identifiers
    "FR0010315770",
    "LU0323578657 / IE00B4L5Y983 / US0378331005",
    "AAPL MSFT NVDA GOOGL AMZN",
    "NAV 1,234.56 EUR (+0.42 %) on 2024-12-31; AUM 3.2bn",
    "EUR/USD 1.0875 — 10Y Bund 2.31% — CAC 40: 7 543,18",
    "3.14159 2.71828 1e-9 42 0x1F 100% 1/3",
    # Mixed scripts and symbols
    "Fund « Europe Crédit » — class A (EUR), ISIN FR0010315770, 東京 office",
    "E-mail: contact@example.com — https://example.com/funds?id=42&lang=fr",
    # Composed and decomposed forms of the same texts
    unicodedata.normalize("NFC", ACCENTED),
    unicodedata.normalize("NFKD", ACCENTED),
    unicodedata.normalize("NFD", VOICED),
]

LONG_TEXTS = [
    # About 400 words: around the 512 tokens of the model.
    long_text("en", 400),
    long_text("fr", 400),
    long_text("de", 400),
    long_text("es", 400),
    long_text("it", 400),
    long_text("ja", 30),
    # About 900 words: well beyond the limit, so both backends must cut at the same place.
    long_text("en", 900, start=5),
    long_text("fr", 900, start=5),
]

TEXTS = SHORT_TEXTS + LONG_TEXTS

CORPUS = [
    "The fund invests in investment-grade corporate bonds issued in euros.",
    "Equity sub-fund investing in small and mid capitalisations of the euro area.",
    "Money market fund with a variable net asset value and daily liquidity.",
    "The ongoing charges of the A share class are 0.85 % per year.",
    "A performance fee of 15 % applies above the benchmark, with a high-water mark.",
    "Subscriptions are centralised every business day before noon, Paris time.",
    "The depositary of the fund is a credit institution established in Luxembourg.",
    "Le fonds est exposé au risque de crédit et au risque de taux d'intérêt.",
    "La durée de placement recommandée est supérieure à cinq ans.",
    "Les revenus de la part D sont distribués chaque année aux porteurs.",
    "Le fonds exclut les producteurs de tabac et de charbon thermique.",
    "Der Fonds darf Derivate zur Absicherung von Währungsrisiken einsetzen.",
    "Die Mindestanlage für institutionelle Anleger beträgt eine Million Euro.",
    "El fondo replica un índice de renta variable de mercados emergentes.",
    "La cartera tiene una duración media de cuatro años y medio.",
    "Il fondo investe in titoli di Stato italiani indicizzati all'inflazione.",
    "Il rimborso delle quote avviene entro cinque giorni lavorativi.",
    "このファンドは日本の中小型株に投資します。",
    "分配金は毎月支払われ、再投資することもできます。",
    "The sub-fund promotes environmental characteristics under article 8 of the regulation.",
]

QUERIES = [
    "What are the ongoing charges of the A share class?",
    "Quels secteurs sont exclus pour des raisons de durabilité ?",
    "Wie werden Währungsrisiken abgesichert?",
    "¿Cuál es la duración de la cartera?",
    "毎月分配金が出るファンドはありますか。",
]

TOP = 5


def test_the_fixed_texts_are_the_ones_described():
    assert len(TEXTS) == 50
    assert len(set(TEXTS)) == 50
    assert len(CORPUS) == len(set(CORPUS)) == 20
    assert len(QUERIES) == 5
    nfc, nfkd, voiced_nfd = SHORT_TEXTS[-3:]
    assert nfc != nfkd and unicodedata.normalize("NFC", nfkd) == nfc
    assert voiced_nfd != VOICED and "゙" in voiced_nfd  # the combining voiced sound mark
    lengths = [len(text.split()) for text in LONG_TEXTS]
    assert all(390 <= lengths[index] <= 420 for index in range(5))
    assert all(890 <= lengths[index] <= 920 for index in (6, 7))


@pytest.fixture(scope="module")
def vectors():
    """Vectors of every fixed text from both backends, with the settings of the environment."""
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("fastembed")
    settings = Settings.from_env()
    result: dict[str, object] = {"parity_min": settings.parity_min}
    for backend in ("torch", "onnx"):
        with make_embedder(dataclasses.replace(settings, backend=backend)) as embedder:
            result[backend] = {
                "passages": embedder.encode_passage(TEXTS),
                "queries": np.stack([embedder.encode_query(text) for text in TEXTS]),
                "corpus": embedder.encode_passage(CORPUS),
                "search": np.stack([embedder.encode_query(text) for text in QUERIES]),
            }
            if backend == "onnx":
                # What produced the vectors above: the file, its hash as the engine checked it,
                # and the execution providers of the onnxruntime session fastembed built.
                result["onnx_model"] = {
                    "name": embedder.model_name,
                    "path": embedder.model_path,
                    "sha256": embedder.model_sha256,
                    "providers": embedder._model.model.model.get_providers(),
                }
    return result


@pytest.mark.models
def test_the_onnx_vectors_come_from_the_pinned_file_on_cpu(vectors):
    loaded = vectors["onnx_model"]
    pinned = ONNX_MODELS[loaded["name"]]
    path = loaded["path"]
    # The file of the pinned revision, in the layout of the Hugging Face cache.
    assert path.parts[-4:] == ("snapshots", pinned.revision, *pinned.model_file.split("/"))
    assert loaded["sha256"] == pinned.sha256
    # Hashed again here, without the engine: the file on disk is the one of the table.
    started = time.perf_counter()
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 20):
            hasher.update(block)
    elapsed = time.perf_counter() - started
    assert hasher.hexdigest() == pinned.sha256
    assert path.stat().st_size == pinned.size
    assert loaded["providers"] == ["CPUExecutionProvider"]
    print(
        f"\nonnx vectors computed from {path}\n  revision {pinned.revision}, {pinned.size} bytes, "
        f"sha256 {hasher.hexdigest()} (hashed in {elapsed:.2f} s), providers {loaded['providers']}"
    )
    # The tokenizer and configuration files next to it are those of the table too.
    directory = path.parents[len(pinned.model_file.split("/")) - 1]
    assert set(pinned.files) == {pinned.model_file, *embedding_mod._ONNX_SUPPORT_FILES}
    print("  files of the model, as found on disk:")
    for file, (expected, size) in pinned.files.items():
        found = (
            hashlib.sha256((directory / file).read_bytes()).hexdigest()
            if file != pinned.model_file
            else hasher.hexdigest()
        )
        print(f"    {file}: {(directory / file).stat().st_size} bytes, sha256 {found}")
        assert (found, (directory / file).stat().st_size) == (expected, size), file


@pytest.mark.models
def test_the_hash_of_the_onnx_file_is_computed_once_per_process(vectors, monkeypatch):
    path = vectors["onnx_model"]["path"]
    monkeypatch.setattr(embedding_mod, "_sha256_cache", {})
    started = time.perf_counter()
    first = embedding_mod._file_sha256(path, cached=True)
    cold = time.perf_counter() - started
    started = time.perf_counter()
    second = embedding_mod._file_sha256(path, cached=True)
    warm = time.perf_counter() - started
    print(f"\nhash check of the ONNX file: {cold:.2f} s the first time, {warm * 1000:.3f} ms afterwards")
    assert first == second == (ONNX_MODELS[vectors["onnx_model"]["name"]].sha256, path.stat().st_size)
    assert warm < cold / 10


@pytest.fixture
def model_copy_dir(tmp_path):
    """An empty model directory for one test, removed when the test ends, passed or failed.

    pytest keeps the temporary directories of its last three runs: without this, each run of
    the altered-ONNX test would leave its 1.1 GB copy of the model behind.
    """
    directory = tmp_path / "model-copy"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        assert not directory.exists(), f"the copy of the model was not removed: {directory}"


@pytest.mark.models
def test_an_altered_onnx_file_is_refused_before_any_inference(vectors, model_copy_dir, monkeypatch):
    fastembed = pytest.importorskip("fastembed")
    tmp_path = model_copy_dir
    loaded = vectors["onnx_model"]
    pinned = ONNX_MODELS[loaded["name"]]
    source = loaded["path"].parents[len(pinned.model_file.split("/")) - 1]  # snapshots/<revision>
    # A model directory of its own: a full copy of the pinned files, one byte of the ONNX file
    # changed (same size, so only the hash can tell). The fixture removes it afterwards.
    copy = tmp_path / source.parents[1].name / "snapshots" / pinned.revision
    # copyfile, not the default copy2: the files of a freshly downloaded cache are read-only,
    # and copy2 would carry that mode over to the copy that has to be altered.
    shutil.copytree(source, copy, copy_function=shutil.copyfile)  # follows the links: real files
    altered = copy / pinned.model_file
    with open(altered, "r+b") as stream:
        stream.seek(pinned.size // 2)
        byte = stream.read(1)
        stream.seek(pinned.size // 2)
        stream.write(bytes([byte[0] ^ 0x01]))
    assert altered.stat().st_size == pinned.size

    def no_model(*args, **kwargs):
        raise AssertionError("the model must not be built from a file that failed the hash check")

    monkeypatch.setattr(fastembed.TextEmbedding, "__init__", no_model)
    spec = ModelSpec.from_settings(Settings.from_env())
    with pytest.raises(ModelIntegrityError) as caught:
        OnnxEmbedder(spec, cache_dir=tmp_path)
    message = str(caught.value)
    print("\n" + message)
    assert f"Expected sha256 {pinned.sha256}" in message
    assert str(altered) in message
    assert f"the file {pinned.model_file!r}" in message
    assert "parity must be re-measured" in message
    found = message.split("found sha256 ")[1][:64]
    assert found != pinned.sha256 and len(set(found) - set("0123456789abcdef")) == 0


@pytest.mark.models
def test_an_altered_tokenizer_file_is_refused_before_any_inference(vectors, model_copy_dir, monkeypatch):
    fastembed = pytest.importorskip("fastembed")
    tmp_path = model_copy_dir  # removed afterwards: where links cannot be made it holds a full copy
    loaded = vectors["onnx_model"]
    pinned = ONNX_MODELS[loaded["name"]]
    source = loaded["path"].parents[len(pinned.model_file.split("/")) - 1]  # snapshots/<revision>
    # A model directory of its own. The ONNX file is the real one, through a link (it is not
    # copied a second time); the small files are copied, and one byte of tokenizer.json is
    # changed (same size, so only the hash can tell).
    copy = tmp_path / source.parents[1].name / "snapshots" / pinned.revision
    for file in pinned.files:
        (copy / file).parent.mkdir(parents=True, exist_ok=True)
        if file != pinned.model_file:
            shutil.copyfile(source / file, copy / file)
            continue
        try:
            os.symlink(os.path.realpath(source / file), copy / file)
        except OSError:  # no right to create links (Windows): a copy, about 1.1 GB
            shutil.copyfile(source / file, copy / file)
    altered = copy / "tokenizer.json"
    expected, size = pinned.files["tokenizer.json"]
    with open(altered, "r+b") as stream:
        stream.seek(size // 2)
        byte = stream.read(1)
        stream.seek(size // 2)
        stream.write(bytes([byte[0] ^ 0x01]))
    assert altered.stat().st_size == size

    def no_model(*args, **kwargs):
        raise AssertionError("the model must not be built with a file that failed the hash check")

    monkeypatch.setattr(fastembed.TextEmbedding, "__init__", no_model)
    spec = ModelSpec.from_settings(Settings.from_env())
    with pytest.raises(ModelIntegrityError) as caught:
        OnnxEmbedder(spec, cache_dir=tmp_path)
    message = str(caught.value)
    print("\n" + message)
    assert "the file 'tokenizer.json'" in message
    assert str(altered) in message
    assert f"Expected sha256 {expected} ({size} bytes)" in message
    assert pinned.sha256 not in message  # the ONNX file passed: it is not the one blamed
    assert "parity must be re-measured" in message
    found = message.split("found sha256 ")[1][:64]
    assert found == hashlib.sha256(altered.read_bytes()).hexdigest() != expected

    # With the pinned tokenizer back in place, the same directory is accepted.
    shutil.copyfile(source / "tokenizer.json", altered)
    monkeypatch.undo()
    with OnnxEmbedder(spec, cache_dir=tmp_path) as embedder:
        assert embedder.model_path == copy / pinned.model_file
        assert embedder.model_sha256 == pinned.sha256
        assert np.allclose(embedder.encode_query(QUERIES[0]), vectors["onnx"]["search"][0], atol=1e-6)


def describe(text: str) -> str:
    return f"{text[:40]!r}… ({len(text)} characters)" if len(text) > 40 else repr(text)


def report(kind: str, cosines: np.ndarray, minimum: float) -> str:
    worst = np.argsort(cosines)[:3]
    lines = [
        f"parity of {kind} vectors, torch against onnx, on {len(cosines)} texts: min {cosines.min():.8f}, "
        + f"mean {cosines.mean():.8f}, largest 1 - cosine {1 - cosines.min():.2e}, required {minimum}",
    ]
    lines += [f"  worst: {cosines[index]:.8f} for text {index}: {describe(TEXTS[index])}" for index in worst]
    return "\n".join(lines)


@pytest.mark.models
@pytest.mark.parametrize("kind", ["passages", "queries"])
def test_torch_and_onnx_give_the_same_vector_for_every_text(vectors, kind):
    torch, onnx = vectors["torch"][kind], vectors["onnx"][kind]
    assert torch.shape == onnx.shape == (len(TEXTS), 768)
    cosines = cosine_parity(torch, onnx)
    message = report(kind, cosines, vectors["parity_min"])
    print("\n" + message)
    assert (cosines >= vectors["parity_min"]).all(), message  # a nan fails too


@pytest.mark.models
def test_the_two_kinds_of_vectors_differ(vectors):
    # Guards the test above: the prefixes are applied, a passage is not its own query.
    for backend in ("torch", "onnx"):
        cosines = cosine_parity(vectors[backend]["passages"], vectors[backend]["queries"])
        assert cosines.max() < 0.9999


def ranking(passages: np.ndarray, queries: np.ndarray) -> list[list[int]]:
    """For each query, the indexes of the ``TOP`` passages with the highest cosine, best first."""
    passages = passages / np.linalg.norm(passages, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    scores = queries.astype(np.float64) @ passages.astype(np.float64).T
    return [np.argsort(-row, kind="stable")[:TOP].tolist() for row in scores]


@pytest.mark.models
def test_onnx_queries_rank_torch_passages_as_torch_queries_do(vectors):
    # The production configuration: passages embedded with torch, queries with onnx.
    passages = vectors["torch"]["corpus"]
    reference = ranking(passages, vectors["torch"]["search"])
    mixed = ranking(passages, vectors["onnx"]["search"])
    lines = [f"top-{TOP} of {len(QUERIES)} queries over {len(CORPUS)} torch passages (torch / onnx queries):"]
    lines += [f"  {QUERIES[i]!r}: {reference[i]} / {mixed[i]}" for i in range(len(QUERIES))]
    message = "\n".join(lines)
    print("\n" + message)
    assert mixed == reference, message
    # And the ranking is a meaningful one: each query finds the chunk written for it first.
    assert [row[0] for row in reference] == [3, 10, 11, 14, 18]
