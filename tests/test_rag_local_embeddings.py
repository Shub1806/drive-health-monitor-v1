import ast
import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.rag import embed, retrieve
from src.rag.build_kb import load_knowledge
from src.rag.embed import (
    CACHE_FORMAT, QUERY_PREFIX, SentenceTransformerEmbedder, embed_documents_cached, embed_model_name,
    embed_model_revision,
)
from src.rag.report import build_report, evidence_only_answer
from src.rag.retrieve import Hit, build_local_index, hit_as_doc, select_hits
from src.rag.prompt import assemble_context
from src.rag.validate import validate_answer
from tests.rag_helpers import RULE_STATS, firing_features, rule_anomaly

DOCS = load_knowledge(resolve_git=lambda: "git commit TEST")
DIM = 512


class FakeEmbedder:
    """Deterministic bag-of-content-words vectors; records every batch it is asked to embed.

    Words under five letters and the query prefix are ignored, so only content words decide similarity.
    """

    def __init__(self, fingerprint="fake|rev=1|dim=512"):
        self.fingerprint = fingerprint
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            vector = np.zeros(DIM)
            words = text.replace(QUERY_PREFIX, "").lower().replace(".", " ").replace(",", " ").replace(";", " ").split()
            for word in (w for w in words if len(w) >= 5):
                vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = np.linalg.norm(vector)
            out.append([float(x) for x in (vector / norm if norm else vector)])
        return out


def doc_dicts(docs=DOCS):
    return [{"doc_id": d.doc_id, "text": d.text} for d in docs]


# --- document vectors are cached and only recomputed when needed ---

def test_the_first_build_embeds_all_ten_documents_in_one_batch_and_writes_the_cache(tmp_path):
    embedder, path = FakeEmbedder(), tmp_path / "cache" / "embeddings.json"

    vectors = embed_documents_cached(doc_dicts(), embedder, path)

    assert len(vectors) == 10 and len(embedder.calls) == 1 and len(embedder.calls[0]) == 10
    cache = json.loads(path.read_text())
    assert cache["format"] == CACHE_FORMAT and cache["fingerprint"] == embedder.fingerprint and cache["dimension"] == DIM
    assert set(cache["docs"]) == {d.doc_id for d in DOCS}
    by_id = {d.doc_id: d for d in DOCS}
    assert cache["docs"]["attr:smart_5_raw"]["text_sha256"] == hashlib.sha256(by_id["attr:smart_5_raw"].text.encode()).hexdigest()


def test_an_unchanged_second_run_embeds_nothing(tmp_path):
    embedder, path = FakeEmbedder(), tmp_path / "embeddings.json"
    first = embed_documents_cached(doc_dicts(), embedder, path)

    second = embed_documents_cached(doc_dicts(), embedder, path)

    assert len(embedder.calls) == 1            # no new batch
    assert second == first


def test_a_changed_document_is_the_only_one_re_embedded(tmp_path):
    embedder, path = FakeEmbedder(), tmp_path / "embeddings.json"
    docs = doc_dicts()
    embed_documents_cached(docs, embedder, path)
    docs[3] = {"doc_id": docs[3]["doc_id"], "text": docs[3]["text"] + " An added sentence."}

    vectors = embed_documents_cached(docs, embedder, path)

    assert len(embedder.calls) == 2 and embedder.calls[1] == [docs[3]["text"]]
    assert len(vectors) == 10
    assert json.loads(path.read_text())["docs"][docs[3]["doc_id"]]["text_sha256"] == hashlib.sha256(docs[3]["text"].encode()).hexdigest()


def test_a_different_model_fingerprint_discards_the_old_cache(tmp_path):
    path = tmp_path / "embeddings.json"
    embed_documents_cached(doc_dicts(), FakeEmbedder("model-a|rev=1|dim=512"), path)
    other = FakeEmbedder("model-b|rev=2|dim=512")

    embed_documents_cached(doc_dicts(), other, path)

    assert len(other.calls) == 1 and len(other.calls[0]) == 10          # everything re-embedded
    assert json.loads(path.read_text())["fingerprint"] == "model-b|rev=2|dim=512"


def test_a_removed_document_is_dropped_from_the_cache(tmp_path):
    embedder, path = FakeEmbedder(), tmp_path / "embeddings.json"
    embed_documents_cached(doc_dicts(), embedder, path)
    docs = doc_dicts()[:-1]
    docs[0] = {"doc_id": docs[0]["doc_id"], "text": docs[0]["text"] + " edited"}      # forces a rewrite

    embed_documents_cached(docs, embedder, path)

    assert set(json.loads(path.read_text())["docs"]) == {d["doc_id"] for d in docs}


@pytest.mark.parametrize("content", ["not json at all", json.dumps({"format": 99, "docs": {}}), json.dumps([1, 2]), ""])
def test_a_corrupt_or_foreign_cache_is_ignored_and_rebuilt(tmp_path, content):
    path = tmp_path / "embeddings.json"
    path.write_text(content)
    embedder = FakeEmbedder()

    vectors = embed_documents_cached(doc_dicts(), embedder, path)

    assert len(vectors) == 10 and len(embedder.calls) == 1
    assert json.loads(path.read_text())["format"] == CACHE_FORMAT


def test_the_cache_is_written_atomically_and_leaves_no_temporary_files(tmp_path):
    embed_documents_cached(doc_dicts(), FakeEmbedder(), tmp_path / "sub" / "embeddings.json")

    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["embeddings.json"]


# --- questions and documents share one embedder ---

def test_the_index_embeds_documents_without_the_prefix_and_questions_with_it_on_the_same_embedder(tmp_path):
    embedder = FakeEmbedder()

    index = build_local_index(embedder, DOCS, tmp_path / "e.json")
    index.search("what does smart 5 count?")

    document_batch, query_batch = embedder.calls[0], embedder.calls[1]
    assert not any(text.startswith(QUERY_PREFIX) for text in document_batch)
    assert query_batch == [QUERY_PREFIX + "what does smart 5 count?"]
    assert index.fingerprint == embedder.fingerprint


def test_hits_carry_the_source_metadata_and_scores(tmp_path):
    index = build_local_index(FakeEmbedder(), DOCS, tmp_path / "e.json")

    hits = index.search("the temperature in degrees Celsius", top_k=3)

    assert all(isinstance(h, Hit) for h in hits) and len(hits) == 3
    assert hits[0].source_name and hits[0].source_reference and hits[0].score >= hits[1].score >= hits[2].score


@pytest.mark.parametrize("question, expected", [
    ("temperature in degrees Celsius held in the lowest byte", "attr:smart_194_raw"),
    ("random split value path length forest of trees", "method:isolation_forest"),
    ("sectors waiting to be remapped after unrecoverable read errors", "attr:smart_197_raw"),
    ("each daily file holds one record per drive per day", "dataset:backblaze_q2_2026"),
    ("points for a z-score above 3 and a rise since the previous day", "method:rule_baseline"),
])
def test_the_local_index_finds_the_relevant_document_first(tmp_path, question, expected):
    index = build_local_index(FakeEmbedder(), DOCS, tmp_path / "e.json")

    assert index.search(question, top_k=3)[0].doc_id == expected


def test_an_unrelated_question_falls_below_the_minimum_similarity(tmp_path):
    index = build_local_index(FakeEmbedder(), DOCS, tmp_path / "e.json")

    assert select_hits(index.search("how to bake sourdough bread"), min_similarity=0.3) == []


def test_documents_from_the_local_index_work_in_the_existing_context_and_validator(tmp_path):
    index = build_local_index(FakeEmbedder(), DOCS, tmp_path / "e.json")
    hits = select_hits(index.search("sectors waiting to be remapped"), min_similarity=0.0)
    docs = [hit_as_doc(h) for h in hits]
    report = build_report(rule_anomaly(), firing_features(), RULE_STATS)

    context = assemble_context("why was this drive flagged?", report, docs)

    assert validate_answer(evidence_only_answer(report, docs), context) == []
    assert context.allowed_ids[0] == report.doc_id and len(docs) == 3


# --- the sentence-transformers wrapper, with a stand-in library ---

class StubModel:
    created = []

    def __init__(self, name, **kwargs):
        self.name, self.kwargs = name, kwargs
        self.encode_calls = []
        StubModel.created.append(self)

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy):
        self.encode_calls.append(dict(texts=list(texts), batch_size=batch_size, normalize=normalize_embeddings, numpy=convert_to_numpy))
        return np.array([[1.0, 0.0, 0.0, 0.0] for _ in texts])

    def get_sentence_embedding_dimension(self):
        return 4

    def __getitem__(self, index):
        return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="abc123")))


@pytest.fixture
def stub_library(monkeypatch):
    StubModel.created = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=StubModel))
    monkeypatch.setattr(embed, "_library_version", lambda: "9.9")
    return StubModel


def test_the_model_loads_once_on_first_use_and_texts_are_normalised(stub_library):
    embedder = SentenceTransformerEmbedder("some/model", None)
    assert stub_library.created == []                       # nothing loaded yet

    out = embedder.embed_texts(["a", "b"])
    embedder.embed_texts(["c"])

    assert len(stub_library.created) == 1
    model = stub_library.created[0]
    assert model.name == "some/model" and model.kwargs == {}
    assert model.encode_calls[0] == dict(texts=["a", "b"], batch_size=16, normalize=True, numpy=True)
    assert out == [[1.0, 0.0, 0.0, 0.0]] * 2 and all(isinstance(x, float) for x in out[0])


def test_a_configured_revision_is_passed_to_the_library(stub_library):
    SentenceTransformerEmbedder("some/model", "deadbeef").embed_texts(["a"])

    assert stub_library.created[0].kwargs == {"revision": "deadbeef"}


def test_the_fingerprint_names_the_model_revision_library_and_dimension(stub_library):
    fingerprint = SentenceTransformerEmbedder("some/model", None).fingerprint

    assert fingerprint == "some/model|rev=abc123|sentence-transformers=9.9|dim=4"
    assert SentenceTransformerEmbedder("other/model", None).fingerprint != fingerprint


def test_the_dimension_uses_the_new_method_name_when_the_library_has_it(stub_library, monkeypatch):
    monkeypatch.setattr(stub_library, "get_embedding_dimension", lambda self: 7, raising=False)
    monkeypatch.setattr(stub_library, "get_sentence_embedding_dimension",
                        lambda self: (_ for _ in ()).throw(AssertionError("deprecated name used")))

    assert SentenceTransformerEmbedder("some/model", None).fingerprint.endswith("dim=7")


def test_the_revision_falls_back_to_the_cached_snapshot_folder_when_the_model_does_not_expose_it(stub_library, monkeypatch):
    # newer library versions no longer expose the commit hash on the loaded model
    monkeypatch.setattr(stub_library, "__getitem__", lambda self, i: (_ for _ in ()).throw(TypeError("not subscriptable")))
    asked = {}

    def snapshot_download(repo_id, revision=None, local_files_only=False):
        asked.update(repo_id=repo_id, revision=revision, local_files_only=local_files_only)
        return "/home/u/.cache/huggingface/hub/models--some--model/snapshots/0123456789abcdef"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))

    fingerprint = SentenceTransformerEmbedder("some/model", "main").fingerprint

    assert "rev=0123456789abcdef" in fingerprint
    assert asked == {"repo_id": "some/model", "revision": "main", "local_files_only": True}   # looks only at what is already downloaded


def test_the_revision_is_reported_unpinned_only_when_nothing_can_be_resolved(stub_library, monkeypatch):
    monkeypatch.setattr(stub_library, "__getitem__", lambda self, i: (_ for _ in ()).throw(TypeError("no")))
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)        # also not importable

    assert "rev=unpinned" in SentenceTransformerEmbedder("some/model", None).fingerprint
    assert "rev=pinned123" in SentenceTransformerEmbedder("some/model", "pinned123").fingerprint


def test_a_missing_library_gives_a_clear_install_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError, match="pip install -r requirements.txt"):
        SentenceTransformerEmbedder("some/model", None).embed_texts(["a"])


def test_model_name_and_revision_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "my/model")
    monkeypatch.setenv("EMBED_MODEL_REVISION", " rev1 ")
    assert embed_model_name() == "my/model" and embed_model_revision() == "rev1"
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setenv("EMBED_MODEL_REVISION", "")
    assert embed_model_name() == embed.DEFAULT_EMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert embed_model_revision() is None


# --- retrieval needs neither Snowflake vectors, Cortex nor FAISS ---

def test_retrieval_modules_have_no_snowflake_vector_cortex_or_faiss_dependency():
    for path in ("src/rag/embed.py", "src/rag/retrieve.py", "eval/evaluate_rag.py"):
        source = pathlib.Path(path).read_text().lower()
        assert "cortex" not in source and "vector(" not in source and "faiss" not in source, path


def test_the_heavy_library_is_imported_only_when_a_model_is_needed():
    tree = ast.parse(pathlib.Path("src/rag/embed.py").read_text())
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in top_level for a in n.names} | {n.module for n in top_level if isinstance(n, ast.ImportFrom)}

    assert "sentence_transformers" not in names


def test_configuration_files_list_the_new_settings_and_packages():
    env = {line.split("=", 1)[0]: line.split("=", 1)[1] for line in pathlib.Path(".env.example").read_text().splitlines() if "=" in line}
    requirements = pathlib.Path("requirements.txt").read_text().split()

    assert {"LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "EMBED_MODEL", "EMBED_MODEL_REVISION"} <= set(env)
    assert env["LLM_API_KEY"] == "" and env["LLM_BASE_URL"] == "" and env["LLM_MODEL"] == ""
    assert env["EMBED_MODEL"] == "BAAI/bge-small-en-v1.5"
    assert "sentence-transformers" in requirements and "requests" in requirements
    assert not any("faiss" in r for r in requirements)
    assert ".cache/" in pathlib.Path(".gitignore").read_text()
