"""Local text embeddings for the knowledge documents and for questions.

Embeddings are computed on this machine with a sentence-transformers model. Document vectors are
cached on disk, keyed by the document's text hash and by a fingerprint of the model, so unchanged
documents are never re-embedded and a different model can never be mixed with an old cache.
Documents and questions are always embedded by the same embedder object.
"""
import hashlib
import json
import os
import pathlib
import tempfile
from importlib import metadata
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# The model card recommends this prefix on search queries, not on the documents themselves.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CACHE_FORMAT = 1
DEFAULT_CACHE_PATH = pathlib.Path(__file__).resolve().parents[2] / ".cache" / "rag" / "embeddings.json"


def embed_model_name() -> str:
    load_dotenv()
    return os.environ.get("EMBED_MODEL", "").strip() or DEFAULT_EMBED_MODEL


def embed_model_revision() -> Optional[str]:
    load_dotenv()
    return os.environ.get("EMBED_MODEL_REVISION", "").strip() or None


def _library_version() -> str:
    try:
        return metadata.version("sentence-transformers")
    except metadata.PackageNotFoundError:
        return "unknown"


class SentenceTransformerEmbedder:
    """Loads the model on first use. `embed_texts` returns one normalised vector per text."""

    def __init__(self, model_name: Optional[str] = None, revision: Optional[str] = None, batch_size: int = 16):
        self.model_name = model_name or embed_model_name()
        self.revision = revision if revision is not None else embed_model_revision()
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed; run `pip install -r requirements.txt` "
                    "before using local embeddings"
                ) from exc
            kwargs = {"revision": self.revision} if self.revision else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def _resolved_revision(self) -> str:
        """The revision actually loaded, so an unpinned model that changed upstream is noticed.

        The commit hash is read from the loaded model when the library exposes it, and otherwise from
        the name of the cached snapshot folder, which is the commit the files were downloaded at.
        """
        try:
            resolved = self._load()[0].auto_model.config._commit_hash
        except Exception:
            resolved = None
        if not resolved:
            try:
                from huggingface_hub import snapshot_download

                resolved = pathlib.Path(
                    snapshot_download(self.model_name, revision=self.revision, local_files_only=True)
                ).name
            except Exception:
                resolved = None
        return resolved or self.revision or "unpinned"

    @property
    def fingerprint(self) -> str:
        model = self._load()
        # the method was renamed in newer library versions; the old name is deprecated there
        dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return (f"{self.model_name}|rev={self._resolved_revision()}"
                f"|sentence-transformers={_library_version()}|dim={dimension()}")

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._load().encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in vectors]


def default_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def _read_cache(path: pathlib.Path) -> Dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict) and data.get("format") == CACHE_FORMAT and isinstance(data.get("docs"), dict):
        return data
    return {}


def _write_cache(path: pathlib.Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as out:
            json.dump(data, out)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_documents_cached(docs: Sequence[Dict], embedder, cache_path: pathlib.Path = DEFAULT_CACHE_PATH) -> Dict[str, List[float]]:
    """doc_id -> vector. Only documents that are new, changed, or from another model are embedded.

    Each doc is a mapping with `doc_id` and `text`. The cache is rewritten only when something changed.
    """
    fingerprint = embedder.fingerprint
    cache = _read_cache(cache_path)
    if cache.get("fingerprint") != fingerprint:
        cache = {"format": CACHE_FORMAT, "fingerprint": fingerprint, "docs": {}}

    vectors: Dict[str, List[float]] = {}
    pending = []
    for doc in docs:
        sha = text_sha256(doc["text"])
        entry = cache["docs"].get(doc["doc_id"])
        if entry and entry.get("text_sha256") == sha and entry.get("vector"):
            vectors[doc["doc_id"]] = entry["vector"]
        else:
            pending.append((doc["doc_id"], sha, doc["text"]))

    if pending:
        fresh = embedder.embed_texts([text for _, _, text in pending])
        for (doc_id, sha, _), vector in zip(pending, fresh):
            vectors[doc_id] = vector
            cache["docs"][doc_id] = {"text_sha256": sha, "vector": vector}
        live = {d["doc_id"] for d in docs}
        cache["docs"] = {k: v for k, v in cache["docs"].items() if k in live}
        cache["dimension"] = len(next(iter(vectors.values())))
        _write_cache(cache_path, cache)
    return vectors
