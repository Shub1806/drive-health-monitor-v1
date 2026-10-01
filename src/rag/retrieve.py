"""Question parsing and knowledge retrieval.

Drive evidence comes from an exact lookup (see report.py). Only the reusable knowledge documents
are searched by similarity, in memory, with vectors computed locally (see embed.py).
"""
import pathlib
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.rag import embed
from src.rag.embed import QUERY_PREFIX

TOP_K = 3
# Chosen on the 6 dev questions only, for BAAI/bge-small-en-v1.5: the midpoint between the dev abstention
# question's top score (0.416) and the lowest correct dev top-1 score (0.686). Not tuned on the evaluation set.
DEFAULT_MIN_SIMILARITY = 0.55


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    source_name: str
    source_reference: str
    source_version: Optional[str]
    score: float


def hit_as_doc(hit: Hit) -> Dict:
    return {"doc_id": hit.doc_id, "title": hit.title, "text": hit.text, "source_name": hit.source_name,
            "source_reference": hit.source_reference, "source_version": hit.source_version}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0


def select_hits(hits: Sequence[Hit], min_similarity: float, top_k: int = TOP_K) -> List[Hit]:
    """Best first, only those at or above the minimum similarity, at most top_k."""
    ranked = sorted(hits, key=lambda h: (-h.score, h.doc_id))
    return [h for h in ranked if h.score >= min_similarity][:top_k]


METHOD_DOC_PREFIX = "method:"


def select_for_drive(hits: Sequence[Hit], method: Optional[str], min_similarity: float, top_k: int = TOP_K) -> List[Hit]:
    """Drive questions: the detector-method document of the run is always included, whatever its score.

    The other detector's method document is dropped so it cannot stand in for it; the remaining
    slots go to the usual thresholded semantic ranking. `hits` must cover the whole index.
    """
    own = f"{METHOD_DOC_PREFIX}{method}" if method else None
    pinned = [h for h in hits if h.doc_id == own]
    if not pinned:
        return select_hits(hits, min_similarity, top_k)
    rest = [h for h in hits if not h.doc_id.startswith(METHOD_DOC_PREFIX)]
    chosen = pinned + select_hits(rest, min_similarity, top_k - 1)
    return sorted(chosen, key=lambda h: (-h.score, h.doc_id))


class LocalKnowledgeIndex:
    """In-memory index over precomputed document vectors, ranked by cosine similarity.

    `embed` embeds a final query text; the instruction prefix is added here, once.
    """

    def __init__(self, docs: Sequence[Mapping], embed: Callable[[str], Sequence[float]]):
        self.docs = list(docs)
        self.embed = embed

    def search(self, question: str, top_k: Optional[int] = TOP_K) -> List[Hit]:
        query = self.embed(QUERY_PREFIX + question)
        hits = [
            Hit(d["doc_id"], d["title"], d["text"], d["source_name"], d["source_reference"],
                d.get("source_version"), cosine(query, d["vector"]))
            for d in self.docs
        ]
        return sorted(hits, key=lambda h: (-h.score, h.doc_id))[:top_k]   # top_k=None: the whole ranking


def build_local_index(embedder=None, docs: Optional[Sequence] = None,
                      cache_path: pathlib.Path = embed.DEFAULT_CACHE_PATH) -> LocalKnowledgeIndex:
    """Index over the knowledge documents, using one embedder for both documents and questions.

    Document vectors come from the on-disk cache when the text and the model are unchanged.
    """
    from src.rag.build_kb import load_knowledge

    docs = list(docs) if docs is not None else load_knowledge()
    embedder = embedder or embed.default_embedder()
    vectors = embed.embed_documents_cached([{"doc_id": d.doc_id, "text": d.text} for d in docs], embedder, cache_path)
    index = LocalKnowledgeIndex(
        [
            {"doc_id": d.doc_id, "title": d.title, "text": d.text, "source_name": d.source_name,
             "source_reference": d.source_reference, "source_version": d.source_version, "vector": vectors[d.doc_id]}
            for d in docs
        ],
        lambda text: embedder.embed_texts([text])[0],
    )
    index.fingerprint = embedder.fingerprint
    return index


# --- question parsing ------------------------------------------------------------------------

_TOKEN = re.compile(r"\b[A-Za-z0-9]{6,20}\b")


def serial_candidates(question: str, limit: int = 25) -> List[str]:
    """Tokens that could be serial numbers: 6 to 20 letters or digits."""
    seen, out = set(), []
    for token in _TOKEN.findall(question):
        upper = token.upper()
        if upper not in seen:
            seen.add(upper)
            out.append(upper)
    return out[:limit]


def looks_like_serial(token: str) -> bool:
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def method_hint(question: str) -> Optional[str]:
    q = question.lower()
    if "isolation" in q or "forest" in q:
        return "isolation_forest"
    if "rule" in q:
        return "rule_baseline"
    return None


def role_hint(question: str) -> Optional[str]:
    q = question.lower()
    if "june" in q:
        return "official"
    if "may" in q.split() or " may " in f" {q} ":
        return "supplemental"
    return None


def filter_by_role(rows: Sequence[Mapping], role: Optional[str]) -> List[Mapping]:
    return [r for r in rows if role is None or r["window_role"] == role]
