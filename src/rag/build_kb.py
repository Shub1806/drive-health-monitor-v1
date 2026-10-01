"""Load the reusable knowledge documents and write their metadata to GOLD.KB_DOCS.

`python -m src.rag.build_kb` validates the documents and prints them; it writes nothing.
`python -m src.rag.build_kb --write` upserts them. The table holds text, sources and hashes only;
embeddings are computed locally and cached on disk (see embed.py).
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.rag import db

KB_VERSION = "kb-v1"
KB_TABLE = "DRIVE_HEALTH.GOLD.KB_DOCS"
KNOWLEDGE_DIR = pathlib.Path(__file__).parent / "knowledge"

EXPECTED_DOC_IDS = (
    "attr:smart_5_raw",
    "attr:smart_9_raw",
    "attr:smart_187_raw",
    "attr:smart_188_raw",
    "attr:smart_194_raw",
    "attr:smart_197_raw",
    "attr:smart_198_raw",
    "method:rule_baseline",
    "method:isolation_forest",
    "dataset:backblaze_q2_2026",
)
SOURCE_TYPES = {"attribute": 7, "method": 2, "dataset": 1}
REQUIRED_FIELDS = ("doc_id", "source_type", "title", "source_name", "source_reference", "accessed_on")


@dataclass
class KnowledgeDoc:
    doc_id: str
    source_type: str
    title: str
    text: str
    source_name: str
    source_reference: str
    source_version: Optional[str]
    accessed_on: str
    additional_sources: List[Dict] = field(default_factory=list)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    return "git commit " + out.stdout.strip()


def parse_document(raw: str, resolve_git: Callable[[], str] = git_head) -> KnowledgeDoc:
    """Parse one markdown file: `key: value` front matter between '---' lines, then the text."""
    lines = raw.strip("\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError("knowledge document must start with '---' front matter")
    end = lines.index("---", 1) if "---" in lines[1:] else -1
    if end < 0:
        raise ValueError("knowledge document front matter is not closed with '---'")

    meta: Dict[str, str] = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise ValueError(f"{meta.get('doc_id', '<unknown>')}: missing source metadata {missing}")

    text = "\n".join(lines[end + 1:]).strip()
    if not text:
        raise ValueError(f"{meta['doc_id']}: empty text")

    additional = json.loads(meta["additional_sources"]) if meta.get("additional_sources") else []
    for source in additional:
        if "git:HEAD" == source.get("version"):
            source["version"] = resolve_git()
    version = meta.get("source_version") or None
    if version == "git:HEAD":
        version = resolve_git()

    return KnowledgeDoc(
        doc_id=meta["doc_id"],
        source_type=meta["source_type"],
        title=meta["title"],
        text=text,
        source_name=meta["source_name"],
        source_reference=meta["source_reference"],
        source_version=version,
        accessed_on=meta["accessed_on"],
        additional_sources=additional,
    )


def load_knowledge(directory: pathlib.Path = KNOWLEDGE_DIR, resolve_git: Callable[[], str] = git_head) -> List[KnowledgeDoc]:
    """Load every document and check that exactly the approved 10 exist, with the expected types."""
    docs = [parse_document(p.read_text(), resolve_git) for p in sorted(directory.glob("*.md"))]
    ids = [d.doc_id for d in docs]
    if sorted(ids) != sorted(EXPECTED_DOC_IDS):
        extra, missing = sorted(set(ids) - set(EXPECTED_DOC_IDS)), sorted(set(EXPECTED_DOC_IDS) - set(ids))
        raise ValueError(f"knowledge documents do not match the approved set: extra {extra}, missing {missing}")
    counts = {t: sum(1 for d in docs if d.source_type == t) for t in SOURCE_TYPES}
    if counts != SOURCE_TYPES:
        raise ValueError(f"unexpected document types {counts}, expected {SOURCE_TYPES}")
    return docs


MERGE_SQL = f"""
MERGE INTO {KB_TABLE} t
USING (SELECT %s AS doc_id, %s AS source_type, %s AS title, %s AS text, %s AS source_name,
              %s AS source_reference, %s AS source_version, %s::DATE AS source_accessed_on,
              PARSE_JSON(%s) AS additional_sources, %s AS text_sha256, %s AS kb_version) s
ON t.doc_id = s.doc_id
WHEN MATCHED THEN UPDATE SET
    source_type = s.source_type, title = s.title, text = s.text, source_name = s.source_name,
    source_reference = s.source_reference, source_version = s.source_version,
    source_accessed_on = s.source_accessed_on, additional_sources = s.additional_sources,
    kb_version = s.kb_version, text_sha256 = s.text_sha256
WHEN NOT MATCHED THEN INSERT
    (doc_id, source_type, title, text, source_name, source_reference, source_version,
     source_accessed_on, additional_sources, text_sha256, kb_version)
VALUES
    (s.doc_id, s.source_type, s.title, s.text, s.source_name, s.source_reference, s.source_version,
     s.source_accessed_on, s.additional_sources, s.text_sha256, s.kb_version)
"""


def merge_params(doc: KnowledgeDoc, kb_version: str = KB_VERSION) -> tuple:
    return (
        doc.doc_id, doc.source_type, doc.title, doc.text, doc.source_name, doc.source_reference,
        doc.source_version, doc.accessed_on, json.dumps(doc.additional_sources), doc.text_sha256, kb_version,
    )


def write_documents(cursor, docs: List[KnowledgeDoc], kb_version: str = KB_VERSION) -> int:
    for doc in docs:
        cursor.execute(MERGE_SQL, merge_params(doc, kb_version))
    return len(docs)


def describe(docs: List[KnowledgeDoc]) -> str:
    lines = [f"{len(docs)} knowledge documents ({KB_VERSION})"]
    for d in docs:
        lines.append(f"  {d.doc_id:<28} {d.text_sha256[:12]}  {len(d.text):>4} chars  source: {d.source_name[:60]}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the knowledge documents and optionally write GOLD.KB_DOCS.")
    parser.add_argument("--write", action="store_true", help="upsert the documents into GOLD.KB_DOCS")
    args = parser.parse_args(argv)

    docs = load_knowledge()
    print(describe(docs))
    if not args.write:
        print("dry run: nothing written (use --write to upsert)")
        return 0

    connection = db.connect()
    try:
        written = write_documents(connection.cursor(), docs)
        print(f"upserted {written} documents")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
