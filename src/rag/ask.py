"""Ask why a drive was flagged, or a general question about the knowledge documents.

    python -m src.rag.ask "why was drive <serial> flagged?"

Drive questions: the stored run(s) for that drive are looked up exactly, a report is built from
persisted rows, and up to three knowledge documents are retrieved by similarity. A drive that
appears in several runs gets one explanation per run, official runs first, at most four.
General questions use the knowledge documents only. Evaluation outcomes are never read or shown.
"""
import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

from src.rag import db, embed, report as report_mod, retrieve
from src.rag.build_kb import KB_VERSION
from src.rag.llm import make_client
from src.rag.prompt import PROMPT_VERSION, assemble_context, build_messages, retry_messages, system_prompt
from src.rag.validate import canonicalize_citations, cited_ids, validate_answer

EXPLANATIONS_TABLE = "DRIVE_HEALTH.GOLD.EXPLANATIONS"
MAX_ATTEMPTS = 2

NO_STORED_RESULT = "No stored anomaly result matches this question, so there is no drive evidence to explain."
NO_KNOWLEDGE = "The knowledge documents do not contain anything relevant to this question."

RECORD_COLUMNS = (
    "explanation_id", "created_at", "kb_version", "question", "question_sha256", "request_kind", "run_id",
    "method", "serial_number", "prompt_version", "llm_model", "llm_params", "embedding_model",
    "retrieved_doc_ids", "retrieval_scores", "min_similarity_used", "drive_report_sha256",
    "retrieved_context_sha256", "cited_doc_ids", "answer_text", "answer_source", "validation_status",
    "validation_errors", "llm_attempts", "prompt_tokens", "completion_tokens", "latency_ms", "cache_key",
)
JSON_COLUMNS = ("llm_params", "retrieved_doc_ids", "retrieval_scores", "cited_doc_ids", "validation_errors")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(question: str, run_id: Optional[str], prompt_version: str, llm_model: str, kb_version: str,
              embed_model: str, report_sha: Optional[str]) -> str:
    parts = [question.strip(), run_id or "", prompt_version, llm_model, kb_version, embed_model, report_sha or ""]
    return sha("\x1f".join(parts))


@dataclass
class GenerationResult:
    answer: str
    answer_source: str          # llm | llm_retry | fallback_evidence_only
    validation_status: str      # passed | passed_after_retry | fallback
    errors: List[Dict] = field(default_factory=list)
    attempts: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: int = 0


def _add(a: Optional[int], b: Optional[int]) -> Optional[int]:
    return b if a is None else a if b is None else a + b


def generate_explanation(context, client, fallback_text: str, validator: Callable = validate_answer) -> GenerationResult:
    """At most two model calls. An answer that still fails validation is replaced by the fallback."""
    base = build_messages(context)
    messages = base
    log: List[Dict] = []
    result = GenerationResult(answer=fallback_text, answer_source="fallback_evidence_only", validation_status="fallback")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            llm = client.complete(messages)
        except Exception as exc:
            log.append({"attempt": attempt, "error": f"model call failed: {str(exc)[:300]}"})
            result.errors, result.attempts = log, attempt
            return result
        result.attempts = attempt
        result.prompt_tokens = _add(result.prompt_tokens, llm.prompt_tokens)
        result.completion_tokens = _add(result.completion_tokens, llm.completion_tokens)
        result.latency_ms += llm.latency_ms

        text = canonicalize_citations(llm.text, context)     # [KNOWLEDGE:<supplied id>] -> [<supplied id>], nothing else
        errors = validator(text, context)
        if not errors:
            result.answer = text
            result.answer_source = "llm" if attempt == 1 else "llm_retry"
            result.validation_status = "passed" if attempt == 1 else "passed_after_retry"
            result.errors = log
            return result
        log.append({"attempt": attempt, "errors": [{"code": e.code, "detail": e.detail[:300]} for e in errors]})
        messages = retry_messages(base, text, errors)

    result.errors = log
    return result


def safe_validation_errors(log: Optional[Sequence[Dict]]) -> List[Dict]:
    """What may be persisted about failed attempts: attempt number and error codes only.

    The runtime log keeps error details (which quote rejected model sentences); they are never stored.
    """
    safe: List[Dict] = []
    for entry in log or []:
        if "errors" in entry:
            safe.append({"attempt": entry["attempt"], "errors": [{"code": e["code"]} for e in entry["errors"]]})
        else:
            safe.append({"attempt": entry["attempt"], "errors": [{"code": "model_call_failed"}]})
    return safe


def explanation_record(*, question: str, context, request_kind: str, anomaly: Optional[Dict], hits: Sequence,
                       min_similarity: float, generation: Optional[GenerationResult], answer: str,
                       answer_source: str, validation_status: str, llm_model: str, embed_model: str,
                       key: str, llm_options: Optional[Dict] = None) -> Dict:
    """The row stored in GOLD.EXPLANATIONS. It carries ids, hashes and text only; no outcome fields exist."""
    generation = generation or GenerationResult(answer, answer_source, validation_status)
    return {
        "explanation_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "kb_version": KB_VERSION,
        "question": question,
        "question_sha256": sha(question.strip()),
        "request_kind": request_kind,
        "run_id": anomaly["run_id"] if anomaly else None,
        "method": anomaly["method"] if anomaly else None,
        "serial_number": anomaly["serial_number"] if anomaly else None,
        "prompt_version": PROMPT_VERSION,
        "llm_model": llm_model,
        "llm_params": llm_options or {},
        "embedding_model": embed_model,
        "retrieved_doc_ids": [h.doc_id for h in hits],
        "retrieval_scores": [float(h.score) for h in hits],
        "min_similarity_used": float(min_similarity),
        "drive_report_sha256": context.report.sha256 if context is not None and context.report else None,
        "retrieved_context_sha256": context.context_sha256 if context is not None else None,
        "cited_doc_ids": sorted(cited_ids(answer)),
        "answer_text": answer,
        "answer_source": answer_source,
        "validation_status": validation_status,
        "validation_errors": safe_validation_errors(generation.errors),
        "llm_attempts": generation.attempts,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "latency_ms": generation.latency_ms,
        "cache_key": key,
    }


def insert_sql() -> str:
    selects = ", ".join("PARSE_JSON(%s)" if c in JSON_COLUMNS else "%s" for c in RECORD_COLUMNS)
    # insert-if-absent on cache_key (the hash of question, run, prompt, model, kb and report): a repeat never duplicates a row
    return (f"INSERT INTO {EXPLANATIONS_TABLE} ({', '.join(RECORD_COLUMNS)}) SELECT {selects} "
            f"WHERE NOT EXISTS (SELECT 1 FROM {EXPLANATIONS_TABLE} WHERE cache_key = %s)")


def insert_params(record: Dict) -> tuple:
    values = tuple(json.dumps(record[c]) if c in JSON_COLUMNS else record[c] for c in RECORD_COLUMNS)
    return values + (record["cache_key"],)


def store_record(cursor, record: Dict) -> bool:
    try:
        cursor.execute(insert_sql(), insert_params(record))
        return True
    except Exception as exc:
        print(f"warning: explanation not stored ({str(exc)[:120]})")
        return False


CACHE_SQL = (
    f"SELECT answer_text, answer_source, validation_status FROM {EXPLANATIONS_TABLE} "
    "WHERE cache_key = %s ORDER BY created_at DESC LIMIT 1"
)


def lookup_cached(cursor, key: str):
    try:
        cursor.execute(CACHE_SQL, (key,))
        return cursor.fetchone()
    except Exception:
        return None


@dataclass
class Outcome:
    answer: str
    answer_source: str
    validation_status: str
    anomaly: Optional[Dict] = None
    from_cache: bool = False
    record: Optional[Dict] = None


@dataclass
class Settings:
    llm_model: str = ""
    embed_model: str = embed.DEFAULT_EMBED_MODEL
    min_similarity: float = retrieve.DEFAULT_MIN_SIMILARITY
    refresh: bool = False
    dry_run: bool = False
    store: bool = True
    llm_options: Dict = field(default_factory=dict)


def _drive_outcome(cursor, question, anomaly, client, index, s: Settings) -> Outcome:
    report = report_mod.report_for_run(cursor, anomaly)
    if s.dry_run:
        text = (f"[dry run] system prompt {PROMPT_VERSION}, no retrieval and no model call.\n\n"
                + system_prompt("drive") + "\n\n" + report.text)
        return Outcome(text, "dry_run", "dry_run", anomaly)

    hits = retrieve.select_for_drive(index.search(question, top_k=None), anomaly["method"], s.min_similarity)
    docs = [retrieve.hit_as_doc(h) for h in hits]
    context = assemble_context(question, report, docs)
    key = cache_key(question, anomaly["run_id"], PROMPT_VERSION, s.llm_model, KB_VERSION, s.embed_model, report.sha256)

    if not s.refresh:
        cached = lookup_cached(cursor, key)
        if cached:
            return Outcome(cached[0], cached[1], cached[2], anomaly, from_cache=True)

    generation = generate_explanation(context, client, report_mod.evidence_only_answer(report, docs))
    record = explanation_record(
        question=question, context=context, request_kind="drive", anomaly=anomaly, hits=hits,
        min_similarity=s.min_similarity, generation=generation, answer=generation.answer,
        answer_source=generation.answer_source, validation_status=generation.validation_status,
        llm_model=s.llm_model, embed_model=s.embed_model, key=key, llm_options=s.llm_options,
    )
    if s.store:
        store_record(cursor, record)
    return Outcome(generation.answer, generation.answer_source, generation.validation_status, anomaly, record=record)


def _general_outcome(cursor, question, client, index, s: Settings) -> Outcome:
    if s.dry_run:
        return Outcome("[dry run] general question: no retrieval and no model call.", "dry_run", "dry_run")
    hits = retrieve.select_hits(index.search(question), s.min_similarity)
    key = cache_key(question, None, PROMPT_VERSION, s.llm_model, KB_VERSION, s.embed_model, None)
    if not hits:
        record = explanation_record(
            question=question, context=None, request_kind="general", anomaly=None, hits=[],
            min_similarity=s.min_similarity, generation=None, answer=NO_KNOWLEDGE, answer_source="no_evidence",
            validation_status="no_evidence", llm_model=s.llm_model, embed_model=s.embed_model, key=key,
        )
        if s.store:
            store_record(cursor, record)
        return Outcome(NO_KNOWLEDGE, "no_evidence", "no_evidence", record=record)

    docs = [retrieve.hit_as_doc(h) for h in hits]
    context = assemble_context(question, None, docs)
    if not s.refresh:
        cached = lookup_cached(cursor, key)
        if cached:
            return Outcome(cached[0], cached[1], cached[2], from_cache=True)
    generation = generate_explanation(context, client, report_mod.evidence_only_answer(None, docs))
    record = explanation_record(
        question=question, context=context, request_kind="general", anomaly=None, hits=hits,
        min_similarity=s.min_similarity, generation=generation, answer=generation.answer,
        answer_source=generation.answer_source, validation_status=generation.validation_status,
        llm_model=s.llm_model, embed_model=s.embed_model, key=key, llm_options=s.llm_options,
    )
    if s.store:
        store_record(cursor, record)
    return Outcome(generation.answer, generation.answer_source, generation.validation_status, record=record)


def explain_question(cursor, question: str, client, index, s: Settings, run_id: Optional[str] = None,
                     method: Optional[str] = None) -> List[Outcome]:
    candidates = retrieve.serial_candidates(question)
    serials = report_mod.existing_serials(cursor, candidates)
    unknown_serial_like = [c for c in candidates if retrieve.looks_like_serial(c) and c not in serials]

    if not serials:
        if unknown_serial_like:
            return [Outcome(f"{NO_STORED_RESULT} (looked for {unknown_serial_like[0]}.)", "no_evidence", "no_evidence")]
        return [_general_outcome(cursor, question, client, index, s)]

    serial = serials[0]
    rows = report_mod.fetch_runs(cursor, serial, run_id, method or retrieve.method_hint(question))
    rows = retrieve.filter_by_role(rows, retrieve.role_hint(question))
    if not rows:
        return [Outcome(f"{NO_STORED_RESULT} (drive {serial}.)", "no_evidence", "no_evidence")]
    return [_drive_outcome(cursor, question, row, client, index, s) for row in rows]


def format_outcome(outcome: Outcome) -> str:
    if outcome.anomaly:
        a = outcome.anomaly
        head = (f"--- run {a['run_id'][:8]} | {a['method']} | {a['window_role']} window | rank {a['rank']} "
                f"| {outcome.validation_status}{' (cached)' if outcome.from_cache else ''} ---")
    else:
        head = f"--- {outcome.validation_status}{' (cached)' if outcome.from_cache else ''} ---"
    return head + "\n" + outcome.answer


def main(argv: Optional[Sequence[str]] = None, connect: Callable = db.connect) -> int:
    parser = argparse.ArgumentParser(description="Explain stored anomaly results.")
    parser.add_argument("question")
    parser.add_argument("--run", help="only this run id")
    parser.add_argument("--method", choices=["rule_baseline", "isolation_forest"])
    parser.add_argument("--refresh", action="store_true", help="ignore a cached explanation")
    parser.add_argument("--dry-run", action="store_true", help="show the drive report and system prompt; no retrieval or model call")
    parser.add_argument("--no-store", action="store_true", help="do not save the explanation")
    parser.add_argument("--min-similarity", type=float, default=None)
    args = parser.parse_args(argv)

    connection = connect()
    try:
        cursor = connection.cursor()
        client = None if args.dry_run else make_client(cursor)
        # One embedder, loaded once, embeds both the documents and the question.
        index = None if args.dry_run else retrieve.build_local_index()
        settings = Settings(
            llm_model=getattr(client, "model", ""),
            embed_model=index.fingerprint if index else embed.embed_model_name(),
            min_similarity=args.min_similarity if args.min_similarity is not None else retrieve.DEFAULT_MIN_SIMILARITY,
            refresh=args.refresh, dry_run=args.dry_run, store=not args.no_store,
            llm_options=getattr(client, "options", {}),
        )
        for outcome in explain_question(cursor, args.question, client, index, settings, args.run, args.method):
            print(format_outcome(outcome))
            print()
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
