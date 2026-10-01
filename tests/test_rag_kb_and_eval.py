import hashlib
import json
import pathlib
import re

import pytest

from eval.evaluate_rag import (
    QUESTION_FILES, evaluate, format_report, hit_at_k, load_questions, reciprocal_rank,
)
from src.rag import build_kb
from src.rag.build_kb import (
    EXPECTED_DOC_IDS, KNOWLEDGE_DIR, SOURCE_TYPES, load_knowledge, merge_params, parse_document, write_documents,
)
from src.rag.retrieve import Hit
from src.rag.validate import CAUSAL_PATTERNS, OUTCOME_PATTERNS, PREDICTION_PATTERNS
from tests.rag_helpers import FakeCursor

GIT = lambda: "git commit TEST"


def doc_text(**overrides):
    fields = {
        "doc_id": "attr:smart_5_raw", "source_type": "attribute", "title": "T", "source_name": "A source",
        "source_reference": "https://example.org", "source_version": "v1", "accessed_on": "2026-10-01",
        "additional_sources": json.dumps([{"name": "x", "reference": "y", "version": "git:HEAD", "accessed_on": "2026-10-01"}]),
    }
    fields.update(overrides)
    front = "\n".join(f"{k}: {v}" for k, v in fields.items() if v is not None)
    return f"---\n{front}\n---\nBody text."


# --- the 10 knowledge documents ---

def test_exactly_the_approved_ten_documents_with_the_approved_types():
    docs = load_knowledge(resolve_git=GIT)

    assert sorted(d.doc_id for d in docs) == sorted(EXPECTED_DOC_IDS) and len(docs) == 10
    assert {t: sum(d.source_type == t for d in docs) for t in SOURCE_TYPES} == {"attribute": 7, "method": 2, "dataset": 1}


def test_every_document_has_source_metadata_and_a_matching_hash():
    for d in load_knowledge(resolve_git=GIT):
        assert d.source_name and d.source_reference and d.accessed_on == "2026-10-01"
        assert d.text and d.text_sha256 == hashlib.sha256(d.text.encode()).hexdigest()
        assert d.additional_sources is not None


def test_documents_that_cite_the_repository_record_the_build_commit():
    docs = {d.doc_id: d for d in load_knowledge(resolve_git=GIT)}

    assert docs["method:rule_baseline"].source_version == "git commit TEST"
    assert docs["method:isolation_forest"].additional_sources[0]["version"] == "git commit TEST"


def test_no_document_mentions_labels_outcomes_causes_or_predictions():
    patterns = [*CAUSAL_PATTERNS, *PREDICTION_PATTERNS, *OUTCOME_PATTERNS]
    for d in load_knowledge(resolve_git=GIT):
        for pattern in patterns:
            assert not re.search(pattern, d.text, flags=re.IGNORECASE), f"{d.doc_id}: {pattern}"


def test_the_isolation_forest_document_states_there_is_no_per_feature_attribution():
    text = {d.doc_id: d.text for d in load_knowledge(resolve_git=GIT)}["method:isolation_forest"]

    assert "does not provide per-feature causal attribution" in text
    assert "not a probability" in text


def test_the_dataset_note_says_outcomes_are_only_for_evaluation():
    text = {d.doc_id: d.text for d in load_knowledge(resolve_git=GIT)}["dataset:backblaze_q2_2026"]

    assert "never used to explain a flag" in text


def test_a_document_without_source_metadata_is_refused():
    for field in ("source_name", "source_reference", "accessed_on", "doc_id", "title"):
        with pytest.raises(ValueError, match="missing source metadata|must start"):
            parse_document(doc_text(**{field: None}), GIT)
    with pytest.raises(ValueError, match="front matter"):
        parse_document("no front matter here", GIT)
    with pytest.raises(ValueError, match="empty text"):
        parse_document(doc_text().replace("Body text.", ""), GIT)


def test_extra_or_missing_documents_are_refused(tmp_path):
    for p in KNOWLEDGE_DIR.glob("*.md"):
        (tmp_path / p.name).write_text(p.read_text())
    (tmp_path / "extra.md").write_text(doc_text(doc_id="attr:smart_999_raw"))
    with pytest.raises(ValueError, match="extra"):
        load_knowledge(tmp_path, GIT)

    (tmp_path / "extra.md").unlink()
    (tmp_path / "attr_smart_5_raw.md").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_knowledge(tmp_path, GIT)


def test_the_merge_binds_every_placeholder_and_carries_no_embedding():
    doc = load_knowledge(resolve_git=GIT)[0]

    assert build_kb.MERGE_SQL.count("%s") == len(merge_params(doc))
    assert "embedding" not in build_kb.MERGE_SQL.lower() and "VECTOR" not in build_kb.MERGE_SQL
    params = merge_params(doc)
    assert params[0] == doc.doc_id and params[-2] == doc.text_sha256 and params[-1] == build_kb.KB_VERSION


def test_writing_upserts_one_row_per_document_and_embeds_nothing():
    docs = load_knowledge(resolve_git=GIT)
    cursor = FakeCursor()

    assert write_documents(cursor, docs) == 10
    assert len(cursor.executed) == 10
    assert "EMBED_TEXT_768" not in cursor.sql_text()


def test_the_dry_run_cli_writes_nothing(capsys, monkeypatch):
    monkeypatch.setattr(build_kb, "git_head", GIT)
    monkeypatch.setattr(build_kb, "load_knowledge", lambda: load_knowledge(resolve_git=GIT))

    assert build_kb.main([]) == 0
    out = capsys.readouterr().out
    assert "10 knowledge documents" in out and "nothing written" in out


# --- evaluation questions and metrics ---

def test_fifteen_evaluation_questions_with_valid_expected_documents_and_one_abstention():
    questions = load_questions(QUESTION_FILES["eval"])

    assert len(questions) == 15 and len({q["id"] for q in questions}) == 15
    assert all(set(q["expected_doc_ids"]) <= set(EXPECTED_DOC_IDS) for q in questions)
    abstain = [q for q in questions if q["abstain"]]
    assert len(abstain) == 1 and abstain[0]["expected_doc_ids"] == []
    assert all(q["expected_doc_ids"] for q in questions if not q["abstain"])


def test_dev_questions_are_separate_from_the_evaluation_questions():
    dev = load_questions(QUESTION_FILES["dev"])
    evaluation = load_questions(QUESTION_FILES["eval"])

    assert len(dev) >= 5 and any(q["abstain"] for q in dev)
    assert not {q["question"].lower() for q in dev} & {q["question"].lower() for q in evaluation}
    assert not {q["id"] for q in dev} & {q["id"] for q in evaluation}


def test_hit_at_k_and_reciprocal_rank():
    ids = ["a", "b", "c", "d"]

    assert hit_at_k(ids, ["a"], 1) and not hit_at_k(ids, ["b"], 1) and hit_at_k(ids, ["b"], 3)
    assert not hit_at_k(ids, ["d"], 3) and hit_at_k(ids, ["x", "c"], 3)
    assert reciprocal_rank(ids, ["b"]) == 0.5 and reciprocal_rank(ids, ["x"]) == 0.0


def hits(*pairs):
    return [Hit(d, d, "t", "s", "r", None, score) for d, score in pairs]


def test_evaluation_reports_hit_at_1_hit_at_3_and_abstention_separately():
    questions = {q["id"]: q for q in load_questions(QUESTION_FILES["eval"])}
    results = [
        (questions["q01"], hits(("attr:smart_5_raw", 0.8), ("attr:smart_9_raw", 0.4))),                       # hit@1
        (questions["q02"], hits(("attr:smart_9_raw", 0.7), ("attr:smart_197_raw", 0.6))),                     # hit@3 only
        (questions["q03"], hits(("method:rule_baseline", 0.5), ("dataset:backblaze_q2_2026", 0.4))),          # miss
        (questions["q15"], hits(("method:rule_baseline", 0.1))),                                              # abstains correctly
    ]

    report = evaluate(results, min_similarity=0.3)

    assert report["retrieval_questions"] == 3
    assert report["hit_at_1"] == pytest.approx(1 / 3) and report["hit_at_3"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert (report["abstention_questions"], report["abstention_passed"]) == (1, 1)

    failing = evaluate([(questions["q15"], hits(("method:rule_baseline", 0.6)))], min_similarity=0.3)
    assert failing["abstention_passed"] == 0
    text = format_report(report)
    assert "hit@1" in text and "hit@3" in text and "abstention: 1/1" in text and "30%" in text


def test_the_evaluation_calls_no_language_model():
    source = pathlib.Path("eval/evaluate_rag.py").read_text()

    assert "COMPLETE" not in source and "make_client" not in source and "llm" not in source.lower().replace("language model", "")
