import json

import numpy as np
import pytest

from src.rag import embed, retrieve
from src.rag.llm import CortexCompleteClient, LLMError, parse_complete_response
from src.rag.retrieve import (
    Hit, LocalKnowledgeIndex, cosine, filter_by_role, looks_like_serial, method_hint,
    role_hint, select_hits, serial_candidates,
)
from tests.rag_helpers import FakeCursor


def hit(doc_id, score):
    return Hit(doc_id, doc_id, "text", "source", "ref", None, score)


# --- question parsing ---

def test_serial_candidates_are_unique_uppercase_tokens_of_6_to_20_characters():
    # plain words of 6+ letters are candidates too; the database lookup decides which are real serials
    out = serial_candidates("Why was drive zhz43wys flagged? Also ZHZ43WYS and ST14000NM0138, ok?")

    assert out == ["ZHZ43WYS", "FLAGGED", "ST14000NM0138"]
    assert serial_candidates("a short one") == []
    assert len(serial_candidates(" ".join(f"ABCDE{i:02d}" for i in range(60)))) == 25


def test_a_serial_like_token_has_both_letters_and_digits():
    assert looks_like_serial("ZHZ43WYS") and looks_like_serial("2CGPDVXP")
    assert not looks_like_serial("BASELINE") and not looks_like_serial("12345678")


def test_method_and_window_hints():
    assert method_hint("what did the isolation forest say?") == "isolation_forest"
    assert method_hint("was it flagged by the rule baseline") == "rule_baseline"
    assert method_hint("why was it flagged") is None
    assert role_hint("flagged in June") == "official"
    assert role_hint("in May it was flagged") == "supplemental"
    assert role_hint("someday it may be flagged") == "supplemental"   # "may" as a word is read as the month
    assert role_hint("why was it flagged") is None
    rows = [{"window_role": "official"}, {"window_role": "supplemental"}]
    assert filter_by_role(rows, "official") == [rows[0]] and filter_by_role(rows, None) == rows


# --- similarity ---

def test_cosine_similarity():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def test_select_hits_orders_best_first_filters_by_minimum_and_caps_at_top_k():
    hits = [hit("c", 0.5), hit("a", 0.9), hit("d", 0.2), hit("b", 0.7), hit("e", 0.65)]

    assert [h.doc_id for h in select_hits(hits, 0.3)] == ["a", "b", "e"]
    assert [h.doc_id for h in select_hits(hits, 0.6)] == ["a", "b", "e"]
    assert [h.doc_id for h in select_hits(hits, 0.8)] == ["a"]
    assert select_hits(hits, 0.95) == []     # nothing relevant: abstain
    assert [h.doc_id for h in select_hits([hit("z", 0.5), hit("y", 0.5)], 0.0)] == ["y", "z"]


VOCAB = ["sector", "remapped", "temperature", "celsius", "forest", "random", "tree", "rows", "quarter", "bread"]


def embed_words(text):
    words = text.lower().replace(".", "").split()
    return [float(words.count(w)) for w in VOCAB]


def local_index():
    docs = [
        ("attr:smart_197_raw", "sector remapped sector"),
        ("attr:smart_194_raw", "temperature celsius temperature"),
        ("method:isolation_forest", "forest random tree"),
        ("dataset:backblaze_q2_2026", "rows quarter rows"),
    ]
    return LocalKnowledgeIndex(
        [{"doc_id": d, "title": d, "text": t, "source_name": "s", "source_reference": "r", "vector": embed_words(t)} for d, t in docs],
        embed_words,
    )


def test_local_index_ranks_the_relevant_document_first():
    top = local_index().search("which sector was remapped", top_k=3)

    assert top[0].doc_id == "attr:smart_197_raw" and len(top) == 3
    assert top[0].score > top[1].score


def test_an_unrelated_question_scores_zero_so_the_abstention_threshold_applies():
    hits = local_index().search("how to bake bread", top_k=3)

    assert select_hits(hits, 0.3) == []






# --- model client ---

def test_the_response_text_and_token_counts_are_read_from_either_shape():
    a = {"choices": [{"messages": "hello"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    b = {"choices": [{"message": {"content": "hi"}}]}

    assert parse_complete_response(json.dumps(a)) == ("hello", 12, 3)
    assert parse_complete_response(b) == ("hi", None, None)
    for bad in ("not json", {"choices": []}, {"nothing": 1}):
        with pytest.raises(LLMError):
            parse_complete_response(bad)


def test_the_cortex_client_sends_the_model_messages_and_deterministic_options():
    reply = json.dumps({"choices": [{"messages": "answer"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
    cursor = FakeCursor().respond("CORTEX.COMPLETE", [(reply,)])
    client = CortexCompleteClient(cursor, "some-model", max_tokens=300)

    result = client.complete([{"role": "user", "content": "hi"}])

    sql, params = cursor.executed[0]
    assert "SNOWFLAKE.CORTEX.COMPLETE" in sql
    assert params[0] == "some-model" and json.loads(params[1]) == [{"role": "user", "content": "hi"}]
    assert json.loads(params[2]) == {"temperature": 0.0, "max_tokens": 300}
    assert (result.text, result.prompt_tokens, result.completion_tokens, result.model) == ("answer", 5, 2, "some-model")


def test_a_failed_model_call_raises_llm_error():
    class Broken(FakeCursor):
        def execute(self, sql, params=None):
            raise RuntimeError("no such model")

    with pytest.raises(LLMError, match="no such model"):
        CortexCompleteClient(Broken(), "m").complete([])


# --- drive questions always carry their own detector-method document ---

def ranking(**scores):
    return [hit(k.replace("__", ":"), v) for k, v in scores.items()]


SEMANTIC = ranking(method__rule_baseline=0.61, attr__smart_188_raw=0.58, attr__smart_9_raw=0.56,
                   method__isolation_forest=0.40, dataset__backblaze_q2_2026=0.30)


def ids(hits):
    return [h.doc_id for h in hits]


def test_a_rule_baseline_drive_always_gets_the_rule_baseline_document():
    chosen = retrieve.select_for_drive(SEMANTIC, "rule_baseline", 0.55)

    assert ids(chosen) == ["method:rule_baseline", "attr:smart_188_raw", "attr:smart_9_raw"]


def test_an_isolation_forest_drive_gets_its_method_document_even_below_the_threshold():
    chosen = retrieve.select_for_drive(SEMANTIC, "isolation_forest", 0.55)

    assert "method:isolation_forest" in ids(chosen)
    assert next(h.score for h in chosen if h.doc_id == "method:isolation_forest") == 0.40    # real score, not inflated


def test_the_wrong_detector_document_never_replaces_or_accompanies_the_right_one():
    chosen = ids(retrieve.select_for_drive(SEMANTIC, "isolation_forest", 0.55))

    assert chosen == ["attr:smart_188_raw", "attr:smart_9_raw", "method:isolation_forest"]
    assert "method:rule_baseline" not in chosen


def test_remaining_slots_stay_semantic_and_thresholded():
    chosen = retrieve.select_for_drive(SEMANTIC, "isolation_forest", 0.59)

    assert ids(chosen) == ["method:isolation_forest"]


def test_a_drive_context_never_exceeds_three_documents():
    for method in ("rule_baseline", "isolation_forest"):
        assert len(retrieve.select_for_drive(SEMANTIC, method, 0.0)) == 3


def test_an_unknown_or_missing_method_falls_back_to_plain_semantic_selection():
    plain = retrieve.select_hits(SEMANTIC, 0.55)

    assert retrieve.select_for_drive(SEMANTIC, None, 0.55) == plain
    assert retrieve.select_for_drive(SEMANTIC, "other", 0.55) == plain
    assert retrieve.select_for_drive([h for h in SEMANTIC if h.doc_id != "method:isolation_forest"], "isolation_forest", 0.55) == plain


def test_the_index_can_return_its_whole_ranking():
    index = local_index()

    assert len(index.search("anything", top_k=None)) == len(index.docs)
