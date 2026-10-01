import json
import re
import sys
from types import SimpleNamespace

import pytest

from src.rag import ask, report as report_mod
from src.rag.ask import (
    MAX_ATTEMPTS, NO_KNOWLEDGE, RECORD_COLUMNS, Settings, cache_key, explain_question, explanation_record,
    generate_explanation, insert_params, insert_sql,
)
from src.rag.prompt import assemble_context, build_messages
from src.rag.report import build_report, evidence_only_answer
from src.rag.retrieve import Hit
from tests.rag_helpers import (
    IF_RUN, IF_STATS, KNOWLEDGE, RULE_RUN, RULE_STATS, SERIAL, FakeClient, FakeCursor, drive_cursor, firing_features,
    if_anomaly, rule_anomaly,
)

QUESTION = f"why was drive {SERIAL} flagged?"
RULE_REPORT = build_report(rule_anomaly(), firing_features(), RULE_STATS)
CONTEXT = assemble_context(QUESTION, RULE_REPORT, KNOWLEDGE)
FALLBACK = evidence_only_answer(RULE_REPORT, KNOWLEDGE)
GOOD = FALLBACK                      # a grounded answer
BAD = GOOD + "\nThis drive will fail soon."


class FakeIndex:
    def __init__(self, hits=None):
        self.hits = hits if hits is not None else [
            Hit(k["doc_id"], k["title"], k["text"], k["source_name"], k["source_reference"], k["source_version"], 0.8 - i / 10)
            for i, k in enumerate(KNOWLEDGE)
        ]
        self.questions = []

    def search(self, question, top_k=3):
        self.questions.append(question)
        return self.hits


def settings(**kw):
    return Settings(llm_model="fake-model", embed_model="embed-model", min_similarity=0.3, **kw)


# --- generation, retry, fallback ---

def test_a_valid_first_answer_is_used_after_one_call():
    client = FakeClient(GOOD)

    result = generate_explanation(CONTEXT, client, FALLBACK)

    assert (result.answer_source, result.validation_status, result.attempts) == ("llm", "passed", 1)
    assert result.answer == GOOD and len(client.calls) == 1 and result.errors == []
    assert (result.prompt_tokens, result.completion_tokens, result.latency_ms) == (100, 50, 7)


def test_an_invalid_answer_is_retried_once_with_the_errors_and_then_accepted():
    client = FakeClient(BAD, GOOD)

    result = generate_explanation(CONTEXT, client, FALLBACK)

    assert (result.answer_source, result.validation_status, result.attempts) == ("llm_retry", "passed_after_retry", 2)
    assert result.answer == GOOD and len(client.calls) == 2
    retry = client.calls[1]
    assert retry[:2] == build_messages(CONTEXT) and retry[2] == {"role": "assistant", "content": BAD}
    assert "failure_prediction" in retry[3]["content"]
    assert result.errors[0]["errors"][0]["code"] == "failure_prediction"
    assert (result.prompt_tokens, result.completion_tokens) == (200, 100)


def test_two_invalid_answers_give_the_evidence_only_fallback_and_never_a_third_call():
    client = FakeClient(BAD, BAD, GOOD)

    result = generate_explanation(CONTEXT, client, FALLBACK)

    assert MAX_ATTEMPTS == 2 and len(client.calls) == 2
    assert result.answer == FALLBACK and BAD not in result.answer
    assert (result.answer_source, result.validation_status, result.attempts) == ("fallback_evidence_only", "fallback", 2)
    assert [e["attempt"] for e in result.errors] == [1, 2]


def test_a_failing_model_call_falls_back_without_retrying():
    client = FakeClient(fail=True)

    result = generate_explanation(CONTEXT, client, FALLBACK)

    assert result.answer == FALLBACK and result.validation_status == "fallback" and len(client.calls) == 1
    assert "simulated model failure" in result.errors[0]["error"]


# --- one explanation per run ---

def test_a_drive_in_several_runs_gets_one_explanation_per_run_official_first():
    supplemental_rule = rule_anomaly(window_role="supplemental", run_id="55555555-aaaa")
    anomalies = [supplemental_rule, if_anomaly(window_role="official"), rule_anomaly(window_role="official")]
    cursor = drive_cursor(anomalies)

    def valid_answer_for(messages):
        # a grounded answer for whichever run's report is in the prompt
        user = messages[1]["content"]
        if f"report:{IF_RUN[:8]}:" in user:
            anomaly, stats = if_anomaly(window_role="official"), IF_STATS
        elif "report:55555555:" in user:
            anomaly, stats = supplemental_rule, RULE_STATS
        else:
            anomaly, stats = rule_anomaly(window_role="official"), RULE_STATS
        return evidence_only_answer(build_report(anomaly, firing_features(), stats), KNOWLEDGE)

    class Echo(FakeClient):
        def complete(self, messages):
            self.calls.append(list(messages))
            return self._result(valid_answer_for(messages), 10, 5, 1, self.model)

    echo = Echo()
    outcomes = explain_question(cursor, QUESTION, echo, FakeIndex(), settings(store=False))

    assert len(outcomes) == 3 and len(echo.calls) == 3
    assert [o.anomaly["method"] for o in outcomes] == ["rule_baseline", "isolation_forest", "rule_baseline"]
    assert [o.anomaly["window_role"] for o in outcomes] == ["official", "official", "supplemental"]
    assert all(o.validation_status == "passed" for o in outcomes)
    prompts = [m[1]["content"] for m in echo.calls]
    assert f"report:{RULE_RUN[:8]}:{SERIAL}" in prompts[0] and f"report:{IF_RUN[:8]}:{SERIAL}" in prompts[1]
    assert "report:55555555:" in prompts[2]


def test_the_drive_report_is_built_from_exact_lookups_and_only_knowledge_is_searched():
    cursor = drive_cursor([rule_anomaly()])
    index = FakeIndex()
    client = FakeClient(GOOD)

    explain_question(cursor, QUESTION, client, index, settings(store=False))

    assert index.questions == [QUESTION]          # the question is searched against the knowledge documents only
    sql = cursor.sql_text()
    assert "JOIN DRIVE_HEALTH.GOLD.MODEL_RUNS" in sql and "FROM DRIVE_HEALTH.GOLD.DRIVE_FEATURES" in sql
    assert not re.search(r"\b(failure_within_30d|label_is_complete|eval_\w+)\b", sql)
    assert not re.search(r"\bfailure\b", sql)


def test_retrieved_knowledge_below_the_minimum_similarity_is_left_out():
    cursor = drive_cursor([rule_anomaly()])
    weak = [Hit("attr:smart_197_raw", "t", "x", "s", "r", None, 0.1)]

    class Capture(FakeClient):
        def complete(self, messages):
            self.calls.append(list(messages))
            return self._result(evidence_only_answer(RULE_REPORT, []), 1, 1, 1, "m")

    capture = Capture()
    outcome = explain_question(cursor, QUESTION, capture, FakeIndex(weak), settings(store=False))[0]

    assert "=== KNOWLEDGE" not in capture.calls[0][1]["content"]
    assert outcome.validation_status == "passed"


# --- dry run, no evidence, cache ---

def test_dry_run_calls_no_model_and_no_cortex_function():
    cursor = drive_cursor([rule_anomaly()])
    client = FakeClient()

    outcome = explain_question(cursor, QUESTION, client, FakeIndex(), settings(dry_run=True))[0]

    assert client.calls == [] and "CORTEX" not in cursor.sql_text().upper()
    assert "[dry run]" in outcome.answer and RULE_REPORT.text in outcome.answer
    assert outcome.validation_status == "dry_run"


def test_an_unknown_drive_gets_a_fixed_reply_and_no_model_call():
    cursor = FakeCursor().respond("SELECT DISTINCT serial_number", [])
    client = FakeClient()
    index = FakeIndex()

    outcome = explain_question(cursor, "why was drive ZZ99AA11 flagged?", client, index, settings(store=False))[0]

    assert outcome.answer_source == "no_evidence" and "ZZ99AA11" in outcome.answer
    assert client.calls == [] and index.questions == []


def test_a_known_drive_with_no_matching_run_gets_a_fixed_reply():
    cursor = drive_cursor([])      # the serial exists, but no stored run matches the filters
    client = FakeClient()

    outcome = explain_question(cursor, QUESTION, client, FakeIndex(), settings(store=False))[0]

    assert outcome.answer_source == "no_evidence" and SERIAL in outcome.answer and client.calls == []


def test_a_general_question_uses_only_knowledge_and_abstains_when_nothing_is_relevant():
    cursor = FakeCursor().respond("SELECT DISTINCT serial_number", [])
    good = "## Answer\nSMART 197 counts sectors waiting to be remapped [attr:smart_197_raw].\n## Limits\nThis describes background documents. It is not a diagnosis, and it does not predict failure."
    client = FakeClient(good)

    outcome = explain_question(cursor, "what does smart 197 count?", client, FakeIndex(), settings(store=False))[0]
    assert outcome.validation_status == "passed" and outcome.anomaly is None

    silent = FakeClient()
    nothing = explain_question(cursor, "best brand to buy?", silent, FakeIndex([]), settings(store=False))[0]
    assert nothing.answer == NO_KNOWLEDGE and nothing.answer_source == "no_evidence" and silent.calls == []


def test_a_cached_explanation_is_returned_without_a_model_call_unless_refreshed():
    cursor = drive_cursor([rule_anomaly()])
    cursor.respond("FROM DRIVE_HEALTH.GOLD.EXPLANATIONS", [("cached answer", "llm", "passed")])
    client = FakeClient(GOOD)

    cached = explain_question(cursor, QUESTION, client, FakeIndex(), settings(store=False))[0]
    assert cached.from_cache and cached.answer == "cached answer" and client.calls == []

    fresh = explain_question(cursor, QUESTION, client, FakeIndex(), settings(store=False, refresh=True))[0]
    assert not fresh.from_cache and len(client.calls) == 1


def test_the_explanation_is_stored_with_ids_hashes_and_no_outcome_fields():
    cursor = drive_cursor([rule_anomaly()])
    client = FakeClient(GOOD)

    outcome = explain_question(cursor, QUESTION, client, FakeIndex(), settings())[0]

    inserts = [(s, p) for s, p in cursor.executed if s.startswith("INSERT INTO")]
    assert len(inserts) == 1
    record = outcome.record
    assert record["run_id"] == RULE_RUN and record["serial_number"] == SERIAL and record["method"] == "rule_baseline"
    assert record["drive_report_sha256"] == RULE_REPORT.sha256 and len(record["retrieved_context_sha256"]) == 64
    assert record["retrieved_doc_ids"] == ["attr:smart_197_raw", "dataset:backblaze_q2_2026"]
    assert all(isinstance(s, float) for s in record["retrieval_scores"])
    assert RULE_REPORT.doc_id in record["cited_doc_ids"]
    assert record["validation_status"] == "passed" and record["llm_attempts"] == 1


# --- schema and keys ---

def test_cache_key_changes_with_every_component():
    base = ("q", RULE_RUN, "prompt_v1", "model", "kb-v1", "embed", "abc")
    key = cache_key(*base)

    assert key == cache_key(*base)
    for i in range(len(base)):
        changed = list(base)
        changed[i] = changed[i] + "x"
        assert cache_key(*changed) != key


def test_stored_columns_match_the_ddl_and_include_no_outcome_column():
    ddl = open("sql/05_rag_tables.sql").read()
    block = ddl.split("GOLD.EXPLANATIONS (")[1].split("PRIMARY KEY")[0]
    ddl_columns = re.findall(r"^\s{4}(\w+)\s", block, flags=re.MULTILINE)

    assert list(RECORD_COLUMNS) == ddl_columns
    assert not set(RECORD_COLUMNS) & set(report_mod.FORBIDDEN_COLUMNS)
    assert not any(c.startswith("eval_") for c in RECORD_COLUMNS)
    sql = insert_sql()
    assert sql.count("PARSE_JSON(%s)") == len(ask.JSON_COLUMNS) and sql.count("%s") == len(RECORD_COLUMNS) + 1
    assert "WHERE NOT EXISTS" in sql and "cache_key = %s" in sql
    record = explanation_record(
        question="q", context=CONTEXT, request_kind="drive", anomaly=rule_anomaly(), hits=[], min_similarity=0.3,
        generation=None, answer=GOOD, answer_source="llm", validation_status="passed", llm_model="m",
        embed_model="e", key="k",
    )
    assert len(insert_params(record)) == len(RECORD_COLUMNS) + 1 and insert_params(record)[-1] == "k"


def test_kb_docs_ddl_has_no_outcome_column_and_matches_the_merge():
    from src.rag import build_kb

    ddl = open("sql/05_rag_tables.sql").read()
    block = ddl.split("GOLD.KB_DOCS (")[1].split("PRIMARY KEY")[0]
    ddl_columns = re.findall(r"^\s{4}(\w+)\s", block, flags=re.MULTILINE)

    assert "VECTOR" not in ddl and "embedding" not in " ".join(ddl_columns[:12])   # retrieval does not use Snowflake vectors
    assert {"doc_id", "source_type", "title", "text", "source_name", "source_reference", "source_version",
            "source_accessed_on", "additional_sources", "text_sha256", "kb_version", "created_at"} == set(ddl_columns)
    assert not set(ddl_columns) & set(report_mod.FORBIDDEN_COLUMNS)
    assert all(c in build_kb.MERGE_SQL for c in ("doc_id", "text_sha256", "kb_version", "source_accessed_on"))


# --- command line ---

def test_the_cli_runs_a_dry_run_and_closes_the_connection(capsys):
    cursor = drive_cursor([rule_anomaly()])
    closed = []
    connection = SimpleNamespace(cursor=lambda: cursor, close=lambda: closed.append(True))

    code = ask.main([QUESTION, "--dry-run"], connect=lambda: connection)

    out = capsys.readouterr().out
    assert code == 0 and closed == [True]
    assert "--- dry_run ---" in out or "dry_run" in out
    assert RULE_REPORT.doc_id in out
    assert "CORTEX" not in cursor.sql_text().upper()
    assert "eval_" not in out


def test_the_cli_refuses_an_unsupported_provider(monkeypatch):
    cursor = FakeCursor()
    connection = SimpleNamespace(cursor=lambda: cursor, close=lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "other")

    with pytest.raises(ValueError, match="is not supported"):
        ask.main([QUESTION], connect=lambda: connection)


# --- detector-method routing ---

def method_index():
    def doc(i, score):
        return Hit(i, i, f"{i} text", "s", "r", None, score)
    return FakeIndex([doc("method:rule_baseline", 0.9), doc("attr:smart_197_raw", 0.8), doc("attr:smart_5_raw", 0.7),
                      doc("method:isolation_forest", 0.2)])


class Capture(FakeClient):
    def complete(self, messages):
        self.calls.append(list(messages))
        return self._result("## x", 1, 1, 1, "m")


def retrieved_ids(anomaly):
    outcome = explain_question(drive_cursor([anomaly]), QUESTION, Capture(), method_index(), settings(store=False))[0]
    return outcome.record["retrieved_doc_ids"]


def test_an_isolation_forest_drive_retrieves_its_own_method_document_not_the_rule_baseline_one():
    got = retrieved_ids(if_anomaly(window_role="official"))

    assert "method:isolation_forest" in got and "method:rule_baseline" not in got
    assert got == ["attr:smart_197_raw", "attr:smart_5_raw", "method:isolation_forest"]


def test_a_rule_baseline_drive_retrieves_the_rule_baseline_document():
    got = retrieved_ids(rule_anomaly(window_role="official"))

    assert got == ["method:rule_baseline", "attr:smart_197_raw", "attr:smart_5_raw"]


def test_a_general_question_stays_purely_semantic():
    outcome = explain_question(FakeCursor(), "what does smart 197 measure?", Capture(), method_index(), settings(store=False))[0]

    assert outcome.record["retrieved_doc_ids"] == ["method:rule_baseline", "attr:smart_197_raw", "attr:smart_5_raw"]


# --- persisted validation_errors hold codes only ---

REJECTED = "The drive will fail soon because the head is wearing out."


def stored_record(client):
    result = generate_explanation(CONTEXT, client, FALLBACK)
    record = explanation_record(question="q", context=CONTEXT, request_kind="drive", anomaly=rule_anomaly(), hits=[], min_similarity=0.3,
                                generation=result, answer=result.answer, answer_source=result.answer_source,
                                validation_status=result.validation_status, llm_model="m", embed_model="e", key="k")
    return result, record


def test_failed_attempt_text_is_absent_from_the_persisted_record():
    from tests.test_rag_citation_ids import answer

    bad = answer(background="Purple sectors orbit the platter and hum loudly [attr:smart_197_raw].")
    result, record = stored_record(FakeClient(bad, bad))

    assert result.validation_status == "fallback"
    assert any("Purple sectors" in e["detail"] for entry in result.errors for e in entry["errors"])   # the runtime log keeps the snippet
    assert "Purple sectors" not in json.dumps(record, default=str)                                      # no column of the record has it
    assert "Purple sectors" not in " ".join(str(v) for v in insert_params(record))


def test_error_codes_and_attempt_numbers_are_preserved():
    bad = GOOD + "\n" + REJECTED
    result, record = stored_record(FakeClient(bad, bad))

    assert [e["attempt"] for e in record["validation_errors"]] == [1, 2]
    for entry, runtime in zip(record["validation_errors"], result.errors):
        assert entry["errors"] == [{"code": e["code"]} for e in runtime["errors"]] and entry["errors"]
        assert all(set(e) == {"code"} for e in entry["errors"])


def test_passed_after_retry_records_the_failure_without_the_rejected_text():
    first = GOOD + "\n" + REJECTED
    result, record = stored_record(FakeClient(first, GOOD))

    assert record["validation_status"] == "passed_after_retry" and record["llm_attempts"] == 2
    assert record["validation_errors"] and record["validation_errors"][0]["attempt"] == 1
    assert record["validation_errors"][0]["errors"] and all(set(e) == {"code"} for e in record["validation_errors"][0]["errors"])
    assert "wearing out" not in json.dumps(record["validation_errors"]) and "wearing out" not in record["answer_text"]


def test_a_passed_answer_stores_an_empty_error_list_and_the_answer_is_unchanged():
    result, record = stored_record(FakeClient(GOOD))
    _, retried = stored_record(FakeClient(GOOD + "\n" + REJECTED, GOOD))

    assert record["validation_errors"] == [] and record["answer_text"] == GOOD
    assert retried["answer_text"] == GOOD and retried["answer_source"] == "llm_retry"


def test_a_failed_model_call_is_stored_as_a_code_not_the_exception_text():
    class Down(FakeClient):
        def complete(self, messages):
            raise RuntimeError("connection refused: secret-ish detail")

    result, record = stored_record(Down())

    assert record["validation_errors"] == [{"attempt": 1, "errors": [{"code": "model_call_failed"}]}]
    assert "secret-ish" not in json.dumps(record, default=str)
