import pytest

from src.rag.ask import explanation_record, generate_explanation
from src.rag.prompt import assemble_context
from src.rag.report import build_report, evidence_only_answer
from src.rag.validate import canonicalize_citations, cited_ids, validate_answer
from tests.rag_helpers import FakeClient, RULE_STATS, SERIAL, firing_features, rule_anomaly
from tests.test_rag_validate_claims import ALL_KNOWLEDGE, LIMITS

REPORT = build_report(rule_anomaly(), firing_features(), RULE_STATS)
RID = REPORT.doc_id
CONTEXT = assemble_context("why was this drive flagged?", REPORT, ALL_KNOWLEDGE)    # attr:smart_197_raw, attr:smart_5_raw, both method docs
FALLBACK = evidence_only_answer(REPORT, ALL_KNOWLEDGE)

QUOTE_197 = "Lower is better."
QUOTE_RULE = "The score counts triggers."


def answer(background="", detector=None, observed=None):
    return (f"## Observed telemetry\n{observed or f'Drive {SERIAL} was flagged on 2026-06-19 [{RID}].'}\n"
            f"## How the detector scored this drive\n{detector or f'The rule baseline scored this drive 8 of 8 [{RID}].'}\n"
            f"## General background (not specific to this drive)\n{background}\n## Limits\n{LIMITS}")


# --- the canonicalization function ---

def test_a_knowledge_prefixed_id_of_a_supplied_document_becomes_the_canonical_id():
    assert canonicalize_citations(f"Lower is better [KNOWLEDGE:attr:smart_197_raw].", CONTEXT) == "Lower is better [attr:smart_197_raw]."


def test_a_knowledge_prefixed_method_document_id_becomes_the_canonical_id():
    assert canonicalize_citations("The score counts triggers [KNOWLEDGE:method:rule_baseline].", CONTEXT) == \
        "The score counts triggers [method:rule_baseline]."


def test_an_unknown_knowledge_id_is_left_as_written_so_it_is_still_rejected():
    text = answer(background="SMART 999 counts things [KNOWLEDGE:attr:smart_999_raw].")

    assert canonicalize_citations(text, CONTEXT) == text
    assert "unknown_citation" in {e.code for e in validate_answer(canonicalize_citations(text, CONTEXT), CONTEXT)}


def test_a_document_that_exists_but_was_not_supplied_in_this_context_is_not_canonicalized():
    only_197 = assemble_context("q", REPORT, [d for d in ALL_KNOWLEDGE if d["doc_id"] == "attr:smart_197_raw"])
    text = "The score counts triggers [KNOWLEDGE:method:rule_baseline]."

    assert canonicalize_citations(text, only_197) == text


def test_a_report_id_behind_the_prefix_is_never_accepted():
    text = answer(observed=f"Drive {SERIAL} was flagged on 2026-06-19 [KNOWLEDGE:{RID}].")

    assert canonicalize_citations(text, CONTEXT) == text
    errors = validate_answer(canonicalize_citations(text, CONTEXT), CONTEXT)
    assert {e.code for e in errors} >= {"unknown_citation"}
    assert any(f"[KNOWLEDGE:{RID}]" in e.detail for e in errors if e.code == "unknown_citation")


def test_ordinary_canonical_citations_are_unchanged():
    text = answer(background=f"{QUOTE_197} [attr:smart_197_raw]", detector=f"The rule baseline scored this drive 8 of 8 [{RID}, method:rule_baseline].")

    assert canonicalize_citations(text, CONTEXT) == text


@pytest.mark.parametrize("bad", [
    "[knowledge:attr:smart_197_raw]",                # different letter case
    "[Knowledge:attr:smart_197_raw]",
    "[KNOWLEDGE: attr:smart_197_raw]",               # a space is not an exact id
    "[KNOWLEDGE:KNOWLEDGE:attr:smart_197_raw]",      # nested prefix
    "[SOURCE:attr:smart_197_raw]",                   # another prefix
    "[docs:attr:smart_197_raw]",
    "[KNOWLEDGE:attr:smart_197_raw_extra]",          # not exactly a supplied id
    "[KNOWLEDGE:]",
])
def test_only_the_exact_knowledge_prefix_is_accepted(bad):
    text = f"Lower is better {bad}."

    assert canonicalize_citations(text, CONTEXT) == text


def test_a_group_is_fixed_part_by_part_and_other_text_is_untouched():
    text = f"Facts [KNOWLEDGE:attr:smart_197_raw, {RID}; KNOWLEDGE:attr:smart_999_raw] and KNOWLEDGE:attr:smart_197_raw outside."

    assert canonicalize_citations(text, CONTEXT) == f"Facts [attr:smart_197_raw, {RID}; KNOWLEDGE:attr:smart_999_raw] and KNOWLEDGE:attr:smart_197_raw outside."


def test_the_validator_itself_stays_strict_about_the_alias():
    raw = answer(background=f"{QUOTE_197} [KNOWLEDGE:attr:smart_197_raw]")

    assert "unknown_citation" in {e.code for e in validate_answer(raw, CONTEXT)}              # raw alias, no canonicalization
    assert validate_answer(canonicalize_citations(raw, CONTEXT), CONTEXT) == []              # after the compatibility step


# --- in the generate / retry flow ---

def test_an_answer_that_only_needs_canonicalizing_passes_on_the_first_attempt_with_canonical_ids():
    raw = answer(background=f"{QUOTE_197} [KNOWLEDGE:attr:smart_197_raw]\n{QUOTE_RULE} [KNOWLEDGE:method:rule_baseline]")
    client = FakeClient(raw)

    result = generate_explanation(CONTEXT, client, FALLBACK)

    assert (result.validation_status, result.attempts, len(client.calls)) == ("passed", 1, 1)
    assert "KNOWLEDGE:" not in result.answer
    assert "[attr:smart_197_raw]" in result.answer and "[method:rule_baseline]" in result.answer


def test_the_stored_citation_ids_are_canonical_only():
    raw = answer(background=f"{QUOTE_197} [KNOWLEDGE:attr:smart_197_raw]\n{QUOTE_RULE} [KNOWLEDGE:method:rule_baseline]")
    result = generate_explanation(CONTEXT, FakeClient(raw), FALLBACK)

    record = explanation_record(question="q", context=CONTEXT, request_kind="drive", anomaly=rule_anomaly(), hits=[],
                                min_similarity=0.55, generation=result, answer=result.answer, answer_source=result.answer_source,
                                validation_status=result.validation_status, llm_model="m", embed_model="e", key="k")

    assert record["cited_doc_ids"] == sorted(["attr:smart_197_raw", "method:rule_baseline", RID])
    assert all(not i.startswith("KNOWLEDGE:") for i in record["cited_doc_ids"])
    assert "KNOWLEDGE:" not in record["answer_text"]
    assert cited_ids(result.answer) <= set(CONTEXT.allowed_ids)


def test_an_unknown_alias_still_fails_and_goes_through_the_normal_retry_and_fallback():
    bad = answer(background="SMART 999 counts things [KNOWLEDGE:attr:smart_999_raw].")
    client = FakeClient(bad, bad)

    result = generate_explanation(CONTEXT, client, FALLBACK)

    assert (result.validation_status, result.answer, len(client.calls)) == ("fallback", FALLBACK, 2)
    assert "unknown_citation" in str(result.errors[0])
    assert "KNOWLEDGE:attr:smart_999_raw" in client.calls[1][-2]["content"]       # shown to the model as written, unfixed


def test_a_report_alias_fails_the_flow_and_is_never_canonicalized():
    bad = answer(observed=f"Drive {SERIAL} was flagged [KNOWLEDGE:{RID}].")
    client = FakeClient(bad, bad)

    result = generate_explanation(CONTEXT, client, FALLBACK)

    assert result.validation_status == "fallback" and result.answer == FALLBACK


def test_the_retry_message_carries_the_canonical_text_of_the_previous_answer():
    first = answer(background=f"{QUOTE_197} [KNOWLEDGE:attr:smart_197_raw]", detector="The rule baseline scored this drive 8 of 8.")
    good = answer(background=f"{QUOTE_197} [attr:smart_197_raw]")
    client = FakeClient(first, good)

    result = generate_explanation(CONTEXT, client, FALLBACK)

    assert result.validation_status == "passed_after_retry"
    assert "KNOWLEDGE:" not in client.calls[1][-2]["content"]
    assert "[attr:smart_197_raw]" in client.calls[1][-2]["content"]
    assert "unknown_citation" not in client.calls[1][-1]["content"]              # the alias is no longer an error to fix


def test_an_answer_without_aliases_is_returned_exactly_as_the_model_wrote_it():
    good = answer(background=f"{QUOTE_197} [attr:smart_197_raw]")

    assert generate_explanation(CONTEXT, FakeClient(good), FALLBACK).answer == good


def test_general_questions_canonicalize_the_same_way():
    context = assemble_context("what is smart 197?", None, ALL_KNOWLEDGE)
    raw = "## Answer\nLower is better [KNOWLEDGE:attr:smart_197_raw].\n## Limits\nThis describes background documents. It is not a diagnosis, and it does not predict failure."

    result = generate_explanation(context, FakeClient(raw), "fallback")

    assert result.validation_status == "passed" and "[attr:smart_197_raw]" in result.answer and "KNOWLEDGE:" not in result.answer
