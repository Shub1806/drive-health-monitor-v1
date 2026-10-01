import pytest

from src.rag.prompt import (
    DRIVE_HEADINGS, PROMPT_VERSION, assemble_context, build_messages, messages_json, retry_messages, system_prompt,
)
from src.rag.report import build_report, evidence_only_answer
from src.rag.validate import ValidationError, numbers_in, split_sections, validate_answer
from tests.rag_helpers import (
    IF_STATS, KNOWLEDGE, RULE_STATS, SERIAL, firing_features, if_anomaly, rule_anomaly,
)

RULE_REPORT = build_report(rule_anomaly(), firing_features(), RULE_STATS)
IF_REPORT = build_report(if_anomaly(), firing_features(), IF_STATS)
QUESTION = "why was this drive flagged?"


def drive_context(report=RULE_REPORT, knowledge=KNOWLEDGE):
    return assemble_context(QUESTION, report, knowledge)


def codes(errors):
    return {e.code for e in errors}


def good_answer(report=RULE_REPORT, knowledge=KNOWLEDGE):
    return evidence_only_answer(report, knowledge)


def test_a_grounded_answer_passes():
    assert validate_answer(good_answer(), drive_context()) == []
    assert validate_answer(good_answer(IF_REPORT), drive_context(IF_REPORT)) == []


def test_a_model_style_answer_with_only_context_numbers_passes():
    rid = RULE_REPORT.doc_id
    answer = f"""## Observed telemetry
Drive {SERIAL} was flagged on 2026-06-19 with a score of 8 of 8 [{rid}]. smart_197_raw had a z-score of 85.4 and rose by +360 [{rid}].
## How the detector scored this drive
The rule baseline counts triggers; all 8 fired [{rid}].
## General background (not specific to this drive)
SMART 197 counts sectors waiting to be remapped [attr:smart_197_raw]. The project uses 31,968,003 drive-day rows [dataset:backblaze_q2_2026].
## Limits
This describes stored detector output and observed telemetry. It is not a diagnosis, and it does not predict failure."""

    assert validate_answer(answer, drive_context()) == []


def test_numbers_ignore_thousands_separators_and_split_dates():
    assert numbers_in("31,968,003 rows on 2026-06-19 with z 85.4") == {"31968003", "2026", "06", "19", "85.4"}


# --- structure and citations ---

def test_a_missing_section_is_reported():
    answer = good_answer().replace("## Limits", "## Something else")

    assert "missing_section" in codes(validate_answer(answer, drive_context()))
    assert set(DRIVE_HEADINGS) == {"Observed telemetry", "How the detector scored this drive", "General background", "Limits"}


def test_a_citation_outside_the_context_is_rejected():
    answer = good_answer() + "\nSee also [attr:smart_999_raw]."

    errors = validate_answer(answer, drive_context())

    assert "unknown_citation" in codes(errors)
    assert any("attr:smart_999_raw" in e.detail for e in errors)


def test_the_telemetry_section_must_cite_the_drive_report():
    answer = good_answer().replace(f" [{RULE_REPORT.doc_id}]", "")

    assert "missing_citation" in codes(validate_answer(answer, drive_context()))


def test_background_must_cite_knowledge_when_it_was_retrieved_and_say_so_when_not():
    no_background_cited = good_answer().replace(" [attr:smart_197_raw]", "").replace(" [dataset:backblaze_q2_2026]", "")
    assert "missing_citation" in codes(validate_answer(no_background_cited, drive_context()))

    empty = good_answer(RULE_REPORT, [])
    assert validate_answer(empty, drive_context(knowledge=[])) == []
    silent = empty.replace("No relevant background documents were retrieved.", "")
    assert "missing_citation" in codes(validate_answer(silent, drive_context(knowledge=[])))


# --- numbers ---

def test_a_rounded_or_invented_number_is_rejected():
    rounded = good_answer().replace("85.4", "85")
    invented = good_answer() + "\nThe drive has 4242 bad sectors [" + RULE_REPORT.doc_id + "]."

    assert "unsupported_number" in codes(validate_answer(rounded, drive_context()))
    errors = validate_answer(invented, drive_context())
    assert any(e.code == "unsupported_number" and "4242" in e.detail for e in errors)


def test_list_numbering_is_not_treated_as_a_telemetry_number():
    answer = good_answer() + "\n1. A numbered note with no figures."

    assert "unsupported_number" not in codes(validate_answer(answer, drive_context()))


# --- wording ---

@pytest.mark.parametrize("sentence", [
    "The values rose because the drive is wearing out.",
    "This was caused by bad sectors.",
    "The rise is due to a failing head.",
    "That leads to data loss.",
])
def test_unsupported_causal_wording_is_rejected(sentence):
    assert "causal_wording" in codes(validate_answer(good_answer() + "\n" + sentence, drive_context()))


@pytest.mark.parametrize("sentence", [
    "This drive will fail soon.",
    "The drive is about to fail.",
    "Failure is likely.",
    "The detector predicts failure for this drive.",
    "Failure looks imminent.",
])
def test_failure_predictions_are_rejected(sentence):
    assert "failure_prediction" in codes(validate_answer(good_answer() + "\n" + sentence, drive_context()))


def test_a_negated_disclaimer_is_allowed():
    answer = good_answer() + "\nThe detector does not predict failure, and it cannot say what causes the values."

    errors = validate_answer(answer, drive_context())

    assert "failure_prediction" not in codes(errors) and "causal_wording" not in codes(errors)


@pytest.mark.parametrize("sentence", [
    "Its failure_within_30d value was 1.",
    "The label_is_complete flag is false.",
    "See eval_failure for the outcome.",
    "The drive has failed since.",
])
def test_outcome_and_label_references_are_rejected(sentence):
    assert "outcome_reference" in codes(validate_answer(good_answer() + "\n" + sentence, drive_context()))


def test_the_non_failure_day_training_phrase_is_not_an_outcome_reference():
    assert "outcome_reference" not in codes(validate_answer(good_answer(IF_REPORT), drive_context(IF_REPORT)))


def test_an_isolation_forest_answer_must_state_there_is_no_per_feature_attribution():
    answer = good_answer(IF_REPORT).replace("It does not provide per-feature causal attribution.", "")

    errors = validate_answer(answer, drive_context(IF_REPORT))

    assert "missing_attribution_statement" in codes(errors)
    assert "missing_attribution_statement" not in codes(validate_answer(good_answer(), drive_context()))


# --- general questions ---

def test_general_mode_needs_an_answer_section_with_a_knowledge_citation():
    context = assemble_context("what is smart 197?", None, KNOWLEDGE)
    ok = "## Answer\nSMART 197 counts sectors waiting to be remapped [attr:smart_197_raw].\n## Limits\nThis describes background documents. It is not a diagnosis, and it does not predict failure."

    assert validate_answer(ok, context) == []
    assert "missing_citation" in codes(validate_answer(ok.replace(" [attr:smart_197_raw]", ""), context))
    assert "missing_section" in codes(validate_answer("Just an answer [attr:smart_197_raw]", context))


def test_split_sections_finds_each_heading():
    sections = split_sections(good_answer(), DRIVE_HEADINGS)

    assert all(body is not None for body in sections.values())
    assert sections["Limits"].strip().startswith("## Limits")


# --- prompt layout ---

def test_the_prompt_contains_only_the_supplied_evidence():
    context = drive_context()
    system, user = build_messages(context)

    assert system["role"] == "system" and user["role"] == "user"
    assert user["content"].startswith(f"QUESTION: {QUESTION}")
    assert f"=== DRIVE REPORT [{RULE_REPORT.doc_id}] ===\n{RULE_REPORT.text}" in user["content"]
    assert user["content"].count("=== KNOWLEDGE [") == 2
    assert "=== KNOWLEDGE [attr:smart_197_raw] SMART 197 - Current Pending Sector Count | source: Backblaze blog ===" in user["content"]
    for text in (system["content"], user["content"]):
        assert "failure_within_30d" not in text and "label_is_complete" not in text and "eval_" not in text
    assert context.allowed_ids == [RULE_REPORT.doc_id, "attr:smart_197_raw", "dataset:backblaze_q2_2026"]


def test_the_system_prompt_carries_the_rules_and_the_mandatory_sentence():
    prompt = system_prompt("drive")

    assert "Use only the text in the DRIVE REPORT and KNOWLEDGE blocks" in prompt
    assert "Do not state or guess a cause" in prompt
    assert "does not predict failure" in prompt
    assert "It does not provide per-feature causal attribution." in prompt
    for heading in DRIVE_HEADINGS:
        assert heading in prompt
    assert "## Answer" in system_prompt("general") and "## Observed telemetry" not in system_prompt("general")


def test_context_hash_is_stable_and_depends_on_the_evidence():
    a, b = drive_context(), drive_context()
    changed = drive_context(knowledge=KNOWLEDGE[:1])

    assert a.context_sha256 == b.context_sha256
    assert a.context_sha256 != changed.context_sha256
    assert assemble_context(QUESTION, None, KNOWLEDGE).mode == "general"


def test_the_retry_message_lists_every_error():
    base = build_messages(drive_context())
    errors = [ValidationError("unsupported_number", "numbers not found in the context: ['85']"),
              ValidationError("causal_wording", "unsupported causal wording: ['because']")]

    messages = retry_messages(base, "first answer", errors)

    assert messages[:2] == base and len(messages) == 4
    assert messages[2] == {"role": "assistant", "content": "first answer"}
    assert "unsupported_number" in messages[3]["content"] and "causal_wording" in messages[3]["content"]
    assert PROMPT_VERSION == "prompt_v1"
    assert '"role": "system"' in messages_json(messages)
