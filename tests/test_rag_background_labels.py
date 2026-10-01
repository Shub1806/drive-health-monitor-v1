import pytest

from src.rag.prompt import assemble_context
from src.rag.report import build_report
from src.rag.validate import validate_answer
from tests.rag_helpers import RULE_STATS, SERIAL, firing_features, rule_anomaly
from tests.test_rag_validate_claims import ALL_KNOWLEDGE, LIMITS

REPORT = build_report(rule_anomaly(), firing_features(), RULE_STATS)
RID = REPORT.doc_id
CONTEXT = assemble_context("why was this drive flagged?", REPORT, ALL_KNOWLEDGE)

TEXT_197 = next(d["text"] for d in ALL_KNOWLEDGE if d["doc_id"] == "attr:smart_197_raw")
TEXT_RULE = next(d["text"] for d in ALL_KNOWLEDGE if d["doc_id"] == "method:rule_baseline")


def answer(background):
    return (f"## Observed telemetry\nDrive {SERIAL} was flagged on 2026-06-19 [{RID}].\n"
            f"## How the detector scored this drive\nThe rule baseline scored this drive 8 of 8 [{RID}].\n"
            f"## General background (not specific to this drive)\n{background}\n## Limits\n{LIMITS}")


def background_errors(line):
    return [e for e in validate_answer(answer(line), CONTEXT) if e.code not in ()]


def codes(line):
    return {e.code for e in background_errors(line)}


def test_the_documents_used_here_have_text():
    assert TEXT_197 and TEXT_RULE


# a sentence the document supports, written as a paraphrase that is only supported once the label is set aside
ATTR_BODY = "SMART 197 Current Pending Sector Count. " + TEXT_197.split(". ", 1)[1].split(". ")[0] + ". " + TEXT_197.split(". ", 2)[2].split(". ")[0] + "."


@pytest.mark.parametrize("label", ["**attr:smart_197_raw**:", "attr:smart_197_raw:", "* **attr:smart_197_raw**:", "- **attr:smart_197_raw**:"])
def test_a_supplied_attribute_label_with_a_supported_sentence_and_matching_citation_is_accepted(label):
    first = TEXT_197.split(". ")[0] + "."
    line = f"{label} {first} [attr:smart_197_raw]"

    assert validate_answer(answer(line), CONTEXT) == []


def test_a_supplied_method_label_with_supported_detector_background_is_accepted():
    first = TEXT_RULE.split(". ")[0] + "."

    assert validate_answer(answer(f"**method:rule_baseline**: {first} [method:rule_baseline]"), CONTEXT) == []


def test_an_unknown_document_label_is_rejected():
    first = TEXT_197.split(". ")[0] + "."

    assert "unsupported_citation" in codes(f"**attr:smart_999_raw**: {first} [attr:smart_197_raw]")


def test_a_label_naming_a_different_document_than_the_one_cited_is_rejected():
    first = TEXT_197.split(". ")[0] + "."

    assert "unsupported_citation" in codes(f"**method:rule_baseline**: {first} [attr:smart_197_raw]")
    assert "unsupported_citation" in codes(f"**attr:smart_197_raw**: {first} [method:rule_baseline]")


def test_a_correct_label_does_not_make_an_unsupported_sentence_acceptable():
    assert "unsupported_citation" in codes("**attr:smart_197_raw**: The drive is noisy and hot and will overheat soon. [attr:smart_197_raw]")


def test_a_label_alone_is_not_evidence():
    assert "unsupported_citation" in codes("**attr:smart_197_raw**: [attr:smart_197_raw]")


def test_a_report_id_used_as_a_background_label_is_rejected():
    first = TEXT_197.split(". ")[0] + "."

    assert "unsupported_citation" in codes(f"**{RID}**: {first} [attr:smart_197_raw]")
    assert "unsupported_citation" in codes(f"**{RID}**: {first} [{RID}]")


def test_only_a_leading_well_formed_label_naming_a_supplied_document_is_stripped():
    from src.rag.validate import _leading_label

    supplied = {d["doc_id"] for d in ALL_KNOWLEDGE}

    assert _leading_label("**attr:smart_197_raw**: text", supplied) == ("attr:smart_197_raw", "text")
    assert _leading_label("attr:smart_197_raw: text", supplied) == ("attr:smart_197_raw", "text")
    assert _leading_label("Note **attr:smart_197_raw**: text", supplied) is None          # not at the start
    assert _leading_label("**attr:smart_197_raw text", supplied) is None                  # malformed
    assert _leading_label("*attr:smart_197_raw*: text", supplied) is None                 # other markup
    assert _leading_label("**attr:smart_999_raw**: text", supplied) is None               # unknown
    assert _leading_label(f"**{RID}**: text", supplied) is None                           # report id
    assert _leading_label("**KNOWLEDGE:attr:smart_197_raw**: text", supplied) is None     # other prefix


def test_sentences_without_a_label_behave_as_before():
    first = TEXT_197.split(". ")[0] + "."

    assert validate_answer(answer(f"{first} [attr:smart_197_raw]"), CONTEXT) == []
    assert "unsupported_citation" in codes("The drive is noisy and hot and will overheat soon. [attr:smart_197_raw]")
    assert "missing_citation" in codes(f"{first}")
