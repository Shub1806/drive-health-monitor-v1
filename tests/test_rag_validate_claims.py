import pytest

from src.rag.build_kb import load_knowledge
from src.rag.prompt import assemble_context
from src.rag.report import build_report, evidence_only_answer
from src.rag.validate import (
    check_claims, extract_claims, section_sentences, split_sections, validate_answer,
)
from src.rag.prompt import DRIVE_HEADINGS
from tests.rag_helpers import (
    IF_STATS, KNOWLEDGE, RULE_STATS, SERIAL, firing_features, if_anomaly, rule_anomaly,
)

LIMITS = "This describes stored detector output and observed telemetry. It is not a diagnosis, and it does not predict failure."
METHOD_DOCS = {
    d.doc_id: {"doc_id": d.doc_id, "title": d.title, "text": d.text, "source_name": d.source_name,
               "source_reference": d.source_reference, "source_version": d.source_version}
    for d in load_knowledge(resolve_git=lambda: "git commit TEST") if d.doc_id.startswith(("method:", "attr:smart_197", "attr:smart_5_"))
}
ALL_KNOWLEDGE = [METHOD_DOCS[i] for i in ("attr:smart_197_raw", "attr:smart_5_raw", "method:rule_baseline", "method:isolation_forest")]

# only the real SMART 197 document and the dataset note: no SMART 198 document and no method documents
REAL_DOCS = [METHOD_DOCS["attr:smart_197_raw"]]
RULE_REPORT = build_report(rule_anomaly(), firing_features(), RULE_STATS)
IF_REPORT = build_report(if_anomaly(), firing_features(), IF_STATS)
RID, IID = RULE_REPORT.doc_id, IF_REPORT.doc_id


def rule_context(knowledge=KNOWLEDGE):
    return assemble_context("why was this drive flagged?", RULE_REPORT, knowledge)


def if_context(knowledge=ALL_KNOWLEDGE):
    return assemble_context("why was this drive flagged?", IF_REPORT, knowledge)


def answer(observed=None, detector=None, background=None, rid=RID, method="rule baseline"):
    observed = observed if observed is not None else f"Drive {SERIAL} was flagged on 2026-06-19 [{rid}]."
    detector = detector if detector is not None else f"The {method} scored this drive 8 of 8 [{rid}]."
    background = background if background is not None else "SMART 197 counts sectors waiting to be remapped [attr:smart_197_raw]."
    return (f"## Observed telemetry\n{observed}\n## How the detector scored this drive\n{detector}\n"
            f"## General background (not specific to this drive)\n{background}\n## Limits\n{LIMITS}")


def codes(text, context):
    return {e.code for e in validate_answer(text, context)}


def details(text, context, code):
    return [e.detail for e in validate_answer(text, context) if e.code == code]


# --- the report carries the structured values the claims are checked against ---

def test_the_report_exposes_structured_facts_for_each_attribute_and_field():
    facts = RULE_REPORT.facts

    assert facts["attributes"]["smart_197_raw"] == {"raw": 600, "mean": 300.0, "delta": 360.0, "z": 85.4}
    assert facts["attributes"]["smart_5_raw"]["z"] == 29.2 and facts["rank"] == 1 and facts["score"] == 8
    assert len(facts["raw_by_date"]["smart_5_raw"]) == 7 and len(facts["raw_by_date"]["smart_9_raw"]) == 1


# --- 1. misattributed telemetry numbers ---

@pytest.mark.parametrize("sentence, wrong_label", [
    ("smart_197_raw had a z-score of 29.2", "smart_197_raw z"),                       # 29.2 is SMART 5's z-score
    ("smart_197_raw rose by 37 since the previous day", "smart_197_raw delta"),        # 37 is SMART 187's change
    ("The 7-day mean for smart_198_raw was 50.0", "smart_198_raw mean"),               # 50.0 is SMART 5's mean
    ("The raw value of smart_198_raw was 100", "smart_198_raw raw"),                   # 100 is SMART 5's raw value
])
def test_a_number_that_exists_but_belongs_to_another_attribute_is_rejected(sentence, wrong_label):
    text = answer(observed=f"{sentence} [{RID}].")

    found = details(text, rule_context(), "misattributed_number")

    assert found and wrong_label in found[0]
    assert "unsupported_number" not in codes(text, rule_context())      # the number itself does exist in the context


def test_an_attribute_id_is_never_split_into_an_attribute_and_a_number():
    # without a number after the cue, a loose pattern can read "smart_187_raw" as smart_18 followed by 7
    for sentence in ("The z-score of smart_187_raw was not reported", "smart_187_raw rose since the previous day",
                     "The 7-day mean for smart_187_raw was not reported"):
        assert extract_claims(sentence) == [], sentence
    claim = extract_claims("The z-score for smart_187_raw was 34.2")[0]
    assert (claim.attribute, claim.field, claim.value) == ("smart_187_raw", "z", 34.2)


def test_the_right_attribute_with_the_right_numbers_is_accepted_for_every_field():
    text = answer(observed=(f"smart_197_raw had a z-score of 85.4, a change since the previous day of +360, "
                            f"a 7-day mean of 300.0 and a raw value of 600 [{RID}]."))

    assert validate_answer(text, rule_context()) == []


def test_claims_are_checked_per_attribute_within_one_sentence():
    good = answer(observed=f"smart_5_raw (z-score 29.2, increased by 16) and smart_197_raw (z-score 85.4, increased by 360) [{RID}].")
    swapped = answer(observed=f"smart_5_raw (z-score 85.4, increased by 360) and smart_197_raw (z-score 29.2, increased by 16) [{RID}].")

    assert validate_answer(good, rule_context()) == []
    assert len(details(swapped, rule_context(), "misattributed_number")) == 4


def test_the_attribute_can_come_before_or_after_the_cue():
    for sentence in ("The z-score for smart_197_raw was 85.4", "smart_197_raw has a z-score of 85.4", "SMART 197 z-score 85.4"):
        assert validate_answer(answer(observed=f"{sentence} [{RID}]."), rule_context()) == [], sentence
    assert "misattributed_number" in codes(answer(observed=f"The z-score for smart_197_raw was 29.2 [{RID}]."), rule_context())


def test_table_rows_are_checked_cell_by_cell():
    right = answer(observed=f"- smart_197_raw: 600 | 300.0 | +360 | 85.4 [{RID}]")
    shuffled = answer(observed=f"- smart_197_raw: 85.4 | 300.0 | +360 | 600 [{RID}]")

    assert validate_answer(right, rule_context()) == []
    assert len(details(shuffled, rule_context(), "misattributed_number")) == 2


def test_trend_values_are_checked_against_the_right_attribute_and_date():
    ok = answer(observed=f"smart_5_raw: 2026-06-19 = 100 [{RID}]")
    wrong_value = answer(observed=f"smart_5_raw: 2026-06-19 = 12 [{RID}]")
    outside_window = answer(observed=f"smart_5_raw: 2026-06-01 = 0 [{RID}]")

    assert validate_answer(ok, rule_context()) == []
    assert "misattributed_number" in codes(wrong_value, rule_context())
    assert any("no smart_5_raw value for 2026-06-01" in d for d in details(outside_window, rule_context(), "misattributed_number"))


def test_a_raw_value_with_a_date_must_be_that_dates_value():
    on_date = answer(observed=f"The raw value of smart_5_raw was 0 on 2026-06-13 [{RID}].")
    wrong = answer(observed=f"The raw value of smart_5_raw was 100 on 2026-06-13 [{RID}].")

    assert validate_answer(on_date, rule_context()) == []
    assert "misattributed_number" in codes(wrong, rule_context())


def test_a_reported_number_for_a_value_the_report_says_is_not_reported_is_rejected():
    features = firing_features(smart_187_raw=dict(raw=None, delta=None, z=None, mean=None))
    report = build_report(rule_anomaly(score=6), features, RULE_STATS)
    context = assemble_context("q", report, KNOWLEDGE)
    detector = f"The rule baseline scored this drive 6 of 8 [{report.doc_id}]."     # six triggers fire without SMART 187
    bad = answer(observed=f"smart_187_raw had a z-score of 29.2 [{report.doc_id}].", detector=detector, rid=report.doc_id)
    good = answer(observed=f"The z-score of smart_187_raw was not reported [{report.doc_id}].", detector=detector, rid=report.doc_id)

    assert "misattributed_number" in codes(bad, context)
    assert "misattributed_number" not in codes(good, context)


def test_a_one_decimal_rounding_difference_between_the_trigger_text_and_the_table_is_tolerated():
    features = firing_features(smart_5_raw=dict(raw=100, delta=16.0, z=29.25, mean=50.0))
    report = build_report(rule_anomaly(triggered_rules="smart_5_raw z=29.3; smart_5_raw increased by 16"), features, RULE_STATS)
    context = assemble_context("q", report, KNOWLEDGE)

    for shown in ("29.2", "29.3"):
        assert "misattributed_number" not in codes(answer(observed=f"smart_5_raw z-score {shown} [{report.doc_id}].", rid=report.doc_id), context)
    assert "misattributed_number" in codes(answer(observed=f"smart_5_raw z-score 29.4 [{report.doc_id}].", rid=report.doc_id), context)


def test_score_rank_and_anomaly_score_claims_are_checked():
    assert "misattributed_number" in codes(answer(detector=f"The rule baseline scored this drive 5 of 8 [{RID}]."), rule_context())
    assert "misattributed_number" in codes(answer(observed=f"Drive {SERIAL} had rank 2 [{RID}]."), rule_context())
    assert "misattributed_number" not in codes(answer(observed=f"Drive {SERIAL} had rank 1 [{RID}]."), rule_context())
    wrong = answer(detector=f"The isolation forest gave an anomaly_score of 0.7124 [{IID}]. It does not provide per-feature causal attribution [{IID}].", rid=IID, method="isolation forest")
    right = wrong.replace("0.7124", "0.8513")
    assert "misattributed_number" in codes(wrong, if_context()) and "misattributed_number" not in codes(right, if_context())


def test_numbers_that_are_not_claims_about_an_attribute_are_left_alone():
    # thresholds and counts mentioned next to a cue word are not telemetry claims
    text = answer(observed=f"smart_197_raw had a z-score above 3 and rose since the previous day, on 2026-06-19 [{RID}].")

    assert extract_claims(f"smart_197_raw had a z-score above 3 and rose since the previous day, on 2026-06-19 [{RID}].") == []
    assert "misattributed_number" not in codes(text, rule_context())


def test_quoting_the_knowledge_documents_never_produces_a_telemetry_claim():
    for doc in load_knowledge(resolve_git=lambda: "x"):
        for sentence in section_sentences("## Answer\n" + doc.text):
            assert extract_claims(sentence) == [], (doc.doc_id, sentence)


# --- 2. citation support ---

def test_telemetry_cited_only_to_a_smart_document_is_rejected_and_cited_to_the_report_is_accepted():
    bad = answer(observed="smart_197_raw had a z-score of 85.4 [attr:smart_197_raw].")
    good = answer(observed=f"smart_197_raw had a z-score of 85.4 [{RID}].")

    assert "unsupported_citation" in codes(bad, rule_context())
    assert any("must cite [report:" in d for d in details(bad, rule_context(), "unsupported_citation"))
    assert validate_answer(good, rule_context()) == []


def test_an_uncited_telemetry_sentence_is_a_missing_citation():
    assert "missing_citation" in codes(answer(observed="smart_197_raw had a z-score of 85.4."), rule_context())


def test_a_detector_statement_cited_only_to_a_smart_document_is_rejected():
    bad = answer(detector="The rule baseline counts triggers [attr:smart_197_raw].")

    errors = details(bad, rule_context(), "unsupported_citation")

    assert errors and "never a SMART document" in errors[0]


def test_a_detector_statement_may_cite_the_report_the_own_method_document_or_both():
    context = rule_context(ALL_KNOWLEDGE)
    for citation in (f"[{RID}]", "[method:rule_baseline]", f"[{RID}, method:rule_baseline]"):
        assert validate_answer(answer(detector=f"The rule baseline counts triggers {citation}."), context) == [], citation


def test_the_other_detectors_document_cannot_support_this_detectors_statement():
    rule_citing_forest = answer(detector=f"The rule baseline counts triggers [{RID}, method:isolation_forest].")
    forest_citing_rule = answer(detector=f"The isolation forest scores all features at once [method:rule_baseline]. It does not provide per-feature causal attribution [{IID}].",
                                rid=IID, method="isolation forest")

    assert "unsupported_citation" in codes(rule_citing_forest, rule_context(ALL_KNOWLEDGE))
    assert "unsupported_citation" in codes(forest_citing_rule, if_context())


def test_forest_detector_claims_are_supported_by_the_report_or_its_method_document():
    ok = answer(detector=(f"The isolation forest scores all features at once [method:isolation_forest]. "
                          f"It does not provide per-feature causal attribution [{IID}]."), rid=IID, method="isolation forest")

    assert validate_answer(ok, if_context()) == []


def test_the_no_attribution_statement_is_refused_for_a_rule_baseline_report():
    bad = answer(detector=f"The rule baseline scored 8 of 8. It does not provide per-feature causal attribution [{RID}].")

    assert "unsupported_detector_claim" in codes(bad, rule_context())
    assert "unsupported_detector_claim" not in codes(answer(rid=IID, method="isolation forest",
                                                            detector=f"It does not provide per-feature causal attribution [{IID}]."), if_context())


def test_background_about_a_smart_attribute_must_cite_that_attributes_document():
    context = rule_context(ALL_KNOWLEDGE)
    ok = answer(background="SMART 197 counts sectors waiting to be remapped [attr:smart_197_raw].")
    wrong_doc = answer(background="SMART 197 counts sectors waiting to be remapped [method:rule_baseline].")
    other_attr = answer(background="SMART 197 counts sectors waiting to be remapped [attr:smart_5_raw].")
    report_only = answer(background=f"SMART 197 counts sectors waiting to be remapped [{RID}].")

    assert validate_answer(ok, context) == []
    for bad in (wrong_doc, other_attr, report_only):
        assert "unsupported_citation" in codes(bad, context)


def test_a_background_sentence_may_mention_a_related_attribute_that_its_own_document_mentions():
    context = rule_context(ALL_KNOWLEDGE)
    ok = answer(background="SMART 197 correlates well with SMART 198 [attr:smart_197_raw].")
    only_related = answer(background="SMART 198 correlates well with SMART 5 [attr:smart_197_raw].")    # neither has a document cited

    assert validate_answer(ok, context) == []
    assert "unsupported_citation" in codes(only_related, context)


def test_background_about_a_detector_must_cite_its_method_document_not_a_smart_document():
    context = rule_context(ALL_KNOWLEDGE)
    ok = answer(background="The rule baseline gives points for each trigger [method:rule_baseline].")
    bad = answer(background="The rule baseline gives points for each trigger [attr:smart_197_raw].")

    assert validate_answer(ok, context) == []
    assert "unsupported_citation" in codes(bad, context)


def test_background_about_an_attribute_whose_document_was_not_retrieved_is_unsupported():
    bad = answer(background="SMART 5 counts remapped sectors [attr:smart_197_raw].")      # attr:smart_5_raw is not in this context

    assert "unsupported_citation" in codes(bad, rule_context())


def test_one_citation_covers_the_uncited_sentences_before_it_in_the_same_line():
    text = answer(observed=f"Drive {SERIAL} was flagged. smart_197_raw had a z-score of 85.4. The cohort is shown below [{RID}].")
    trailing = answer(observed=f"Drive {SERIAL} was flagged [{RID}]. smart_197_raw had a z-score of 85.4.")
    lone = answer(observed=f"Drive {SERIAL} was flagged.\n[{RID}]")

    assert validate_answer(text, rule_context()) == []
    assert "missing_citation" in codes(trailing, rule_context())      # the last sentence has nothing after it
    assert "missing_citation" in codes(lone, rule_context())          # a citation does not reach back across lines


def test_a_sentence_unit_is_tied_to_its_own_line():
    units = section_sentences(f"## Observed telemetry\nFirst [{RID}]. Second.\n- Third [{RID}]\nFourth")

    assert units == [f"First [{RID}].", "Second.", f"- Third [{RID}]", "Fourth"]


def test_the_fallback_answers_still_pass_with_the_stricter_rules():
    for report, knowledge in ((RULE_REPORT, ALL_KNOWLEDGE), (IF_REPORT, ALL_KNOWLEDGE), (RULE_REPORT, []), (IF_REPORT, [])):
        context = assemble_context("q", report, knowledge)
        assert validate_answer(evidence_only_answer(report, knowledge), context) == []


def test_general_questions_use_the_same_citation_support_rule():
    context = assemble_context("what is smart 197?", None, ALL_KNOWLEDGE)
    ok = "## Answer\nSMART 197 counts sectors waiting to be remapped [attr:smart_197_raw].\n## Limits\nThis describes background documents. It is not a diagnosis, and it does not predict failure."

    assert validate_answer(ok, context) == []
    assert "unsupported_citation" in codes(ok.replace("attr:smart_197_raw", "method:rule_baseline"), context)
    assert "missing_citation" in codes(ok.replace(" [attr:smart_197_raw]", ""), context)


def test_the_error_details_say_what_to_fix_so_the_retry_can_act_on_them():
    bad = answer(observed="smart_197_raw had a z-score of 29.2 [attr:smart_197_raw].")

    messages = [e.detail for e in validate_answer(bad, rule_context())]

    assert any("smart_197_raw z: the report has 85.4, but the answer says 29.2" in m for m in messages)
    assert any("every sentence about the drive must cite" in m and "cites ['attr:smart_197_raw']" in m for m in messages)


# --- lead-in lines: a bare introduction to cited evidence needs no citation of its own ---

TELEMETRY_LEAD_IN = "Telemetry on the flagged date (raw value | 7-day mean | change since previous day | z-score versus same-model drives):"
TREND_LEAD_IN = "Raw values over the 7 days up to the flagged date:"
CITED_ROW = f"- smart_197_raw: 600 | 300.0 | +360 | 85.4 [{RID}]"
CITED_TREND = f"- smart_5_raw: 2026-06-19 = 100 [{RID}]"
FIRST = f"Drive {SERIAL} was flagged on 2026-06-19 [{RID}]."


def test_a_structural_lead_in_followed_by_cited_evidence_is_accepted():
    for lead_in, row in ((TELEMETRY_LEAD_IN, CITED_ROW), (TREND_LEAD_IN, CITED_TREND)):
        text = answer(observed=f"{FIRST}\n{lead_in}\n{row}")

        assert validate_answer(text, rule_context()) == [], lead_in


def test_a_lead_in_is_accepted_only_when_the_next_line_cites_the_drive_report():
    uncited_next = answer(observed=f"{FIRST}\n{TELEMETRY_LEAD_IN}\n- smart_197_raw: 600 | 300.0 | +360 | 85.4")
    wrong_doc_next = answer(observed=f"{FIRST}\n{TELEMETRY_LEAD_IN}\n- smart_197_raw: 600 | 300.0 | +360 | 85.4 [attr:smart_197_raw]")
    last_line = answer(observed=f"{FIRST}\n{TELEMETRY_LEAD_IN}")

    for text in (uncited_next, wrong_doc_next, last_line):
        found = codes(text, rule_context())
        assert found & {"missing_citation", "unsupported_citation"}, text
    # the lead-in is rejected as well as the line that follows it
    assert len(details(uncited_next, rule_context(), "missing_citation")[0].split("' cites nothing")) >= 3


def test_a_line_with_a_factual_payload_is_not_a_lead_in_even_before_cited_evidence():
    for line in ("smart_197_raw had a z-score of 85.4:",             # a telemetry value
                 "Telemetry for smart_5_raw follows:",                # names an attribute
                 "The z-score was above 3 for these attributes:",     # a number
                 f"Drive {SERIAL} was flagged. Telemetry follows:"):  # more than one sentence
        text = answer(observed=f"{FIRST}\n{line}\n{CITED_ROW}")

        assert "missing_citation" in codes(text, rule_context()), line


def test_a_factual_telemetry_sentence_with_no_citation_is_still_rejected():
    text = answer(observed=f"{FIRST}\nsmart_197_raw had a z-score of 85.4.\n{CITED_ROW}")

    errors = details(text, rule_context(), "missing_citation")

    assert errors and "smart_197_raw had a z-score of 85.4." in errors[0]


def test_the_lead_in_exemption_does_not_extend_to_detector_statements():
    uncited_claim = answer(detector="The rule baseline scored this drive 8 of 8.")
    lead_in_there = answer(detector=f"Trigger check on the flagged date:\nThe rule baseline scored this drive 8 of 8 [{RID}].")

    assert "missing_citation" in codes(uncited_claim, rule_context())
    assert "missing_citation" in codes(lead_in_there, rule_context())          # exempt in Observed telemetry only


def test_a_real_model_style_layout_with_lead_ins_and_cited_bullets_passes():
    first = f"Drive {SERIAL} was flagged on 2026-06-19 [{IID}]."          # this report's id, not the rule report's
    observed = (f"{first} The anomaly_score was 0.8513 [{IID}].\n"
                f"Largest same-model z-scores on the flagged date were: smart_197_raw z-score 85.4, smart_198_raw z-score 85.4 [{IID}].\n"
                f"{TELEMETRY_LEAD_IN}\n- smart_197_raw: 600 | 300.0 | +360 | 85.4 [{IID}]\n- smart_198_raw: 600 | 300.0 | +360 | 85.4 [{IID}]\n"
                f"{TREND_LEAD_IN}\n- smart_5_raw: 2026-06-19 = 100 [{IID}]")
    text = answer(observed=observed, rid=IID, method="isolation forest",
                  detector=f"The detector is isolation_forest [{IID}]. It does not provide per-feature causal attribution [{IID}].")

    assert validate_answer(text, if_context()) == []


# --- background support comes from the cited document's own text ---

def bg(sentence, context=None):
    return codes(answer(background=sentence), context or rule_context(ALL_KNOWLEDGE))


def test_a_sentence_quoted_from_the_smart_197_document_may_mention_smart_198():
    quoted = "Backblaze found it correlates well with SMART 198 and notes that not all manufacturers report both attributes [attr:smart_197_raw]."

    assert validate_answer(answer(background=quoted), rule_context(REAL_DOCS)) == []     # even with no SMART 198 document in context


def test_a_sentence_quoted_from_the_document_may_mention_the_rule_baseline_as_project_use():
    quoted = "Project use: one of the four critical attributes in the rule baseline [attr:smart_197_raw]."

    assert validate_answer(answer(background=quoted), rule_context(REAL_DOCS)) == []     # method:rule_baseline need not be cited


def test_quoting_ignores_case_spacing_and_the_citation_mark():
    spaced = "project use: one of the four  critical attributes in the rule baseline . [attr:smart_197_raw]"

    assert validate_answer(answer(background=spaced), rule_context(REAL_DOCS)) == []


def test_several_quoted_sentences_under_one_citation_pass_and_one_unsupported_sentence_fails_the_unit():
    two = ("Lower is better. Backblaze found it correlates well with SMART 198 and notes that not all manufacturers report both attributes "
           "[attr:smart_197_raw].")
    spoiled = two.replace("Lower is better.", "SMART 198 always fails first.")

    assert validate_answer(answer(background=two), rule_context(REAL_DOCS)) == []
    assert "unsupported_citation" in bg(spoiled, rule_context(REAL_DOCS))


def test_an_unrelated_detector_claim_cited_only_to_a_smart_document_is_rejected():
    for claim in ("The rule baseline gives triple weight to SMART 197 [attr:smart_197_raw].",
                  "The isolation forest uses SMART 197 to build its trees [attr:smart_197_raw]."):
        assert "unsupported_citation" in bg(claim), claim


def test_a_near_quote_with_an_added_claim_is_not_supported():
    spoiled = "Backblaze found it correlates well with SMART 198 and says SMART 198 always fails first [attr:smart_197_raw]."

    assert "unsupported_citation" in bg(spoiled)


def test_a_paraphrase_must_be_supported_by_the_cited_documents_text():
    assert bg("SMART 197 counts sectors waiting to be remapped [attr:smart_197_raw].") == set()
    assert "unsupported_citation" in bg("SMART 197 counts sectors that are failing [attr:smart_197_raw].")       # 'failing' is not in the document
    assert "unsupported_citation" in bg("SMART 197 counts 5 sectors waiting to be remapped [attr:smart_197_raw].")   # neither is the number


def test_the_other_detectors_method_document_does_not_support_this_detectors_sentences():
    for sentence in ("The rule baseline gives points for each trigger [method:isolation_forest].",
                     "The score counts triggers. It is not a probability and it does not say why the values changed [method:isolation_forest]."):
        assert "unsupported_citation" in bg(sentence), sentence
    assert bg("The rule baseline gives points for each trigger [method:rule_baseline].") == set()


def test_a_sentence_about_one_attribute_is_not_supported_by_another_attributes_document():
    assert "unsupported_citation" in bg("SMART 197 counts sectors waiting to be remapped [attr:smart_5_raw].")
    assert "unsupported_citation" in bg("SMART 198 correlates well with SMART 5 [attr:smart_197_raw].")


def test_unknown_ids_and_the_drive_report_still_do_not_support_background():
    assert "unknown_citation" in bg("SMART 197 counts sectors waiting to be remapped [KNOWLEDGE:attr:smart_197_raw].")
    assert "unsupported_citation" in bg(f"SMART 197 counts sectors waiting to be remapped [{RID}].")


def test_a_reference_line_is_accepted_only_for_the_document_it_names_and_cites():
    listing = "- SMART 197 - Current Pending Sector Count (source: Backblaze blog) [attr:smart_197_raw]"
    context = rule_context(ALL_KNOWLEDGE)
    title = METHOD_DOCS["attr:smart_197_raw"]["title"]
    source = METHOD_DOCS["attr:smart_197_raw"]["source_name"]

    assert "unsupported_citation" in bg(listing, context)                                       # not the exact title and source
    assert bg(f"- {title} (source: {source}) [attr:smart_197_raw]", context) == set()
    assert "unsupported_citation" in bg(f"- {title} (source: {source}) [method:rule_baseline]", context)


def test_telemetry_number_validation_is_not_weakened_by_the_new_rules():
    wrong = answer(observed=f"{FIRST}\n{TELEMETRY_LEAD_IN}\n- smart_197_raw: 85.4 | 300.0 | +360 | 600 [{RID}]")

    assert "misattributed_number" in codes(wrong, rule_context())
    assert "unsupported_number" in codes(answer(observed=f"smart_197_raw had a z-score of 99.9 [{RID}]."), rule_context())


def test_identifiers_are_split_into_words_when_checking_support():
    from src.rag.validate import _content_words

    assert _content_words("The rule_baseline detector") == {"rule", "baseline", "detector"}


def test_a_paraphrase_of_the_method_document_that_uses_the_identifier_and_the_title_word_is_supported():
    # observed from a model: supported by the rule baseline document, whose title is "Rule baseline detector"
    supported = "The rule_baseline detector uses four critical attributes (SMART 5, 187, 197 and 198) [method:rule_baseline]."
    unsupported = "The rule_baseline detector uses four critical attributes and weights each by drive age [method:rule_baseline]."
    context = rule_context(ALL_KNOWLEDGE)

    assert validate_answer(answer(background=supported), context) == []
    assert "unsupported_citation" in codes(answer(background=unsupported), context)
    assert "unsupported_citation" in codes(answer(background=supported.replace("method:rule_baseline", "method:isolation_forest")), context)


def test_a_verbatim_quote_is_still_a_quote_when_the_model_prefixes_a_list_marker():
    rule_doc = METHOD_DOCS["method:rule_baseline"]["text"]
    first_two = " ".join(rule_doc.split(". ")[:2]) + "."
    context = if_context()                      # a rule baseline document in the context of an isolation forest report

    for marker in ("*   ", "- ", "• ", "1. "):
        text = answer(background=f"{marker}{first_two} [method:rule_baseline]", rid=IID, method="isolation forest",
                      detector=f"The detector is isolation_forest [{IID}]. It does not provide per-feature causal attribution [{IID}].")
        assert "unsupported_citation" not in codes(text, context), marker


def test_a_list_marker_does_not_turn_an_unsupported_sentence_into_a_quote():
    context = rule_context(ALL_KNOWLEDGE)

    assert "unsupported_citation" in bg("*   The rule baseline doubles the score for drives older than five years [method:rule_baseline].", context)
    assert "unsupported_citation" in bg("- SMART 197 counts sectors that are failing [attr:smart_197_raw].", context)
