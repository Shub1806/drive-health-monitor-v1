import json
import re
from datetime import date, timedelta

import pytest

from src.rag import report as R
from src.rag.report import build_report, evidence_only_answer, safe_params, sort_runs
from tests.rag_helpers import (
    FLAGGED, IF_PARAMS, IF_RUN, IF_STATS, KNOWLEDGE, RULE_PARAMS, RULE_RUN, RULE_STATS, SERIAL,
    FakeCursor, feature_row, firing_features, if_anomaly, rule_anomaly, window_features,
)

FORBIDDEN = re.compile(r"\b(failure|failure_within_30d|label_is_complete|eval_\w+)\b", re.IGNORECASE)


def rule_report(**anomaly):
    return build_report(rule_anomaly(**anomaly), firing_features(), RULE_STATS)


def if_report(features=None, **anomaly):
    return build_report(if_anomaly(**anomaly), features or firing_features(), IF_STATS)


def test_attributes_match_the_silver_columns():
    from src.silver import SMART_COLUMNS

    assert R.ATTRIBUTES == SMART_COLUMNS
    assert set(R.CRITICAL) <= set(R.ATTRIBUTES)


# --- rule_baseline report ---

def test_rule_report_layout_and_values():
    report = rule_report()
    text = report.text

    assert report.doc_id == f"report:{RULE_RUN[:8]}:{SERIAL}"
    assert text.splitlines()[0] == f"DRIVE REPORT {report.doc_id}"
    assert f"Drive {SERIAL} | model ST14000NM0138 | flagged date 2026-06-19 | rank 1 of 1000 stored (of 356764 drives scored in the window)" in text
    assert "Score: 8 of 8" in text
    assert "Stored trigger list: smart_5_raw z=29.2; smart_5_raw increased by 16; smart_187_raw z=34.2" in text
    assert "- smart_197_raw: z-score 85.4 (fired); change since previous day +360 (fired)" in text
    assert "- smart_197_raw: 600 | 300.0 | +360 | 85.4" in text
    assert "Cohort: 500 drives of this model reported on the flagged date." in text
    assert text.splitlines()[-1] == R.FOOTER


def test_rule_statement_uses_the_stored_thresholds_and_says_the_score_is_not_a_probability():
    text = rule_report().text

    assert "z-score above 3 versus same-model drives" in text
    assert "at least 30 drives" in text
    assert "it is not a probability" in text


def test_trend_lists_the_seven_days_up_to_the_flagged_date_for_the_critical_attributes():
    lines = rule_report().trend_lines

    assert lines[0] == "Raw values over the 7 days up to the flagged date:"
    assert len(lines) == 1 + len(R.CRITICAL)
    assert lines[1].count("=") == 7
    assert "2026-06-13 = 0" in lines[1] and "2026-06-19 = 100" in lines[1]


def test_a_report_carries_no_date_later_than_the_flagged_date():
    import re

    for make in (rule_report, if_report):
        report = make()
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", report.text)

        assert max(dates) <= str(FLAGGED), (make.__name__, max(dates))
        assert "window starting 2026-06-01 (official)" in report.text
        assert "2026-06-30" not in report.text                      # the run's window end


def test_the_report_is_deterministic_and_its_hash_follows_the_content():
    a, b = rule_report(), rule_report()
    changed = build_report(rule_anomaly(), firing_features(smart_5_raw=dict(raw=101, delta=16.0, z=29.2)), RULE_STATS)

    assert a.text == b.text and a.sha256 == b.sha256
    assert changed.sha256 != a.sha256


def test_missing_values_read_not_reported_and_missing_days_not_recorded():
    features = firing_features(smart_187_raw=dict(raw=None, delta=None, z=None, mean=None))
    features = [r for r in features if r["date"] != FLAGGED - timedelta(days=3)]

    report = build_report(rule_anomaly(), features, RULE_STATS)

    assert "- smart_187_raw: not reported | not reported | not reported | not reported" in report.text
    assert "- smart_187_raw: z-score not reported (not fired); change since previous day not reported (not fired)" in report.text
    assert f"{FLAGGED - timedelta(days=3)} = not recorded" in report.text


def test_a_small_cohort_does_not_fire_the_z_trigger_and_the_mismatch_is_noted():
    features = firing_features()
    features[-1]["cohort_size"] = 29  # z-scores above 3 do not count below 30 drives

    report = build_report(rule_anomaly(score=8), features, RULE_STATS)

    assert "z-score 85.4 (not fired)" in report.text
    assert "recomputed trigger count 4 differs from the stored score 8" in report.text
    assert report.consistent is False


def test_a_consistent_report_has_no_note():
    report = rule_report()

    assert report.consistent is True
    assert "Note:" not in report.text


def test_telemetry_after_the_flagged_date_is_refused():
    features = firing_features() + [feature_row(FLAGGED + timedelta(days=1))]

    with pytest.raises(ValueError, match="after the flagged date"):
        build_report(rule_anomaly(), features, RULE_STATS)


def test_a_missing_flagged_day_is_an_error_not_a_guess():
    features = [r for r in firing_features() if r["date"] != FLAGGED]

    with pytest.raises(ValueError, match="no telemetry"):
        build_report(rule_anomaly(), features, RULE_STATS)


# --- isolation_forest report ---

def test_if_report_states_the_score_range_and_the_no_attribution_sentence():
    text = if_report().text

    assert "Detector: isolation_forest. anomaly_score 0.8513 (higher is more unusual)." in text
    assert "range from 0.7124 to 0.8513" in text
    assert "It does not provide per-feature causal attribution." in text
    assert "observed telemetry, not the model's reasons" in text
    assert "all 21 features at once" in text
    assert "Training: non-failure-day rows dated 2026-04-07 to 2026-05-31, without labels, 21 features." in text


def test_if_report_has_no_trigger_machinery():
    text = if_report().text

    assert "Trigger check" not in text and "Stored trigger list" not in text and "Score:" not in text


def test_if_report_describes_the_largest_z_scores_and_rising_attributes_without_claiming_reasons():
    text = if_report().text

    assert ("Largest same-model z-scores on the flagged date (description of observed values): "
            "smart_197_raw z-score 85.4, smart_198_raw z-score 85.4, smart_187_raw z-score 34.2") in text
    assert "Attributes whose raw value rose since the previous day: smart_5_raw, smart_187_raw, smart_197_raw, smart_198_raw" in text


def test_if_report_with_nothing_unusual():
    text = if_report(features=window_features()).text

    assert "Attributes whose raw value rose since the previous day: none" in text


# --- no outcomes, no future ---

@pytest.mark.parametrize("make", [rule_report, if_report])
def test_report_text_never_contains_label_or_outcome_fields(make):
    text = make().text

    assert not re.search(r"failure_within_30d|label_is_complete|eval_|precision|recall|base rate", text, re.IGNORECASE)
    assert "failure = 0 and" not in text   # the stored params carry that sentence; the report must not


def test_only_a_few_settings_are_taken_from_the_stored_params():
    assert safe_params("rule_baseline", RULE_PARAMS) == {"z_threshold": 3.0, "min_cohort_size": 30}
    assert safe_params("isolation_forest", IF_PARAMS) == {"train_window": ["2026-04-07", "2026-05-31"], "n_features": 21}


def test_sql_selects_explicit_columns_and_never_an_outcome_column():
    statements = [R.FEATURES_SQL, R.RUN_STATS_SQL, R.anomaly_sql(), R.anomaly_sql(RULE_RUN, "rule_baseline")]

    for sql in statements:
        assert not FORBIDDEN.search(sql), sql
        assert "SELECT *" not in sql and ".*" not in sql
    assert set(R.FEATURE_COLUMNS) <= set(re.findall(r"\w+", R.FEATURES_SQL))
    assert not set(R.FORBIDDEN_COLUMNS) & set(R.FEATURE_COLUMNS)
    assert "date BETWEEN DATEADD(day, -6, %s) AND %s" in R.FEATURES_SQL


def test_fetch_features_binds_the_flagged_date_as_the_upper_bound():
    cursor = FakeCursor().respond("DRIVE_FEATURES", [("a",)], ["date"])

    rows = R.fetch_features(cursor, SERIAL, FLAGGED)

    assert rows == [{"date": "a"}]
    assert cursor.executed[0][1] == (SERIAL, FLAGGED, FLAGGED)


def test_runs_come_back_official_first_then_rule_before_forest_at_most_four():
    rows = [
        rule_anomaly(window_role="supplemental", run_id="s-rule"),
        if_anomaly(window_role="official", run_id="o-if"),
        if_anomaly(window_role="supplemental", run_id="s-if"),
        rule_anomaly(window_role="official", run_id="o-rule"),
        rule_anomaly(window_role="official", run_id="extra"),
    ]

    ordered = [r["run_id"] for r in sort_runs(rows)]

    assert ordered[:2] == ["o-rule", "extra"] or ordered[:2] == ["extra", "o-rule"]
    assert ordered[2] == "o-if" and len(ordered) == 4


def test_run_lookup_binds_serial_filters_and_limit():
    cursor = FakeCursor()

    R.fetch_runs(cursor, SERIAL, run_id=RULE_RUN, method="rule_baseline")

    sql, params = cursor.executed[0]
    assert "a.serial_number = %s AND a.run_id = %s AND a.method = %s" in sql
    assert params == (SERIAL, RULE_RUN, "rule_baseline", 4)


def test_existing_serials_checks_the_database_case_insensitively():
    cursor = FakeCursor().respond("SELECT DISTINCT serial_number", [(SERIAL,)])

    found = R.existing_serials(cursor, ["zhz43wys", "NOTADRIVE1"])

    assert found == [SERIAL]
    assert cursor.executed[0][1] == ("NOTADRIVE1", "ZHZ43WYS")
    assert R.existing_serials(cursor, []) == []


# --- deterministic fallback ---

@pytest.mark.parametrize("make", [rule_report, if_report])
@pytest.mark.parametrize("knowledge", [KNOWLEDGE, []])
def test_the_fallback_answer_passes_the_same_validator_as_a_model_answer(make, knowledge):
    from src.rag.prompt import assemble_context
    from src.rag.validate import validate_answer

    report = make()
    answer = evidence_only_answer(report, knowledge)
    context = assemble_context("why was this drive flagged?", report, knowledge)

    assert validate_answer(answer, context) == []
    assert answer == evidence_only_answer(report, knowledge)   # deterministic


def test_the_general_fallback_cites_documents_and_makes_no_claims():
    from src.rag.prompt import assemble_context
    from src.rag.validate import validate_answer

    answer = evidence_only_answer(None, KNOWLEDGE)

    assert "No stored drive report applies" in answer
    assert validate_answer(answer, assemble_context("what is smart 197?", None, KNOWLEDGE)) == []
