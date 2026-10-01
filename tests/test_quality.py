from datetime import date, timedelta

import pytest

from src.quality import (
    DataQualityError,
    MAX_SMART_NULL_RATE,
    assert_quality,
    check_cast_loss,
    check_duplicate_keys,
    check_label_complete,
    check_labels,
    check_null_rates,
    check_row_accounting,
    check_rows_per_day,
)


def days(counts, start=date(2026, 4, 1)):
    return {start + timedelta(days=i): c for i, c in enumerate(counts)}


def clean_null_counts(total):
    counts = {"date": 0, "serial_number": 0, "model": 0, "failure": 0}
    counts.update({column: 0 for column in MAX_SMART_NULL_RATE})
    return counts


def test_rows_per_day_passes_on_steady_data():
    assert check_rows_per_day(days([100, 98, 103, 101])) == []


def test_rows_per_day_flags_a_missing_calendar_day():
    counts = days([100, 100, 100, 100])
    del counts[date(2026, 4, 3)]

    failures = check_rows_per_day(counts)

    assert len(failures) == 1
    assert "1 day(s) with no rows" in failures[0]
    assert "2026-04-03" in failures[0]


def test_rows_per_day_flags_a_partial_day():
    failures = check_rows_per_day(days([100, 100, 30, 100, 100]))

    assert len(failures) == 1
    assert "below 50%" in failures[0]


def test_rows_per_day_flags_empty_input():
    assert check_rows_per_day({}) != []


def test_duplicate_keys():
    assert check_duplicate_keys(0) == []
    assert "3" in check_duplicate_keys(3)[0]


def test_null_rates_pass_on_clean_data():
    assert check_null_rates(clean_null_counts(1000), 1000) == []


def test_null_rates_allow_mostly_empty_columns_within_their_ceiling():
    counts = clean_null_counts(1000)
    counts["smart_187_raw"] = 680  # unreported by many models, expected
    assert check_null_rates(counts, 1000) == []


def test_null_rates_flag_any_null_in_identity_column():
    counts = clean_null_counts(1000)
    counts["serial_number"] = 1

    failures = check_null_rates(counts, 1000)

    assert len(failures) == 1
    assert "serial_number" in failures[0]


def test_null_rates_flag_smart_column_over_ceiling():
    counts = clean_null_counts(1000)
    counts["smart_9_raw"] = 200  # 20% against a 5% ceiling

    failures = check_null_rates(counts, 1000)

    assert len(failures) == 1
    assert "smart_9_raw" in failures[0]


def test_null_rates_flag_missing_counts_and_empty_table():
    counts = clean_null_counts(1000)
    del counts["smart_5_raw"]
    assert any("smart_5_raw" in f for f in check_null_rates(counts, 1000))
    assert check_null_rates(clean_null_counts(0), 0) != []


def test_cast_loss_detects_values_nulled_by_a_cast():
    before = {"smart_5_raw": 100, "smart_9_raw": 100}
    after = {"smart_5_raw": 100, "smart_9_raw": 97}

    failures = check_cast_loss(before, after)

    assert len(failures) == 1
    assert "smart_9_raw" in failures[0]
    assert check_cast_loss(before, before) == []


def test_cast_loss_flags_a_column_missing_from_silver():
    assert check_cast_loss({"smart_5_raw": 10}, {}) != []


def test_row_accounting():
    assert check_row_accounting(1000, 990, 10) == []
    failures = check_row_accounting(1000, 985, 10)
    assert len(failures) == 1
    assert "expected 990" in failures[0]


def test_labels_pass_when_clean():
    assert check_labels([0, 1, 0], 0) == []


def test_labels_flag_bad_failure_values_and_unlabelled_failures():
    failures = check_labels([0, 1, 2], 4)

    assert len(failures) == 2
    assert "[2]" in failures[0]
    assert "4 failure row(s)" in failures[1]


def test_assert_quality_is_quiet_when_nothing_failed():
    assert_quality([])


def test_assert_quality_raises_and_lists_every_failure():
    with pytest.raises(DataQualityError) as err:
        assert_quality(["first problem", "second problem"])

    message = str(err.value)
    assert "2 data quality check(s) failed" in message
    assert "first problem" in message and "second problem" in message


def label_days(complete_through, last_day, per_day=10):
    """Per-day (complete, incomplete) counts: complete up to `complete_through`, incomplete after."""
    start = date(2026, 4, 1)
    counts = {}
    for i in range((last_day - start).days + 1):
        day = start + timedelta(days=i)
        counts[day] = (per_day, 0) if day <= complete_through else (0, per_day)
    return counts


END = date(2026, 6, 30)
LAST_COMPLETE = date(2026, 5, 31)


def test_label_complete_passes_when_flags_follow_the_date_rule():
    assert check_label_complete(label_days(LAST_COMPLETE, END), 0, END) == []


def test_label_complete_flags_nulls():
    failures = check_label_complete(label_days(LAST_COMPLETE, END), 3, END)

    assert any("3 null" in f for f in failures)


def test_label_complete_flags_a_recent_day_marked_complete():
    counts = label_days(LAST_COMPLETE, END)
    counts[date(2026, 6, 15)] = (10, 0)

    failures = check_label_complete(counts, 0, END)

    assert any("disagree with the date rule" in f and "2026-06-15" in f for f in failures)


def test_label_complete_flags_an_old_day_marked_incomplete():
    counts = label_days(LAST_COMPLETE, END)
    counts[date(2026, 4, 10)] = (0, 10)

    failures = check_label_complete(counts, 0, END)

    assert any("disagree with the date rule" in f and "2026-04-10" in f for f in failures)


def test_label_complete_boundary_day_is_complete():
    # dataset_end - 30 days is the last complete day; the next day is not.
    counts = {LAST_COMPLETE: (10, 0), date(2026, 6, 1): (0, 10), END: (0, 10)}

    assert check_label_complete(counts, 0, END) == []


def test_label_complete_requires_incomplete_rows_and_complete_rows_over_a_long_span():
    all_complete = label_days(END, END)
    all_incomplete = label_days(date(2026, 3, 31), END)

    assert any("no incomplete rows" in f for f in check_label_complete(all_complete, 0, END))
    assert any("no complete rows" in f for f in check_label_complete(all_incomplete, 0, END))


def test_label_complete_short_span_needs_no_complete_rows():
    counts = label_days(date(2026, 3, 31), date(2026, 4, 3))

    assert check_label_complete(counts, 0, date(2026, 4, 3)) == []
