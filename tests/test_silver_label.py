from datetime import date, timedelta

from src.silver import add_labels

START = date(2026, 4, 1)


def d(offset):
    return START + timedelta(days=offset)


def label_rows(spark, rows, dataset_end):
    """rows: (serial_number, day offset from START, failure). Returns {(serial, offset): (label, complete)}."""
    df = spark.createDataFrame(
        [(serial, d(offset), failure) for serial, offset, failure in rows],
        "serial_number string, date date, failure int",
    )
    out = add_labels(df, dataset_end=dataset_end).collect()
    return {
        (r["serial_number"], (r["date"] - START).days): (r["failure_within_30d"], r["label_is_complete"])
        for r in out
    }


def daily(serial, days, failure_on=None):
    return [(serial, i, 1 if i == failure_on else 0) for i in range(days)]


def test_failure_today_labels_the_failure_row_itself(spark):
    result = label_rows(spark, daily("A", 3, failure_on=2), dataset_end=d(100))

    assert result[("A", 2)][0] == 1


def test_failure_exactly_30_days_later_counts(spark):
    result = label_rows(spark, daily("A", 31, failure_on=30), dataset_end=d(100))

    assert result[("A", 0)][0] == 1


def test_failure_31_days_later_does_not_count(spark):
    result = label_rows(spark, daily("A", 32, failure_on=31), dataset_end=d(100))

    assert result[("A", 0)][0] == 0
    assert result[("A", 1)][0] == 1


def test_window_follows_calendar_days_across_gaps(spark):
    # Only 4 rows exist, but the failure is 40 calendar days after day 0 and 30 after day 10.
    rows = [("A", 0, 0), ("A", 10, 0), ("A", 20, 0), ("A", 40, 1)]

    result = label_rows(spark, rows, dataset_end=d(100))

    assert result[("A", 0)][0] == 0  # 40 days out, even though it is only 3 rows away
    assert result[("A", 10)][0] == 1  # exactly 30 days out
    assert result[("A", 20)][0] == 1


def test_no_future_failure_means_label_zero(spark):
    result = label_rows(spark, daily("A", 40), dataset_end=d(100))

    assert all(label == 0 for label, _ in result.values())


def test_labels_do_not_leak_between_drives(spark):
    rows = daily("A", 5, failure_on=4) + daily("B", 5)

    result = label_rows(spark, rows, dataset_end=d(100))

    assert result[("A", 0)][0] == 1
    assert all(result[("B", i)][0] == 0 for i in range(5))


def test_label_is_complete_near_dataset_end(spark):
    # Dataset ends on offset 60, so a full 30-day future window exists through offset 30.
    rows = [("A", 30, 0), ("A", 31, 0), ("A", 60, 0)]

    result = label_rows(spark, rows, dataset_end=d(60))

    assert result[("A", 30)][1] is True
    assert result[("A", 31)][1] is False
    assert result[("A", 60)][1] is False


def test_incomplete_window_can_still_carry_a_positive_label(spark):
    rows = [("A", 59, 0), ("A", 60, 1)]

    result = label_rows(spark, rows, dataset_end=d(60))

    assert result[("A", 59)] == (1, False)
