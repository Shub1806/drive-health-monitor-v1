from datetime import date, datetime, timedelta

import pytest

import src.silver as silver
from src.quality import DataQualityError
from src.silver import (
    PRECAST_PREFIX,
    SILVER_COLUMNS,
    SMART_COLUMNS,
    build_prepared,
    check_written_table,
    collect_daily_metrics,
    evaluate_metrics,
    run_silver,
    summarise_daily_metrics,
)
from tests.test_silver_transform import BRONZE_SCHEMA, bronze_df, bronze_row

DAYS = ["2026-04-01", "2026-04-02", "2026-04-03"]


def clean_rows(drives=("A", "B")):
    return [bronze_row(serial=s, day=d) for d in DAYS for s in drives]


def prepared_for(spark, rows):
    return build_prepared(bronze_df(spark, rows), dataset_end=date(2026, 4, 3))


def daily_row(day, rows=100, **overrides):
    row = {
        "date": day,
        "rows": rows,
        "duplicate_keys": 0,
        "min_failure": 0,
        "max_failure": 0,
        "failure_unlabelled": 0,
        "label_complete": 0,
        "label_incomplete": rows,
        "label_null": 0,
    }
    for column in ("serial_number", "model", "failure") + SMART_COLUMNS:
        row["null_" + column] = 0
    for column in SMART_COLUMNS:
        row["precast_" + column] = rows
    row.update(overrides)
    return row


# --- summarise / evaluate on plain dicts ---

def test_summarise_folds_daily_rows_into_totals():
    daily = [
        daily_row(date(2026, 4, 1), rows=100, null_smart_187_raw=40, precast_smart_187_raw=60, max_failure=1),
        daily_row(date(2026, 4, 2), rows=110, null_smart_187_raw=44, precast_smart_187_raw=66),
    ]

    metrics = summarise_daily_metrics(daily)

    assert metrics["total_rows"] == 210
    assert metrics["day_counts"] == {date(2026, 4, 1): 100, date(2026, 4, 2): 110}
    assert metrics["null_counts"]["smart_187_raw"] == 84
    assert metrics["pre_cast_non_null"]["smart_187_raw"] == 126
    assert metrics["post_cast_non_null"]["smart_187_raw"] == 126
    assert metrics["failure_values"] == {0, 1}
    assert metrics["label_counts"][date(2026, 4, 1)] == (0, 100)


def test_summarise_counts_null_dates_separately():
    daily = [daily_row(date(2026, 4, 1), rows=100), daily_row(None, rows=3)]

    metrics = summarise_daily_metrics(daily)

    assert metrics["null_counts"]["date"] == 3
    assert list(metrics["day_counts"]) == [date(2026, 4, 1)]
    assert metrics["total_rows"] == 103


def test_evaluate_metrics_passes_on_clean_daily_rows():
    daily = [daily_row(date(2026, 4, 1) + timedelta(days=i)) for i in range(3)]
    metrics = summarise_daily_metrics(daily)

    assert evaluate_metrics(metrics, bronze_rows=300, dataset_end=date(2026, 4, 3)) == []


def test_evaluate_metrics_reports_every_failing_rule_together():
    daily = [
        daily_row(date(2026, 4, 1), duplicate_keys=2, null_serial_number=1),
        daily_row(date(2026, 4, 2), precast_smart_9_raw=95, max_failure=2),
        daily_row(date(2026, 4, 3)),
    ]
    metrics = summarise_daily_metrics(daily)

    failures = evaluate_metrics(metrics, bronze_rows=300, dataset_end=date(2026, 4, 3))
    text = "\n".join(failures)

    assert "duplicate keys" in text
    assert "serial_number" in text
    assert "cast loss: smart_9_raw" in text
    assert "other than 0/1" in text


def test_evaluate_metrics_flags_more_silver_rows_than_bronze():
    daily = [daily_row(date(2026, 4, 1) + timedelta(days=i)) for i in range(3)]

    failures = evaluate_metrics(summarise_daily_metrics(daily), bronze_rows=299, dataset_end=date(2026, 4, 3))

    assert any("more than bronze" in f for f in failures)


def test_check_written_table():
    assert check_written_table(10, 10, list(SILVER_COLUMNS)) == []
    assert check_written_table(10, 10, [c.upper() for c in SILVER_COLUMNS]) == []
    assert any("expected 10" in f for f in check_written_table(10, 9, list(SILVER_COLUMNS)))
    assert any("columns" in f for f in check_written_table(10, 10, list(SILVER_COLUMNS)[:-1]))


# --- collect on a tiny local DataFrame ---

def test_collect_daily_metrics_on_clean_data(spark):
    daily = collect_daily_metrics(prepared_for(spark, clean_rows()))
    metrics = summarise_daily_metrics(daily)

    assert metrics["total_rows"] == 6
    assert set(metrics["day_counts"].values()) == {2}
    assert metrics["duplicate_keys"] == 0
    assert metrics["pre_cast_non_null"] == metrics["post_cast_non_null"]
    assert metrics["failure_values"] == {0}
    assert metrics["label_null_count"] == 0
    assert all(null == 0 for null in metrics["null_counts"].values())


def test_collect_daily_metrics_detects_cast_loss_and_nulls(spark):
    rows = clean_rows()
    bad = list(bronze_row(serial="A", day="2026-04-01"))
    smart_start = BRONZE_SCHEMA.split(", ").index("smart_5_raw string")
    bad[smart_start] = "not-a-number"  # present before the cast, null after
    rows[0] = tuple(bad)

    metrics = summarise_daily_metrics(collect_daily_metrics(prepared_for(spark, rows)))
    failures = evaluate_metrics(metrics, bronze_rows=6, dataset_end=date(2026, 4, 3))

    assert metrics["pre_cast_non_null"]["smart_5_raw"] == 6
    assert metrics["post_cast_non_null"]["smart_5_raw"] == 5
    assert any("cast loss: smart_5_raw" in f for f in failures)


def test_dedup_happens_once_and_duplicates_do_not_reach_the_metrics(spark):
    rows = clean_rows() + [bronze_row(serial="A", day="2026-04-01", load_ts=datetime(2026, 8, 1))]

    metrics = summarise_daily_metrics(collect_daily_metrics(prepared_for(spark, rows)))

    assert metrics["total_rows"] == 6
    assert metrics["duplicate_keys"] == 0


def test_prepared_frame_carries_helper_columns_that_the_table_drops(spark):
    prepared = prepared_for(spark, clean_rows())

    assert all(PRECAST_PREFIX + c in prepared.columns for c in SMART_COLUMNS)
    assert tuple(prepared.select(*SILVER_COLUMNS).columns) == SILVER_COLUMNS


# --- run order: check first, write only if clean ---

@pytest.fixture
def bronze_view(spark, monkeypatch):
    def register(rows):
        bronze_df(spark, rows).createOrReplaceTempView("bronze_run_test")
        monkeypatch.setattr(silver, "BRONZE_TABLE", "bronze_run_test")

    return register


def test_run_silver_writes_after_clean_checks(spark, bronze_view):
    bronze_view(clean_rows())
    written, verified = [], []

    metrics = run_silver(spark, write_fn=written.append, verify_fn=lambda s, n: verified.append(n))

    assert len(written) == 1
    assert tuple(written[0].columns) == SILVER_COLUMNS
    assert verified == [6]
    assert metrics["total_rows"] == 6


def test_run_silver_does_not_write_when_a_check_fails(spark, bronze_view):
    rows = clean_rows()
    broken = list(rows[0])
    broken[BRONZE_SCHEMA.split(", ").index("model string")] = None
    rows[0] = tuple(broken)
    bronze_view(rows)
    written = []

    with pytest.raises(DataQualityError) as err:
        run_silver(spark, write_fn=written.append, verify_fn=lambda s, n: None)

    assert written == []
    assert "model" in str(err.value)
