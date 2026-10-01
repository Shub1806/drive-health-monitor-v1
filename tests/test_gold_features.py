import math
from datetime import date, timedelta

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType

import src.gold as gold
from src.gold import (
    FEATURE_COLUMNS,
    GOLD_COLUMNS,
    LABEL_COLUMNS,
    SMART_COLUMNS,
    build_prepared,
    collect_daily_metrics,
    collect_silver_daily,
    delta_col,
    evaluate_metrics,
    roll_col,
    run_gold,
    summarise_daily_metrics,
    summarise_silver_daily,
    z_col,
)
from src.quality import DataQualityError

START = date(2026, 4, 1)
SILVER_SCHEMA = (
    "date date, serial_number string, model string, capacity_bytes long, failure int, "
    + ", ".join(f"{c} long" for c in SMART_COLUMNS)
    + ", failure_within_30d int, label_is_complete boolean"
)


def srow(serial, offset, model="M1", failure=0, label=0, complete=True, **smart):
    values = [smart.get(c, 0) for c in SMART_COLUMNS]
    return (START + timedelta(days=offset), serial, model, 4000787030016, failure, *values, label, complete)


def silver_df(spark, rows):
    return spark.createDataFrame(rows, SILVER_SCHEMA)


def features(spark, rows):
    """Run the feature build and index results by (serial, day offset)."""
    out = build_prepared(silver_df(spark, rows)).collect()
    return {(r["serial_number"], (r["date"] - START).days): r for r in out}


# --- column design ---

def test_required_smart_columns_are_exactly_the_seven_including_smart_9():
    assert SMART_COLUMNS == (
        "smart_5_raw",
        "smart_9_raw",
        "smart_187_raw",
        "smart_188_raw",
        "smart_194_raw",
        "smart_197_raw",
        "smart_198_raw",
    )


def test_every_smart_column_has_all_three_features_in_the_output():
    for attribute in SMART_COLUMNS:
        for name in (roll_col(attribute), delta_col(attribute), z_col(attribute)):
            assert name in GOLD_COLUMNS
    assert "smart_9_raw_roll7_mean" in GOLD_COLUMNS
    assert "smart_9_raw_delta_1d" in GOLD_COLUMNS
    assert "smart_9_raw_cohort_z" in GOLD_COLUMNS


def test_gold_has_36_columns_in_the_designed_order():
    assert len(GOLD_COLUMNS) == 36
    assert GOLD_COLUMNS[:3] == ("date", "serial_number", "model")
    assert GOLD_COLUMNS[3:10] == SMART_COLUMNS
    assert GOLD_COLUMNS[-3:] == ("failure", "failure_within_30d", "label_is_complete")
    assert FEATURE_COLUMNS[-2:] == ("obs_days_7d", "cohort_size")


def test_labels_are_not_features():
    assert set(LABEL_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(LABEL_COLUMNS).isdisjoint(SMART_COLUMNS)


def test_features_do_not_change_when_labels_change(spark):
    base = [srow("A", i, smart_5_raw=i) for i in range(5)] + [srow("B", i, smart_5_raw=2 * i) for i in range(5)]
    flipped = [
        srow(r[1], (r[0] - START).days, smart_5_raw=r[5], failure=1, label=1, complete=False) for r in base
    ]

    feature_names = list(FEATURE_COLUMNS)
    before = build_prepared(silver_df(spark, base)).select("serial_number", "date", *feature_names).collect()
    after = build_prepared(silver_df(spark, flipped)).select("serial_number", "date", *feature_names).collect()

    key = lambda r: (r["serial_number"], r["date"])
    assert sorted(before, key=key) == sorted(after, key=key)


def test_build_keeps_every_row_and_select_gives_exactly_the_gold_columns(spark):
    rows = [srow("A", i) for i in range(4)] + [srow("B", i) for i in range(3)]

    prepared = build_prepared(silver_df(spark, rows))

    assert prepared.count() == 7
    assert tuple(prepared.select(*GOLD_COLUMNS).columns) == GOLD_COLUMNS


def test_feature_output_types_are_double_and_raw_columns_stay_bigint(spark):
    schema = {f.name: f.dataType for f in build_prepared(silver_df(spark, [srow("A", 0)])).schema.fields}

    for attribute in SMART_COLUMNS:
        assert isinstance(schema[roll_col(attribute)], DoubleType)
        assert isinstance(schema[delta_col(attribute)], DoubleType)
        assert isinstance(schema[z_col(attribute)], DoubleType)
        assert isinstance(schema[attribute], LongType)  # stored values are not touched


def test_rolling_mean_keeps_a_non_integer_average(spark):
    rows = [srow("A", 0, smart_5_raw=0), srow("A", 1, smart_5_raw=0), srow("A", 2, smart_5_raw=1)]

    result = features(spark, rows)

    assert result[("A", 2)][roll_col("smart_5_raw")] == pytest.approx(1 / 3, abs=1e-12)


def test_cohort_z_uses_the_unrounded_mean(spark):
    # 169 zeros and one 1: mean 1/170. A mean rounded to 3 decimals would shift z.
    rows = [srow(f"D{i}", 0, smart_5_raw=0) for i in range(169)] + [srow("X", 0, smart_5_raw=1)]
    n = 170
    mean = 1 / n
    std = math.sqrt((169 * mean**2 + (1 - mean) ** 2) / (n - 1))

    result = features(spark, rows)

    assert result[("X", 0)][z_col("smart_5_raw")] == pytest.approx((1 - mean) / std, rel=1e-9)


# --- rolling mean ---

def test_rolling_mean_includes_today_and_the_previous_six_calendar_days(spark):
    rows = [srow("A", i, smart_5_raw=i + 1) for i in range(8)]  # values 1..8 on days 0..7

    result = features(spark, rows)

    assert result[("A", 0)][roll_col("smart_5_raw")] == 1.0
    assert result[("A", 6)][roll_col("smart_5_raw")] == pytest.approx(4.0)  # days 0..6
    assert result[("A", 7)][roll_col("smart_5_raw")] == pytest.approx(5.0)  # days 1..7, day 0 dropped
    assert result[("A", 6)]["obs_days_7d"] == 7
    assert result[("A", 7)]["obs_days_7d"] == 7
    assert result[("A", 0)]["obs_days_7d"] == 1


def test_rolling_window_follows_calendar_days_across_gaps(spark):
    rows = [srow("A", 0, smart_5_raw=1), srow("A", 1, smart_5_raw=2), srow("A", 2, smart_5_raw=3), srow("A", 10, smart_5_raw=100)]

    result = features(spark, rows)

    assert result[("A", 2)][roll_col("smart_5_raw")] == pytest.approx(2.0)
    assert result[("A", 10)][roll_col("smart_5_raw")] == 100.0  # earlier rows are 8+ days back
    assert result[("A", 10)]["obs_days_7d"] == 1


def test_rolling_mean_skips_nulls_and_is_null_only_when_all_null(spark):
    rows = [
        srow("A", 0, smart_9_raw=10),
        srow("A", 1, smart_9_raw=None),
        srow("A", 2, smart_9_raw=30),
        srow("B", 0, smart_9_raw=None),
        srow("B", 1, smart_9_raw=None),
    ]

    result = features(spark, rows)

    assert result[("A", 2)][roll_col("smart_9_raw")] == pytest.approx(20.0)
    assert result[("A", 1)][roll_col("smart_9_raw")] == pytest.approx(10.0)
    assert result[("B", 1)][roll_col("smart_9_raw")] is None


def test_rolling_windows_do_not_mix_drives(spark):
    rows = [srow("A", i, smart_5_raw=1) for i in range(3)] + [srow("B", i, smart_5_raw=9) for i in range(3)]

    result = features(spark, rows)

    assert result[("A", 2)][roll_col("smart_5_raw")] == 1.0
    assert result[("B", 2)][roll_col("smart_5_raw")] == 9.0


# --- delta ---

def test_delta_is_today_minus_yesterday_and_null_on_the_first_row(spark):
    rows = [srow("A", 0, smart_5_raw=2), srow("A", 1, smart_5_raw=5), srow("A", 2, smart_5_raw=4)]

    result = features(spark, rows)

    assert result[("A", 0)][delta_col("smart_5_raw")] is None
    assert result[("A", 1)][delta_col("smart_5_raw")] == 3.0
    assert result[("A", 2)][delta_col("smart_5_raw")] == -1.0


def test_delta_is_null_across_a_date_gap(spark):
    rows = [srow("A", 0, smart_5_raw=1), srow("A", 1, smart_5_raw=2), srow("A", 3, smart_5_raw=9), srow("A", 4, smart_5_raw=10)]

    result = features(spark, rows)

    assert result[("A", 1)][delta_col("smart_5_raw")] == 1.0
    assert result[("A", 3)][delta_col("smart_5_raw")] is None  # day 2 is missing
    assert result[("A", 4)][delta_col("smart_5_raw")] == 1.0


def test_delta_is_null_when_current_or_previous_value_is_null(spark):
    rows = [
        srow("A", 0, smart_9_raw=10),
        srow("A", 1, smart_9_raw=None),
        srow("A", 2, smart_9_raw=30),
        srow("A", 3, smart_9_raw=35),
    ]

    result = features(spark, rows)

    assert result[("A", 1)][delta_col("smart_9_raw")] is None  # current null
    assert result[("A", 2)][delta_col("smart_9_raw")] is None  # previous null
    assert result[("A", 3)][delta_col("smart_9_raw")] == 5.0


def test_smart_9_gets_features_like_every_other_attribute(spark):
    rows = [srow("A", 0, smart_9_raw=100), srow("A", 1, smart_9_raw=124)]

    result = features(spark, rows)

    assert result[("A", 1)][delta_col("smart_9_raw")] == 24.0
    assert result[("A", 1)][roll_col("smart_9_raw")] == pytest.approx(112.0)


# --- cohort z-score ---

def test_cohort_z_scores_use_sample_std_of_same_model_and_date(spark):
    # values 0,0,0,4: mean 1, sample std 2 -> z(4)=1.5, z(0)=-0.5
    rows = [srow("A", 0, smart_5_raw=0), srow("B", 0, smart_5_raw=0), srow("C", 0, smart_5_raw=0), srow("D", 0, smart_5_raw=4)]

    result = features(spark, rows)

    assert result[("D", 0)][z_col("smart_5_raw")] == pytest.approx(1.5)
    assert result[("A", 0)][z_col("smart_5_raw")] == pytest.approx(-0.5)
    assert result[("A", 0)]["cohort_size"] == 4


def test_zero_std_gives_zero_z(spark):
    rows = [srow("A", 0, smart_5_raw=7), srow("B", 0, smart_5_raw=7), srow("C", 0, smart_5_raw=7)]

    result = features(spark, rows)

    assert all(result[(s, 0)][z_col("smart_5_raw")] == 0.0 for s in "ABC")


def test_null_std_gives_null_z(spark):
    rows = [srow("A", 0, smart_5_raw=7)]  # a cohort of one has no spread

    result = features(spark, rows)

    assert result[("A", 0)][z_col("smart_5_raw")] is None


def test_null_raw_gives_null_z_and_is_left_out_of_cohort_statistics(spark):
    rows = [srow("A", 0, smart_187_raw=None), srow("B", 0, smart_187_raw=2), srow("C", 0, smart_187_raw=4)]

    result = features(spark, rows)

    assert result[("A", 0)][z_col("smart_187_raw")] is None
    # only B and C reported: mean 3, sample std sqrt(2)
    assert result[("C", 0)][z_col("smart_187_raw")] == pytest.approx(1 / math.sqrt(2))
    assert result[("A", 0)]["cohort_size"] == 3  # cohort_size still counts every drive


def test_one_reporting_drive_in_a_cohort_gives_null_z(spark):
    rows = [srow("A", 0, smart_187_raw=None), srow("B", 0, smart_187_raw=5), srow("C", 0, smart_187_raw=None)]

    result = features(spark, rows)

    assert result[("B", 0)][z_col("smart_187_raw")] is None


def test_cohorts_do_not_mix_models_or_dates(spark):
    rows = [
        srow("A", 0, model="M1", smart_5_raw=0),
        srow("B", 0, model="M1", smart_5_raw=2),
        srow("C", 0, model="M2", smart_5_raw=1000),
        srow("D", 0, model="M2", smart_5_raw=1000),
        srow("A", 1, model="M1", smart_5_raw=50),
        srow("B", 1, model="M1", smart_5_raw=50),
    ]

    result = features(spark, rows)

    assert result[("B", 0)][z_col("smart_5_raw")] == pytest.approx(math.sqrt(2) / 2)  # mean 1, std sqrt(2)
    assert result[("C", 0)][z_col("smart_5_raw")] == 0.0
    assert result[("A", 1)][z_col("smart_5_raw")] == 0.0


# --- quality metrics on real feature output ---

def clean_rows():
    rows = []
    for offset in (0, 1, 2, 3):
        for serial in ("A", "B", "C"):
            if serial == "C" and offset == 2:
                continue  # a gap in drive C's dates
            rows.append(srow(serial, offset, smart_5_raw=offset + (2 if serial == "A" else 0), smart_187_raw=None if serial == "B" else offset))
    return rows


def metrics_for(spark, rows):
    silver = silver_df(spark, rows)
    metrics = summarise_daily_metrics(collect_daily_metrics(build_prepared(silver)))
    silver_summary = summarise_silver_daily(collect_silver_daily(silver))
    return metrics, silver_summary


def test_clean_features_pass_every_structural_check(spark):
    metrics, silver_summary = metrics_for(spark, clean_rows())

    assert {rule: n for rule, n in metrics["violations"].items() if n} == {}
    assert evaluate_metrics(metrics, silver_summary) == []
    assert metrics["total_rows"] == 11


def test_structural_check_catches_a_delta_on_a_first_row_and_across_a_gap(spark):
    prepared = build_prepared(silver_df(spark, clean_rows())).withColumn(
        delta_col("smart_5_raw"), F.lit(1.0)
    )

    metrics = summarise_daily_metrics(collect_daily_metrics(prepared))
    broken = {rule for rule, n in metrics["violations"].items() if n}

    assert "smart_5_raw: delta not NULL on a drive's first row" in broken
    assert "smart_5_raw: delta not NULL across a date gap" in broken


def test_structural_check_catches_delta_on_null_inputs(spark):
    prepared = build_prepared(silver_df(spark, clean_rows())).withColumn(
        delta_col("smart_187_raw"), F.lit(1.0)
    )

    broken = {r for r, n in summarise_daily_metrics(collect_daily_metrics(prepared))["violations"].items() if n}

    assert "smart_187_raw: delta not NULL although the current raw value is NULL" in broken


def test_structural_check_catches_a_missing_delta(spark):
    prepared = build_prepared(silver_df(spark, clean_rows())).withColumn(
        delta_col("smart_5_raw"), F.lit(None).cast("double")
    )

    broken = {r for r, n in summarise_daily_metrics(collect_daily_metrics(prepared))["violations"].items() if n}

    assert "smart_5_raw: delta NULL although both days are present and consecutive" in broken


def test_structural_check_catches_bad_z_scores(spark):
    base = build_prepared(silver_df(spark, clean_rows()))

    absurd = base.withColumn(z_col("smart_5_raw"), F.lit(99.0))
    missing = base.withColumn(z_col("smart_5_raw"), F.lit(None).cast("double"))
    nan = base.withColumn(z_col("smart_5_raw"), F.lit(float("nan")))

    def broken(df):
        return {r for r, n in summarise_daily_metrics(collect_daily_metrics(df))["violations"].items() if n}

    assert "smart_5_raw: z-score is NaN, infinite, or beyond the (n-1)/sqrt(n) bound" in broken(absurd)
    assert "smart_5_raw: z-score is NaN, infinite, or beyond the (n-1)/sqrt(n) bound" in broken(nan)
    assert any("z-score NULL although the value and a non-zero std" in r for r in broken(missing))


def test_zero_std_rule_is_enforced(spark):
    rows = [srow(s, 0, smart_5_raw=7) for s in "ABC"]
    prepared = build_prepared(silver_df(spark, rows)).withColumn(z_col("smart_5_raw"), F.lit(0.5))

    broken = {r for r, n in summarise_daily_metrics(collect_daily_metrics(prepared))["violations"].items() if n}

    assert "smart_5_raw: z-score is not 0.0 although the cohort std is 0" in broken


def test_z_bound_allows_the_exact_extreme(spark):
    # 0,0,0,4 puts one drive at |z| = (n-1)/sqrt(n) = 1.5 exactly; it must not count as a violation.
    rows = [srow("A", 0), srow("B", 0), srow("C", 0), srow("D", 0, smart_5_raw=4)]

    metrics = summarise_daily_metrics(collect_daily_metrics(build_prepared(silver_df(spark, rows))))

    assert metrics["violations"]["smart_5_raw: z-score is NaN, infinite, or beyond the (n-1)/sqrt(n) bound"] == 0
    assert metrics["max_abs_z"]["smart_5_raw"] == pytest.approx(1.5)


# --- run order ---

@pytest.fixture
def silver_view(spark, monkeypatch):
    def register(rows):
        silver_df(spark, rows).createOrReplaceTempView("silver_gold_test")
        monkeypatch.setattr(gold, "SILVER_TABLE", "silver_gold_test")

    return register


def test_run_gold_writes_36_columns_after_clean_checks(spark, silver_view):
    silver_view(clean_rows())
    written, verified = [], []

    metrics = run_gold(spark, write_fn=written.append, verify_fn=lambda s, n: verified.append(n))

    assert len(written) == 1
    assert tuple(written[0].columns) == GOLD_COLUMNS
    assert verified == [11]
    assert metrics["total_rows"] == 11


def test_run_gold_does_not_write_when_a_check_fails(spark, silver_view, monkeypatch):
    silver_view(clean_rows())
    real_build = gold.build_prepared
    monkeypatch.setattr(
        gold, "build_prepared", lambda df: real_build(df).withColumn(delta_col("smart_5_raw"), F.lit(1.0))
    )
    written = []

    with pytest.raises(DataQualityError) as err:
        run_gold(spark, write_fn=written.append, verify_fn=lambda s, n: None)

    assert written == []
    assert "delta not NULL" in str(err.value)


def test_run_gold_fails_when_gold_rows_do_not_match_silver(spark, silver_view, monkeypatch):
    silver_view(clean_rows())
    real_build = gold.build_prepared
    monkeypatch.setattr(gold, "build_prepared", lambda df: real_build(df).filter(F.col("serial_number") != "C"))
    written = []

    with pytest.raises(DataQualityError) as err:
        run_gold(spark, write_fn=written.append, verify_fn=lambda s, n: None)

    assert written == []
    assert "silver match" in str(err.value)
