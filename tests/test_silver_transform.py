from datetime import datetime

from pyspark.sql.types import DateType, IntegerType, LongType

import src.silver as silver
from src.silver import SILVER_COLUMNS, SMART_COLUMNS, cast_columns, deduplicate

BRONZE_SCHEMA = (
    "date string, serial_number string, model string, capacity_bytes string, failure string, "
    "datacenter string, "
    + ", ".join(f"{c} string" for c in SMART_COLUMNS)
    + ", source_file string, load_ts timestamp"
)


def bronze_row(serial="A", day="2026-04-01", smart="1", load_ts=datetime(2026, 7, 1, 12),
               source_file="2026-04-01.csv.gz", failure="0", model="M1"):
    smart_values = [smart] * len(SMART_COLUMNS)
    return (day, serial, model, "4000787030016", failure, "dc1", *smart_values, source_file, load_ts)


def bronze_df(spark, rows):
    return spark.createDataFrame(rows, BRONZE_SCHEMA)


def test_cast_columns_types(spark):
    typed = cast_columns(bronze_df(spark, [bronze_row(smart="42")]))
    types = {f.name: f.dataType for f in typed.schema.fields}

    assert isinstance(types["date"], DateType)
    assert isinstance(types["capacity_bytes"], LongType)
    assert isinstance(types["failure"], IntegerType)
    assert all(isinstance(types[c], LongType) for c in SMART_COLUMNS)


def test_cast_columns_values_and_null_smart_stays_null(spark):
    row = list(bronze_row(smart="7"))
    row[BRONZE_SCHEMA.split(", ").index("smart_187_raw string")] = None  # not reported by this model

    typed = cast_columns(bronze_df(spark, [tuple(row)])).collect()[0]

    assert typed["capacity_bytes"] == 4000787030016  # larger than a 32-bit int
    assert typed["smart_5_raw"] == 7
    assert typed["smart_187_raw"] is None
    assert typed["smart_9_raw"] == 7


def test_dedup_keeps_the_newest_load(spark):
    rows = [
        bronze_row(smart="1", load_ts=datetime(2026, 7, 1, 10)),
        bronze_row(smart="2", load_ts=datetime(2026, 7, 2, 10)),
        bronze_row(smart="3", load_ts=datetime(2026, 7, 1, 23)),
    ]

    result = deduplicate(cast_columns(bronze_df(spark, rows))).collect()

    assert len(result) == 1
    assert result[0]["smart_5_raw"] == 2


def test_dedup_breaks_load_ts_ties_by_source_file_descending(spark):
    same_time = datetime(2026, 7, 1, 10)
    rows = [
        bronze_row(smart="1", load_ts=same_time, source_file="2026-04-01.csv.gz"),
        bronze_row(smart="2", load_ts=same_time, source_file="2026-04-01_reload.csv.gz"),
    ]

    result = deduplicate(cast_columns(bronze_df(spark, rows))).collect()

    assert len(result) == 1
    assert result[0]["smart_5_raw"] == 2


def test_dedup_keeps_distinct_drives_and_days_and_drops_load_columns(spark):
    rows = [
        bronze_row(serial="A", day="2026-04-01"),
        bronze_row(serial="A", day="2026-04-02"),
        bronze_row(serial="B", day="2026-04-01"),
    ]

    result = deduplicate(cast_columns(bronze_df(spark, rows)))

    assert result.count() == 3
    assert "source_file" not in result.columns
    assert "load_ts" not in result.columns


def test_build_silver_final_columns_and_order(spark, monkeypatch):
    bronze_df(spark, [bronze_row(), bronze_row(day="2026-04-02")]).createOrReplaceTempView("bronze_test")
    monkeypatch.setattr(silver, "BRONZE_TABLE", "bronze_test")

    result = silver.build_silver(spark)

    assert result.columns == [
        "date",
        "serial_number",
        "model",
        "capacity_bytes",
        "failure",
        "smart_5_raw",
        "smart_9_raw",
        "smart_187_raw",
        "smart_188_raw",
        "smart_194_raw",
        "smart_197_raw",
        "smart_198_raw",
        "failure_within_30d",
        "label_is_complete",
    ]
    assert tuple(result.columns) == SILVER_COLUMNS
    assert result.count() == 2
