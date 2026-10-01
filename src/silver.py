"""Silver layer: typed, deduplicated drive-day rows with the 30-day failure label.

Reads DRIVE_HEALTH.BRONZE.DRIVE_STATS_RAW. Missing SMART values stay NULL on purpose: a
drive that does not report an attribute is not the same as a drive reporting zero, and
filling zeros would make unmeasured drives look healthy. Gold decides how to handle nulls.

The run checks quality on the transformed DataFrame *before* writing, so a failed check
never replaces SILVER.DRIVE_STATS with bad data.
"""
from datetime import date, timedelta
from typing import Callable, Dict, List, Mapping, Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src import quality

BRONZE_TABLE = "DRIVE_HEALTH.BRONZE.DRIVE_STATS_RAW"
SILVER_TABLE = "DRIVE_HEALTH.SILVER.DRIVE_STATS"

SMART_COLUMNS = (
    "smart_5_raw",
    "smart_9_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_198_raw",
)

SILVER_COLUMNS = (
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    *SMART_COLUMNS,
    "failure_within_30d",
    "label_is_complete",
)

LABEL_HORIZON_DAYS = 30

# Helper columns carrying each SMART column's pre-cast "was reported" flag, so cast loss
# can be measured in the same pass as the other checks without deduplicating twice.
PRECAST_PREFIX = "_precast_reported_"


def cast_columns(df: DataFrame) -> DataFrame:
    """Turn the all-VARCHAR bronze columns into proper types."""
    typed = (
        df.withColumn("date", F.col("date").cast("date"))
        .withColumn("capacity_bytes", F.col("capacity_bytes").cast("bigint"))
        .withColumn("failure", F.col("failure").cast("int"))
    )
    for column in SMART_COLUMNS:
        typed = typed.withColumn(column, F.col(column).cast("bigint"))
    return typed


def deduplicate(df: DataFrame) -> DataFrame:
    """Keep one row per (serial_number, date): the most recently loaded, ties broken by file name."""
    newest_first = Window.partitionBy("serial_number", "date").orderBy(
        F.col("load_ts").desc(), F.col("source_file").desc()
    )
    return (
        df.withColumn("_row_rank", F.row_number().over(newest_first))
        .filter(F.col("_row_rank") == 1)
        .drop("_row_rank", "source_file", "load_ts")
    )


def mark_precast_reported(df: DataFrame) -> DataFrame:
    """Record, per SMART column, whether the raw (pre-cast) value was present."""
    for column in SMART_COLUMNS:
        df = df.withColumn(
            PRECAST_PREFIX + column, F.when(F.col(column).isNotNull(), 1).otherwise(0)
        )
    return df


def add_labels(df: DataFrame, dataset_end: Optional[date] = None) -> DataFrame:
    """Add failure_within_30d and label_is_complete.

    failure_within_30d is 1 if the drive fails on this date or any of the next 30 days
    (offsets 0..30, so a failure exactly 30 days out counts). The window is by calendar
    date, not row count, because drives have missing days and a row-based window would
    reach past 30 days.

    label_is_complete is true when a full 30 days of future data exists in the dataset.
    Rows near the end can still be labelled 1, but a 0 there may only mean the failure
    has not happened yet, so evaluation should be able to exclude them.
    """
    if dataset_end is None:
        dataset_end = df.agg(F.max("date")).collect()[0][0]

    day_number = F.datediff(F.col("date"), F.lit("1970-01-01").cast("date"))
    next_30_days = (
        Window.partitionBy("serial_number")
        .orderBy(day_number)
        .rangeBetween(0, LABEL_HORIZON_DAYS)
    )
    last_complete_day = dataset_end - timedelta(days=LABEL_HORIZON_DAYS)

    return df.withColumn(
        "failure_within_30d", F.max("failure").over(next_30_days).cast("int")
    ).withColumn("label_is_complete", F.col("date") <= F.lit(last_complete_day).cast("date"))


def build_prepared(raw: DataFrame, dataset_end: Optional[date] = None) -> DataFrame:
    """Dedup once, note which SMART values were present, cast, then label.

    The result still carries the helper columns; `.select(*SILVER_COLUMNS)` gives the table.
    """
    deduped = deduplicate(raw)
    return add_labels(cast_columns(mark_precast_reported(deduped)), dataset_end)


def build_silver(spark) -> DataFrame:
    """Bronze table -> silver DataFrame. Nothing is written here."""
    return build_prepared(spark.table(BRONZE_TABLE)).select(*SILVER_COLUMNS)


# --- quality metrics -------------------------------------------------------------------

_NULL_COLUMNS = ("serial_number", "model", "failure") + SMART_COLUMNS


def _count_if(condition) -> F.Column:
    return F.sum(F.when(condition, 1).otherwise(0))


def collect_daily_metrics(prepared: DataFrame) -> List[Dict]:
    """One aggregation pass over the prepared DataFrame, one result row per date.

    Everything the quality rules need is a per-day count, so a single pass answers all of
    them. Doing it per day (91 small rows) keeps the heavy transform from running again
    for each metric.
    """
    aggregates = [
        F.count("*").alias("rows"),
        # Rows minus distinct drives is the number of repeated (serial_number, date) keys.
        (F.count("*") - F.countDistinct("serial_number")).alias("duplicate_keys"),
        F.min("failure").alias("min_failure"),
        F.max("failure").alias("max_failure"),
        _count_if(
            (F.col("failure") == 1)
            & (F.col("failure_within_30d").isNull() | (F.col("failure_within_30d") != 1))
        ).alias("failure_unlabelled"),
        _count_if(F.col("label_is_complete")).alias("label_complete"),
        _count_if(~F.col("label_is_complete")).alias("label_incomplete"),
        _count_if(F.col("label_is_complete").isNull()).alias("label_null"),
    ]
    aggregates += [_count_if(F.col(c).isNull()).alias("null_" + c) for c in _NULL_COLUMNS]
    aggregates += [F.sum(PRECAST_PREFIX + c).alias("precast_" + c) for c in SMART_COLUMNS]

    return [row.asDict() for row in prepared.groupBy("date").agg(*aggregates).collect()]


def summarise_daily_metrics(daily: List[Mapping]) -> Dict:
    """Fold per-day rows into the plain totals the quality rules take."""
    def total(key):
        return sum(int(row[key] or 0) for row in daily)

    dated = [row for row in daily if row["date"] is not None]
    undated = [row for row in daily if row["date"] is None]

    null_counts = {c: total("null_" + c) for c in _NULL_COLUMNS}
    null_counts["date"] = sum(int(row["rows"]) for row in undated)

    total_rows = total("rows")
    failure_values = set()
    for row in daily:
        failure_values.update(v for v in (row["min_failure"], row["max_failure"]) if v is not None)

    return {
        "total_rows": total_rows,
        "day_counts": {row["date"]: int(row["rows"]) for row in dated},
        "null_counts": null_counts,
        "pre_cast_non_null": {c: total("precast_" + c) for c in SMART_COLUMNS},
        "post_cast_non_null": {c: total_rows - null_counts[c] for c in SMART_COLUMNS},
        "duplicate_keys": total("duplicate_keys"),
        "failure_values": failure_values,
        "failure_rows_without_label": total("failure_unlabelled"),
        "label_counts": {
            row["date"]: (int(row["label_complete"] or 0), int(row["label_incomplete"] or 0))
            for row in dated
        },
        "label_null_count": total("label_null"),
    }


def evaluate_metrics(metrics: Mapping, bronze_rows: int, dataset_end: date) -> List[str]:
    """Run every quality rule against the metrics and return all failure messages."""
    failures = []
    failures += quality.check_rows_per_day(metrics["day_counts"])
    failures += quality.check_duplicate_keys(metrics["duplicate_keys"])
    failures += quality.check_null_rates(metrics["null_counts"], metrics["total_rows"])
    failures += quality.check_cast_loss(metrics["pre_cast_non_null"], metrics["post_cast_non_null"])
    failures += quality.check_labels(
        metrics["failure_values"], metrics["failure_rows_without_label"]
    )
    failures += quality.check_label_complete(
        metrics["label_counts"], metrics["label_null_count"], dataset_end, LABEL_HORIZON_DAYS
    )
    # Dedup can only remove rows. More silver rows than bronze rows means something is wrong.
    if metrics["total_rows"] > bronze_rows:
        failures.append(
            f"row accounting: silver would have {metrics['total_rows']} rows, "
            f"more than bronze's {bronze_rows}"
        )
    return failures


def check_written_table(planned_rows: int, written_rows: int, written_columns: List[str]) -> List[str]:
    """After the write: same row count as planned, and exactly the designed columns."""
    failures = []
    if written_rows != planned_rows:
        failures.append(f"written table: {written_rows} rows, expected {planned_rows}")
    # Snowflake may store unquoted names in upper case, so compare case-insensitively.
    if [c.lower() for c in written_columns] != list(SILVER_COLUMNS):
        failures.append(f"written table: columns are {written_columns}, expected {list(SILVER_COLUMNS)}")
    return failures


# --- run -------------------------------------------------------------------------------

def write_silver(df: DataFrame) -> None:
    df.write.mode("overwrite").saveAsTable(SILVER_TABLE)


def verify_written_table(spark, planned_rows: int) -> None:
    """Read the table back and check it against the plan."""
    written = spark.table(SILVER_TABLE)
    quality.assert_quality(check_written_table(planned_rows, written.count(), written.columns))
    print("written table schema:", {f.name: f.dataType.simpleString() for f in written.schema.fields})
    for row in written.limit(5).collect():
        print("  sample:", row.asDict())


def run_silver(
    spark,
    write_fn: Callable[[DataFrame], None] = write_silver,
    verify_fn: Callable[..., None] = verify_written_table,
) -> Dict:
    """Transform, check, and only then write. Returns the metrics for reporting."""
    raw = spark.table(BRONZE_TABLE)
    bronze_rows = raw.count()
    # Taken from bronze, not the transformed frame, so it does not force the window work.
    dataset_end = raw.agg(F.max(F.col("date").cast("date"))).collect()[0][0]
    print(f"bronze rows: {bronze_rows}, dataset end: {dataset_end}")

    prepared = build_prepared(raw, dataset_end)
    metrics = summarise_daily_metrics(collect_daily_metrics(prepared))
    print(
        f"silver rows planned: {metrics['total_rows']} "
        f"(duplicates removed: {bronze_rows - metrics['total_rows']})"
    )

    quality.assert_quality(evaluate_metrics(metrics, bronze_rows, dataset_end))
    print("quality checks passed; writing", SILVER_TABLE)

    write_fn(prepared.select(*SILVER_COLUMNS))
    verify_fn(spark, metrics["total_rows"])
    return metrics


def main() -> None:
    from src.config import get_spark_session

    spark = get_spark_session()
    try:
        run_silver(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
