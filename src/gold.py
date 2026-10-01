"""Gold layer: per-drive-per-day features built from SILVER.DRIVE_STATS.

Every feature at date D uses only rows dated D or earlier, so nothing from the future can
leak into a score. failure / failure_within_30d / label_is_complete are carried along as
labels for evaluation and are never inputs to a feature.

Missing SMART values stay NULL all the way through (no imputation), and each feature has an
explicit rule for gaps in a drive's dates and for zero or undefined spread in its cohort.
"""
from datetime import date
from typing import Callable, Dict, List, Mapping

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src import quality
from src.silver import SILVER_TABLE, SMART_COLUMNS

GOLD_TABLE = "DRIVE_HEALTH.GOLD.DRIVE_FEATURES"

ROLLING_DAYS = 7

CONTEXT_COLUMNS = ("date", "serial_number", "model")
LABEL_COLUMNS = ("failure", "failure_within_30d", "label_is_complete")


def roll_col(attribute: str) -> str:
    return f"{attribute}_roll7_mean"


def delta_col(attribute: str) -> str:
    return f"{attribute}_delta_1d"


def z_col(attribute: str) -> str:
    return f"{attribute}_cohort_z"


FEATURE_COLUMNS = (
    *[roll_col(a) for a in SMART_COLUMNS],
    *[delta_col(a) for a in SMART_COLUMNS],
    *[z_col(a) for a in SMART_COLUMNS],
    "obs_days_7d",
    "cohort_size",
)

GOLD_COLUMNS = (*CONTEXT_COLUMNS, *SMART_COLUMNS, *FEATURE_COLUMNS, *LABEL_COLUMNS)

# Helper columns used to compute and then verify the features; they are not written.
PREV_DATE = "_prev_date"
PREV = "_prev_"
COHORT_MEAN = "_cohort_mean_"
COHORT_STD = "_cohort_std_"
COHORT_N = "_cohort_n_"


def _day_number() -> F.Column:
    return F.datediff(F.col("date"), F.lit("1970-01-01").cast("date"))


def add_rolling_features(df: DataFrame) -> DataFrame:
    """Mean over the drive's rows in the last 7 calendar days, today included.

    The window is by calendar day, not row count: a drive with a missing day averages fewer
    rows instead of reaching further back. Nulls are skipped, so the mean is NULL only when
    every value in the window is NULL. obs_days_7d says how many rows the window held, which
    also marks the first days of the quarter where earlier history is not in the data.
    """
    window = (
        Window.partitionBy("serial_number")
        .orderBy(_day_number())
        .rangeBetween(-(ROLLING_DAYS - 1), 0)
    )
    for attribute in SMART_COLUMNS:
        # Cast first: Snowflake's AVG over an integer column keeps only 3 decimals.
        df = df.withColumn(roll_col(attribute), F.avg(F.col(attribute).cast("double")).over(window))
    return df.withColumn("obs_days_7d", F.count(F.lit(1)).over(window).cast("int"))


def add_delta_features(df: DataFrame) -> DataFrame:
    """Today minus yesterday, only when yesterday's row exists and both values are present.

    A delta across a date gap would blend a 1-day change with a multi-day one, so it is NULL.
    The first row of a drive has no previous row, so it is NULL too.
    """
    window = Window.partitionBy("serial_number").orderBy(_day_number())
    df = df.withColumn(PREV_DATE, F.lag("date").over(window))
    consecutive = F.datediff(F.col("date"), F.col(PREV_DATE)) == 1
    for attribute in SMART_COLUMNS:
        df = df.withColumn(PREV + attribute, F.lag(attribute).over(window))
        both_present = F.col(attribute).isNotNull() & F.col(PREV + attribute).isNotNull()
        df = df.withColumn(
            delta_col(attribute),
            F.when(
                consecutive & both_present,
                (F.col(attribute) - F.col(PREV + attribute)).cast("double"),
            ),
        )
    return df


def add_cohort_features(df: DataFrame) -> DataFrame:
    """Z-score against the same model on the same date (the row itself is in the cohort).

    Mean and spread use only drives that reported the attribute. NULL if the drive did not
    report it or the spread is undefined (fewer than 2 reporting drives). Zero spread means
    every reporting drive has the same value, so the deviation is 0. The division only runs
    when the spread is non-zero. cohort_size counts all drives of the model that day, so
    detection can decide to ignore small cohorts.
    """
    cohort = Window.partitionBy("model", "date")
    df = df.withColumn("cohort_size", F.count(F.lit(1)).over(cohort).cast("int"))
    for attribute in SMART_COLUMNS:
        value = F.col(attribute)
        as_double = value.cast("double")  # same precision reason as the rolling mean
        df = (
            df.withColumn(COHORT_MEAN + attribute, F.avg(as_double).over(cohort))
            .withColumn(COHORT_STD + attribute, F.stddev_samp(as_double).over(cohort))
            .withColumn(COHORT_N + attribute, F.count(attribute).over(cohort))
        )
        std = F.col(COHORT_STD + attribute)
        df = df.withColumn(
            z_col(attribute),
            F.when(value.isNull() | std.isNull(), F.lit(None).cast("double"))
            .when(std == 0, F.lit(0.0))
            .otherwise((value - F.col(COHORT_MEAN + attribute)) / std),
        )
    return df


def build_prepared(silver: DataFrame) -> DataFrame:
    """All features, plus the helper columns the quality checks need."""
    return add_cohort_features(add_delta_features(add_rolling_features(silver)))


def read_silver(spark) -> DataFrame:
    """Silver comes back with upper-case names; work in lower case like the rest of the code."""
    table = spark.table(SILVER_TABLE)
    return table.toDF(*[c.lower() for c in table.columns])


# --- quality metrics -------------------------------------------------------------------

def _count_if(condition) -> F.Column:
    return F.sum(F.when(condition, 1).otherwise(0))


def _attribute_aggregates(attribute: str) -> List[F.Column]:
    """Per-attribute counts of rows that break a structural rule, plus plain null counts."""
    raw, prev = F.col(attribute), F.col(PREV + attribute)
    roll, delta, z = F.col(roll_col(attribute)), F.col(delta_col(attribute)), F.col(z_col(attribute))
    std, n = F.col(COHORT_STD + attribute), F.col(COHORT_N + attribute)
    has_prev_row = F.col(PREV_DATE).isNotNull()
    gap_days = F.datediff(F.col("date"), F.col(PREV_DATE))
    # |z| can never exceed (n-1)/sqrt(n) for a sample standard deviation over n values.
    # greatest(n, 1) keeps the expression defined for rows where n is 0.
    bound = (n - 1) / F.sqrt(F.greatest(n, F.lit(1)))
    within_bound = F.abs(z) <= bound * (1 + 1e-9) + 1e-9

    def agg(kind, column):
        return column.alias(f"{kind}__{attribute}")

    return [
        agg("null_raw", _count_if(raw.isNull())),
        agg("null_roll", _count_if(roll.isNull())),
        agg("null_delta", _count_if(delta.isNull())),
        agg("null_z", _count_if(z.isNull())),
        agg("max_abs_z", F.max(F.abs(z))),
        agg("roll_missing", _count_if(raw.isNotNull() & roll.isNull())),
        agg("delta_first_row_not_null", _count_if(~has_prev_row & delta.isNotNull())),
        agg("delta_gap_not_null", _count_if(has_prev_row & (gap_days != 1) & delta.isNotNull())),
        agg("delta_current_null_not_null", _count_if(raw.isNull() & delta.isNotNull())),
        agg("delta_previous_null_not_null", _count_if(prev.isNull() & delta.isNotNull())),
        agg(
            "delta_missing",
            _count_if(has_prev_row & (gap_days == 1) & raw.isNotNull() & prev.isNotNull() & delta.isNull()),
        ),
        agg("z_raw_null_not_null", _count_if(raw.isNull() & z.isNotNull())),
        agg("z_std_null_not_null", _count_if(std.isNull() & z.isNotNull())),
        agg("z_std_zero_not_zero", _count_if(raw.isNotNull() & (std == 0) & (z.isNull() | (z != 0)))),
        agg("z_missing", _count_if(raw.isNotNull() & std.isNotNull() & (std != 0) & z.isNull())),
        agg("z_not_finite_or_out_of_bound", _count_if(z.isNotNull() & ~within_bound)),
    ]


def collect_daily_metrics(prepared: DataFrame) -> List[Dict]:
    """One aggregation pass, one result row per date (see silver.collect_daily_metrics)."""
    aggregates = [
        F.count("*").alias("rows"),
        (F.count("*") - F.countDistinct("serial_number")).alias("duplicate_keys"),
        _count_if(F.col("serial_number").isNull()).alias("null_serial_number"),
        _count_if(F.col("model").isNull()).alias("null_model"),
        _count_if(F.col("failure").isNull()).alias("null_failure"),
        F.sum("failure").alias("failure_sum"),
        F.sum("failure_within_30d").alias("label_sum"),
        _count_if(~F.col("obs_days_7d").between(1, ROLLING_DAYS)).alias("obs_days_out_of_range"),
        _count_if(F.col("cohort_size") < 1).alias("cohort_size_invalid"),
    ]
    for attribute in SMART_COLUMNS:
        aggregates += _attribute_aggregates(attribute)
    return [row.asDict() for row in prepared.groupBy("date").agg(*aggregates).collect()]


_VIOLATION_KINDS = {
    "roll_missing": "rolling mean is NULL although the raw value is present",
    "delta_first_row_not_null": "delta not NULL on a drive's first row",
    "delta_gap_not_null": "delta not NULL across a date gap",
    "delta_current_null_not_null": "delta not NULL although the current raw value is NULL",
    "delta_previous_null_not_null": "delta not NULL although the previous raw value is NULL",
    "delta_missing": "delta NULL although both days are present and consecutive",
    "z_raw_null_not_null": "z-score not NULL although the raw value is NULL",
    "z_std_null_not_null": "z-score not NULL although the cohort std is NULL",
    "z_std_zero_not_zero": "z-score is not 0.0 although the cohort std is 0",
    "z_missing": "z-score NULL although the value and a non-zero std are present",
    "z_not_finite_or_out_of_bound": "z-score is NaN, infinite, or beyond the (n-1)/sqrt(n) bound",
}


def summarise_daily_metrics(daily: List[Mapping]) -> Dict:
    """Fold per-day rows into totals; violations map a readable rule name to its count."""
    def total(key):
        return sum(int(row[key] or 0) for row in daily)

    dated = [row for row in daily if row["date"] is not None]
    undated_rows = sum(int(row["rows"]) for row in daily if row["date"] is None)

    null_counts = {
        "date": undated_rows,
        "serial_number": total("null_serial_number"),
        "model": total("null_model"),
        "failure": total("null_failure"),
    }
    null_counts.update({a: total("null_raw__" + a) for a in SMART_COLUMNS})

    violations = {
        "obs_days_7d outside 1..7": total("obs_days_out_of_range"),
        "cohort_size below 1": total("cohort_size_invalid"),
    }
    for attribute in SMART_COLUMNS:
        for kind, description in _VIOLATION_KINDS.items():
            violations[f"{attribute}: {description}"] = total(f"{kind}__{attribute}")

    return {
        "total_rows": total("rows"),
        "day_counts": {row["date"]: int(row["rows"]) for row in dated},
        "null_counts": null_counts,
        "duplicate_keys": total("duplicate_keys"),
        "failure_sum": total("failure_sum"),
        "label_sum": total("label_sum"),
        "violations": violations,
        "feature_null_counts": {
            col: total(f"{kind}__{a}")
            for a in SMART_COLUMNS
            for kind, col in (
                ("null_roll", roll_col(a)),
                ("null_delta", delta_col(a)),
                ("null_z", z_col(a)),
            )
        },
        "max_abs_z": {
            a: max((row[f"max_abs_z__{a}"] for row in daily if row[f"max_abs_z__{a}"] is not None), default=None)
            for a in SMART_COLUMNS
        },
    }


def collect_silver_daily(silver: DataFrame) -> List[Dict]:
    """Per-date row count and label totals from Silver, to compare against Gold."""
    return [
        row.asDict()
        for row in silver.groupBy("date")
        .agg(
            F.count("*").alias("rows"),
            F.sum("failure").alias("failure_sum"),
            F.sum("failure_within_30d").alias("label_sum"),
        )
        .collect()
    ]


def summarise_silver_daily(daily: List[Mapping]) -> Dict:
    return {
        "total_rows": sum(int(r["rows"]) for r in daily),
        "day_counts": {r["date"]: int(r["rows"]) for r in daily if r["date"] is not None},
        "failure_sum": sum(int(r["failure_sum"] or 0) for r in daily),
        "label_sum": sum(int(r["label_sum"] or 0) for r in daily),
    }


def evaluate_metrics(metrics: Mapping, silver_summary: Mapping) -> List[str]:
    """Run every Gold quality rule and return all failure messages."""
    failures = []
    failures += quality.check_rows_per_day(metrics["day_counts"])
    failures += quality.check_duplicate_keys(metrics["duplicate_keys"])
    failures += quality.check_null_rates(metrics["null_counts"], metrics["total_rows"])
    failures += quality.check_zero_violations(metrics["violations"])
    failures += quality.check_matches_silver(metrics, silver_summary)
    return failures


def check_written_table(planned_rows: int, written_rows: int, written_columns: List[str]) -> List[str]:
    failures = []
    if written_rows != planned_rows:
        failures.append(f"written table: {written_rows} rows, expected {planned_rows}")
    # Snowflake may store unquoted names in upper case, so compare case-insensitively.
    if [c.lower() for c in written_columns] != list(GOLD_COLUMNS):
        failures.append(f"written table: columns are {written_columns}, expected {list(GOLD_COLUMNS)}")
    return failures


# --- run -------------------------------------------------------------------------------

def write_gold(df: DataFrame) -> None:
    df.write.mode("overwrite").saveAsTable(GOLD_TABLE)


def verify_written_table(spark, planned_rows: int) -> None:
    written = spark.table(GOLD_TABLE)
    quality.assert_quality(check_written_table(planned_rows, written.count(), written.columns))
    print("written table schema:", {f.name: f.dataType.simpleString() for f in written.schema.fields})
    for row in written.limit(3).collect():
        print("  sample:", row.asDict())


def run_gold(
    spark,
    write_fn: Callable[[DataFrame], None] = write_gold,
    verify_fn: Callable[..., None] = verify_written_table,
) -> Dict:
    """Build features, check them, and only then write. Returns the metrics for reporting."""
    silver = read_silver(spark)
    silver_summary = summarise_silver_daily(collect_silver_daily(silver))
    print(f"silver rows: {silver_summary['total_rows']}")

    prepared = build_prepared(silver)
    metrics = summarise_daily_metrics(collect_daily_metrics(prepared))
    print(f"gold rows planned: {metrics['total_rows']}")
    print("max |z| per attribute:", metrics["max_abs_z"])
    print("feature null counts:", metrics["feature_null_counts"])

    quality.assert_quality(evaluate_metrics(metrics, silver_summary))
    print("quality checks passed; writing", GOLD_TABLE)

    write_fn(prepared.select(*GOLD_COLUMNS))
    verify_fn(spark, metrics["total_rows"])
    return metrics


def main() -> None:
    from src.config import get_spark_session

    spark = get_spark_session()
    try:
        run_gold(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
