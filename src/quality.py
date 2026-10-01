"""Data quality rules for the silver layer.

Each check takes plain numbers (counts gathered by a Spark query elsewhere) and returns a
list of failure messages, so the rules can be tested without a Snowflake connection.
`assert_quality` turns any collected failures into one loud error.
"""
from datetime import date, timedelta
from typing import Iterable, List, Mapping, Tuple

# A day with less than this share of the median day's rows is treated as a partial load.
MIN_DAY_FRACTION_OF_MEDIAN = 0.5

# These identify a row; a null in any of them makes the row unusable.
REQUIRED_COLUMNS = ("date", "serial_number", "model", "failure")

# Ceilings sit above the rates seen in Q2 2026 bronze (5/9/194/198 under 1%, 197 about 2.4%,
# 187/188 about 68% because many models never report them). They catch a column going
# mostly empty, not normal model-to-model variation.
MAX_SMART_NULL_RATE = {
    "smart_5_raw": 0.05,
    "smart_9_raw": 0.05,
    "smart_187_raw": 0.80,
    "smart_188_raw": 0.80,
    "smart_194_raw": 0.05,
    "smart_197_raw": 0.10,
    "smart_198_raw": 0.05,
}


class DataQualityError(Exception):
    """Raised when one or more data quality rules fail."""


def check_rows_per_day(day_counts: Mapping[date, int]) -> List[str]:
    """Every calendar day in range has rows, and none is far below a typical day."""
    if not day_counts:
        return ["rows per day: no rows found"]

    failures = []
    first, last = min(day_counts), max(day_counts)
    expected_days = (last - first).days + 1
    missing = [
        first + timedelta(days=i)
        for i in range(expected_days)
        if day_counts.get(first + timedelta(days=i), 0) == 0
    ]
    if missing:
        failures.append(
            f"rows per day: {len(missing)} day(s) with no rows, first missing {missing[0]}"
        )

    ordered = sorted(day_counts.values())
    median = ordered[len(ordered) // 2]
    floor = median * MIN_DAY_FRACTION_OF_MEDIAN
    thin = sorted(day for day, count in day_counts.items() if 0 < count < floor)
    if thin:
        failures.append(
            f"rows per day: {len(thin)} day(s) below {MIN_DAY_FRACTION_OF_MEDIAN:.0%} "
            f"of the median ({median}), first {thin[0]}"
        )
    return failures


def check_duplicate_keys(duplicate_key_count: int) -> List[str]:
    """(serial_number, date) must be unique."""
    if duplicate_key_count > 0:
        return [f"duplicate keys: {duplicate_key_count} (serial_number, date) pair(s) repeat"]
    return []


def check_null_rates(
    null_counts: Mapping[str, int],
    total_rows: int,
    max_smart_rates: Mapping[str, float] = MAX_SMART_NULL_RATE,
) -> List[str]:
    """No nulls in identity columns; SMART null rates stay under their ceilings."""
    if total_rows <= 0:
        return ["null rates: table is empty"]

    failures = []
    for column in REQUIRED_COLUMNS:
        nulls = null_counts.get(column)
        if nulls is None:
            failures.append(f"null rates: no count supplied for required column {column}")
        elif nulls > 0:
            failures.append(f"null rates: {column} has {nulls} null(s), none allowed")

    for column, ceiling in max_smart_rates.items():
        nulls = null_counts.get(column)
        if nulls is None:
            failures.append(f"null rates: no count supplied for {column}")
            continue
        rate = nulls / total_rows
        if rate > ceiling:
            failures.append(f"null rates: {column} is {rate:.2%} null, ceiling {ceiling:.0%}")
    return failures


def check_cast_loss(
    bronze_non_null: Mapping[str, int], silver_non_null: Mapping[str, int]
) -> List[str]:
    """A cast that quietly produces null shows up as fewer non-null values after the cast.

    Duplicates removed in silver can legitimately lower the counts, so callers should pass
    bronze counts taken after dedup.
    """
    failures = []
    for column, before in bronze_non_null.items():
        after = silver_non_null.get(column)
        if after is None:
            failures.append(f"cast loss: {column} missing from silver counts")
        elif after != before:
            failures.append(f"cast loss: {column} had {before} values before casting, {after} after")
    return failures


def check_row_accounting(bronze_rows: int, silver_rows: int, duplicates_removed: int) -> List[str]:
    """Silver should differ from bronze only by the duplicates that were removed."""
    expected = bronze_rows - duplicates_removed
    if silver_rows != expected:
        return [
            f"row accounting: silver has {silver_rows} rows, expected {expected} "
            f"({bronze_rows} bronze - {duplicates_removed} duplicates)"
        ]
    return []


def check_labels(failure_values: Iterable[int], failure_rows_without_label: int) -> List[str]:
    """`failure` is strictly 0/1 and every failure day carries label 1 (offset 0 is in the window)."""
    failures = []
    unexpected = sorted(set(failure_values) - {0, 1})
    if unexpected:
        failures.append(f"labels: failure has values other than 0/1: {unexpected}")
    if failure_rows_without_label > 0:
        failures.append(
            f"labels: {failure_rows_without_label} failure row(s) have failure_within_30d = 0"
        )
    return failures


def check_label_complete(
    label_counts: Mapping[date, Tuple[int, int]],
    null_count: int,
    dataset_end: date,
    horizon_days: int = 30,
) -> List[str]:
    """label_is_complete must match the date rule, using per-day (complete, incomplete) row counts.

    A day is complete when `horizon_days` of later data exists, i.e. date <= dataset_end - horizon.
    """
    failures = []
    if null_count > 0:
        failures.append(f"label_is_complete: {null_count} null value(s)")

    last_complete_day = dataset_end - timedelta(days=horizon_days)
    wrong_days = sorted(
        day
        for day, (complete, incomplete) in label_counts.items()
        if (incomplete > 0 if day <= last_complete_day else complete > 0)
    )
    if wrong_days:
        failures.append(
            f"label_is_complete: {len(wrong_days)} day(s) disagree with the date rule, "
            f"first {wrong_days[0]}"
        )

    total_complete = sum(complete for complete, _ in label_counts.values())
    total_incomplete = sum(incomplete for _, incomplete in label_counts.values())
    if total_incomplete == 0:
        failures.append("label_is_complete: no incomplete rows; the final days should be incomplete")
    span_days = (max(label_counts) - min(label_counts)).days + 1 if label_counts else 0
    if span_days > horizon_days and total_complete == 0:
        failures.append("label_is_complete: no complete rows despite a span longer than the horizon")
    return failures


def check_zero_violations(violations: Mapping[str, int]) -> List[str]:
    """Each entry counts rows that break a structural rule; every count must be zero."""
    return [f"{rule}: {count} row(s)" for rule, count in violations.items() if count > 0]


def check_matches_silver(gold: Mapping, silver: Mapping) -> List[str]:
    """Gold adds columns, never rows or labels, so these totals must equal Silver's."""
    failures = []
    if gold["total_rows"] != silver["total_rows"]:
        failures.append(
            f"silver match: gold has {gold['total_rows']} rows, silver has {silver['total_rows']}"
        )
    if gold["day_counts"] != silver["day_counts"]:
        differing = sorted(
            day
            for day in set(gold["day_counts"]) | set(silver["day_counts"])
            if gold["day_counts"].get(day) != silver["day_counts"].get(day)
        )
        failures.append(
            f"silver match: rows per day differ on {len(differing)} day(s), first {differing[0]}"
        )
    for key in ("failure_sum", "label_sum"):
        if gold[key] != silver[key]:
            failures.append(f"silver match: {key} is {gold[key]} in gold, {silver[key]} in silver")
    return failures


def assert_quality(failures: Iterable[str]) -> None:
    """Raise one error listing every failed rule, or return quietly if none failed."""
    failures = list(failures)
    if failures:
        raise DataQualityError(
            f"{len(failures)} data quality check(s) failed:\n- " + "\n- ".join(failures)
        )
