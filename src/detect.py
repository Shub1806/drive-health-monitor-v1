"""Rule-based anomaly baseline.

Scoring and ranking are label-blind: the scoring functions only ever receive the columns in
SCORING_INPUT_COLUMNS, so failure, failure_within_30d and label_is_complete cannot influence a
score, a flagged date or a rank. Labels are joined only after the ranking is fixed, and only
to evaluate it.

Rule (per drive-day, four critical attributes): one point if the cohort z-score is above 3
(for cohorts of at least 30 drives), one point if the raw value rose since yesterday. The score
is the number of points, 0 to 8. A NULL never earns a point, because a value that was not
reported is not evidence of a problem.
"""
import argparse
import json
import uuid
from datetime import date, datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src import quality
from src.gold import GOLD_TABLE, delta_col, z_col

METHOD = "rule_baseline"
ANOMALIES_TABLE = "DRIVE_HEALTH.GOLD.ANOMALIES"
MODEL_RUNS_TABLE = "DRIVE_HEALTH.GOLD.MODEL_RUNS"

CRITICAL_COLUMNS = ("smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw")
Z_THRESHOLD = 3.0
MIN_COHORT_SIZE = 30
MAX_SCORE = 2 * len(CRITICAL_COLUMNS)

K_VALUES = (50, 100, 500)
TOP_N = 1000

# The only columns scoring may see. Labels are deliberately absent.
SCORING_INPUT_COLUMNS = (
    "serial_number",
    "model",
    "date",
    "cohort_size",
    *[z_col(a) for a in CRITICAL_COLUMNS],
    *[delta_col(a) for a in CRITICAL_COLUMNS],
)
LABEL_COLUMNS = ("failure", "failure_within_30d", "label_is_complete")

# Stands in for "no critical z-score" so the tie-break key is never NULL. NULL ordering
# differs between engines, a fixed sentinel does not.
NO_Z_SENTINEL = -1e9

# Official evaluation is the spec's last month. Its labels are censored because the data
# ends on June 30. May is supplemental validation with complete labels.
WINDOWS = (
    ("official", date(2026, 6, 1), date(2026, 6, 30)),
    ("supplemental", date(2026, 5, 1), date(2026, 5, 31)),
)


def _zt(attribute: str) -> str:
    return "_zt_" + attribute


def _it(attribute: str) -> str:
    return "_it_" + attribute


# --- stage 0/1: label-blind scoring ----------------------------------------------------

def select_scoring_inputs(gold: DataFrame, start: date, end: date) -> DataFrame:
    """Window by date and keep only the scoring inputs; labels are dropped here."""
    return gold.filter(F.col("date").between(start, end)).select(*SCORING_INPUT_COLUMNS)


def score_rows(df: DataFrame) -> DataFrame:
    """Add score (0..8), max_critical_z and the per-trigger flags to each drive-day."""
    leaked = [c for c in LABEL_COLUMNS if c in df.columns]
    if leaked:
        raise ValueError(f"label columns must not reach scoring: {leaked}")

    for attribute in CRITICAL_COLUMNS:
        df = df.withColumn(
            _zt(attribute),
            F.when(
                (F.col(z_col(attribute)) > Z_THRESHOLD) & (F.col("cohort_size") >= MIN_COHORT_SIZE), 1
            ).otherwise(0),
        ).withColumn(_it(attribute), F.when(F.col(delta_col(attribute)) > 0, 1).otherwise(0))

    triggers = [F.col(name) for a in CRITICAL_COLUMNS for name in (_zt(a), _it(a))]
    z_key = F.greatest(*[F.coalesce(F.col(z_col(a)), F.lit(NO_Z_SENTINEL)) for a in CRITICAL_COLUMNS])
    return (
        df.withColumn("score", sum(triggers, F.lit(0)))
        .withColumn("_z_key", z_key)
        .withColumn("max_critical_z", F.when(F.col("_z_key") > NO_Z_SENTINEL, F.col("_z_key")))
    )


def select_flagged_rows(scored: DataFrame) -> DataFrame:
    """One row per drive: the first day it reached its peak severity.

    Ordered by score, then the largest critical z, then the earliest date. Failure-day rows
    compete like any other row, since nothing here knows about failures.
    """
    best_first = Window.partitionBy("serial_number").orderBy(
        F.col("score").desc(), F.col("_z_key").desc(), F.col("date").asc()
    )
    return (
        scored.withColumn("_pick", F.row_number().over(best_first))
        .filter(F.col("_pick") == 1)
        .drop("_pick")
    )


def rank_top_drives(flagged: DataFrame, top_n: int = TOP_N) -> DataFrame:
    """Rank drives by score, then max critical z, then serial number; keep the top_n."""
    order = [F.col("score").desc(), F.col("_z_key").desc(), F.col("serial_number").asc()]
    top = flagged.orderBy(*order).limit(top_n)
    return top.withColumn("rank", F.row_number().over(Window.orderBy(*order)).cast("int"))


def add_trigger_text(df: DataFrame) -> DataFrame:
    """Readable explanation of which triggers fired; built only for the small top-N frame."""
    parts = []
    for attribute in CRITICAL_COLUMNS:
        parts.append(
            F.when(
                F.col(_zt(attribute)) == 1,
                F.concat(F.lit(f"{attribute} z="), F.round(F.col(z_col(attribute)), 1).cast("string")),
            )
        )
        parts.append(
            F.when(
                F.col(_it(attribute)) == 1,
                F.concat(
                    F.lit(f"{attribute} increased by "),
                    F.col(delta_col(attribute)).cast("bigint").cast("string"),
                ),
            )
        )
    return df.withColumn("triggered_rules", F.when(F.col("score") > 0, F.concat_ws("; ", *parts)))


# --- stage 4: evaluation, after ranks are fixed ------------------------------------------

def window_labels(gold: DataFrame, start: date, end: date) -> DataFrame:
    """The only place labels are read: keys plus label columns for the window."""
    return gold.filter(F.col("date").between(start, end)).select(
        "serial_number", "date", *LABEL_COLUMNS
    )


def attach_labels(ranked: DataFrame, labels: DataFrame) -> DataFrame:
    """Join labels onto already-ranked drives by (serial_number, flagged date)."""
    flagged = labels.select(
        F.col("serial_number").alias("_l_serial"),
        F.col("date").alias("_l_date"),
        F.col("failure").alias("eval_failure"),
        F.col("failure_within_30d").alias("eval_failure_within_30d"),
        F.col("label_is_complete").alias("eval_label_is_complete"),
    )
    joined = ranked.join(
        flagged,
        (ranked["serial_number"] == flagged["_l_serial"]) & (ranked["date"] == flagged["_l_date"]),
        "left",
    )
    return joined.drop("_l_serial", "_l_date")


def window_label_summary(labels: DataFrame) -> Dict:
    """Drives scored, failing drives (P), and how much of the window has complete labels."""
    row = labels.agg(
        F.countDistinct("serial_number").alias("n_drives"),
        F.countDistinct(F.when(F.col("failure_within_30d") == 1, F.col("serial_number"))).alias("n_positive"),
        F.count("*").alias("rows"),
        F.sum(F.when(F.col("label_is_complete"), 1).otherwise(0)).alias("complete_rows"),
    ).collect()[0]
    rows = int(row["rows"])
    return {
        "n_drives": int(row["n_drives"]),
        "n_positive": int(row["n_positive"]),
        "label_complete_share": (int(row["complete_rows"] or 0) / rows) if rows else 0.0,
    }


def is_early_warning_hit(failure: Optional[int], failure_within_30d: Optional[int]) -> bool:
    """A hit needs the flagged day to be before the failure and within 30 days of it.

    If the flagged day is the failure day itself, the drive had already failed: not a hit.
    """
    return failure == 0 and failure_within_30d == 1


def precision_at_k(hit_flags: Sequence[bool], k: int) -> Optional[float]:
    """Share of the top k ranked drives that are hits. Fewer than k drives means all of them."""
    top = list(hit_flags)[:k]
    return sum(top) / len(top) if top else None


def recall_at_k(hit_flags: Sequence[bool], k: int, n_positive: int) -> Optional[float]:
    """Hits among the top k divided by all failing drives in the window."""
    return sum(list(hit_flags)[:k]) / n_positive if n_positive else None


def base_rate(n_positive: int, n_drives: int) -> Optional[float]:
    return n_positive / n_drives if n_drives else None


def evaluate_ranking(
    hit_flags: Sequence[bool], n_positive: int, n_drives: int, ks: Sequence[int] = K_VALUES
) -> Dict[str, Optional[float]]:
    metrics = {"base_rate": base_rate(n_positive, n_drives)}
    for k in ks:
        metrics[f"precision_at_{k}"] = precision_at_k(hit_flags, k)
        metrics[f"recall_at_{k}"] = recall_at_k(hit_flags, k, n_positive)
    return metrics


def evaluation_note(label_complete_share: float, n_positive: int, dataset_end: date, ks: Sequence[int] = K_VALUES) -> str:
    """Plain-language caveat stored with the run so a censored result cannot be misread."""
    if label_complete_share >= 1.0:
        note = "Complete labels: every window row has a full 30-day future window in the data."
    else:
        scope = "no" if label_complete_share == 0 else f"only {label_complete_share:.0%} of"
        note = (
            f"Censored labels: label_is_complete is false for {'all' if label_complete_share == 0 else 'part of'} "
            f"window rows ({scope} window rows have a full 30-day future window) because the dataset ends "
            f"{dataset_end}. Failures after that date are invisible, so precision@k and the base rate are lower "
            "bounds; recall may move either way because hits and failing drives are both undercounted."
        )
    capped = [k for k in ks if n_positive < k]
    if capped:
        caps = ", ".join(f"precision@{k} <= {n_positive / k:.0%}" for k in capped)
        note += f" Only {n_positive} failing drives are observable, so {caps}."
    return note


def model_params() -> str:
    return json.dumps(
        {
            "critical_attributes": list(CRITICAL_COLUMNS),
            "z_threshold": Z_THRESHOLD,
            "min_cohort_size": MIN_COHORT_SIZE,
            "increase_rule": "delta_1d > 0",
            "score": "count of fired triggers, 0..8",
            "flagged_day": "score desc, max_critical_z desc, date asc",
            "drive_rank": "score desc, max_critical_z desc, serial_number asc",
            "hit": "flagged row has failure = 0 and failure_within_30d = 1",
            "top_n": TOP_N,
        },
        sort_keys=True,
    )


# --- run -------------------------------------------------------------------------------

def read_gold(spark) -> DataFrame:
    table = spark.table(GOLD_TABLE)
    return table.toDF(*[c.lower() for c in table.columns])


def detect_window(spark, gold: DataFrame, role: str, start: date, end: date, dataset_end: date) -> Dict:
    """Score, rank and evaluate one window. Returns the run row and the top drives to store."""
    # label-blind part
    scored = score_rows(select_scoring_inputs(gold, start, end))
    top = add_trigger_text(rank_top_drives(select_flagged_rows(scored)))

    # evaluation part: labels are read only now
    labels = window_labels(gold, start, end)
    evaluated = attach_labels(top, labels).orderBy("rank")
    rows = evaluated.collect()
    summary = window_label_summary(labels)

    quality.assert_quality(
        quality.check_detection_results(
            [r["rank"] for r in rows],
            [r["serial_number"] for r in rows],
            [r["score"] for r in rows],
            MAX_SCORE,
            TOP_N,
        )
    )

    hits = [is_early_warning_hit(r["eval_failure"], r["eval_failure_within_30d"]) for r in rows]
    metrics = evaluate_ranking(hits, summary["n_positive"], summary["n_drives"])
    run = {
        "run_id": str(uuid.uuid4()),
        "run_ts": datetime.now(timezone.utc).replace(tzinfo=None),  # naive UTC for a TIMESTAMP column
        "method": METHOD,
        "params": model_params(),
        "window_start": start,
        "window_end": end,
        "window_role": role,
        "window_labels_complete": summary["label_complete_share"] >= 1.0,
        "label_complete_row_share": summary["label_complete_share"],
        "n_drives_scored": summary["n_drives"],
        "n_positive_drives": summary["n_positive"],
        **metrics,
        "evaluation_note": evaluation_note(summary["label_complete_share"], summary["n_positive"], dataset_end),
    }
    anomalies = evaluated.select(
        F.lit(run["run_id"]).alias("run_id"),
        F.lit(METHOD).alias("method"),
        "rank",
        "serial_number",
        "model",
        F.col("date").alias("flagged_date"),
        "score",
        "max_critical_z",
        "triggered_rules",
        F.lit(start).alias("window_start"),
        F.lit(end).alias("window_end"),
        "eval_failure",
        "eval_failure_within_30d",
        "eval_label_is_complete",
        # Last, so an append works whether columns are matched by name or by position.
        F.col("score").cast("double").alias("anomaly_score"),
    )
    return {"run": run, "anomalies": anomalies, "top_rows": rows}


MODEL_RUNS_SCHEMA = (
    "run_id string, run_ts timestamp, method string, params string, window_start date, window_end date, "
    "window_role string, window_labels_complete boolean, label_complete_row_share double, "
    "n_drives_scored long, n_positive_drives long, base_rate double, "
    "precision_at_50 double, precision_at_100 double, precision_at_500 double, "
    "recall_at_50 double, recall_at_100 double, recall_at_500 double, evaluation_note string"
)


def write_results(spark, result: Mapping) -> None:
    """Append one MODEL_RUNS row and the top drives; history is kept by run_id."""
    run = result["run"]
    columns = [f.split(" ")[0] for f in MODEL_RUNS_SCHEMA.split(", ")]
    spark.createDataFrame([tuple(run[c] for c in columns)], MODEL_RUNS_SCHEMA).write.mode(
        "append"
    ).saveAsTable(MODEL_RUNS_TABLE)
    result["anomalies"].write.mode("append").saveAsTable(ANOMALIES_TABLE)


def format_run(run: Mapping) -> str:
    def pct(value):
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        f"[{run['window_role']}] {run['window_start']} to {run['window_end']} "
        f"(labels complete: {run['label_complete_row_share']:.0%} of rows)",
        f"  drives scored: {run['n_drives_scored']} | failing drives (P): {run['n_positive_drives']} "
        f"| base rate: {run['base_rate']:.3%}",
    ]
    for k in K_VALUES:
        lines.append(
            f"  k={k}: precision {pct(run[f'precision_at_{k}'])} | recall {pct(run[f'recall_at_{k}'])}"
        )
    lines.append(f"  note: {run['evaluation_note']}")
    return "\n".join(lines)


def run_detection(spark, write: bool = False, write_fn=write_results) -> List[Dict]:
    gold = read_gold(spark)
    dataset_end = gold.agg(F.max("date")).collect()[0][0]
    # Compute and check every window before persisting anything. detect_window raises if a
    # result check fails, so a bad window stops the run with nothing written.
    results = []
    for role, start, end in WINDOWS:
        result = detect_window(spark, gold, role, start, end, dataset_end)
        print(format_run(result["run"]))
        results.append(result)

    if write:
        for result in results:
            write_fn(spark, result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-based anomaly baseline.")
    parser.add_argument("--write", action="store_true", help="append results to GOLD.MODEL_RUNS and GOLD.ANOMALIES")
    args = parser.parse_args()

    from src.config import get_spark_session

    spark = get_spark_session()
    try:
        run_detection(spark, write=args.write)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
