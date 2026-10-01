"""Isolation Forest detector.

Trained on non-failure-day rows: failure = 0 is used only to drop the same-day failure row.
That can leave pre-failure rows of drives that fail later in the training data, a small
natural contamination (on the order of 0.1% to 0.3% of sampled rows) that an unsupervised
forest tolerates. failure_within_30d and label_is_complete never select training rows, and no
label is ever a model feature.

Each evaluation window has its own model, trained entirely on dates before that window
starts. Scoring, per-drive selection and ranking see only the 21 feature columns; labels are
joined afterwards, only to evaluate the ranking.
"""
import argparse
import json
import platform
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import sklearn
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from sklearn.ensemble import IsolationForest

from src import detect, quality
from src.gold import delta_col, roll_col, z_col
from src.silver import SMART_COLUMNS

METHOD = "isolation_forest"

# Order matters: it is the column order of the training matrix.
FEATURE_COLUMNS = (
    *[roll_col(a) for a in SMART_COLUMNS],
    *[delta_col(a) for a in SMART_COLUMNS],
    *[z_col(a) for a in SMART_COLUMNS],
)
Z_FEATURES = tuple(z_col(a) for a in SMART_COLUMNS)
LABEL_COLUMNS = detect.LABEL_COLUMNS

MIN_COHORT_SIZE = 30  # a z-score from a smaller cohort is not trusted
SEED = 42
TRAIN_CAP = 200_000

FOREST_PARAMS = {
    "n_estimators": 200,
    "max_samples": 256,
    "max_features": 1.0,
    "contamination": "auto",
    "bootstrap": False,
    "random_state": SEED,
}

# (role, evaluation start, evaluation end, training start, training end)
# Training ends before the evaluation window starts, and it starts on April 7 so every
# rolling window is a full 7 days.
IF_WINDOWS = (
    ("official", date(2026, 6, 1), date(2026, 6, 30), date(2026, 4, 7), date(2026, 5, 31)),
    ("supplemental", date(2026, 5, 1), date(2026, 5, 31), date(2026, 4, 7), date(2026, 4, 30)),
)


# --- training rows: deterministic and label-free ---------------------------------------

def training_candidates(gold: DataFrame, start: date, end: date) -> DataFrame:
    """Rows in the training window, minus same-day failure rows.

    failure is read here only to drop the failure row itself; the frame that leaves this
    function carries no label column.
    """
    return gold.filter(F.col("date").between(start, end) & (F.col("failure") == 0)).select(
        "serial_number", "date", "cohort_size", *FEATURE_COLUMNS
    )


def _drive_hash(seed: int) -> F.Column:
    return F.sha2(F.concat_ws("|", F.col("serial_number"), F.lit(str(seed))), 256)


def _row_hash(seed: int) -> F.Column:
    return F.sha2(
        F.concat_ws("|", F.col("serial_number"), F.col("date").cast("string"), F.lit(str(seed))), 256
    )


def sample_training_rows(candidates: DataFrame, cap: int = TRAIN_CAP, seed: int = SEED) -> DataFrame:
    """At most `cap` rows, exactly one per drive, chosen without looking at any label.

    1. Keep the `cap` drives with the smallest sha2(serial_number|seed); ties by serial_number.
       The drive hash does not depend on how many rows a drive has, so long-lived drives get
       no extra chance of being kept.
    2. For each kept drive keep the one row with the smallest sha2(serial_number|date|seed);
       ties by earliest date.
    """
    drives = (
        candidates.select("serial_number")
        .distinct()
        .withColumn("_drive_hash", _drive_hash(seed))
        .orderBy(F.col("_drive_hash").asc(), F.col("serial_number").asc())
        .limit(cap)
    )
    one_per_drive = Window.partitionBy("serial_number").orderBy(
        F.col("_row_hash").asc(), F.col("date").asc()
    )
    return (
        candidates.join(drives, "serial_number")
        .withColumn("_row_hash", _row_hash(seed))
        .withColumn("_pick", F.row_number().over(one_per_drive))
        .filter(F.col("_pick") == 1)
        .drop("_pick", "_row_hash", "_drive_hash")
    )


def prepare_model_frame(df: DataFrame) -> DataFrame:
    """Apply the cohort-size guard and keep exactly the identifiers plus the 21 features.

    cohort_size is used only here, to blank z-scores computed from small cohorts.
    """
    leaked = [c for c in LABEL_COLUMNS if c in df.columns]
    if leaked:
        raise ValueError(f"label columns must not reach the model: {leaked}")
    for column in Z_FEATURES:
        df = df.withColumn(column, F.when(F.col("cohort_size") >= MIN_COHORT_SIZE, F.col(column)))
    return df.select("serial_number", "date", *FEATURE_COLUMNS)


def scoring_inputs(gold: DataFrame, start: date, end: date) -> DataFrame:
    """Every row of the evaluation window, with no label column selected."""
    return prepare_model_frame(
        gold.filter(F.col("date").between(start, end)).select(
            "serial_number", "date", "cohort_size", *FEATURE_COLUMNS
        )
    )


# --- preprocessing and model -------------------------------------------------------------

def signed_log1p(values: np.ndarray) -> np.ndarray:
    """log1p that keeps the sign, so negative deltas and z-scores are handled; NaN stays NaN."""
    return np.sign(values) * np.log1p(np.abs(values))


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(FEATURE_COLUMNS)].to_numpy(dtype="float64", na_value=np.nan)


class Preprocessor:
    """Signed log1p, then median imputation with medians learned only from training data."""

    def __init__(self):
        self.medians_ = None

    def fit(self, frame: pd.DataFrame) -> "Preprocessor":
        transformed = signed_log1p(feature_matrix(frame))
        all_nan = [c for c, empty in zip(FEATURE_COLUMNS, np.isnan(transformed).all(axis=0)) if empty]
        if all_nan:
            raise ValueError(f"training feature(s) entirely NaN: {all_nan}")
        self.medians_ = np.nanmedian(transformed, axis=0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("Preprocessor.transform called before fit")
        transformed = signed_log1p(feature_matrix(frame))
        return np.where(np.isnan(transformed), self.medians_, transformed)


class TrainedModel:
    def __init__(self, preprocessor: Preprocessor, forest: IsolationForest, n_train_rows: int):
        self.preprocessor = preprocessor
        self.forest = forest
        self.n_train_rows = n_train_rows


def fit_model(sample: pd.DataFrame) -> TrainedModel:
    # Spark returns rows in no fixed order, and the forest's random draws depend on row order,
    # so fix the order to make the same sample always produce the same model.
    sample = sample.sort_values(["serial_number", "date"], kind="mergesort").reset_index(drop=True)
    preprocessor = Preprocessor().fit(sample)
    forest = IsolationForest(**FOREST_PARAMS, n_jobs=-1)
    forest.fit(preprocessor.transform(sample))
    return TrainedModel(preprocessor, forest, len(sample))


def score_frame(model: TrainedModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Anomaly score per row; higher means more anomalous (sklearn reports the opposite sign)."""
    scores = -model.forest.score_samples(model.preprocessor.transform(frame))
    return pd.DataFrame(
        {
            "serial_number": frame["serial_number"].to_numpy(),
            "date": pd.to_datetime(frame["date"]).to_numpy(),
            "anomaly_score": scores,
        }
    )


# --- one result per drive ----------------------------------------------------------------

def update_best(best: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Merge scored rows into the per-drive best: highest score, earliest date on a tie.

    The outcome does not depend on how rows are split into chunks or in which order they
    arrive.
    """
    combined = scored if best.empty else pd.concat([best, scored], ignore_index=True)
    combined = combined.sort_values(
        ["anomaly_score", "date"], ascending=[False, True], kind="mergesort"
    )
    return combined.drop_duplicates("serial_number", keep="first").reset_index(drop=True)


def rank_best(best: pd.DataFrame, top_n: int = detect.TOP_N) -> pd.DataFrame:
    """Rank drives by anomaly_score desc, then serial_number asc; keep the top_n."""
    ranked = best.sort_values(["anomaly_score", "serial_number"], ascending=[False, True], kind="mergesort")
    ranked = ranked.head(top_n).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype="int64")
    return ranked


def iter_day_chunks(frame: DataFrame, start: date, end: date):
    """One pandas chunk per calendar day, so memory stays at a single day of rows."""
    day = start
    while day <= end:
        chunk = frame.filter(F.col("date") == F.lit(day).cast("date")).toPandas()
        if len(chunk):
            yield chunk
        day += timedelta(days=1)


def score_window(model: TrainedModel, chunks) -> Dict:
    """Score every row of the window; keep each drive's best day. Returns the table and row count."""
    best = pd.DataFrame({"serial_number": [], "date": pd.to_datetime([]), "anomaly_score": []})
    n_rows = 0
    for chunk in chunks:
        n_rows += len(chunk)
        best = update_best(best, score_frame(model, chunk))
    return {"best": best, "n_rows": n_rows}


# --- run ---------------------------------------------------------------------------------

def _json_default(value):
    """Spark Connect returns numpy integers from count(); turn numpy scalars into plain numbers."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def model_params(model: TrainedModel, window: Dict, counts: Dict) -> str:
    return json.dumps(
        {
            "method": METHOD,
            "forest": {**FOREST_PARAMS, "n_jobs": -1},
            "features": list(FEATURE_COLUMNS),
            "cohort_guard": f"z-scores set to NULL where cohort_size < {MIN_COHORT_SIZE}",
            "preprocessing": "signed log1p, then median imputation fitted on the training sample only",
            "medians": [float(m) for m in model.preprocessor.medians_],
            "training": {
                "window": [str(window["train_start"]), str(window["train_end"])],
                "selection": "non-failure-day rows: failure = 0 only removes the same-day failure row; "
                "failure_within_30d and label_is_complete are not used",
                "contamination_note": "may include pre-failure rows of drives that fail later",
                "sample": "drive-level sha2(serial_number|seed) keeps the smallest hashes up to the cap; "
                "then one row per drive by sha2(serial_number|date|seed)",
                "seed": SEED,
                "cap": TRAIN_CAP,
                **counts,
            },
            "score": "anomaly_score = -score_samples, higher is more anomalous",
            "flagged_day": "highest anomaly_score, earliest date at the maximum",
            "drive_rank": "anomaly_score desc, serial_number asc",
            "hit": "flagged row has failure = 0 and failure_within_30d = 1",
            "top_n": detect.TOP_N,
            "versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        sort_keys=True,
        default=_json_default,
    )


def run_window(
    spark,
    gold: DataFrame,
    role: str,
    start: date,
    end: date,
    train_start: date,
    train_end: date,
    dataset_end: date,
    cap: int = TRAIN_CAP,
) -> Dict:
    """Train on the window's own training dates, score every evaluation row, rank, then evaluate."""
    if train_end >= start:
        raise ValueError(f"training must end before the evaluation window starts ({train_end} >= {start})")

    # training: label-free
    candidates = training_candidates(gold, train_start, train_end)
    counts = {
        "n_candidate_rows": candidates.count(),
        "n_candidate_drives": candidates.select("serial_number").distinct().count(),
    }
    sample = prepare_model_frame(sample_training_rows(candidates, cap)).toPandas()
    quality.assert_quality(
        quality.check_training_sample(
            len(sample), sample["serial_number"].nunique(), len(FEATURE_COLUMNS),
            counts["n_candidate_drives"], cap, len(FEATURE_COLUMNS),
        )
    )
    counts["n_training_rows"] = len(sample)
    model = fit_model(sample)

    # scoring and ranking: label-free
    scored = score_window(model, iter_day_chunks(scoring_inputs(gold, start, end), start, end))
    counts["n_rows_scored"] = scored["n_rows"]
    ranked = rank_best(scored["best"])

    # evaluation: labels are read only now, after the ranking is fixed
    schema = "serial_number string, date date, anomaly_score double, rank int"
    top = spark.createDataFrame(
        [(r.serial_number, r.date.date(), float(r.anomaly_score), int(r.rank)) for r in ranked.itertuples()],
        schema,
    )
    names = gold.filter(F.col("date").between(start, end)).select("serial_number", "date", "model")
    labels = detect.window_labels(gold, start, end)
    evaluated = detect.attach_labels(top.join(names, ["serial_number", "date"], "left"), labels).orderBy("rank")
    rows = evaluated.collect()
    summary = detect.window_label_summary(labels)

    quality.assert_quality(
        quality.check_detection_results(
            [r["rank"] for r in rows], [r["serial_number"] for r in rows],
            [r["anomaly_score"] for r in rows], 1.0, detect.TOP_N,
        )
    )

    hits = [detect.is_early_warning_hit(r["eval_failure"], r["eval_failure_within_30d"]) for r in rows]
    metrics = detect.evaluate_ranking(hits, summary["n_positive"], summary["n_drives"])
    run = {
        "run_id": str(uuid.uuid4()),
        "run_ts": datetime.now(timezone.utc).replace(tzinfo=None),  # naive UTC for a TIMESTAMP column
        "method": METHOD,
        "params": model_params(model, {"train_start": train_start, "train_end": train_end}, counts),
        "window_start": start,
        "window_end": end,
        "window_role": role,
        "window_labels_complete": summary["label_complete_share"] >= 1.0,
        "label_complete_row_share": summary["label_complete_share"],
        "n_drives_scored": summary["n_drives"],
        "n_positive_drives": summary["n_positive"],
        **metrics,
        "evaluation_note": detect.evaluation_note(
            summary["label_complete_share"], summary["n_positive"], dataset_end
        ),
    }
    # Same columns, order and types as the rule baseline's output; anomaly_score is last.
    anomalies = evaluated.select(
        F.lit(run["run_id"]).alias("run_id"),
        F.lit(METHOD).alias("method"),
        "rank",
        "serial_number",
        "model",
        F.col("date").alias("flagged_date"),
        F.lit(None).cast("int").alias("score"),
        F.lit(None).cast("double").alias("max_critical_z"),
        F.lit(None).cast("string").alias("triggered_rules"),
        F.lit(start).alias("window_start"),
        F.lit(end).alias("window_end"),
        "eval_failure",
        "eval_failure_within_30d",
        "eval_label_is_complete",
        "anomaly_score",
    )
    return {"run": run, "anomalies": anomalies, "top_rows": rows}


def run_isolation_forest(
    spark,
    write: bool = False,
    write_fn=detect.write_results,
    windows: Sequence = IF_WINDOWS,
    cap: int = TRAIN_CAP,
) -> List[Dict]:
    """Train, score and check every window first; persist only if all of them passed."""
    gold = detect.read_gold(spark)
    dataset_end = gold.agg(F.max("date")).collect()[0][0]
    results = []
    for role, start, end, train_start, train_end in windows:
        result = run_window(spark, gold, role, start, end, train_start, train_end, dataset_end, cap)
        print(detect.format_run(result["run"]))
        results.append(result)

    if write:
        for result in results:
            write_fn(spark, result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Isolation Forest detector.")
    parser.add_argument("--write", action="store_true", help="append results to GOLD.MODEL_RUNS and GOLD.ANOMALIES")
    args = parser.parse_args()

    from src.config import get_spark_session

    spark = get_spark_session()
    try:
        run_isolation_forest(spark, write=args.write)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
