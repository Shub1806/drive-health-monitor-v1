import hashlib
import json
import sys
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.iforest as iforest
from src import detect
from src.iforest import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    Preprocessor,
    fit_model,
    prepare_model_frame,
    rank_best,
    run_isolation_forest,
    run_window,
    sample_training_rows,
    score_frame,
    score_window,
    signed_log1p,
    training_candidates,
    update_best,
)
from src.quality import DataQualityError
from src.silver import SMART_COLUMNS

FEATURES = list(FEATURE_COLUMNS)
GOLD_SCHEMA = (
    "date date, serial_number string, model string, cohort_size int, "
    + ", ".join(f"{c} double" for c in FEATURES)
    + ", failure int, failure_within_30d int, label_is_complete boolean"
)
D7 = date(2026, 4, 7)


def day(offset):
    return D7 + timedelta(days=offset)


def grow(serial, when, feats=None, cohort_size=100, failure=0, label=0, complete=False, model="M1"):
    feats = feats or {}
    return (when, serial, model, cohort_size, *[feats.get(c, 0.0) for c in FEATURES], failure, label, complete)


def gold_df(spark, rows):
    return spark.createDataFrame(rows, GOLD_SCHEMA)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def drive_hash(serial, seed=42):
    return sha(f"{serial}|{seed}")


def row_hash(serial, when, seed=42):
    return sha(f"{serial}|{when}|{seed}")


def candidate_rows(n_drives=30, days=3):
    return [grow(f"D{i:03d}", day(d)) for i in range(n_drives) for d in range(days)]


def sample_pairs(df):
    return sorted((r["serial_number"], r["date"]) for r in df.collect())


# --- features and labels ---

def test_the_exact_21_feature_columns():
    expected = (
        [f"{a}_roll7_mean" for a in SMART_COLUMNS]
        + [f"{a}_delta_1d" for a in SMART_COLUMNS]
        + [f"{a}_cohort_z" for a in SMART_COLUMNS]
    )
    assert list(FEATURE_COLUMNS) == expected
    assert len(FEATURE_COLUMNS) == 21 and len(set(FEATURE_COLUMNS)) == 21


def test_no_label_raw_smart_cohort_size_or_obs_days_among_the_features():
    assert set(LABEL_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(SMART_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert not {"cohort_size", "obs_days_7d", "serial_number", "model", "date"} & set(FEATURE_COLUMNS)


def test_model_frame_is_identifiers_plus_the_21_features_and_refuses_labels(spark):
    df = gold_df(spark, [grow("A", day(0))])

    prepared = prepare_model_frame(df.select("serial_number", "date", "cohort_size", *FEATURES))
    assert prepared.columns == ["serial_number", "date", *FEATURES]

    with pytest.raises(ValueError, match="label columns"):
        prepare_model_frame(df)


def test_cohort_size_guard_blanks_only_z_scores_below_30(spark):
    rows = [
        grow("small", day(0), cohort_size=29, feats={c: 5.0 for c in FEATURES}),
        grow("exact", day(0), cohort_size=30, feats={c: 5.0 for c in FEATURES}),
    ]
    df = gold_df(spark, rows).select("serial_number", "date", "cohort_size", *FEATURES)

    out = {r["serial_number"]: r for r in prepare_model_frame(df).collect()}

    z_names = [c for c in FEATURES if c.endswith("_cohort_z")]
    other = [c for c in FEATURES if not c.endswith("_cohort_z")]
    assert all(out["small"][c] is None for c in z_names)
    assert all(out["small"][c] == 5.0 for c in other)  # roll means and deltas are untouched
    assert all(out["exact"][c] == 5.0 for c in z_names)


# --- training rows ---

def test_training_candidates_drop_only_failure_day_rows_and_carry_no_labels(spark):
    rows = [
        grow("A", day(0)),
        grow("A", day(1), failure=1, label=1),                 # failure day: removed
        grow("B", day(0), label=1, complete=False),            # pre-failure row: kept, label ignored
        grow("C", day(0), label=0, complete=True),
        grow("C", day(5)),                                     # outside the window
    ]

    out = training_candidates(gold_df(spark, rows), day(0), day(2))

    assert sample_pairs(out) == [("A", day(0)), ("B", day(0)), ("C", day(0))]
    assert set(out.columns).isdisjoint(LABEL_COLUMNS)


def test_training_selection_ignores_failure_within_30d_and_label_is_complete(spark):
    base = [grow(f"D{i}", day(0)) for i in range(6)]
    flipped = [grow(f"D{i}", day(0), label=1, complete=True) for i in range(6)]

    a = training_candidates(gold_df(spark, base), day(0), day(0))
    b = training_candidates(gold_df(spark, flipped), day(0), day(0))

    assert sample_pairs(sample_training_rows(a, cap=4)) == sample_pairs(sample_training_rows(b, cap=4))


def test_deterministic_drive_selection_matches_an_independent_sha256(spark):
    candidates = training_candidates(gold_df(spark, candidate_rows(30)), day(0), day(2))

    chosen = {r["serial_number"] for r in sample_training_rows(candidates, cap=10).collect()}

    serials = [f"D{i:03d}" for i in range(30)]
    expected = sorted(serials, key=lambda s: (drive_hash(s), s))[:10]
    assert chosen == set(expected)


def test_exactly_one_row_per_drive_chosen_by_the_row_hash(spark):
    candidates = training_candidates(gold_df(spark, candidate_rows(30, days=4)), day(0), day(3))

    out = sample_training_rows(candidates, cap=100).collect()

    assert len(out) == 30 and len({r["serial_number"] for r in out}) == 30
    for r in out:
        best_day = min((day(d) for d in range(4)), key=lambda x: (row_hash(r["serial_number"], x), x))
        assert r["date"] == best_day


def test_cap_limits_rows_and_a_larger_cap_keeps_every_drive(spark):
    candidates = training_candidates(gold_df(spark, candidate_rows(30)), day(0), day(2))

    assert sample_training_rows(candidates, cap=10).count() == 10
    assert sample_training_rows(candidates, cap=30).count() == 30
    assert sample_training_rows(candidates, cap=1000).count() == 30


def test_long_lived_drives_get_no_extra_chance_of_selection(spark):
    one_row_each = training_candidates(gold_df(spark, candidate_rows(30, days=1)), day(0), day(9))
    many_rows = training_candidates(gold_df(spark, candidate_rows(30, days=9)), day(0), day(9))

    few = {r["serial_number"] for r in sample_training_rows(one_row_each, cap=10).collect()}
    many = {r["serial_number"] for r in sample_training_rows(many_rows, cap=10).collect()}

    assert few == many


def test_same_seed_gives_the_same_sample_and_a_different_seed_can_change_it(spark):
    candidates = training_candidates(gold_df(spark, candidate_rows(30)), day(0), day(2))

    first = sample_pairs(sample_training_rows(candidates, cap=10, seed=42))
    again = sample_pairs(sample_training_rows(candidates, cap=10, seed=42))
    other = sample_pairs(sample_training_rows(candidates, cap=10, seed=43))

    assert first == again
    assert first != other


# --- preprocessing ---

def feature_frame(values, serials=None, dates=None):
    values = np.asarray(values, dtype="float64")
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, "date", dates if dates is not None else day(0))
    frame.insert(0, "serial_number", serials if serials is not None else [f"S{i}" for i in range(len(frame))])
    return frame


def test_signed_log1p_keeps_sign_zero_and_nan():
    out = signed_log1p(np.array([-3.0, 0.0, 3.0, np.nan]))

    assert out[0] == pytest.approx(-np.log1p(3.0))
    assert out[1] == 0.0
    assert out[2] == pytest.approx(np.log1p(3.0))
    assert np.isnan(out[3])
    assert signed_log1p(np.array([-1e12]))[0] == pytest.approx(-np.log1p(1e12))


def test_medians_come_from_the_training_matrix_only_and_ignore_nan():
    train = np.zeros((5, 21))
    train[:, 0] = [0, 1, 2, 3, 100]
    train[1, 1] = np.nan
    train[:, 1] = np.where(np.isnan(train[:, 1]), np.nan, 4.0)
    train[:, 1][[0, 2, 3, 4]] = [4.0, 6.0, 8.0, 10.0]

    pre = Preprocessor().fit(feature_frame(train))

    assert pre.medians_[0] == pytest.approx(np.log1p(2.0))
    assert pre.medians_[1] == pytest.approx(np.median(signed_log1p(np.array([4.0, 6.0, 8.0, 10.0]))))


def test_scoring_reuses_training_medians_and_never_refits_them():
    rng = np.random.default_rng(0)
    pre = Preprocessor().fit(feature_frame(rng.normal(size=(50, 21))))
    medians_before = pre.medians_.copy()
    chunk = rng.normal(loc=500.0, size=(4, 21))   # a very different distribution
    chunk[0, 3] = np.nan

    out = pre.transform(feature_frame(chunk))

    assert out[0, 3] == medians_before[3]                       # filled with the TRAINING median
    assert np.array_equal(pre.medians_, medians_before)         # unchanged by scoring
    assert out[1, 5] == pytest.approx(signed_log1p(np.array([chunk[1, 5]]))[0])
    assert not np.isnan(out).any()


def test_an_entirely_nan_training_feature_raises():
    values = np.zeros((10, 21))
    values[:, 7] = np.nan

    with pytest.raises(ValueError, match=FEATURES[7]):
        Preprocessor().fit(feature_frame(values))


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        Preprocessor().transform(feature_frame(np.zeros((1, 21))))


# --- model, scores, ranking ---

@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(1)
    return fit_model(feature_frame(rng.normal(scale=0.3, size=(400, 21))))


def test_a_planted_outlier_gets_the_highest_anomaly_score(trained):
    rng = np.random.default_rng(2)
    values = rng.normal(scale=0.3, size=(60, 21))
    values[17] = 25.0
    chunk = feature_frame(values, serials=[f"S{i}" for i in range(60)])

    scored = score_frame(trained, chunk)

    assert scored["anomaly_score"].idxmax() == 17
    assert scored["anomaly_score"].between(0, 1).all()
    assert scored["anomaly_score"][17] > scored["anomaly_score"].drop(17).max()


def test_same_data_and_seed_reproduce_identical_scores():
    rng = np.random.default_rng(3)
    train = feature_frame(rng.normal(size=(200, 21)))
    chunk = feature_frame(rng.normal(size=(30, 21)))

    first = score_frame(fit_model(train), chunk)["anomaly_score"].to_numpy()
    second = score_frame(fit_model(train), chunk)["anomaly_score"].to_numpy()

    assert np.array_equal(first, second)


def test_the_model_does_not_depend_on_the_order_of_the_training_rows():
    rng = np.random.default_rng(6)
    train = feature_frame(rng.normal(size=(300, 21)), dates=[day(i % 5) for i in range(300)])
    shuffled = train.sample(frac=1.0, random_state=0).reset_index(drop=True)
    chunk = feature_frame(rng.normal(size=(40, 21)))

    a = fit_model(train)
    b = fit_model(shuffled)

    assert np.array_equal(a.preprocessor.medians_, b.preprocessor.medians_)
    assert np.array_equal(score_frame(a, chunk)["anomaly_score"], score_frame(b, chunk)["anomaly_score"])


def test_chunked_scoring_equals_one_shot_scoring(trained):
    rng = np.random.default_rng(4)
    serials = [f"S{i % 25}" for i in range(75)]
    dates = [day(i // 25) for i in range(75)]
    everything = feature_frame(rng.normal(scale=0.5, size=(75, 21)), serials=serials, dates=dates)

    one_shot = score_window(trained, [everything])
    chunked = score_window(trained, [everything.iloc[:20], everything.iloc[20:51], everything.iloc[51:]])
    reversed_chunks = score_window(trained, [everything.iloc[51:], everything.iloc[:20], everything.iloc[20:51]])

    def table(result):
        return result["best"].sort_values("serial_number").reset_index(drop=True)

    pd.testing.assert_frame_equal(table(one_shot), table(chunked))
    pd.testing.assert_frame_equal(table(one_shot), table(reversed_chunks))
    assert one_shot["n_rows"] == chunked["n_rows"] == 75


def test_earliest_date_at_the_maximum_regardless_of_arrival_order():
    def rows(items):
        return pd.DataFrame(
            {"serial_number": [i[0] for i in items], "date": pd.to_datetime([i[1] for i in items]),
             "anomaly_score": [i[2] for i in items]}
        )

    early_tie, late_tie, higher_later = day(3), day(5), day(9)
    batch = rows([("X", late_tie, 0.7), ("X", early_tie, 0.7), ("X", day(1), 0.4), ("Y", day(2), 0.5), ("Y", higher_later, 0.6)])
    reverse = rows(list(reversed([(s, d, a) for s, d, a in zip(batch.serial_number, batch.date, batch.anomaly_score)])))

    best_batch = update_best(rows([]).iloc[0:0], batch).set_index("serial_number")
    best_reverse = update_best(rows([]).iloc[0:0], reverse).set_index("serial_number")

    assert best_batch.loc["X", "date"] == pd.Timestamp(early_tie)
    assert best_batch.loc["Y", "date"] == pd.Timestamp(higher_later)
    assert best_reverse.loc["X", "date"] == pd.Timestamp(early_tie)
    # merging in two steps gives the same answer
    first = update_best(rows([]).iloc[0:0], rows([("X", late_tie, 0.7)]))
    second = update_best(first, rows([("X", early_tie, 0.7)]))
    assert second.set_index("serial_number").loc["X", "date"] == pd.Timestamp(early_tie)


def test_ranking_is_score_desc_then_serial_asc_with_contiguous_ranks():
    best = pd.DataFrame(
        {"serial_number": ["C", "A", "B", "D"], "date": pd.to_datetime([day(0)] * 4),
         "anomaly_score": [0.6, 0.6, 0.9, 0.3]}
    )

    ranked = rank_best(best, top_n=3)

    assert list(ranked["serial_number"]) == ["B", "A", "C"]
    assert list(ranked["rank"]) == [1, 2, 3]


# --- end to end on tiny local data ---

def end_to_end_rows(labels=None):
    """60 training drives on days 0-2, 40 scored drives on days 3-5 plus one planted outlier."""
    rng = np.random.default_rng(5)
    labels = labels or {}
    rows = []
    for i in range(60):
        for d in range(3):
            rows.append(grow(f"T{i:02d}", day(d), feats={c: float(rng.normal(scale=0.3)) for c in FEATURES}))
    for i in range(40):
        for d in range(3, 6):
            rows.append(grow(f"S{i:02d}", day(d), feats={c: float(rng.normal(scale=0.3)) for c in FEATURES}, **labels.get("normal", {})))
    for d in range(3, 6):
        big = {c: 30.0 for c in FEATURES} if d == 4 else {c: 0.1 for c in FEATURES}
        rows.append(grow("OUT", day(d), feats=big, **labels.get("outlier", {"label": 1})))
    return rows


EVAL = dict(start=day(3), end=day(5), train_start=day(0), train_end=day(2), dataset_end=date(2026, 6, 30))


def run_tiny(spark, rows, cap=50, role="official", **overrides):
    window = {**EVAL, **overrides}
    return run_window(spark, gold_df(spark, rows), role, window["start"], window["end"],
                      window["train_start"], window["train_end"], window["dataset_end"], cap)


def test_end_to_end_ranks_the_planted_outlier_first_and_records_the_training(spark):
    result = run_tiny(spark, end_to_end_rows())
    run, rows = result["run"], result["top_rows"]

    assert rows[0]["serial_number"] == "OUT" and rows[0]["rank"] == 1
    assert rows[0]["date"] == day(4)
    assert [r["rank"] for r in rows] == list(range(1, 42))
    assert run["method"] == "isolation_forest" and run["window_role"] == "official"
    assert run["n_drives_scored"] == 41 and run["n_positive_drives"] == 1
    assert run["precision_at_50"] == pytest.approx(1 / 41) and run["recall_at_50"] == 1.0

    params = json.loads(run["params"])
    assert params["features"] == FEATURES
    assert params["forest"]["n_estimators"] == 200 and params["forest"]["random_state"] == 42
    assert params["forest"]["max_samples"] == 256 and params["forest"]["contamination"] == "auto"
    assert params["training"]["n_candidate_rows"] == 180
    assert params["training"]["n_candidate_drives"] == 60
    assert params["training"]["n_training_rows"] == 50
    assert params["training"]["window"] == [str(day(0)), str(day(2))]
    assert params["training"]["seed"] == 42 and params["training"]["cap"] == 200_000
    assert "failure_within_30d and label_is_complete are not used" in params["training"]["selection"]
    assert len(params["medians"]) == 21


def test_params_json_accepts_numpy_counts_as_returned_by_spark_connect(trained):
    counts = {"n_candidate_rows": np.int64(180), "n_candidate_drives": np.int64(60), "n_training_rows": np.int64(50),
              "n_rows_scored": np.int64(123)}

    params = json.loads(iforest.model_params(trained, {"train_start": day(0), "train_end": day(2)}, counts))

    assert params["training"]["n_candidate_rows"] == 180
    assert params["training"]["n_rows_scored"] == 123
    with pytest.raises(TypeError):
        iforest.model_params(trained, {"train_start": day(0), "train_end": day(2)}, {"bad": object()})


def test_training_must_end_before_the_evaluation_window_starts(spark):
    with pytest.raises(ValueError, match="before the evaluation window"):
        run_tiny(spark, end_to_end_rows(), train_end=day(3))


def test_scores_and_ranks_are_label_blind(spark):
    plain = end_to_end_rows()
    flipped = end_to_end_rows(labels={
        "normal": {"label": 1, "complete": True, "failure": 1},
        "outlier": {"label": 0, "complete": True},
    })
    # Training rows: flip failure_within_30d and label_is_complete but keep failure = 0, because
    # failure may legitimately remove a same-day failure row and would change the sample.
    flipped = [r[:-3] + (0, 1, True) if r[0] < day(3) else r for r in flipped]

    a = run_tiny(spark, plain)["top_rows"]
    b = run_tiny(spark, flipped)["top_rows"]

    assert [(r["rank"], r["serial_number"], r["date"], r["anomaly_score"]) for r in a] == \
           [(r["rank"], r["serial_number"], r["date"], r["anomaly_score"]) for r in b]
    assert a[0]["eval_failure_within_30d"] == 1 and b[0]["eval_failure_within_30d"] == 0  # labels did differ


def test_failure_day_rows_are_scored_and_a_selected_one_is_a_miss(spark):
    rows = end_to_end_rows(labels={"outlier": {"label": 1}})
    rows = [r[:-3] + (1, 1, False) if (r[1] == "OUT" and r[0] == day(4)) else r for r in rows]  # the peak row is the failure day

    result = run_tiny(spark, rows)

    assert result["top_rows"][0]["serial_number"] == "OUT" and result["top_rows"][0]["eval_failure"] == 1
    assert result["run"]["precision_at_50"] == 0.0   # selected failure-day row is not an early-warning hit


def test_anomaly_output_schema_matches_the_rule_baseline_and_nulls_the_rule_fields(spark):
    rows = end_to_end_rows()
    gold = gold_df(spark, rows)

    forest = run_tiny(spark, rows)
    rule = detect.detect_window(spark, gold, "official", day(3), day(5), date(2026, 6, 30))

    columns = lambda frame: [(f.name, f.dataType.simpleString()) for f in frame.schema.fields]
    assert columns(forest["anomalies"]) == columns(rule["anomalies"])
    assert forest["anomalies"].columns[-1] == "anomaly_score" == rule["anomalies"].columns[-1]
    assert forest["anomalies"].columns == [
        "run_id", "method", "rank", "serial_number", "model", "flagged_date", "score", "max_critical_z",
        "triggered_rules", "window_start", "window_end", "eval_failure", "eval_failure_within_30d",
        "eval_label_is_complete", "anomaly_score",
    ]
    for row in forest["anomalies"].collect():
        assert row["score"] is None and row["max_critical_z"] is None and row["triggered_rules"] is None
        assert 0 < row["anomaly_score"] < 1
    for row in rule["anomalies"].collect():
        assert row["anomaly_score"] == float(row["score"])


# --- run control: both models before any write ---

def tiny_windows():
    return (
        ("official", day(3), day(5), day(0), day(2)),
        ("supplemental", day(4), day(5), day(0), day(2)),
    )


def test_nothing_is_written_without_the_write_flag(spark, monkeypatch):
    monkeypatch.setattr(detect, "read_gold", lambda s: gold_df(spark, end_to_end_rows()))
    written = []

    results = run_isolation_forest(spark, write=False, write_fn=lambda s, r: written.append(r),
                                   windows=tiny_windows(), cap=50)

    assert written == []
    assert [r["run"]["window_role"] for r in results] == ["official", "supplemental"]


def test_with_write_both_windows_are_written_in_order(spark, monkeypatch):
    monkeypatch.setattr(detect, "read_gold", lambda s: gold_df(spark, end_to_end_rows()))
    written = []

    run_isolation_forest(spark, write=True, write_fn=lambda s, r: written.append(r["run"]["window_role"]),
                         windows=tiny_windows(), cap=50)

    assert written == ["official", "supplemental"]


def test_both_models_are_computed_before_anything_is_written(spark, monkeypatch):
    monkeypatch.setattr(detect, "read_gold", lambda s: gold_df(spark, end_to_end_rows()))
    computed, written = [], []
    real = iforest.run_window

    def second_window_fails(spark_, gold, role, *args, **kwargs):
        if role == "supplemental":
            assert computed == ["official"] and written == []
            raise DataQualityError("simulated check failure in the second window")
        computed.append(role)
        return real(spark_, gold, role, *args, **kwargs)

    monkeypatch.setattr(iforest, "run_window", second_window_fails)

    with pytest.raises(DataQualityError, match="simulated"):
        run_isolation_forest(spark, write=True, write_fn=lambda s, r: written.append(r),
                             windows=tiny_windows(), cap=50)

    assert computed == ["official"] and written == []


def test_main_persists_only_with_the_write_flag(monkeypatch):
    calls = []
    monkeypatch.setattr("src.config.get_spark_session", lambda: SimpleNamespace(stop=lambda: calls.append("stop")))
    monkeypatch.setattr(iforest, "run_isolation_forest", lambda spark, write=False, **kw: calls.append(("run", write)))

    monkeypatch.setattr(sys, "argv", ["iforest"])
    iforest.main()
    monkeypatch.setattr(sys, "argv", ["iforest", "--write"])
    iforest.main()

    assert calls == [("run", False), "stop", ("run", True), "stop"]


def test_the_two_windows_use_the_approved_training_dates():
    assert iforest.IF_WINDOWS == (
        ("official", date(2026, 6, 1), date(2026, 6, 30), date(2026, 4, 7), date(2026, 5, 31)),
        ("supplemental", date(2026, 5, 1), date(2026, 5, 31), date(2026, 4, 7), date(2026, 4, 30)),
    )
    assert iforest.TRAIN_CAP == 200_000 and iforest.SEED == 42
    assert iforest.FOREST_PARAMS == {
        "n_estimators": 200, "max_samples": 256, "max_features": 1.0,
        "contamination": "auto", "bootstrap": False, "random_state": 42,
    }
