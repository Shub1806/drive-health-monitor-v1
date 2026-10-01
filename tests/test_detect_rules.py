from datetime import date, timedelta

import pytest

import src.detect as detect
from src.detect import (
    CRITICAL_COLUMNS,
    LABEL_COLUMNS,
    MAX_SCORE,
    SCORING_INPUT_COLUMNS,
    add_trigger_text,
    attach_labels,
    base_rate,
    detect_window,
    evaluate_ranking,
    evaluation_note,
    is_early_warning_hit,
    precision_at_k,
    rank_top_drives,
    recall_at_k,
    run_detection,
    score_rows,
    select_flagged_rows,
    select_scoring_inputs,
    window_label_summary,
    window_labels,
)
from src.gold import delta_col, z_col
from src.quality import DataQualityError

JUNE = date(2026, 6, 1)
GOLD_SCHEMA = (
    "date date, serial_number string, model string, cohort_size int, "
    + ", ".join(f"{z_col(a)} double" for a in CRITICAL_COLUMNS)
    + ", "
    + ", ".join(f"{delta_col(a)} double" for a in CRITICAL_COLUMNS)
    + ", failure int, failure_within_30d int, label_is_complete boolean"
)


def grow(serial, offset, start=JUNE, model="M1", cohort_size=100, z=None, delta=None,
         failure=0, label=0, complete=False):
    z, delta = z or {}, delta or {}
    return (
        start + timedelta(days=offset),
        serial,
        model,
        cohort_size,
        *[z.get(a) for a in CRITICAL_COLUMNS],
        *[delta.get(a) for a in CRITICAL_COLUMNS],
        failure,
        label,
        complete,
    )


def gold_df(spark, rows):
    return spark.createDataFrame(rows, GOLD_SCHEMA)


def scored(spark, rows):
    inputs = select_scoring_inputs(gold_df(spark, rows), JUNE, JUNE + timedelta(days=60))
    return {(r["serial_number"], (r["date"] - JUNE).days): r for r in score_rows(inputs).collect()}


def pipeline(spark, rows, top_n=1000):
    gold = gold_df(spark, rows)
    inputs = select_scoring_inputs(gold, JUNE, JUNE + timedelta(days=60))
    top = rank_top_drives(select_flagged_rows(score_rows(inputs)), top_n)
    return [(r["rank"], r["serial_number"], r["date"], r["score"]) for r in top.orderBy("rank").collect()]


# --- label-blind by construction ---

def test_scoring_inputs_exclude_every_label_column():
    assert set(LABEL_COLUMNS).isdisjoint(SCORING_INPUT_COLUMNS)


def test_only_the_four_critical_attributes_feed_the_rule():
    names = " ".join(SCORING_INPUT_COLUMNS)
    assert all(a in names for a in CRITICAL_COLUMNS)
    assert all(a not in names for a in ("smart_9_raw", "smart_188_raw", "smart_194_raw"))
    assert CRITICAL_COLUMNS == ("smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw")


def test_select_scoring_inputs_drops_labels_and_other_columns(spark):
    gold = gold_df(spark, [grow("A", 0, failure=1, label=1)])

    assert tuple(select_scoring_inputs(gold, JUNE, JUNE).columns) == SCORING_INPUT_COLUMNS


def test_score_rows_refuses_a_frame_that_carries_labels(spark):
    with pytest.raises(ValueError, match="label columns"):
        score_rows(gold_df(spark, [grow("A", 0)]))


def test_flipping_labels_changes_no_score_flagged_date_or_rank(spark):
    signals = dict(z={"smart_5_raw": 5.0}, delta={"smart_5_raw": 2.0})
    healthy = [grow("A", i, **signals) for i in range(3)] + [grow("B", i, delta={"smart_197_raw": 1.0}) for i in range(3)]
    flipped = [grow("A", i, failure=1, label=1, complete=True, **signals) for i in range(3)] + [
        grow("B", i, failure=1, label=1, complete=True, delta={"smart_197_raw": 1.0}) for i in range(3)
    ]

    assert pipeline(spark, healthy) == pipeline(spark, flipped)


def test_attaching_labels_does_not_change_ranks(spark):
    rows = [
        grow("A", 0, z={"smart_5_raw": 9.0}, delta={"smart_5_raw": 1.0}, failure=1, label=1),
        grow("B", 0, delta={"smart_5_raw": 1.0}, label=1),
        grow("C", 0),
    ]
    gold = gold_df(spark, rows)
    top = rank_top_drives(select_flagged_rows(score_rows(select_scoring_inputs(gold, JUNE, JUNE))))
    before = [(r["rank"], r["serial_number"]) for r in top.orderBy("rank").collect()]

    after = attach_labels(top, window_labels(gold, JUNE, JUNE)).orderBy("rank").collect()

    assert [(r["rank"], r["serial_number"]) for r in after] == before
    assert after[0]["eval_failure"] == 1


# --- trigger logic ---

def test_each_trigger_adds_one_point_and_the_score_is_the_count(spark):
    rows = [
        grow("none", 0),
        grow("one_z", 0, z={"smart_5_raw": 3.5}),
        grow("one_inc", 0, delta={"smart_187_raw": 1.0}),
        grow("z_and_inc_same_attr", 0, z={"smart_197_raw": 4.0}, delta={"smart_197_raw": 2.0}),
        grow("all_eight", 0,
             z={a: 10.0 for a in CRITICAL_COLUMNS}, delta={a: 1.0 for a in CRITICAL_COLUMNS}),
    ]

    result = scored(spark, rows)

    assert result[("none", 0)]["score"] == 0
    assert result[("one_z", 0)]["score"] == 1
    assert result[("one_inc", 0)]["score"] == 1
    assert result[("z_and_inc_same_attr", 0)]["score"] == 2
    assert result[("all_eight", 0)]["score"] == MAX_SCORE == 8


def test_z_must_be_strictly_above_3_and_delta_strictly_positive(spark):
    rows = [grow("edge", 0, z={"smart_5_raw": 3.0}, delta={"smart_5_raw": 0.0}),
            grow("just_over", 0, z={"smart_5_raw": 3.0001}, delta={"smart_5_raw": 0.0}),
            grow("decrease", 0, delta={"smart_5_raw": -4.0})]

    result = scored(spark, rows)

    assert result[("edge", 0)]["score"] == 0
    assert result[("just_over", 0)]["score"] == 1
    assert result[("decrease", 0)]["score"] == 0


def test_null_values_never_trigger_and_give_no_max_z(spark):
    rows = [grow("nulls", 0)]  # every z and delta is NULL

    row = scored(spark, rows)[("nulls", 0)]

    assert row["score"] == 0
    assert row["max_critical_z"] is None


def test_null_z_does_not_block_other_triggers(spark):
    rows = [grow("A", 0, z={"smart_5_raw": None, "smart_197_raw": 6.0}, delta={"smart_5_raw": 2.0})]

    row = scored(spark, rows)[("A", 0)]

    assert row["score"] == 2
    assert row["max_critical_z"] == 6.0


def test_cohort_size_guard_applies_to_z_but_not_to_increases(spark):
    rows = [
        grow("small", 0, cohort_size=29, z={"smart_5_raw": 8.0}, delta={"smart_5_raw": 1.0}),
        grow("exact", 0, cohort_size=30, z={"smart_5_raw": 8.0}),
    ]

    result = scored(spark, rows)

    assert result[("small", 0)]["score"] == 1  # only the increase counts
    assert result[("exact", 0)]["score"] == 1  # 30 is large enough for the z trigger


def test_max_critical_z_is_the_largest_across_attributes(spark):
    rows = [grow("A", 0, z={"smart_5_raw": 1.0, "smart_187_raw": 7.5, "smart_198_raw": -2.0})]

    assert scored(spark, rows)[("A", 0)]["max_critical_z"] == 7.5


# --- flagged day and ranking ---

def test_flagged_day_is_the_earliest_day_at_the_peak(spark):
    same = dict(z={"smart_5_raw": 5.0})
    rows = [grow("A", 1, **same), grow("A", 3, **same), grow("A", 5, **same)]

    flagged = {r["serial_number"]: r["date"] for r in select_flagged_rows(score_rows(
        select_scoring_inputs(gold_df(spark, rows), JUNE, JUNE + timedelta(days=9)))).collect()}

    assert flagged["A"] == JUNE + timedelta(days=1)


def test_higher_score_beats_an_earlier_day_and_higher_z_breaks_score_ties(spark):
    rows = [
        grow("S", 1, z={"smart_5_raw": 9.0}),                      # score 1
        grow("S", 4, z={"smart_5_raw": 4.0}, delta={"smart_5_raw": 1.0}),  # score 2, later
        grow("Z", 1, z={"smart_5_raw": 4.0}),                      # score 1, z 4
        grow("Z", 6, z={"smart_5_raw": 8.0}),                      # score 1, z 8, later
    ]

    picked = {r["serial_number"]: r["date"] for r in select_flagged_rows(score_rows(
        select_scoring_inputs(gold_df(spark, rows), JUNE, JUNE + timedelta(days=9)))).collect()}

    assert picked["S"] == JUNE + timedelta(days=4)
    assert picked["Z"] == JUNE + timedelta(days=6)


def test_ranking_orders_by_score_then_z_then_serial_with_contiguous_ranks(spark):
    rows = [
        grow("D", 0, z={"smart_5_raw": 5.0}),
        grow("C", 0, z={"smart_5_raw": 5.0}),   # ties D on score and z; serial breaks the tie
        grow("B", 0, z={"smart_5_raw": 9.0}),   # higher z
        grow("A", 0, z={"smart_5_raw": 4.0}, delta={"smart_5_raw": 1.0}),  # score 2
        grow("E", 0),
    ]

    ranked = pipeline(spark, rows)

    assert [serial for _, serial, _, _ in ranked] == ["A", "B", "C", "D", "E"]
    assert [rank for rank, *_ in ranked] == [1, 2, 3, 4, 5]


def test_top_n_limits_the_stored_drives_and_one_row_per_drive(spark):
    rows = [grow(f"D{i:02d}", day, z={"smart_5_raw": 5.0}) for i in range(6) for day in range(3)]

    ranked = pipeline(spark, rows, top_n=4)

    assert len(ranked) == 4
    assert len({serial for _, serial, _, _ in ranked}) == 4


def test_trigger_text_names_the_fired_rules(spark):
    rows = [grow("A", 0, z={"smart_197_raw": 5.26}, delta={"smart_5_raw": 3.0}), grow("B", 0)]
    gold = gold_df(spark, rows)
    top = rank_top_drives(select_flagged_rows(score_rows(select_scoring_inputs(gold, JUNE, JUNE))))

    text = {r["serial_number"]: r["triggered_rules"] for r in add_trigger_text(top).collect()}

    assert "smart_5_raw increased by 3" in text["A"]
    assert "smart_197_raw z=5.3" in text["A"]
    assert text["B"] is None


# --- evaluation ---

def test_hit_requires_a_pre_failure_row_inside_the_30_day_window():
    assert is_early_warning_hit(0, 1) is True
    assert is_early_warning_hit(1, 1) is False  # selected row is the failure day
    assert is_early_warning_hit(0, 0) is False
    assert is_early_warning_hit(None, 1) is False


def test_failure_day_selection_is_label_blind_and_counts_as_a_miss(spark):
    rows = [
        # the failure-day row has the highest score, so it is selected without knowing it is a failure
        grow("F", 0, delta={"smart_5_raw": 1.0}, label=1),
        grow("F", 1, z={"smart_5_raw": 6.0}, delta={"smart_5_raw": 2.0}, failure=1, label=1),
        grow("H", 0, delta={"smart_5_raw": 1.0}, label=1),
    ]
    gold = gold_df(spark, rows)
    top = rank_top_drives(select_flagged_rows(score_rows(select_scoring_inputs(gold, JUNE, JUNE + timedelta(days=1)))))

    evaluated = attach_labels(top, window_labels(gold, JUNE, JUNE + timedelta(days=1))).orderBy("rank").collect()
    hits = {r["serial_number"]: is_early_warning_hit(r["eval_failure"], r["eval_failure_within_30d"]) for r in evaluated}

    assert evaluated[0]["serial_number"] == "F" and evaluated[0]["date"] == JUNE + timedelta(days=1)
    assert hits == {"F": False, "H": True}


def test_precision_recall_and_base_rate_arithmetic():
    flags = [True, False, True, True, False, False]

    assert precision_at_k(flags, 2) == 0.5
    assert precision_at_k(flags, 4) == 0.75
    assert recall_at_k(flags, 4, n_positive=6) == pytest.approx(0.5)
    assert base_rate(3, 1000) == 0.003


def test_fewer_drives_than_k_uses_every_drive_and_empty_input_is_none():
    flags = [True, False, False]

    assert precision_at_k(flags, 50) == pytest.approx(1 / 3)
    assert recall_at_k(flags, 50, n_positive=2) == 0.5
    assert precision_at_k([], 50) is None
    assert recall_at_k(flags, 50, n_positive=0) is None
    assert base_rate(0, 0) is None


def test_evaluate_ranking_reports_every_metric_and_no_accuracy():
    metrics = evaluate_ranking([True] * 10 + [False] * 90, n_positive=40, n_drives=1000, ks=(10, 50, 100))

    assert metrics["precision_at_10"] == 1.0
    assert metrics["precision_at_50"] == pytest.approx(0.2)
    assert metrics["recall_at_100"] == pytest.approx(10 / 40)
    assert metrics["base_rate"] == 0.04
    assert not any("accuracy" in key for key in metrics)


def test_window_label_summary(spark):
    rows = [
        grow("A", 0, label=1, complete=True),
        grow("A", 1, label=0, complete=True),
        grow("B", 0, label=0, complete=False),
        grow("B", 1, label=0, complete=False),
        grow("C", 0, label=1, complete=False, failure=1),
    ]

    summary = window_label_summary(window_labels(gold_df(spark, rows), JUNE, JUNE + timedelta(days=1)))

    assert summary["n_drives"] == 3
    assert summary["n_positive"] == 2  # A and C; C counts although only its failure-day row exists
    assert summary["label_complete_share"] == pytest.approx(2 / 5)


def test_censored_note_is_explicit_and_complete_note_is_not():
    june = evaluation_note(0.0, 340, date(2026, 6, 30))
    may = evaluation_note(1.0, 855, date(2026, 6, 30))

    assert "Censored" in june and "2026-06-30" in june and "lower bound" in june
    assert "precision@500 <= 68%" in june
    assert "Complete labels" in may and "Censored" not in may
    assert "<=" not in may


# --- one window and the whole run, on tiny local data ---

def window_rows():
    return [
        # outside the June window: must be ignored even though it scores 8
        grow("F1", -20, z={a: 9.0 for a in CRITICAL_COLUMNS}, delta={a: 1.0 for a in CRITICAL_COLUMNS}),
        # failing drive caught early: best day is 4 days before it fails
        grow("F1", 1, label=1),
        grow("F1", 4, z={"smart_5_raw": 5.0}, delta={"smart_5_raw": 2.0}, label=1),
        grow("F1", 8, delta={"smart_5_raw": 1.0}, failure=1, label=1),
        # failing drive whose failure-day row scores highest: not an early warning
        grow("F2", 2, delta={"smart_5_raw": 1.0}, label=1),
        grow("F2", 3, z={"smart_5_raw": 6.0, "smart_197_raw": 6.0}, delta={"smart_5_raw": 2.0}, failure=1, label=1),
        # noisy but healthy drive
        grow("N1", 5, z={"smart_5_raw": 4.0}, delta={"smart_5_raw": 1.0}),
        grow("OK", 5),
    ]


def test_detect_window_end_to_end(spark):
    gold = gold_df(spark, window_rows())

    result = detect_window(spark, gold, "official", JUNE, JUNE + timedelta(days=29), date(2026, 6, 30))
    run = result["run"]

    assert [r["serial_number"] for r in result["top_rows"]] == ["F2", "F1", "N1", "OK"]
    assert [r["rank"] for r in result["top_rows"]] == [1, 2, 3, 4]
    assert run["n_drives_scored"] == 4 and run["n_positive_drives"] == 2
    assert run["base_rate"] == 0.5
    assert run["precision_at_50"] == 0.25      # only F1 is an early-warning hit
    assert run["recall_at_50"] == 0.5
    assert run["window_role"] == "official" and run["window_labels_complete"] is False
    assert run["label_complete_row_share"] == 0.0
    assert "Censored" in run["evaluation_note"]
    assert tuple(result["anomalies"].columns) == (
        "run_id", "method", "rank", "serial_number", "model", "flagged_date", "score",
        "max_critical_z", "triggered_rules", "window_start", "window_end",
        "eval_failure", "eval_failure_within_30d", "eval_label_is_complete",
    )


def two_window_gold(spark):
    return gold_df(spark, window_rows() + [
        grow("M", 3, start=date(2026, 5, 1), z={"smart_5_raw": 5.0}, label=1, complete=True)])


def test_without_write_nothing_is_persisted(spark, monkeypatch):
    monkeypatch.setattr(detect, "read_gold", lambda s: two_window_gold(spark))
    written = []

    results = run_detection(spark, write=False, write_fn=lambda s, r: written.append(r))

    assert written == []
    assert [r["run"]["window_role"] for r in results] == ["official", "supplemental"]


def test_with_write_both_windows_are_written_in_order(spark, monkeypatch):
    monkeypatch.setattr(detect, "read_gold", lambda s: two_window_gold(spark))
    written = []

    run_detection(spark, write=True, write_fn=lambda s, r: written.append(r["run"]["window_role"]))

    assert written == ["official", "supplemental"]


def test_nothing_is_written_until_both_windows_have_passed_their_checks(spark, monkeypatch):
    monkeypatch.setattr(detect, "read_gold", lambda s: two_window_gold(spark))
    computed, written = [], []
    real_detect_window = detect.detect_window

    def second_window_fails(spark_, gold, role, start, end, dataset_end):
        if role == "supplemental":
            assert computed == ["official"] and written == []  # first window done, nothing persisted yet
            raise DataQualityError("simulated check failure in the second window")
        computed.append(role)
        return real_detect_window(spark_, gold, role, start, end, dataset_end)

    monkeypatch.setattr(detect, "detect_window", second_window_fails)

    with pytest.raises(DataQualityError, match="simulated"):
        run_detection(spark, write=True, write_fn=lambda s, r: written.append(r))

    assert computed == ["official"]
    assert written == []  # the first window passed, but the writer is never called


def test_a_failing_first_window_also_writes_nothing(spark, monkeypatch):
    monkeypatch.setattr(detect, "read_gold", lambda s: two_window_gold(spark))
    written = []

    def first_window_fails(*args, **kwargs):
        raise DataQualityError("simulated check failure in the first window")

    monkeypatch.setattr(detect, "detect_window", first_window_fails)

    with pytest.raises(DataQualityError):
        run_detection(spark, write=True, write_fn=lambda s, r: written.append(r))

    assert written == []


def test_main_does_not_persist_unless_write_flag_is_given(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    fake_spark = SimpleNamespace(stop=lambda: calls.append("stop"))
    monkeypatch.setattr("src.config.get_spark_session", lambda: fake_spark)
    monkeypatch.setattr(detect, "run_detection", lambda spark, write=False, **kw: calls.append(("run", write)))

    monkeypatch.setattr(sys, "argv", ["detect"])
    detect.main()
    monkeypatch.setattr(sys, "argv", ["detect", "--write"])
    detect.main()

    assert calls == [("run", False), "stop", ("run", True), "stop"]


def test_official_window_is_june_and_supplemental_is_may():
    assert detect.WINDOWS == (
        ("official", date(2026, 6, 1), date(2026, 6, 30)),
        ("supplemental", date(2026, 5, 1), date(2026, 5, 31)),
    )
