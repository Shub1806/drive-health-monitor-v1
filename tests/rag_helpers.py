"""Shared fakes and sample data for the RAG tests. Nothing here touches Snowflake."""
import json
from datetime import date, datetime, timedelta

from src.rag.report import ATTRIBUTES, CRITICAL

FLAGGED = date(2026, 6, 19)
RULE_RUN = "50fb2245-b713-4eb3-ad34-22c7fc4c8568"
IF_RUN = "311e6554-9daf-4a00-9b6d-a62701e8107c"
SERIAL = "ZHZ43WYS"
MODEL = "ST14000NM0138"
RULE_STATS = {"stored": 1000, "min_score": 3.0, "max_score": 8.0}
IF_STATS = {"stored": 1000, "min_score": 0.7124, "max_score": 0.8513}

# today's values for the four critical attributes: all eight rule triggers fire
CRITICAL_TODAY = {
    "smart_5_raw": dict(raw=100, delta=16.0, z=29.2),
    "smart_187_raw": dict(raw=40, delta=37.0, z=34.2),
    "smart_197_raw": dict(raw=600, delta=360.0, z=85.4),
    "smart_198_raw": dict(raw=600, delta=360.0, z=85.4),
}

RULE_PARAMS = json.dumps({
    "z_threshold": 3.0, "min_cohort_size": 30, "critical_attributes": list(CRITICAL),
    "hit": "flagged row has failure = 0 and failure_within_30d = 1", "top_n": 1000,
})
IF_PARAMS = json.dumps({
    "features": [f"{a}_{s}" for s in ("roll7_mean", "delta_1d", "cohort_z") for a in ATTRIBUTES],
    "forest": {"n_estimators": 200, "random_state": 42},
    "training": {
        "window": ["2026-04-07", "2026-05-31"],
        "selection": "failure = 0 only removes the same-day failure row; failure_within_30d and label_is_complete are not used",
        "seed": 42,
    },
    "hit": "flagged row has failure = 0 and failure_within_30d = 1",
})


def feature_row(day, cohort=500, **today):
    """One DRIVE_FEATURES row. `today` can override raw/delta/z per attribute."""
    row = {"date": day, "serial_number": SERIAL, "model": MODEL, "cohort_size": cohort}
    for a in ATTRIBUTES:
        spec = today.get(a, {})
        row[a] = spec.get("raw", 0)
        row[f"{a}_roll7_mean"] = spec.get("mean", 0.0)
        row[f"{a}_delta_1d"] = spec.get("delta", 0.0)
        row[f"{a}_cohort_z"] = spec.get("z", 0.0)
    return row


def window_features(flagged=FLAGGED, cohort=500, **overrides):
    rows = []
    for offset in range(6, -1, -1):
        day = flagged - timedelta(days=offset)
        rows.append(feature_row(day, cohort, **(overrides if offset == 0 else {})))
    return rows


def firing_features(**extra):
    spec = {a: dict(v, mean=v["raw"] / 2) for a, v in CRITICAL_TODAY.items()}
    spec.update(extra)
    return window_features(**spec)


def rule_anomaly(**overrides):
    row = {
        "run_id": RULE_RUN, "method": "rule_baseline", "rank": 1, "serial_number": SERIAL, "model": MODEL,
        "flagged_date": FLAGGED, "score": 8, "max_critical_z": 85.4, "anomaly_score": 8.0,
        "triggered_rules": "smart_5_raw z=29.2; smart_5_raw increased by 16; smart_187_raw z=34.2",
        "window_start": date(2026, 6, 1), "window_end": date(2026, 6, 30), "window_role": "official",
        "n_drives_scored": 356764, "params": RULE_PARAMS, "run_ts": datetime(2026, 10, 1, 1, 0),
    }
    row.update(overrides)
    return row


def if_anomaly(**overrides):
    row = rule_anomaly(run_id=IF_RUN, method="isolation_forest", score=None, max_critical_z=None,
                       triggered_rules=None, anomaly_score=0.8513, params=IF_PARAMS)
    row.update(overrides)
    return row


KNOWLEDGE = [
    {"doc_id": "attr:smart_197_raw", "title": "SMART 197 - Current Pending Sector Count",
     "text": "SMART 197 counts sectors waiting to be remapped. Lower is better.",
     "source_name": "Backblaze blog", "source_reference": "https://example.org/a", "source_version": "2016-10-06"},
    {"doc_id": "dataset:backblaze_q2_2026", "title": "Backblaze Drive Stats dataset",
     "text": "This project uses 31,968,003 drive-day rows from 359,101 drives.",
     "source_name": "Backblaze data page", "source_reference": "https://example.org/b", "source_version": None},
]


class FakeCursor:
    """Records every statement and answers from canned rules: (substring of the SQL, rows, column names)."""

    def __init__(self):
        self.executed = []
        self.rules = []
        self._rows = []
        self._columns = []
        self.rowcount = 0

    def respond(self, contains, rows, columns=None):
        self.rules.append((contains, rows, columns))
        return self

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        for contains, rows, columns in self.rules:
            if contains in sql:
                self._rows, self._columns = list(rows), columns or []
                return self
        self._rows, self._columns = [], []
        return self

    @property
    def description(self):
        return [(c,) for c in self._columns]

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def sql_text(self):
        return "\n".join(s for s, _ in self.executed)


ANOMALY_COLUMNS = ["run_id", "method", "rank", "serial_number", "model", "flagged_date", "score", "max_critical_z",
                   "triggered_rules", "anomaly_score", "window_start", "window_end", "window_role",
                   "n_drives_scored", "params", "run_ts"]
FEATURE_COLUMNS = list(feature_row(FLAGGED).keys())


def as_row(d, columns):
    return tuple(d[c] for c in columns)


def drive_cursor(anomalies, features=None, stats=(1000, 0.7124, 0.8513), serials=(SERIAL,)):
    """A cursor that knows one drive: its stored runs, telemetry, run statistics, and an empty cache."""
    cursor = FakeCursor()
    cursor.respond("SELECT DISTINCT serial_number", [(s,) for s in serials])
    cursor.respond("JOIN DRIVE_HEALTH.GOLD.MODEL_RUNS", [as_row(a, ANOMALY_COLUMNS) for a in anomalies], ANOMALY_COLUMNS)
    cursor.respond("FROM DRIVE_HEALTH.GOLD.DRIVE_FEATURES",
                   [as_row(f, FEATURE_COLUMNS) for f in (features or firing_features())], FEATURE_COLUMNS)
    cursor.respond("MIN(anomaly_score)", [stats])
    return cursor


class FakeClient:
    """Returns scripted answers, one per call, and records the messages it was given."""

    model = "fake-model"
    options = {"temperature": 0.0, "max_tokens": 900}

    def __init__(self, *answers, fail=False):
        from src.rag.llm import LLMResult

        self.answers = list(answers)
        self.calls = []
        self.fail = fail
        self._result = LLMResult

    def complete(self, messages):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("simulated model failure")
        text = self.answers.pop(0)
        return self._result(text, 100, 50, 7, self.model)
