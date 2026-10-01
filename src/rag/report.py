"""Exact drive reports, built on demand from persisted rows.

A report is looked up by (run_id, serial_number). It is never embedded and never similarity
searched. It uses only the stored detector output and telemetry dated on or before the flagged
date. It never reads the failure columns, the label columns or any eval_* column: the SQL below
selects explicit column lists and the builder refuses telemetry from after the flagged date.
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Mapping, Optional, Sequence

ANOMALIES_TABLE = "DRIVE_HEALTH.GOLD.ANOMALIES"
MODEL_RUNS_TABLE = "DRIVE_HEALTH.GOLD.MODEL_RUNS"
FEATURES_TABLE = "DRIVE_HEALTH.GOLD.DRIVE_FEATURES"

ATTRIBUTES = (
    "smart_5_raw",
    "smart_9_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_198_raw",
)
CRITICAL = ("smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw")
TREND_DAYS = 7
MAX_RUNS = 4

# Anything that would carry an outcome. Used by tests and checked against every SQL string here.
FORBIDDEN_COLUMNS = ("failure", "failure_within_30d", "label_is_complete", "eval_failure",
                     "eval_failure_within_30d", "eval_label_is_complete")

FEATURE_COLUMNS = (
    "date",
    "serial_number",
    "model",
    "cohort_size",
    *ATTRIBUTES,
    *[f"{a}_roll7_mean" for a in ATTRIBUTES],
    *[f"{a}_delta_1d" for a in ATTRIBUTES],
    *[f"{a}_cohort_z" for a in ATTRIBUTES],
)

ANOMALY_COLUMNS = (
    "a.run_id", "a.method", "a.rank", "a.serial_number", "a.model", "a.flagged_date", "a.score",
    "a.max_critical_z", "a.triggered_rules", "a.anomaly_score", "a.window_start", "a.window_end",
    "m.window_role", "m.n_drives_scored", "m.params", "m.run_ts",
)

FEATURES_SQL = (
    f"SELECT {', '.join(FEATURE_COLUMNS)} FROM {FEATURES_TABLE} "
    f"WHERE serial_number = %s AND date BETWEEN DATEADD(day, -{TREND_DAYS - 1}, %s) AND %s ORDER BY date"
)
RUN_STATS_SQL = (
    f"SELECT COUNT(*), MIN(anomaly_score), MAX(anomaly_score) FROM {ANOMALIES_TABLE} WHERE run_id = %s"
)
SERIALS_SQL_TEMPLATE = f"SELECT DISTINCT serial_number FROM {ANOMALIES_TABLE} WHERE UPPER(serial_number) IN ({{marks}})"

RULE_STATEMENT = (
    "Each of four critical attributes can earn one point for a z-score above {z} versus same-model drives that day "
    "(only when the cohort has at least {cohort} drives) and one point for a raw value that rose since the previous day. "
    "A missing value earns no point. The score counts triggers; it is not a probability."
)
IF_STATEMENT = (
    "This detector scores the combination of all {n} features at once. It does not provide per-feature causal attribution. "
    "The values below are observed telemetry, not the model's reasons."
)
FOOTER = "No information after the flagged date is included."
LIMITS_LINES = (
    "This describes stored detector output and observed telemetry.",
    "It is not a diagnosis, and it does not predict failure.",
)
NOT_REPORTED = "not reported"
NOT_RECORDED = "not recorded"


def anomaly_sql(run_id: Optional[str] = None, method: Optional[str] = None) -> str:
    extra = (" AND a.run_id = %s" if run_id else "") + (" AND a.method = %s" if method else "")
    return (
        f"SELECT {', '.join(ANOMALY_COLUMNS)} FROM {ANOMALIES_TABLE} a JOIN {MODEL_RUNS_TABLE} m ON a.run_id = m.run_id "
        f"WHERE a.serial_number = %s{extra} "
        "ORDER BY CASE m.window_role WHEN 'official' THEN 0 ELSE 1 END, "
        "CASE a.method WHEN 'rule_baseline' THEN 0 ELSE 1 END, m.run_ts DESC LIMIT %s"
    )


def _dicts(cursor, sql: str, params: Sequence) -> List[Dict]:
    cursor.execute(sql, tuple(params))
    names = [d[0].lower() for d in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def sort_runs(rows: Sequence[Mapping]) -> List[Dict]:
    """Official runs first, then rule_baseline before isolation_forest, then newest."""
    def key(row):
        return (0 if row["window_role"] == "official" else 1, 0 if row["method"] == "rule_baseline" else 1)
    ordered = sorted(rows, key=lambda r: r.get("run_ts") or 0, reverse=True)
    return sorted(ordered, key=key)[:MAX_RUNS]


def fetch_runs(cursor, serial: str, run_id: Optional[str] = None, method: Optional[str] = None) -> List[Dict]:
    params = [serial] + ([run_id] if run_id else []) + ([method] if method else []) + [MAX_RUNS]
    return sort_runs(_dicts(cursor, anomaly_sql(run_id, method), params))


def fetch_features(cursor, serial: str, flagged_date: date) -> List[Dict]:
    return _dicts(cursor, FEATURES_SQL, (serial, flagged_date, flagged_date))


def fetch_run_stats(cursor, run_id: str) -> Dict:
    cursor.execute(RUN_STATS_SQL, (run_id,))
    count, low, high = cursor.fetchone()
    return {"stored": int(count), "min_score": low, "max_score": high}


def existing_serials(cursor, candidates: Sequence[str]) -> List[str]:
    """Which of these tokens are serial numbers with a stored anomaly result."""
    candidates = sorted({c.upper() for c in candidates})
    if not candidates:
        return []
    sql = SERIALS_SQL_TEMPLATE.format(marks=", ".join(["%s"] * len(candidates)))
    cursor.execute(sql, tuple(candidates))
    return sorted(row[0] for row in cursor.fetchall())


# --- formatting ---------------------------------------------------------------------------

def _int(value) -> str:
    return NOT_REPORTED if value is None else str(int(round(value)))


def _one(value) -> str:
    return NOT_REPORTED if value is None else f"{value:.1f}"


def _delta(value) -> str:
    return NOT_REPORTED if value is None else f"{int(round(value)):+d}"


def _g(value) -> str:
    return f"{value:g}"


def safe_params(method: str, params_json: Optional[str]) -> Dict:
    """Pick the few non-outcome settings the report needs; the raw JSON is never passed on."""
    params = json.loads(params_json) if params_json else {}
    if method == "rule_baseline":
        return {"z_threshold": params.get("z_threshold", 3.0), "min_cohort_size": params.get("min_cohort_size", 30)}
    training = params.get("training", {})
    return {"train_window": training.get("window"), "n_features": len(params.get("features", []))}


@dataclass
class DriveReport:
    doc_id: str
    run_id: str
    method: str
    serial_number: str
    flagged_date: date
    window_role: str
    text: str
    telemetry_lines: List[str] = field(default_factory=list)
    trend_lines: List[str] = field(default_factory=list)
    detector_lines: List[str] = field(default_factory=list)
    consistent: bool = True
    # The structured values the text was formatted from, so an answer's claims can be checked against
    # the right attribute and field instead of against the text as a whole.
    facts: Dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def report_id(run_id: str, serial: str) -> str:
    return f"report:{run_id[:8]}:{serial}"


def build_report(anomaly: Mapping, features: Sequence[Mapping], run_stats: Mapping) -> DriveReport:
    """Build the deterministic, method-aware report for one stored anomaly row."""
    flagged = anomaly["flagged_date"]
    if any(row["date"] > flagged for row in features):
        raise ValueError("telemetry dated after the flagged date must not be used")
    window = {row["date"]: row for row in features if row["date"] >= flagged - timedelta(days=TREND_DAYS - 1)}
    today = window.get(flagged)
    if today is None:
        raise ValueError(f"no telemetry for {anomaly['serial_number']} on the flagged date {flagged}")

    method = anomaly["method"]
    params = safe_params(method, anomaly.get("params"))
    doc_id = report_id(anomaly["run_id"], anomaly["serial_number"])
    cohort = today["cohort_size"]

    header = [
        f"DRIVE REPORT {doc_id}",
        # only the window's start: its end date is later than the flagged date, and a report must not carry later dates
        f"Run {anomaly['run_id'][:8]} | method {method} | window starting {anomaly['window_start']} ({anomaly['window_role']})",
        f"Drive {anomaly['serial_number']} | model {anomaly['model']} | flagged date {flagged} | "
        f"rank {anomaly['rank']} of {run_stats['stored']} stored (of {anomaly['n_drives_scored']} drives scored in the window)",
    ]

    z_of = lambda a: today[f"{a}_cohort_z"]
    delta_of = lambda a: today[f"{a}_delta_1d"]
    consistent = True
    detector: List[str] = []

    if method == "rule_baseline":
        z_limit, min_cohort = params["z_threshold"], params["min_cohort_size"]
        detector.append("Detector: rule_baseline. " + RULE_STATEMENT.format(z=_g(z_limit), cohort=min_cohort))
        detector.append(f"Score: {anomaly['score']} of 8")
        detector.append(f"Stored trigger list: {anomaly['triggered_rules']}")
        detector.append("Trigger check on the flagged date:")
        fired = 0
        for a in CRITICAL:
            z, d = z_of(a), delta_of(a)
            z_fired = z is not None and z > z_limit and cohort >= min_cohort
            inc_fired = d is not None and d > 0
            fired += int(z_fired) + int(inc_fired)
            detector.append(
                f"- {a}: z-score {_one(z)} ({'fired' if z_fired else 'not fired'}); "
                f"change since previous day {_delta(d)} ({'fired' if inc_fired else 'not fired'})"
            )
        if fired != anomaly["score"]:
            consistent = False
            detector.append(f"Note: recomputed trigger count {fired} differs from the stored score {anomaly['score']}.")
    else:
        low, high = run_stats["min_score"], run_stats["max_score"]
        detector.append(
            f"Detector: isolation_forest. anomaly_score {anomaly['anomaly_score']:.4f} (higher is more unusual). "
            f"The stored top {run_stats['stored']} scores for this run range from {low:.4f} to {high:.4f}."
        )
        detector.append(IF_STATEMENT.format(n=params["n_features"]))
        if params["train_window"]:
            detector.append(
                f"Training: non-failure-day rows dated {params['train_window'][0]} to {params['train_window'][1]}, "
                f"without labels, {params['n_features']} features."
            )
        by_z = sorted(
            ((a, z_of(a)) for a in ATTRIBUTES if z_of(a) is not None),
            key=lambda item: (-abs(item[1]), ATTRIBUTES.index(item[0])),
        )[:3]
        extremes = ", ".join(f"{a} z-score {_one(z)}" for a, z in by_z) or "none"
        rising = ", ".join(a for a in ATTRIBUTES if delta_of(a) is not None and delta_of(a) > 0) or "none"
        detector.append(f"Largest same-model z-scores on the flagged date (description of observed values): {extremes}")
        detector.append(f"Attributes whose raw value rose since the previous day: {rising}")

    telemetry = ["Telemetry on the flagged date (raw value | 7-day mean | change since previous day | z-score versus same-model drives):"]
    for a in ATTRIBUTES:
        telemetry.append(
            f"- {a}: {_int(today[a])} | {_one(today[f'{a}_roll7_mean'])} | {_delta(delta_of(a))} | {_one(z_of(a))}"
        )

    days = [flagged - timedelta(days=offset) for offset in range(TREND_DAYS - 1, -1, -1)]
    trend = [f"Raw values over the {TREND_DAYS} days up to the flagged date:"]
    for a in CRITICAL:
        cells = "; ".join(f"{d} = {_int(window[d][a]) if d in window else NOT_RECORDED}" for d in days)
        trend.append(f"- {a}: {cells}")

    facts = {
        "flagged_date": flagged,
        "method": method,
        "rank": anomaly["rank"],
        "score": anomaly["score"],
        "anomaly_score": anomaly["anomaly_score"],
        "attributes": {
            a: {"raw": today[a], "mean": today[f"{a}_roll7_mean"], "delta": delta_of(a), "z": z_of(a)}
            for a in ATTRIBUTES
        },
        # the report lists the 7-day trend for the critical attributes only; other attributes have today's value
        "raw_by_date": {
            a: ({d: (window[d][a] if d in window else None) for d in days} if a in CRITICAL else {flagged: today[a]})
            for a in ATTRIBUTES
        },
    }

    tail = [f"Cohort: {cohort} drives of this model reported on the flagged date.", FOOTER]
    text = "\n".join(header + detector + telemetry + trend + tail)
    return DriveReport(
        doc_id=doc_id, run_id=anomaly["run_id"], method=method, serial_number=anomaly["serial_number"],
        flagged_date=flagged, window_role=anomaly["window_role"], text=text, telemetry_lines=telemetry,
        trend_lines=trend, detector_lines=detector, consistent=consistent, facts=facts,
    )


def report_for_run(cursor, anomaly: Mapping) -> DriveReport:
    features = fetch_features(cursor, anomaly["serial_number"], anomaly["flagged_date"])
    return build_report(anomaly, features, fetch_run_stats(cursor, anomaly["run_id"]))


# --- deterministic evidence-only fallback ----------------------------------------------------

def evidence_only_answer(report: Optional[DriveReport], knowledge: Sequence[Mapping]) -> str:
    """The answer used when the LLM output cannot be validated: the report's own lines, with citations."""
    cite = f" [{report.doc_id}]" if report else ""
    if report is None:
        lines = ["## Answer"]
        lines += [f"- {k['title']} (source: {k['source_name']}) [{k['doc_id']}]" for k in knowledge]
        lines += ["", "## Limits", "No stored drive report applies to this question."] + list(LIMITS_LINES)
        return "\n".join(lines)

    lines = ["## Observed telemetry"]
    lines += [f"{line}{cite}" for line in report.telemetry_lines + report.trend_lines]
    lines += ["", "## How the detector scored this drive"]
    lines += [f"{line}{cite}" for line in report.detector_lines]
    lines += ["", "## General background (not specific to this drive)"]
    if knowledge:
        lines += [f"- {k['title']} (source: {k['source_name']}) [{k['doc_id']}]" for k in knowledge]
    else:
        lines.append("No relevant background documents were retrieved.")
    lines += ["", "## Limits"] + list(LIMITS_LINES)
    return "\n".join(lines)
