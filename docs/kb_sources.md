# Knowledge document sources

Every document in `src/rag/knowledge/` carries its sources in its own front matter, and
`python -m src.rag.build_kb` refuses a document without them. This page lists what each source was
used for and how the project-data claims can be reproduced. All pages were read on 2026-10-01.
The page text was read through a summarising tool, so the wording of every claim stays close to
what the page said; please check the quoted passages against the pages before relying on them.

## External sources

| Source | Used for | Version or date shown |
|---|---|---|
| Backblaze, "What S.M.A.R.T. Hard Drive Errors Actually Tell Us About Failures" (Andy Klein), https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/ | SMART 5: a small value such as two remapped sectors means little on its own. SMART 197 and 198: they correlate well, and not all manufacturers report both. SMART 187 and 188: listed among the attributes Backblaze tracks | published 2016-10-06 |
| Wikipedia, Self-Monitoring, Analysis and Reporting Technology, https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology | What the raw value of attributes 5, 9, 187, 188, 194, 197 and 198 counts; lower is better; power-on hours can be in hours, minutes or seconds depending on the manufacturer; the temperature is in the lowest byte of the raw value | no revision date shown on the page |
| Backblaze, "Hard Drive Reliability & Test Data", https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data | One record per drive per day; basic drive information plus raw and normalized SMART statistics; quarterly updates; the user is solely responsible for how the data is used | no version or date shown on the page |
| scikit-learn user guide, Novelty and Outlier Detection, https://scikit-learn.org/stable/modules/outlier_detection.html | How IsolationForest isolates observations (random feature, random split value), path length as a measure of normality, shorter paths for anomalies | scikit-learn 1.9.1 |

## Project sources

| Source | Used for |
|---|---|
| `src/detect.py` | The rule baseline's triggers, score, flagged date and ranking |
| `src/iforest.py` | The Isolation Forest's features, training rows, preprocessing, score sign and flagged date |
| `GOLD.MODEL_RUNS.params` | The per-run thresholds, training window and feature list |

The method documents record the repository commit they were built from.

## What is not claimed

- The Backblaze data page does not define the `failure` column, so no document defines it.
- No attribute document claims that an attribute predicts failure beyond what Backblaze states.
- Nothing in the knowledge documents gives a cause for a particular drive's values.

## Reproducing the project-data claims

The claims in the SMART 187 and SMART 188 documents come from a profile of the Gold features over
the rows the detectors train on (non-failure-day rows, 2026-04-07 to 2026-05-31). This query
reproduces it. The figures below were produced by an equivalent profile run during development, not
by this exact text, so rerun it to confirm them before the documents are treated as final.

```sql
SELECT
    AVG(IFF(smart_187_raw_roll7_mean IS NULL, 1, 0)) AS share_187_missing,
    AVG(IFF(smart_188_raw_roll7_mean IS NULL, 1, 0)) AS share_188_missing,
    COUNT(*)                                          AS rows_counted,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY smart_188_raw_roll7_mean) AS smart_188_mean7_median,
    MAX(smart_188_raw_roll7_mean)                     AS smart_188_mean7_max
FROM DRIVE_HEALTH.GOLD.DRIVE_FEATURES
WHERE date BETWEEN '2026-04-07' AND '2026-05-31'
  AND failure = 0;
```

Figures from the development profile: about 68% of rows had no value (67.9% for the SMART 187
7-day mean, 68.0% for the SMART 188 7-day mean; a 7-day mean is missing only when the raw value is
missing). The SMART 188 7-day mean had a median of 0 and a maximum of about 3.2 trillion, which is
the basis for "differ by many orders of magnitude between rows". The dataset note's counts
(31,968,003 drive-day rows, 359,101 drives, 2026-04-01 to 2026-06-30) come from the bronze and
silver tables.

The `failure = 0` filter reproduces the same non-failure-day rows the detectors train on. It is used
here only to describe the data and never appears in an explanation.
