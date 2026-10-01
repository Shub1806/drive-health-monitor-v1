-- Add a continuous score to GOLD.ANOMALIES so methods with float scores can share the table.
-- Run this once, BEFORE the next write from detect.py or iforest.py: both now append an
-- anomaly_score column, and the append fails if the table does not have it yet.
-- To undo: ALTER TABLE DRIVE_HEALTH.GOLD.ANOMALIES DROP COLUMN anomaly_score;

USE SCHEMA DRIVE_HEALTH.GOLD;

-- 1) Add the column. Existing rows keep every value; the new column starts out NULL.
ALTER TABLE DRIVE_HEALTH.GOLD.ANOMALIES ADD COLUMN IF NOT EXISTS anomaly_score FLOAT;

-- 2) Backfill the existing rule_baseline rows: their integer score is their anomaly score.
--    Safe to re-run; rows that already have a value are left alone.
UPDATE DRIVE_HEALTH.GOLD.ANOMALIES
SET anomaly_score = score
WHERE method = 'rule_baseline' AND anomaly_score IS NULL;

-- 3) Check. Expect every rule_baseline row to have a score.
SELECT method,
       COUNT(*) AS row_count,
       COUNT(anomaly_score) AS rows_with_anomaly_score,
       MIN(anomaly_score) AS min_anomaly_score,
       MAX(anomaly_score) AS max_anomaly_score
FROM DRIVE_HEALTH.GOLD.ANOMALIES
GROUP BY method;
