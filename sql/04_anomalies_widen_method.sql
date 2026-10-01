-- Widen ANOMALIES.method so method names longer than 'rule_baseline' (13 characters) fit.
-- The table was created from the first rule_baseline write, which sized the column to 13
-- characters; 'isolation_forest' is 16. Run this once, BEFORE the first Isolation Forest write.
-- Raising a VARCHAR length keeps every stored value; Snowflake does not allow lowering it again.

ALTER TABLE DRIVE_HEALTH.GOLD.ANOMALIES ALTER COLUMN method SET DATA TYPE VARCHAR(64);

-- Check: the method column should now be VARCHAR(64).
DESCRIBE TABLE DRIVE_HEALTH.GOLD.ANOMALIES;
