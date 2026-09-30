-- Drive Health Monitor: one-time account setup.
-- Safe to re-run; every statement uses IF NOT EXISTS / OR REPLACE where it cannot lose data.

-- Compute: smallest size, suspends after 60s idle so the trial credits last.
CREATE WAREHOUSE IF NOT EXISTS DRIVE_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- Storage layers (medallion): raw -> cleaned -> features/results.
CREATE DATABASE IF NOT EXISTS DRIVE_HEALTH;
CREATE SCHEMA IF NOT EXISTS DRIVE_HEALTH.BRONZE;
CREATE SCHEMA IF NOT EXISTS DRIVE_HEALTH.SILVER;
CREATE SCHEMA IF NOT EXISTS DRIVE_HEALTH.GOLD;

USE WAREHOUSE DRIVE_WH;
USE SCHEMA DRIVE_HEALTH.BRONZE;

-- Backblaze daily files: comma separated, one header row, blanks mean "not reported".
-- Column count mismatches are left as errors on purpose so a schema change fails the load loudly.
CREATE FILE FORMAT IF NOT EXISTS DRIVE_HEALTH.BRONZE.BACKBLAZE_CSV
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('')
    EMPTY_FIELD_AS_NULL = TRUE
    TRIM_SPACE = TRUE;

-- Landing area for the local CSVs (uploaded with PUT, gzip-compressed on upload).
CREATE STAGE IF NOT EXISTS DRIVE_HEALTH.BRONZE.RAW_STAGE
    FILE_FORMAT = DRIVE_HEALTH.BRONZE.BACKBLAZE_CSV
    COMMENT = 'Backblaze Drive Stats CSV uploads';
