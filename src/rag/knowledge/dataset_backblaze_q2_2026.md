---
doc_id: dataset:backblaze_q2_2026
source_type: dataset
title: Backblaze Drive Stats dataset, Q2 2026
source_name: Backblaze, "Hard Drive Reliability & Test Data"
source_reference: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
source_version: no version or date shown on the page
accessed_on: 2026-10-01
additional_sources: [{"name": "Project data profile of the loaded tables", "reference": "docs/kb_sources.md", "version": "bronze and silver tables, 2026-04-01 to 2026-06-30", "accessed_on": "2026-10-01"}]
---
Backblaze Drive Stats. Each daily file holds one record per drive per day, with basic drive information and the raw and normalized SMART statistics the drive reported. Backblaze publishes updates every quarter, and its terms say the user is solely responsible for how the data is used. This project uses one quarter, 2026-04-01 to 2026-06-30: 31,968,003 drive-day rows from 359,101 drives. Explanations in this project use SMART telemetry only. Outcome information recorded in the dataset is used to evaluate the detectors and is never used to explain a flag.
