---
doc_id: attr:smart_187_raw
source_type: attribute
title: SMART 187 - Reported Uncorrectable Errors
source_name: Wikipedia, Self-Monitoring, Analysis and Reporting Technology
source_reference: https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology
source_version: no revision date shown
accessed_on: 2026-10-01
additional_sources: [{"name": "Backblaze blog, \"What S.M.A.R.T. Hard Drive Errors Actually Tell Us About Failures\" (Andy Klein)", "reference": "https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/", "version": "published 2016-10-06", "accessed_on": "2026-10-01"}, {"name": "Project data profile of GOLD.DRIVE_FEATURES", "reference": "docs/kb_sources.md", "version": "non-failure-day rows dated 2026-04-07 to 2026-05-31", "accessed_on": "2026-10-01"}, {"name": "Project code, src/detect.py", "reference": "src/detect.py", "version": "git:HEAD", "accessed_on": "2026-10-01"}]
---
SMART 187, Reported Uncorrectable Errors. Background: the raw value counts errors that could not be recovered using hardware ECC. Lower is better. Backblaze lists it among the SMART attributes it tracks. Project data: about 68% of drive-day rows (non-failure-day rows from 2026-04-07 to 2026-05-31) have no value for it. A missing value is treated as not reported and is never treated as zero. Project use: one of the four critical attributes in the rule baseline.
