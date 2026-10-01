---
doc_id: attr:smart_188_raw
source_type: attribute
title: SMART 188 - Command Timeout
source_name: Wikipedia, Self-Monitoring, Analysis and Reporting Technology
source_reference: https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology
source_version: no revision date shown
accessed_on: 2026-10-01
additional_sources: [{"name": "Backblaze blog, \"What S.M.A.R.T. Hard Drive Errors Actually Tell Us About Failures\" (Andy Klein)", "reference": "https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/", "version": "published 2016-10-06", "accessed_on": "2026-10-01"}, {"name": "Project data profile of GOLD.DRIVE_FEATURES", "reference": "docs/kb_sources.md", "version": "non-failure-day rows dated 2026-04-07 to 2026-05-31", "accessed_on": "2026-10-01"}]
---
SMART 188, Command Timeout. Background: the raw value counts operations the drive aborted on timeout. It should normally be zero. Backblaze lists it among the SMART attributes it tracks. Project data: about 68% of drive-day rows (same window) have no value, and where a value exists its 7-day mean differs by many orders of magnitude between rows, so this project compares a drive with same-model peers instead of reading the raw number alone. Project use: context attribute; it is not one of the four critical attributes.
