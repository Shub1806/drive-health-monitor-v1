---
doc_id: attr:smart_198_raw
source_type: attribute
title: SMART 198 - Uncorrectable Sector Count
source_name: Backblaze blog, "What S.M.A.R.T. Hard Drive Errors Actually Tell Us About Failures" (Andy Klein)
source_reference: https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/
source_version: published 2016-10-06
accessed_on: 2026-10-01
additional_sources: [{"name": "Wikipedia, Self-Monitoring, Analysis and Reporting Technology", "reference": "https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology", "version": "no revision date shown", "accessed_on": "2026-10-01"}, {"name": "Project code, src/detect.py", "reference": "src/detect.py", "version": "git:HEAD", "accessed_on": "2026-10-01"}]
---
SMART 198, Uncorrectable Sector Count. Background: the raw value is the total count of uncorrectable errors when reading or writing a sector. Lower is better. Backblaze found it correlates well with SMART 197 and notes that not all manufacturers report both attributes. Project use: one of the four critical attributes in the rule baseline.
