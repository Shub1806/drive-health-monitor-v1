---
doc_id: attr:smart_5_raw
source_type: attribute
title: SMART 5 - Reallocated Sectors Count
source_name: Backblaze blog, "What S.M.A.R.T. Hard Drive Errors Actually Tell Us About Failures" (Andy Klein)
source_reference: https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/
source_version: published 2016-10-06
accessed_on: 2026-10-01
additional_sources: [{"name": "Wikipedia, Self-Monitoring, Analysis and Reporting Technology", "reference": "https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology", "version": "no revision date shown", "accessed_on": "2026-10-01"}, {"name": "Project code, src/detect.py", "reference": "src/detect.py", "version": "git:HEAD", "accessed_on": "2026-10-01"}]
---
SMART 5, Reallocated Sectors Count. Background: the raw value counts bad sectors that the drive found and remapped. Lower is better. Backblaze notes that a small value, for example two remapped sectors, means little on its own. Project use: one of the four critical attributes (SMART 5, 187, 197 and 198) in the rule baseline.
