---
doc_id: method:rule_baseline
source_type: method
title: Rule baseline detector
source_name: Project code, src/detect.py
source_reference: src/detect.py
source_version: git:HEAD
accessed_on: 2026-10-01
additional_sources: [{"name": "Stored run parameters, GOLD.MODEL_RUNS.params", "reference": "GOLD.MODEL_RUNS", "version": "per run", "accessed_on": "2026-10-01"}]
---
Rule baseline. For each drive-day, four critical attributes (SMART 5, 187, 197 and 198) can each earn two points. One point is for a cohort z-score above 3 against drives of the same model on the same date, counted only when that cohort has at least 30 drives. The other point is for a raw value that rose since the previous day. A missing value earns no point. The score is the number of points, from 0 to 8. For each drive the flagged date is the first day it reached its highest score. Drives are ranked by score, then by the largest critical z-score, then by serial number. The score counts triggers. It is not a probability and it does not say why the values changed.
