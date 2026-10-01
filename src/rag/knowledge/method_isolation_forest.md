---
doc_id: method:isolation_forest
source_type: method
title: Isolation Forest detector
source_name: scikit-learn user guide, Novelty and Outlier Detection (version 1.9.1)
source_reference: https://scikit-learn.org/stable/modules/outlier_detection.html
source_version: scikit-learn 1.9.1
accessed_on: 2026-10-01
additional_sources: [{"name": "Project code, src/iforest.py", "reference": "src/iforest.py", "version": "git:HEAD", "accessed_on": "2026-10-01"}]
---
Isolation Forest. scikit-learn's IsolationForest isolates observations by randomly choosing a feature and then a split value between that feature's minimum and maximum. The number of splits needed to isolate a sample is its path length; averaged over a forest of random trees it is a measure of normality, and shorter paths point to likely anomalies. In this project the detector is trained without labels on drive-day rows dated before the evaluation window (failure-day rows are left out of training), using 21 features: the 7-day mean, the 1-day change and the same-model z-score of each of 7 SMART attributes, after a signed log transform and median imputation. The anomaly_score is the negative of scikit-learn's score_samples, so a higher score means a more unusual drive-day; that sign choice is this project's. For each drive the highest-scoring day is its flagged date. The detector scores the combination of all 21 features at once. It does not provide per-feature causal attribution. The score is not a probability.
