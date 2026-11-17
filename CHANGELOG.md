# Changelog

All notable changes to this project follow the [Keep a Changelog](https://keepachangelog.com/) format.

## [1.0.0] - 2026-11-17

Initial public release accompanying the FIDES manuscript.

### Added
- Stage 1 pretraining on SRPBS traveling subjects and HCP S1200.
- Stage 2 LOSO and 10-fold training on ABIDE and REST-meta-MDD.
- Cross-dataset evaluation on SRPBS multi-disorder.
- Four shortcut audits: site probe, identity fingerprint, scanner ICC, nuisance probes.
- Scanner-overlap analysis, scanner-disjoint Stage 1 retrain, unseen-site pools.
- Severity-subset analysis with ADOS (ABIDE) and HAMD-17 (REST-meta-MDD).
- Per-site statistical comparison with corrected tests and Holm correction.
- Ablation study over twelve variants.
- Pretrained Stage 1 and Stage 2 checkpoints hosted on Zenodo.
- Deterministic splits, seeds and baseline configs for every reported number.

### Known limitations
- Traveling-subject cohort is small (9 healthy men), so scanner invariance
  guarantees hold only on that cohort's support (see Section VII).
- Severity scores are missing for ~35 % of patients; the main training uses a
  diagnosis-only approximation (A5).
- Baseline reimplementations differ from the authors' original code in minor
  engineering details; grids and selections are documented under `baselines/`.
