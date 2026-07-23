# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]


### Fixed

- fix: drop Python 3.9 from test matrix ([#53](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/53)) by @purvanshjoshi on 2026-07-23

### Added

- chore: add CODEOWNERS file ([#57](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/57)) by @purvanshjoshi on 2026-07-23

### Documentation

- docs: update CHANGELOG.md (PR #57) ([#58](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/58)) by @github-actions[bot] on 2026-07-23
## [1.0.0] - 2026-07-16

### Added
- Phase 2 visualizations and CNN filter motifs ([#3](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/3))
- Tomtom Motif Matching Results ([#7](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/7))
- PDF format of Tomtom results ([#8](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/8))
- Benchmark AttentionCNN and add Tomtom TSV/XML ([#9](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/9))
- Update Tomtom PDF with Manual Visualizations ([#10](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/10))
- Solidify Phase 1 baselines with dinucleotide/motif features and negative sampling strategies ([#14](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/14))
- Feature ablation notebook and diagnostic plots ([#15](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/15))
- Phase 2 deep learning baselines with multi-class and cross-cell-type generalization ([#17](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/17))
- Tests CI workflow, negative sampling reproducibility seed, and experiment logger ([#21](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/21))
- Standalone feature modules and single-sample SHAP plots ([#22](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/22))
- 100% Phase 1 Completion (Logs, Analysis, Docs) ([#23](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/23))
- Auto changelog workflow on PR merge ([#24](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/24))
- Professional CI/CD workflows (Dependabot, mypy, coverage, pre-commit, matrix tests, Docker, notebooks, license check) ([#25](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/25))
- Contributing tip to README ([#40](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/40))

### Changed
- Bump `actions/upload-artifact` from 4 to 7 ([#31](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/31))
- Bump `docker/build-push-action` from 5 to 7 ([#30](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/30))
- Bump `docker/metadata-action` from 5 to 6 ([#29](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/29))
- Bump `docker/login-action` from 3 to 4 ([#28](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/28))
- Update `logomaker` requirement from >=0.8 to >=0.8.7 ([#35](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/35))
- Update `jupyter` requirement from >=1.0.0 to >=1.1.1 ([#34](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/34))
- Add requirements.txt ([#6](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/6))
- Restructure visualization folders ([#2](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/2))

### Fixed
- Fix typo in CONTRIBUTING.md ([#43](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/43))
- Changelog section insertion + remove PR creation ([#42](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/42))
- Changelog workflow creates PR (branch protection) ([#39](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/39))
- License-check format flag + mypy non-blocking ([#38](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/38))
- Replace semantic-release with tag-based release (branch protection) ([#37](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/37))
- Dependabot config and semantic-release version ([#26](https://github.com/PxA-Labs/interpretable-regulatory-genomics/pull/26))
