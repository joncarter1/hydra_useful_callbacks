# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-05-13

### Added
- `MLFlowCallback` is now idempotent across launcher retries. Replays from retry-on-failure launchers (Modal `function.retries`, Submitit `--requeue`) resume the same MLflow run rather than allocating a fresh child run per attempt — checkpoint and metric streams now survive container preemption. Implemented via a deterministic tag (`hydra_useful_callbacks_logical_key`) scoped by parent run id (nested multirun) or a host-allocated UUID (single-run / non-nested), set atomically on run creation and looked up via `mlflow.search_runs` on subsequent attempts.
- Integration tests for the new idempotent allocation against a real MLflow file store.

### Changed
- README rewritten to document each callback's behaviour, show typical YAML wiring, and update install/dev instructions for the current `uv`-based workflow.

[Unreleased]: https://github.com/joncarter1/hydra_useful_callbacks/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/joncarter1/hydra_useful_callbacks/releases/tag/v0.2.0
