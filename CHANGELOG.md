# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] — 2026-05-15

### Fixed
- `MLFlowCallback` no longer crashes single-RUN jobs. The idempotent-resume branch added in 0.2.0 read the job number via `getattr(config.hydra.job, 'num', 0)`; in single-RUN mode Hydra leaves `hydra.job.num` mandatory-missing (`???`), and attribute access on such a node raises `MissingMandatoryValue` rather than `AttributeError`, so the `0` default never applied and any single-run launcher replay died on entry to this path. Now uses `OmegaConf.select(config, 'hydra.job.num', default=0)`, which returns the default for both absent and `???` values.

### Changed
- `on_job_end` now gates submitit-log collection on `config.hydra.mode == RunMode.MULTIRUN` (the explicit check used elsewhere in the callback) instead of `'num' in config.hydra.job`. The old check was not broken — `in` returns `False` for a `???` key — but relied on a subtle OmegaConf interaction; the explicit form is version-independent and self-documenting.
- Test suite now builds real OmegaConf configs instead of `MagicMock`/`SimpleNamespace`, so mandatory-missing (`???`) semantics are actually exercised. The mock-based helper is why the 0.2.0 regression shipped against a green suite.

## [0.2.0] — 2026-05-13

### Added
- `MLFlowCallback` is now idempotent across launcher retries. Replays from retry-on-failure launchers (Modal `function.retries`, Submitit `--requeue`) resume the same MLflow run rather than allocating a fresh child run per attempt — checkpoint and metric streams now survive container preemption. Implemented via a deterministic tag (`hydra_useful_callbacks_logical_key`) scoped by parent run id (nested multirun) or a host-allocated UUID (single-run / non-nested), set atomically on run creation and looked up via `mlflow.search_runs` on subsequent attempts.
- Integration tests for the new idempotent allocation against a real MLflow file store.

### Changed
- README rewritten to document each callback's behaviour, show typical YAML wiring, and update install/dev instructions for the current `uv`-based workflow.

[Unreleased]: https://github.com/joncarter1/hydra_useful_callbacks/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/joncarter1/hydra_useful_callbacks/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/joncarter1/hydra_useful_callbacks/releases/tag/v0.2.0
