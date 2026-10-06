# Changelog

## Unreleased - modular refactor line

### Changed

- `cellpaint_pipeline/skills.py` (1646 lines) and `cellpaint_pipeline/cli.py`
  (1604 lines) become the `skills/` and `cli/` packages, plus four new modules:
  `ports.py` (subprocess / environment / path ports), `registry.py` (one source
  of truth for the automation entry points), `capabilities.py` (the capability
  catalogue, which removes the reporting -> workflow dependency) and `errors.py`.
  Line counts are `len(text.splitlines())` on `src/cellpaint_pipeline/`.
- `run_pipeline_skill` accepts `**legacy_kwargs` instead of 36 explicit keyword
  parameters: `(config, skill_key, *, inputs=None, **legacy_kwargs)`.  Every
  historical keyword still works, the seven ignored ones still emit
  `DeprecationWarning`, and supplying the same input twice is still rejected.
  One observable difference: an unknown keyword now raises
  `TypeError: Unexpected keyword argument for run_pipeline_skill: <name>`
  instead of the interpreter's default message.
- `cellpaint_pipeline.skills` resolves its re-exports on first access (PEP 562)
  instead of importing the catalogue, the registry and every runner eagerly.
  Importing the package is roughly eight times cheaper, and importing a leaf
  such as `cellpaint_pipeline.skills.definitions` no longer pulls the package.

### Removed

- Namespace narrowing.  Splitting the two single-file modules drops the 31
  module-level names they leaked by accident.  `skills` no longer exposes
  `Any`, `Callable`, `Path`, `ProjectConfig`, `asdict`, `dataclass`, `datetime`,
  `field`, `is_dataclass`, `json`, `replace`, `timezone`; `evaluation` no longer
  exposes `PCA`, `StandardScaler`, `np`, `pairwise_distances`, `pd`, `plt`;
  `runner` no longer exposes `datetime`, `deque`, `os`, `re`, `subprocess`,
  `timezone`; `cli` no longer exposes `Path`, `argparse`, `import_module`,
  `json`; `adapters.deepprofiler_project` no longer exposes `os`.  All of them
  were implementation imports rather than API, and none was reachable through a
  `patch()` seam that any test relies on.  The two that did belong to the
  interface - `skills.ExecutionResult` and `cli.ProjectConfig` - are kept.

### Fixed

- `configs/*.json`: `data_access.data_cache_root` and `index_cache_root` are
  resolved relative to `workspace_root`, but the shipped values were written
  relative to the config directory, so both resolved to
  `<repo>/demo/demo/workspace/cache/...`, a path that cannot exist.
- Re-running the demo after renaming its folder left the recorded paths
  pointing at the old name; the shipped artefacts are re-pointed.

## 0.1.0 - 2026-03-24

### Added

- Stable release helper scripts:
  - `scripts/run_release_smoke_test.sh`
  - `scripts/build_release_bundle.sh`
- Release documentation:
  - `docs/release_quickstart.md`
  - `docs/first_run_guide.md`
  - `RELEASE_NOTES.md`
- Portable distribution config template:
  - `configs/project_config.portable.example.json`
- Env-driven OpenClaw provider setup for AutoDL and Docker tracks.

### Changed

- Rewrote release-facing documentation in English.
- Standardized OpenClaw guidance around TUI-first usage.
- Updated Docker integration to use generic OpenAI-compatible provider environment variables instead of provider-specific hardcoding.
- Updated release packaging to exclude local runtime state and secret-bearing files.

### Removed

- Secret-bearing OpenClaw backup configs from the repository-managed tree.
- Local temporary build artifacts, caches, and leftover agent runtime state.

### Validated

- Release smoke test passes on the current validated environment.
- Source release bundle generation completes successfully.
