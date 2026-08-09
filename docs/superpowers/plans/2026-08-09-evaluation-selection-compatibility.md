# Evaluation Selection Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select archived evaluations by checkpoint, scenario coverage, and delay while preserving all existing analysis outputs and adding terrain-adaptation outputs with explicit end-of-run audit warnings.

**Architecture:** Add one small, simulator-independent discovery module that owns evaluation metadata, ranking, and capability checks. Existing general and gait discovery use that shared ranking; `analyze_run_range.py` selects a primary source and capable gait/terrain fallbacks, then emits one consolidated audit after all outputs are attempted.

**Tech Stack:** Python 3, dataclasses, pathlib, JSON, pandas, NumPy, Matplotlib, pytest.

## Global Constraints

- Ranking order is highest checkpoint, greatest fixed-scenario count, zero action delay, then deterministic path order.
- Delayed, rebuttal-only, and single-scenario evaluations remain eligible.
- The selected primary evaluation supplies existing metrics; optional gait or terrain analysis may use the highest-ranked capable fallback and must disclose it.
- Do not change training, simulation, `scripts/eval.py`, existing metric formulas, or existing output filenames.
- Keep `walk_x_flat_terrain_1.0mps` versus `fast_walk_x_uneven_terrain` as the matched step-height and terrain pair.
- Keep the strict contact-force threshold greater than 1 N.
- Preserve the main checkout's uncommitted `fast_walk_x_uneven_terrain` rebuttal-scenario addition when applying the branch.
- Do not run `just eval-all`, training, or any GPU simulator command during implementation because VRAM is occupied.

---

### Task 1: Centralize evaluation metadata and ranking

**Files:**
- Create: `scripts/evaluation_discovery.py`
- Modify: `tests/test_evaluation_discovery.py`

**Interfaces:**
- Produces: `EvaluationMetadata(checkpoint: int | None, action_delay_steps: int, evaluation_scope: str, fixed_scenario_count: int)`.
- Produces: `extract_evaluation_metadata(summary: Mapping[str, Any], json_path: Path) -> EvaluationMetadata`.
- Produces: `evaluation_rank_key(entry: Any) -> tuple[int, int, int, str]`, suitable for ascending `sorted()`.
- Produces: `rank_evaluation_entries(entries: Iterable[T]) -> list[T]`.
- Produces: `select_preferred_evaluation(entries: Iterable[T]) -> T | None`.

- [ ] **Step 1: Replace the current preference tests with failing ranking tests**

Add literal fixtures to `tests/test_evaluation_discovery.py` that demonstrate the requested priority independently of implementation details:

```python
def test_higher_checkpoint_wins_even_with_fewer_scenarios(tmp_path: Path) -> None:
    older_full = _write_summary(
        tmp_path / "eval_checkpoint_100_seed_46_action_delay_0",
        _make_summary(checkpoint_scenarios=range(20), action_delay_steps=0),
    )
    newer_single = _write_summary(
        tmp_path / "eval_checkpoint_101_seed_46_action_delay_0_scenario_flat",
        _make_summary(
            checkpoint_scenarios=["flat"],
            action_delay_steps=0,
            selected_fixed_scenario="flat",
        ),
    )
    selected, _ = discover_metrics_summary_files(
        tmp_path, r"^cot_(\d+(?:\.\d+)?)$", (0.6, 1.6)
    )
    assert selected[(ENV_NAME, RUN_NAME)].json_path == newer_single.resolve()
    assert selected[(ENV_NAME, RUN_NAME)].json_path != older_full.resolve()


def test_more_scenarios_win_at_the_same_checkpoint(tmp_path: Path) -> None:
    restricted = _write_summary(
        tmp_path / "eval_checkpoint_100_seed_46_action_delay_0_rebuttal_scenarios",
        _make_summary(checkpoint_scenarios=range(5), action_delay_steps=0),
    )
    full = _write_summary(
        tmp_path / "eval_checkpoint_100_seed_46_action_delay_1",
        _make_summary(checkpoint_scenarios=range(21), action_delay_steps=1),
    )
    selected, _ = discover_metrics_summary_files(
        tmp_path, r"^cot_(\d+(?:\.\d+)?)$", (0.6, 1.6)
    )
    assert selected[(ENV_NAME, RUN_NAME)].json_path == full.resolve()
    assert selected[(ENV_NAME, RUN_NAME)].json_path != restricted.resolve()


def test_zero_delay_wins_only_after_checkpoint_and_scenario_count(tmp_path: Path) -> None:
    delayed = _write_summary(
        tmp_path / "eval_checkpoint_100_seed_46_action_delay_2",
        _make_summary(checkpoint_scenarios=range(21), action_delay_steps=2),
    )
    zero_delay = _write_summary(
        tmp_path / "eval_checkpoint_100_seed_46_action_delay_0",
        _make_summary(checkpoint_scenarios=range(21), action_delay_steps=0),
    )
    selected, _ = discover_metrics_summary_files(
        tmp_path, r"^cot_(\d+(?:\.\d+)?)$", (0.6, 1.6)
    )
    assert selected[(ENV_NAME, RUN_NAME)].json_path == zero_delay.resolve()
    assert selected[(ENV_NAME, RUN_NAME)].json_path != delayed.resolve()
```

Update `_make_summary` to write complete `fixed_command_scenarios`, `fixed_command_scenarios_metrics`, `selected_fixed_scenario`, and gait payload fields matching a real `eval.py` summary.

- [ ] **Step 2: Run the ranking tests and verify the current code fails for the intended reasons**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/pytest -q \
  tests/test_evaluation_discovery.py -k 'higher_checkpoint or more_scenarios or zero_delay'
```

Expected: the current full-scope-first and delay-exclusion behavior fails at least the higher-checkpoint and delayed-full cases.

- [ ] **Step 3: Implement the shared metadata and ranking module**

Create `scripts/evaluation_discovery.py` with the following behavior:

```python
@dataclass(frozen=True)
class EvaluationMetadata:
    checkpoint: int | None
    action_delay_steps: int
    evaluation_scope: str
    fixed_scenario_count: int


def extract_evaluation_metadata(summary: Mapping[str, Any], json_path: Path) -> EvaluationMetadata:
    checkpoint = extract_checkpoint(json_path)
    delay = int(summary.get("action_delay_steps", infer_delay_from_path(json_path)))
    if summary.get("selected_fixed_scenario"):
        scope = "single_scenario"
    elif summary.get("rebuttal_scenarios_only") is True:
        scope = "rebuttal_only"
    else:
        scope = "full"
    scenarios = summary.get("fixed_command_scenarios")
    metrics = summary.get("fixed_command_scenarios_metrics")
    scenario_count = len(scenarios) if isinstance(scenarios, list) else len(metrics) if isinstance(metrics, dict) else 0
    return EvaluationMetadata(checkpoint, delay, scope, scenario_count)


def evaluation_rank_key(entry: Any) -> tuple[int, int, int, str]:
    checkpoint = entry.checkpoint if entry.checkpoint is not None else -1
    return (-checkpoint, -entry.fixed_scenario_count, 0 if entry.action_delay_steps == 0 else 1, str(entry.json_path))


def rank_evaluation_entries(entries: Iterable[T]) -> list[T]:
    return sorted(entries, key=evaluation_rank_key)
```

Unparseable checkpoints rank below every parseable checkpoint but remain eligible. Path ordering resolves the final tie without raising.

- [ ] **Step 4: Use shared metadata in both run-data dataclasses**

Add `fixed_scenario_count: int` to `JsonRunData` and `RebuttalGaitRunData`. Remove the duplicate metadata parser from `gait_dynamics_aggregate.py`; both modules import from `evaluation_discovery.py`.

- [ ] **Step 5: Run the focused tests and verify green**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/pytest -q tests/test_evaluation_discovery.py
```

Expected: all discovery tests pass.

- [ ] **Step 6: Commit the shared ranking**

```bash
git add scripts/evaluation_discovery.py scripts/analyze_run_range.py \
  scripts/gait_dynamics_aggregate.py tests/test_evaluation_discovery.py
git commit -m "fix: rank evaluation artifacts consistently"
```

---

### Task 2: Select primary and capable gait/terrain sources

**Files:**
- Modify: `scripts/analyze_run_range.py`
- Modify: `scripts/gait_dynamics_aggregate.py`
- Modify: `scripts/terrain_adaptation_analysis.py`
- Modify: `tests/test_evaluation_discovery.py`
- Modify: `tests/test_terrain_adaptation_analysis.py`

**Interfaces:**
- Consumes: `rank_evaluation_entries()` and metadata from Task 1.
- Produces: `discover_metrics_summary_candidates(...) -> tuple[dict[tuple[str, str], list[JsonRunData]], pd.DataFrame]`.
- Produces: `select_json_run_candidates(candidates, capability=None) -> dict[tuple[str, str], JsonRunData]`.
- Produces: `discover_consumer_evaluations(root_dir, cot_scenario_pattern, cot_velocity_range, flat_tag, uneven_tag) -> tuple[primary_index, gait_index, terrain_index, manifest]`.
- Produces: `supports_terrain_analysis(entry: JsonRunData, flat_tag: str, uneven_tag: str) -> bool`.
- Existing `discover_metrics_summary_files(...)` remains a compatibility wrapper returning `(primary_index, manifest)`.

- [ ] **Step 1: Write failing tests for capable fallbacks**

Add tests with a higher-checkpoint single-scenario primary and a lower-checkpoint full default evaluation:

```python
def test_gait_uses_primary_when_primary_contains_gait_payload(tmp_path: Path) -> None:
    primary = _write_realistic_full_summary(tmp_path, checkpoint=101, delay=0, include_gait=True)
    _write_realistic_full_summary(tmp_path, checkpoint=100, delay=0, include_gait=True)
    primary_index, gait_index, _, _ = discover_consumer_evaluations(
        tmp_path, r"^cot_(\d+(?:\.\d+)?)$", (0.6, 1.6), FLAT_TAG, UNEVEN_TAG
    )
    key = (ENV_NAME, RUN_NAME)
    assert gait_index[key].json_path == primary_index[key].json_path == primary.resolve()


def test_gait_uses_highest_ranked_capable_fallback(tmp_path: Path) -> None:
    primary = _write_single_scenario_summary(tmp_path, checkpoint=101, include_gait=False)
    fallback = _write_realistic_full_summary(tmp_path, checkpoint=100, delay=0, include_gait=True)
    primary_index, gait_index, _, _ = discover_consumer_evaluations(
        tmp_path, r"^cot_(\d+(?:\.\d+)?)$", (0.6, 1.6), FLAT_TAG, UNEVEN_TAG
    )
    key = (ENV_NAME, RUN_NAME)
    assert primary_index[key].json_path == primary.resolve()
    assert gait_index[key].json_path == fallback.resolve()


def test_terrain_uses_highest_ranked_candidate_with_pair_and_arrays(tmp_path: Path) -> None:
    _write_single_scenario_summary(tmp_path, checkpoint=101, include_gait=True)
    terrain_fallback = _write_realistic_full_summary(
        tmp_path, checkpoint=100, delay=1, include_sim_data=True
    )
    _, _, terrain_index, _ = discover_consumer_evaluations(
        tmp_path, r"^cot_(\d+(?:\.\d+)?)$", (0.6, 1.6), FLAT_TAG, UNEVEN_TAG
    )
    assert terrain_index[(ENV_NAME, RUN_NAME)].json_path == terrain_fallback.resolve()
```

- [ ] **Step 2: Run the fallback tests and verify red**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/pytest -q \
  tests/test_evaluation_discovery.py -k 'uses_primary or capable_fallback or candidate_with_pair'
```

Expected: FAIL because candidate lists and capability-aware selection do not yet exist.

- [ ] **Step 3: Separate discovery from selection without breaking the public wrapper**

Refactor JSON discovery so it retains `dict[(env_name, run_name), list[JsonRunData]]`. Implement the existing `discover_metrics_summary_files()` as:

```python
def discover_metrics_summary_files(...):
    candidates, manifest = discover_metrics_summary_candidates(...)
    selected = select_json_run_candidates(candidates)
    return selected, mark_selected_candidates(manifest, selected, consumer="primary")
```

Every candidate remains eligible. Manifest rows include `fixed_scenario_count`, scope, delay, selected status, and the common ranking reason.

- [ ] **Step 4: Add capability-aware selection**

Use the same ranking after filtering:

```python
primary = select_json_run_candidates(candidates)
gait = select_json_run_candidates(candidates, capability=has_gait_dynamics_payload)
terrain = select_json_run_candidates(
    candidates,
    capability=lambda entry: supports_terrain_analysis(entry, flat_tag, uneven_tag),
)
```

`supports_terrain_analysis` requires both named scenarios in `fixed_command_scenarios`, matching commands, and `plots/sim_data.npz`. It does not load the NPZ during discovery; detailed validation remains in `analyze_terrain_sources()`.

- [ ] **Step 5: Feed consumer-specific sources into existing output code**

Keep `SeriesData.json_runs` and existing metrics tied to `primary`. Build gait series from `gait`. Build `TerrainRunSource` objects from `terrain`, retaining labels and date filters. Do not alter any metric formula or existing output filename.

- [ ] **Step 6: Run all focused tests and verify green**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/pytest -q \
  tests/test_evaluation_discovery.py tests/test_terrain_adaptation_analysis.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit capability-aware fallbacks**

```bash
git add scripts/analyze_run_range.py scripts/gait_dynamics_aggregate.py \
  scripts/terrain_adaptation_analysis.py tests/test_evaluation_discovery.py \
  tests/test_terrain_adaptation_analysis.py
git commit -m "feat: select capable analysis fallbacks"
```

---

### Task 3: Add final audit warnings and dynamic gait-scenario reporting

**Files:**
- Modify: `scripts/analyze_run_range.py`
- Modify: `scripts/gait_dynamics_aggregate.py`
- Modify: `tests/test_evaluation_discovery.py`
- Create: `tests/test_gait_dynamics_aggregate.py`

**Interfaces:**
- Consumes: primary, gait, and terrain selections from Task 2.
- Produces: `build_evaluation_audit_messages(...) -> list[str]`.
- Produces: `log_evaluation_audit_summary(messages: list[str]) -> None`.
- Produces: dynamic scenario-coverage text based on `REBUTTAL_DYNAMICS_SCENARIOS` and observed per-run rows.

- [ ] **Step 1: Write failing tests for duplicate, delay, and fallback audit messages**

Use real selection records rather than mocking logging internals:

```python
def test_audit_reports_duplicates_delayed_winner_and_consumer_fallback() -> None:
    messages = build_evaluation_audit_messages(
        candidates={KEY: [DELAYED_PRIMARY, ZERO_DELAY_OLDER]},
        primary={KEY: DELAYED_PRIMARY},
        gait={KEY: ZERO_DELAY_OLDER},
        terrain={KEY: DELAYED_PRIMARY},
        terrain_manifest=pd.DataFrame(),
    )
    joined = "\n".join(messages)
    assert "multiple evaluation candidates" in joined
    assert str(DELAYED_PRIMARY.json_path) in joined
    assert "action delay 1" in joined
    assert "gait-dynamics fallback" in joined
```

Add a separate test showing that a delayed candidate is selected—not excluded—when it has the highest checkpoint.

- [ ] **Step 2: Write a failing dynamic scenario-report test**

Create `tests/test_gait_dynamics_aggregate.py` with a five-scenario dataframe and call `_render_aggregate_report()`:

```python
def test_report_describes_observed_scenario_coverage_without_hardcoded_four(monkeypatch) -> None:
    monkeypatch.setattr(
        gait_dynamics_aggregate,
        "REBUTTAL_DYNAMICS_SCENARIOS",
        FIVE_SCENARIOS,
    )
    per_run = _five_scenario_per_run_dataframe()
    per_scenario, _, overall = aggregate_rebuttal_gait_dynamics(per_run)
    report = _render_aggregate_report(per_run, per_scenario, overall)
    assert "five scenarios" not in report.lower()
    assert "four scenarios" not in report.lower()
    assert "5 configured scenarios" in report
```

Also test a mixed four/five-scenario fixture produces a data-quality warning naming the incomplete run.

- [ ] **Step 3: Run the new tests and verify red**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/pytest -q \
  tests/test_evaluation_discovery.py tests/test_gait_dynamics_aggregate.py
```

Expected: FAIL because the audit builder does not exist and the report still hardcodes four scenarios.

- [ ] **Step 4: Implement pure audit-message construction**

Build messages after all series filters are known. Include:

- candidate count and selected primary path for every duplicate group;
- ranking summary `checkpoint > scenario count > zero delay > path`;
- each selected primary/gait/terrain source whose delay is greater than zero;
- each gait or terrain path differing from the primary path; and
- each skipped terrain manifest row with its warning.

Deduplicate identical delayed-path warnings while naming all consumers. Emit one final block immediately before `Finished. Outputs written to ...`:

```python
if audit_messages:
    logging.warning("Evaluation selection audit:\n%s", "\n".join(f"- {m}" for m in audit_messages))
else:
    logging.info("Evaluation selection audit: no duplicate, delay, fallback, or terrain-skip warnings.")
```

- [ ] **Step 5: Make gait report scenario wording and coverage dynamic**

Replace “four scenarios” with the configured count derived from `len(REBUTTAL_DYNAMICS_SCENARIOS)`. Add `fast_walk_x_uneven_terrain: "Uneven 1.0 m/s"` to `SCENARIO_DISPLAY_NAMES`. Report missing scenarios per `(label, run_name)` rather than only an aggregate row-count mismatch. Preserve all existing metrics and plot families.

- [ ] **Step 6: Run focused tests and verify green**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/pytest -q \
  tests/test_evaluation_discovery.py tests/test_gait_dynamics_aggregate.py \
  tests/test_terrain_adaptation_analysis.py
```

Expected: all tests pass with no unexpected warnings.

- [ ] **Step 7: Commit audit and dynamic reporting**

```bash
git add scripts/analyze_run_range.py scripts/gait_dynamics_aggregate.py \
  tests/test_evaluation_discovery.py tests/test_gait_dynamics_aggregate.py
git commit -m "feat: report evaluation selection audit"
```

---

### Task 4: Verify the default eval command and complete LP/LEP offline aggregation

**Files:**
- Verify only: `justfile`
- Verify only: `scripts/eval.py`
- Verify only: generated analysis directory under `figures/analysis_runs/`

**Interfaces:**
- Consumes: the user's LP/LEP paths and aggregate command.
- Produces: existing analysis artifacts plus `terrain_adaptation_*.csv`, `terrain_adaptation_report.md`, and `plot_terrain_adaptation.{pdf,png}`.

- [ ] **Step 1: Verify the proposed eval-all commands expand to default full evals**

Run dry-run expansion only:

```bash
just --dry-run eval-all '/local_storage1/loukas-backup/logs/clean_rl/cluster_runs/env_id_102_b2_no_energy_min_proper_seeding/constraints-as-terminations/logs/clean_rl/env_id_102_b2_no_energy_min_proper_seeding' 1
just --dry-run eval-all '/local_storage1/loukas-backup/logs/clean_rl/cluster_runs/env_id_97_regression_perf_shorter_push_disturbance_seeding/constraints-as-terminations/logs/clean_rl/env_id_97_regression_perf_shorter_push_disturbance_seeding' 1
```

Expected: the generated `just eval` calls do not include `--fixed_scenario`, `--rebuttal_scenarios_only`, `--skip_cot_sweep`, or a nonzero `--delay_actions`. Do not execute the commands without `--dry-run` during this task.

- [ ] **Step 2: Run static and focused verification**

Run:

```bash
git diff --check main...HEAD
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m py_compile \
  scripts/evaluation_discovery.py scripts/analyze_run_range.py \
  scripts/gait_dynamics_aggregate.py scripts/terrain_adaptation_analysis.py
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/pytest -q \
  tests/test_evaluation_discovery.py tests/test_gait_dynamics_aggregate.py \
  tests/test_terrain_adaptation_analysis.py
```

Expected: no diff-check or compilation errors and all focused tests pass.

- [ ] **Step 3: Run the exact CPU-only aggregate command on archived LP/LEP data**

Run:

```bash
MPLCONFIGDIR=/tmp/lp-lep-rebuttal-mpl \
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python scripts/analyze_run_range.py \
  --labels LP LEP \
  --env_names env_id_102_b2_no_energy_min_proper_seeding env_id_97_regression_perf_shorter_push_disturbance_seeding \
  --start_times ALL ALL \
  --end_times ALL ALL \
  --metrics_summary_root_dir '/local_storage1/loukas-backup/logs/clean_rl/cluster_runs/env_id_102_b2_no_energy_min_proper_seeding/constraints-as-terminations/logs/clean_rl/env_id_102_b2_no_energy_min_proper_seeding' \
  --metrics_summary_root_dir '/local_storage1/loukas-backup/logs/clean_rl/cluster_runs/env_id_97_regression_perf_shorter_push_disturbance_seeding/constraints-as-terminations/logs/clean_rl/env_id_97_regression_perf_shorter_push_disturbance_seeding' \
  --skip_wandb \
  --output_name lp_vs_lep_rebuttal_gait_dynamics \
  --export_formats pdf png \
  --plot_style corl
```

Expected: exit code 0. This reads archived JSON/NPZ files and does not initialize Isaac Sim or use GPU VRAM.

- [ ] **Step 4: Audit generated artifacts and selection manifests**

Confirm the newest output directory contains the existing files:

```text
per_run_summary.csv
aggregate_summary.csv
cot_sweep_stats.csv
step_height_boxplot_samples.csv
step_height_boxplot_summary.csv
rebuttal_gait_dynamics_per_run.csv
rebuttal_gait_dynamics_per_scenario.csv
rebuttal_gait_dynamics_overall.csv
plot_rebuttal_headline_dynamics.pdf
plot_rebuttal_support_dynamics.pdf
plot_rebuttal_base_excitation.pdf
plot_rebuttal_impact_and_joint_demand.pdf
plot_rebuttal_completion_rate.pdf
```

Confirm the additional terrain files:

```text
terrain_adaptation_per_scenario.csv
terrain_adaptation_paired.csv
terrain_adaptation_summary.csv
terrain_adaptation_manifest.csv
terrain_adaptation_report.md
plot_terrain_adaptation.pdf
plot_terrain_adaptation.png
```

Inspect `discovered_metrics_summary_files.csv`, `discovered_rebuttal_gait_dynamics_files.csv`, and the final log audit. For every seed, verify the selected checkpoint is maximal; within it, scenario count is maximal; within that tie, delay zero is preferred.

- [ ] **Step 5: Compare unchanged metric families for identical selected sources**

For runs where the pre-change and post-change manifests select the same JSON path, compare `per_run_summary.csv`, COT, violation, tracking, and gait CSV values with exact or floating-point-tolerant equality. Differences are allowed only when the selected source changed according to the approved ranking. Record source changes separately from metric changes.

- [ ] **Step 6: Verify the branch does not alter eval or training code**

Run:

```bash
git diff --exit-code main...HEAD -- scripts/eval.py scripts/clean_rl cat_envs
```

Expected: no diff. The later main-branch application must preserve the dirty `scripts/rebuttal_report.py` fifth-scenario line.

- [ ] **Step 7: Commit any final test-only corrections**

If verification required a correction, rerun Steps 2–6 and commit only the verified correction. If no correction was required, do not create an empty commit.
